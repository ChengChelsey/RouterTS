#!/usr/bin/env python3
import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from routerts.config import (
    CANDIDATE_MODEL_SET_20,
    DEFAULT_GAP_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_EVAL_SLIDING_WINDOW,
)
from routerts.features.catch22 import split_ts, extract_catch22_features
from routerts.selector.routing import predict_cluster
from routerts.selector.predictor import (
    predict_detector_scores,
    gap_weighted_voting,
    load_anomaly_score,
)
from routerts.provenance.explainer import ClusterDecisionTracer

try:
    from TSB_AD.evaluation.metrics import get_metrics
    _TSB_AD_AVAILABLE = True
except ImportError:
    _TSB_AD_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def extract_features(data: np.ndarray, window_size: int) -> np.ndarray:
    # Extract catch22 features from a time series, returns (n_windows, 22)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    data_scaled = StandardScaler().fit_transform(data)
    data_split = split_ts(data_scaled, window_size=window_size)
    meta_features = np.vstack([extract_catch22_features(window) for window in data_split])
    meta_features = np.nan_to_num(meta_features, nan=0.0, posinf=0.0, neginf=0.0)
    return meta_features


def parse_args():
    parser = argparse.ArgumentParser(
        description='RouterTS Online Inference Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--classifier_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--file_list', type=str, required=True)
    parser.add_argument('--score_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--meta_csv', type=str, default=None,
                        help='Meta-feature CSV (required for --trace)')

    parser.add_argument('--domain', type=str, default='ID', choices=['ID', 'OOD'])
    parser.add_argument('--variant', type=str, default=None,
                        help='Single-domain LOO variant (e.g., WebService)')

    parser.add_argument('--gap_threshold', type=float, default=DEFAULT_GAP_THRESHOLD)
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE)

    parser.add_argument('--trace', action='store_true',
                        help='Enable decision provenance')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-processed files')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.trace and not args.meta_csv:
        parse_args().error('--trace requires --meta_csv')

    if args.domain == 'ID' and args.variant is not None:
        logger.warning('--variant is ignored when --domain is ID')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Loading classifier: {args.classifier_path}')
    with open(args.classifier_path, 'rb') as f:
        clf = pickle.load(f)
    n_clusters = clf['n_clusters']
    logger.info(f'Classifier loaded: {n_clusters} clusters, method={clf.get("method", "unknown")}')

    file_list_df = pd.read_csv(args.file_list)
    if 'file_name' not in file_list_df.columns:
        raise ValueError(f"file_list must have 'file_name' column, got: {list(file_list_df.columns)}")

    has_domain = 'domain_name' in file_list_df.columns
    if args.domain == 'OOD' and not has_domain:
        raise ValueError("--domain OOD requires 'domain_name' column in file_list")

    file_domain_map = {}
    if has_domain:
        file_domain_map = dict(zip(
            file_list_df['file_name'],
            file_list_df['domain_name'],
        ))

    if args.domain == 'OOD' and args.variant is not None:
        files_to_process = file_list_df[
            file_list_df['domain_name'] == args.variant
        ]['file_name'].tolist()
        output_name = f'OOD_{args.variant}'
        logger.info(f'Single-domain LOO: {len(files_to_process)} files from "{args.variant}"')
    elif args.domain == 'OOD':
        files_to_process = file_list_df['file_name'].tolist()
        output_name = 'OOD'
        logger.info(f'Auto LOO: {len(files_to_process)} files')
    else:
        files_to_process = file_list_df['file_name'].tolist()
        output_name = 'ID'
        logger.info(f'ID mode: {len(files_to_process)} files')

    output_csv = output_dir / f'{output_name}.csv'
    processed_files = set()
    if args.resume and output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        processed_files = set(existing_df['file'].tolist())
        logger.info(f'Resuming: {len(processed_files)} files already processed')

    tracer = None
    trace_output_dir = None
    trace_summary_rows = []

    if args.trace:
        trace_output_dir = output_dir / 'traces' / output_name
        trace_output_dir.mkdir(parents=True, exist_ok=True)
        tracer = ClusterDecisionTracer(
            classifier_path=args.classifier_path,
            model_dir=args.model_dir,
            meta_feature_path=args.meta_csv,
        )
        logger.info('Decision tracer initialized')

    columns = [
        'file', 'true_cluster', 'used_cluster', 'Time', 'flag',
    ]
    if _TSB_AD_AVAILABLE:
        columns += ['AUC-ROC', 'AUC-PR', 'VUS-ROC', 'VUS-PR',
                     'Standard-F1', 'PA-F1', 'Event-based-F1',
                     'R-based-F1', 'Affiliation-F']
    columns += list(CANDIDATE_MODEL_SET_20)

    write_rows = []
    model_dir = Path(args.model_dir)

    count = 0
    for filename in files_to_process:
        if filename in processed_files:
            continue

        count += 1
        start_time = time.time()

        file_path = os.path.join(args.dataset_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f'[Skip] File not found: {filename}')
            continue

        df = pd.read_csv(file_path).dropna()

        label_col = None
        for col_name in ['Label', 'label', 'is_anomaly', 'anomaly']:
            if col_name in df.columns:
                label_col = col_name
                break
        if label_col is None:
            label_col = df.columns[-1]

        feature_cols = [col for col in df.columns if col != label_col]
        data = df[feature_cols].values.astype(float)
        label = df[label_col].astype(int).to_numpy()

        meta_features = extract_features(data, args.window_size)
        cluster_id = predict_cluster(meta_features, clf=clf)

        if args.domain == 'ID':
            variant = 'ID'
        elif args.variant is not None:
            variant = args.variant
        else:
            variant = file_domain_map.get(filename, 'ID')

        model_path = model_dir / f'SATzilla_Cluster_C{cluster_id}' / f'{variant}.pkl'

        try:
            preds = predict_detector_scores(meta_features, str(model_path))
            selected_model, probs_dict, _ = gap_weighted_voting(
                preds, CANDIDATE_MODEL_SET_20, args.gap_threshold,
            )
            score = load_anomaly_score(
                filename, selected_model, args.score_dir, len(label),
            )
            flag = True

        except Exception as e:
            logger.error(f'[Error] {filename}: {e}')
            score = np.zeros(len(label))
            flag = False
            probs_dict = {m: 0.0 for m in CANDIDATE_MODEL_SET_20}

        end_time = time.time()
        run_time = end_time - start_time

        row = {
            'file': filename,
            'true_cluster': f'C{cluster_id}',
            'used_cluster': f'C{cluster_id}',
            'Time': round(run_time, 3),
            'flag': flag,
        }

        if _TSB_AD_AVAILABLE and flag:
            try:
                evaluation_result = get_metrics(score, label, slidingWindow=DEFAULT_EVAL_SLIDING_WINDOW)
                row['AUC-ROC'] = evaluation_result.get('AUC-ROC', 0)
                row['AUC-PR'] = evaluation_result.get('AUC-PR', 0)
                row['VUS-ROC'] = evaluation_result.get('VUS-ROC', 0)
                row['VUS-PR'] = evaluation_result.get('VUS-PR', 0)
                row['Standard-F1'] = evaluation_result.get('Standard-F1', 0)
                row['PA-F1'] = evaluation_result.get('PA-F1', 0)
                row['Event-based-F1'] = evaluation_result.get('Event-based-F1', 0)
                row['R-based-F1'] = evaluation_result.get('R-based-F1', 0)
                row['Affiliation-F'] = evaluation_result.get('Affiliation-F', 0)
            except Exception as e:
                logger.warning(f'[Metrics] Failed for {filename}: {e}')
                for col in ['AUC-ROC', 'AUC-PR', 'VUS-ROC', 'VUS-PR',
                             'Standard-F1', 'PA-F1', 'Event-based-F1',
                             'R-based-F1', 'Affiliation-F']:
                    row[col] = 0
        else:
            for col in ['AUC-ROC', 'AUC-PR', 'VUS-ROC', 'VUS-PR',
                         'Standard-F1', 'PA-F1', 'Event-based-F1',
                         'R-based-F1', 'Affiliation-F']:
                row[col] = 0

        for model_name in CANDIDATE_MODEL_SET_20:
            row[model_name] = probs_dict.get(model_name, 0.0)

        write_rows.append(row)

        if tracer is not None and flag and np.ndim(preds) == 2:
            try:
                scores_dict = {}
                for det_name in CANDIDATE_MODEL_SET_20:
                    if probs_dict.get(det_name, 0) > 0:
                        s = load_anomaly_score(filename, det_name, args.score_dir, len(label))
                        if np.any(s != 0):
                            scores_dict[det_name] = s

                trace = tracer.explain_full(
                    sample_id=filename.replace('.csv', ''),
                    catch22_windows=meta_features,
                    window_scores_matrix=preds,
                    gap_threshold=args.gap_threshold,
                    cluster_id=cluster_id,
                    window_scores_source_cluster_id=cluster_id,
                    score_source_policy='trust_input',
                    scores_dict=scores_dict if scores_dict else None,
                )

                trace_path = trace_output_dir / f'{filename.replace(".csv", "")}.trace.json'
                with open(trace_path, 'w') as f:
                    json.dump(asdict(trace), f, indent=2, default=str)

                trace_summary_rows.append({
                    'file': filename,
                    'true_cluster': cluster_id,
                    'used_cluster': cluster_id,
                    'routed_cluster': trace.routing.cluster_id,
                    'case_type': trace.selection.ambiguity.case_type,
                    'top1_weight': trace.selection.ambiguity.top1_weight,
                    'n_eff': trace.selection.ambiguity.n_eff,
                    'top1_detector': (trace.selection.ambiguity.sorted_weights[0][0]
                                      if trace.selection.ambiguity.sorted_weights else ''),
                    'margin': trace.routing.margin,
                })
            except Exception as e:
                logger.warning(f'[Trace] Failed for {filename}: {e}')

        if count % 10 == 0:
            logger.info(f'Progress: {count}/{len(files_to_process)} files processed')
            _save_results(write_rows, columns, output_csv)

    _save_results(write_rows, columns, output_csv)
    logger.info(f'Results saved to {output_csv} ({len(write_rows)} files)')

    if trace_summary_rows:
        summary_path = output_dir / f'{output_name}_trace_summary.csv'
        pd.DataFrame(trace_summary_rows).to_csv(summary_path, index=False)
        logger.info(f'Trace summary saved to {summary_path}')

    if write_rows and _TSB_AD_AVAILABLE:
        df_results = pd.DataFrame(write_rows)
        valid = df_results[df_results['flag'] == True]
        if len(valid) > 0:
            logger.info(f'=== {output_name} Summary ===')
            logger.info(f'  Files: {len(valid)}/{len(write_rows)}')
            for metric in ['VUS-PR', 'VUS-ROC', 'AUC-PR', 'AUC-ROC']:
                if metric in valid.columns:
                    logger.info(f'  {metric}: {valid[metric].mean():.4f} (mean)')


def _save_results(rows, columns, path):
    # Save or overwrite results CSV
    if not rows:
        return
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df[columns].to_csv(path, index=False)


if __name__ == '__main__':
    main()
