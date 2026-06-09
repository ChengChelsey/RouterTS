#!/usr/bin/env python3

import argparse
import sys
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from routerts.config import CANDIDATE_MODEL_SET_20, DEFAULT_MIN_WINDOWS
from routerts.clustering.ward import load_all_window_features, train_ward_clustering
from routerts.clustering.tagc import select_optimal_k
from routerts.clustering.consolidation import load_dataset_window_counts


def load_algo_perf(meta_csv: str, unique_ds: list) -> np.ndarray:
    # Load per-dataset algorithm performance vectors from meta-feature CSV
    df = pd.read_csv(meta_csv, index_col=0)
    algo_cols = list(df.columns[:len(CANDIDATE_MODEL_SET_20)])
    ds_names = df.index.to_series().apply(lambda x: re.sub(r'_\d+$', '', x)).values

    algo_perf = []
    for ds in unique_ds:
        mask = ds_names == ds
        algo_perf.append(df.loc[mask, algo_cols].values.mean(axis=0))
    return np.array(algo_perf)


def main():
    parser = argparse.ArgumentParser(
        description='RouterTS Offline Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--meta_csv', type=str, required=True)
    parser.add_argument('--file_list', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--manual_k', type=int, default=None,
                        help='Skip TAGC and use this k directly')
    parser.add_argument('--k_range', type=int, nargs='+', default=None,
                        help='Candidate k values for TAGC (default: 5..15)')
    parser.add_argument('--min_windows', type=int, default=DEFAULT_MIN_WINDOWS)
    parser.add_argument('--skip_satzilla', action='store_true',
                        help='Skip model training (only run clustering)')

    args = parser.parse_args()

    print("=" * 70)
    print("RouterTS — Offline Training Pipeline")
    print("=" * 70)
    print(f"  meta_csv:   {args.meta_csv}")
    print(f"  file_list:  {args.file_list}")
    print(f"  output_dir: {args.output_dir}")
    print()

    print("[Step 1] Loading catch22 features...")
    unique_ds, raw_catch22, all_features, ds_names = load_all_window_features(args.meta_csv)
    print(f"  {len(unique_ds)} datasets, catch22 shape: {raw_catch22.shape}")

    ds_windows = load_dataset_window_counts(args.meta_csv, unique_ds)
    print(f"  Window counts loaded for {len(ds_windows)} datasets")

    algo_perf = load_algo_perf(args.meta_csv, unique_ds)
    print(f"  Algorithm performance matrix: {algo_perf.shape}")

    if args.manual_k is not None:
        best_k = args.manual_k
        print(f"\n[Step 2] Using manual k = {best_k} (TAGC skipped)")
    else:
        k_range = args.k_range or list(range(5, 16))
        print(f"\n[Step 2] TAGC k-selection (k_range = {k_range})...")
        best_k, scores_df = select_optimal_k(
            raw_catch22=raw_catch22,
            algo_perf=algo_perf,
            ds_windows=ds_windows,
            unique_ds=unique_ds,
            k_range=k_range,
            min_windows=args.min_windows,
        )
        print(f"\n  TAGC selected k* = {best_k}")
        print(f"\n  TAGC scores:")
        print(scores_df[['k', 'n_effective', 'tagc_score']].to_string(index=False))

        tagc_path = Path(args.output_dir) / 'tagc_scores.csv'
        tagc_path.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(tagc_path, index=False)
        print(f"\n  ✓ TAGC scores saved to {tagc_path}")

    print(f"\n[Step 3] Ward clustering (k={best_k})...")
    result = train_ward_clustering(
        unique_ds=unique_ds,
        raw_catch22=raw_catch22,
        ds_windows=ds_windows,
        k=best_k,
        output_dir=args.output_dir,
        meta_csv=args.meta_csv,
        file_list=args.file_list,
        min_windows=args.min_windows,
        train_satzilla_flag=not args.skip_satzilla,
    )

    print("\n" + "=" * 70)
    print("Offline Training Complete")
    print("=" * 70)
    print(f"  Config:          agg_raw_k{best_k}")
    print(f"  Effective k:     {result['n_clusters']}")
    merged_str = " (clusters were consolidated)" if result.get('merged') else ""
    print(f"  Merged:          {result.get('merged', False)}{merged_str}")
    print(f"  Cluster sizes:   {[int((result['labels'] == c).sum()) for c in range(result['n_clusters'])]}")
    print(f"  Output dir:      {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
