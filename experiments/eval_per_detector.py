#!/usr/bin/env python3

#Loads precomputed anomaly scores for one detector over a file list and computes TSB-AD evaluation metrics

import argparse
import os
import time

import numpy as np
import pandas as pd

try:
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank
    _TSB_AD_AVAILABLE = True
except ImportError:
    _TSB_AD_AVAILABLE = False


METRIC_KEYS = ['AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC',
               'Standard-F1', 'PA-F1', 'Event-based-F1',
               'R-based-F1', 'Affiliation-F']


def main():
    parser = argparse.ArgumentParser(description='Per-detector VUS-PR evaluation')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--file_list', type=str, required=True)
    parser.add_argument('--score_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--ad_name', type=str, default='Sub_IForest',
                        help='Detector name, or one of Random_C / Random_D / Oracle')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', dest='resume', action='store_true',
                       help='Resume from existing <save_dir>/<ad_name>.csv')
    group.add_argument('--no_resume', dest='resume', action='store_false',
                       help='Start from scratch (overwrite)')
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if not _TSB_AD_AVAILABLE:
        raise SystemExit("tsb-ad is required for evaluation: pip install 'tsb-ad>=1.5'")

    os.makedirs(args.save_dir, exist_ok=True)
    out_csv = os.path.join(args.save_dir, args.ad_name + '.csv')

    write_csv, done = [], set()
    if args.resume and os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        write_csv = prev.values.tolist()
        done = set(prev['file'].tolist())

    cols = ['file', 'Time'] + METRIC_KEYS
    file_list = pd.read_csv(args.file_list)['file_name'].values
    for filename in file_list:
        if filename in done:
            continue
        print(f'Evaluating {filename} by {args.ad_name}')
        t0 = time.time()
        df = pd.read_csv(os.path.join(args.dataset_dir, filename)).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        try:
            if args.ad_name == 'Random_C':
                anomaly_score = np.mean(
                    [np.random.uniform(size=data.shape[0]) for _ in range(5)], axis=0)
            elif args.ad_name == 'Random_D':
                anomaly_score = np.mean(
                    [(np.random.uniform(size=data.shape[0]) > 0.5).astype(float)
                     for _ in range(5)], axis=0)
            elif args.ad_name == 'Oracle':
                anomaly_score = label
            else:
                anomaly_score = np.load(
                    os.path.join(args.score_dir, args.ad_name, filename[:-4] + '.npy'))
            if len(anomaly_score) < len(label):
                pad = len(label) - len(anomaly_score)
                anomaly_score = np.pad(anomaly_score, (0, pad),
                                       'constant', constant_values=(0, anomaly_score[-1]))
            res = get_metrics(anomaly_score, label, slidingWindow=slidingWindow)
            list_w = [res.get(k, 0) for k in METRIC_KEYS]
        except Exception:
            list_w = [0] * len(METRIC_KEYS)
        write_csv.append([filename, round(time.time() - t0, 3)] + list_w)
        pd.DataFrame(write_csv, columns=cols).to_csv(out_csv, index=False)


if __name__ == '__main__':
    main()
