#!/usr/bin/env python3
"""Generate per-detector anomaly-score (.npy) files with the TSB-AD detectors.
Use this to produce the scores the online pipeline needs from a freshly
downloaded TSB-AD-U dataset. Example:

    python scripts/generate_scores.py \\
        --dataset_dir /path/to/TSB-AD-U \\
        --file_list data/test_split.csv \\
        --score_dir scores
        
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from routerts.config import CANDIDATE_MODEL_SET_20
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.model_wrapper import (
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool,
    run_Semisupervise_AD,
    run_Unsupervise_AD,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run TSB-AD detectors to generate .npy anomaly scores.",
    )
    p.add_argument(
        "--dataset_dir", required=True,
        help="Root of the TSB-AD-U CSV files, one time series per file.",
    )
    p.add_argument(
        "--file_list", required=True,
        help="CSV with a 'file_name' column (e.g. data/test_split.csv).",
    )
    p.add_argument(
        "--score_dir", required=True,
        help="Output root for scores, e.g. scores/ (<score_dir>/<Detector>/<stem>.npy).",
    )
    p.add_argument(
        "--ad_names", nargs="+", default=None,
        help="Detector names to run. Defaults to all 20 RouterTS candidates.",
    )
    p.add_argument(
        "--max_files", type=int, default=None,
        help="Optionally limit the number of time series (quick testing).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate scores even if the .npy already exists.",
    )
    p.add_argument(
        "--moment_win_size", type=int, default=None,
        help="Optional override for MOMENT_FT win_size (debug empty batch).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    score_root = Path(args.score_dir)
    ad_names = args.ad_names if args.ad_names else list(CANDIDATE_MODEL_SET_20)

    df_files = pd.read_csv(args.file_list)
    if "file_name" not in df_files.columns:
        raise ValueError("file_list must contain a 'file_name' column")
    file_names = df_files["file_name"].tolist()
    if args.max_files is not None:
        file_names = file_names[: args.max_files]

    for ad_name in ad_names:
        hp = Optimal_Uni_algo_HP_dict.get(ad_name, {}).copy()
        if ad_name == "MOMENT_FT" and args.moment_win_size is not None:
            hp["win_size"] = args.moment_win_size
        out_dir = score_root / ad_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== AD_Name: {ad_name} | HP: {hp} ===")

        for fname in file_names:
            csv_path = dataset_dir / fname
            stem = Path(fname).stem
            npy_path = out_dir / f"{stem}.npy"

            if npy_path.exists() and not args.overwrite:
                print(f"[Skip] {ad_name} / {fname} (already exists)")
                continue

            print(f"[Run ] {ad_name} on {fname}")
            try:
                df = pd.read_csv(csv_path).dropna()
                data = df.iloc[:, :-1].values.astype(float)

                if ad_name in Semisupervise_AD_Pool:
                    train_index = int(fname.split(".")[0].split("_")[-3])
                    data_train = data[:train_index, :]
                    output = run_Semisupervise_AD(ad_name, data_train, data, **hp)
                elif ad_name in Unsupervise_AD_Pool:
                    output = run_Unsupervise_AD(ad_name, data, **hp)
                else:
                    raise ValueError(f"{ad_name} is neither in the semi- nor unsupervised AD pool")

                if isinstance(output, str):
                    raise RuntimeError(output)
                scores = np.asarray(output, dtype=float)
                np.save(npy_path, scores)
            except Exception as exc:  # noqa: BLE001
                print(f"[Fail] {ad_name} on {fname}: {exc}")
                if ad_name == "MOMENT_FT":
                    print("[Fallback] writing zeros for MOMENT_FT")
                    n = len(df) if 'df' in locals() else 0
                    np.save(npy_path, np.zeros(n, dtype=float))
                continue


if __name__ == "__main__":
    main()
