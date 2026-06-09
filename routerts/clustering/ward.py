"""Ward hierarchical clustering with micro-cluster consolidation and per-cluster model training."""

import re
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from routerts.config import CANDIDATE_MODEL_SET_20, CATCH22_NAMES, DOMAINS
from routerts.clustering.consolidation import (
    merge_micro_clusters,
    compute_cluster_window_counts,
)


def load_all_window_features(meta_csv: str) -> tuple:
    df = pd.read_csv(meta_csv, index_col=0)
    feat_cols = [f'val_{i}' for i in range(22)]
    features = np.nan_to_num(df[feat_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    ds_names = df.index.to_series().apply(lambda x: re.sub(r'_\d+$', '', x)).values
    unique_ds = sorted(set(ds_names))

    raw_catch22 = []
    for ds in unique_ds:
        mask = ds_names == ds
        raw_catch22.append(features[mask].mean(axis=0))

    return unique_ds, np.array(raw_catch22), features, ds_names


def save_classifier(path: str, clf_dict: dict) -> None:
    # Save classifier dict to pkl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(clf_dict, f)
    print(f"  ✓ classifier → {path}")


def save_cluster_features(path: str, raw_catch22: np.ndarray, labels: np.ndarray) -> None:
    if raw_catch22 is None:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_clusters = len(np.unique(labels))
    global_mean = raw_catch22.mean(axis=0)
    global_std = raw_catch22.std(axis=0)
    global_std[global_std == 0] = 1.0

    all_means = np.array([raw_catch22[labels == c].mean(axis=0) for c in range(n_clusters)])
    discrimination = all_means.max(axis=0) - all_means.min(axis=0)

    rows = []
    for c in range(n_clusters):
        mask = labels == c
        cluster_mean = raw_catch22[mask].mean(axis=0)
        z_scores = (cluster_mean - global_mean) / global_std

        significant = np.where(np.abs(z_scores) > 0.5)[0]
        sig_idx = significant[np.argsort(np.abs(z_scores[significant]))[::-1]][:5]
        sig_desc = "; ".join(
            f"{CATCH22_NAMES[i]} ({'high' if z_scores[i] > 0 else 'low'}, {abs(z_scores[i]):.1f} std)"
            for i in sig_idx
        ) if len(sig_idx) > 0 else "No significant features"

        combined = np.abs(z_scores) * discrimination
        top_indices = np.argsort(combined)[::-1][:3]
        disc_feats = []
        for idx in top_indices:
            if discrimination[idx] > 1.0:
                disc_feats.append(f"{CATCH22_NAMES[idx]} (Δz={discrimination[idx]:.2f}, z={z_scores[idx]:.2f})")
        disc_desc = "; ".join(disc_feats) if disc_feats else "None"

        rows.append({
            'cluster': f'C{c}',
            'n_datasets': int(mask.sum()),
            'signature_features': sig_desc,
            'high_discrimination_features': disc_desc,
        })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  ✓ cluster_features → {Path(path).name}")


def save_cluster_list(path: str, dataset_names: list, labels: np.ndarray) -> str:
    # Save cluster-dataset mapping CSV
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {'cluster': f'C{labels[i]}', 'dataset_name': name.replace('.csv', '') if name.endswith('.csv') else name}
        for i, name in enumerate(dataset_names)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    n_clusters = len(np.unique(labels))
    sizes = [int((labels == c).sum()) for c in range(n_clusters)]
    print(f"  ✓ cluster_list → {Path(path).name} ({len(df)} entries, distribution: {sizes})")
    return str(path)


def train_satzilla(
    config_name: str,
    cluster_list_path: str,
    n_clusters: int,
    meta_csv: str,
    file_list: str,
    weights_dir: str,
    domains: list = None,
) -> None:

    import subprocess

    if domains is None:
        domains = DOMAINS

    satzilla_script = str(Path(__file__).resolve().parents[2] / 'routerts' / 'selector' / 'trainer.py')
    save_root = Path(weights_dir) / config_name
    total = n_clusters * len(domains)

    print(f"  Training SATzilla ({n_clusters} clusters × {len(domains)} domains = {total} models, parallel)...")

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['NUMEXPR_NUM_THREADS'] = '1'

    project_root = str(Path(__file__).resolve().parents[2])
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{project_root}:{existing}" if existing else project_root

    procs = []
    for ci in range(n_clusters):
        cluster = f"C{ci}"
        save_dir = str(save_root / f"SATzilla_Cluster_C{ci}")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        for domain in domains:
            cmd = [
                os.sys.executable, satzilla_script,
                '--data_path', meta_csv,
                '--file_list', file_list,
                '--save_dir', save_dir,
                '--classifier', 'random_forest',
                '--domain', domain,
                '--cluster_list', cluster_list_path,
                '--cluster', cluster,
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            procs.append((proc, f"{cluster}/{domain}"))

    failed = []
    for proc, label in procs:
        ret = proc.wait()
        if ret != 0:
            failed.append(f"  {label}: exit={ret}")

    if failed:
        print(f"  ✗ {len(failed)} failed:")
        for f in failed:
            print(f)
    else:
        print(f"  ✓ SATzilla: {total} models → {config_name}/")


def train_ward_clustering(
    unique_ds: list,
    raw_catch22: np.ndarray,
    ds_windows: dict,
    k: int,
    output_dir: str,
    config_name: str = None,
    meta_csv: str = None,
    file_list: str = None,
    min_windows: int = 100,
    train_satzilla_flag: bool = True,
) -> dict:
   
    if config_name is None:
        config_name = f"agg_raw_k{k}"

    output_path = Path(output_dir)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_catch22)

    agg = AgglomerativeClustering(n_clusters=k, linkage='ward', metric='euclidean')
    labels = agg.fit_predict(X_scaled)

    labels, merge_info = merge_micro_clusters(labels, unique_ds, ds_windows, X_scaled, min_windows)
    n_clusters = len(np.unique(labels))

    centers = np.array([X_scaled[labels == c].mean(axis=0) for c in range(n_clusters)])

    merged = (n_clusters != k)
    if merged:
        print(f"  k={k} → consolidated to k={n_clusters}")

    classifier_dir = output_path / 'testbed' / 'classifier'
    classifier_dir.mkdir(parents=True, exist_ok=True)
    clf_dict = {
        'scaler': scaler,
        'centers': centers,
        'method': 'agglomerative',
        'n_clusters': n_clusters,
        'raw_k': k,
        'train_labels': labels,
        'train_dataset_names': unique_ds,
        'merged': merged,
    }
    save_classifier(str(classifier_dir / f'classifier_{config_name}.pkl'), clf_dict)

    cluster_list_dir = output_path / 'testbed' / 'file_list'
    cluster_list_dir.mkdir(parents=True, exist_ok=True)
    cl_path = save_cluster_list(
        str(cluster_list_dir / f'cluster_dataset_list_{config_name}.csv'),
        unique_ds, labels,
    )

    cluster_feat_dir = output_path / 'testbed' / 'cluster_list'
    cluster_feat_dir.mkdir(parents=True, exist_ok=True)
    save_cluster_features(
        str(cluster_feat_dir / f'cluster_features_{config_name}.csv'),
        raw_catch22, labels,
    )

    if train_satzilla_flag and meta_csv and file_list:
        weights_dir = str(output_path / 'weights')
        train_satzilla(config_name, cl_path, n_clusters, meta_csv, file_list, weights_dir)

    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'centers': centers,
        'scaler': scaler,
        'config_name': config_name,
        'merged': merged,
    }
