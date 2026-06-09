import re
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def load_dataset_window_counts(meta_csv: str, unique_ds: list) -> dict:
    df = pd.read_csv(meta_csv, index_col=0)
    ds_names = df.index.to_series().apply(lambda x: re.sub(r'_\d+$', '', x)).values
    unique, counts = np.unique(ds_names, return_counts=True)
    return dict(zip(unique, counts))


def compute_cluster_window_counts(labels: np.ndarray, unique_ds: list, ds_windows: dict) -> dict:
    return {
        c: sum(ds_windows.get(unique_ds[i], 0) for i in np.where(labels == c)[0])
        for c in np.unique(labels)
    }


def merge_micro_clusters(
    labels: np.ndarray,
    unique_ds: list,
    ds_windows: dict,
    X_scaled: np.ndarray,
    min_windows: int = 100,
) -> tuple:
    # Dissolve clusters below min_windows, reassign to nearest surviving centroid
    labels = labels.copy()
    cw = compute_cluster_window_counts(labels, unique_ds, ds_windows)
    micro = set(c for c, w in cw.items() if w < min_windows)
    normal = sorted(c for c, w in cw.items() if w >= min_windows)

    if not micro or not normal:
        return labels, None

    normal_centers = np.array([X_scaled[labels == c].mean(axis=0) for c in normal])
    for mc in micro:
        mask = labels == mc
        dists = pairwise_distances(X_scaled[mask], normal_centers)
        for i, idx in enumerate(np.where(mask)[0]):
            labels[idx] = normal[dists[i].argmin()]

    old_unique = sorted(set(labels))
    renumber_map = {old: new for new, old in enumerate(old_unique)}
    labels = np.array([renumber_map[l] for l in labels])

    n_final = len(old_unique)
    centers = np.array([X_scaled[labels == c].mean(axis=0) for c in range(n_final)])

    info = {
        'micro_clusters': micro,
        'renumber_map': {c: renumber_map[c] for c in normal},
        'centers': centers,
    }
    return labels, info
