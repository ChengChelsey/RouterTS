"""TAGC — Task-Aware Granularity Criterion for selecting the number of Ward clusters."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

from routerts.clustering.consolidation import merge_micro_clusters
from routerts.config import DEFAULT_LOOCV_ALPHA, DEFAULT_MIN_WINDOWS


def compute_loocv_mse(X_cluster: np.ndarray, Y_cluster: np.ndarray) -> float:
    # LOOCV MSE via Ridge regression; returns 0.1 for clusters with < 3 samples
    n = len(X_cluster)
    if n < 3:
        return 0.1

    preds = np.zeros_like(Y_cluster)
    for train_idx, test_idx in LeaveOneOut().split(X_cluster):
        try:
            model = Ridge(alpha=DEFAULT_LOOCV_ALPHA).fit(X_cluster[train_idx], Y_cluster[train_idx])
            preds[test_idx] = model.predict(X_cluster[test_idx])
        except Exception:
            preds[test_idx] = Y_cluster[train_idx].mean(axis=0)

    return np.mean((Y_cluster - preds) ** 2)


def select_optimal_k(
    raw_catch22: np.ndarray,
    algo_perf: np.ndarray,
    ds_windows: dict,
    unique_ds: list,
    k_range: list = None,
    min_windows: int = DEFAULT_MIN_WINDOWS,
) -> tuple:
    if k_range is None:
        k_range = list(range(5, 16))

    N = len(unique_ds)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_catch22)

    results = []
    prev_inter = None

    for k in k_range:
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward', metric='euclidean')
        labels = clustering.fit_predict(X_scaled)

        labels, info = merge_micro_clusters(labels, unique_ds, ds_windows, X_scaled, min_windows)
        n_eff = len(set(labels))

        if n_eff < 2:
            results.append({
                'k': k, 'n_effective': n_eff, 'mse': np.nan, 'inter': np.nan,
                'r_inter': np.nan, 'A_likelihood': np.nan, 'B_structure': np.nan,
                'C_complexity': np.nan, 'tagc_score': -np.inf,
            })
            prev_inter = None
            continue

        total_mse, total_n = 0.0, 0
        for c in range(n_eff):
            mask = labels == c
            if mask.sum() >= 2:
                mse = compute_loocv_mse(X_scaled[mask], algo_perf[mask])
                total_mse += mse * mask.sum()
                total_n += mask.sum()
        mse = total_mse / total_n if total_n > 0 else 1.0

        centers = np.array([X_scaled[labels == c].mean(axis=0) for c in range(n_eff)])
        inter = np.mean(pdist(centers)) if len(centers) > 1 else 0.0

        r_inter = inter / prev_inter if (prev_inter is not None and prev_inter > 0) else 1.0

        A = -N * np.log(mse) if mse > 0 else -N * np.log(1e-12)
        B = N * np.log(r_inter) if r_inter > 0 else -N * np.log(1e-12)
        C = -n_eff * np.log(N)
        tagc_score = A + B + C

        results.append({
            'k': k, 'n_effective': n_eff, 'mse': mse, 'inter': inter,
            'r_inter': r_inter, 'A_likelihood': A, 'B_structure': B,
            'C_complexity': C, 'tagc_score': tagc_score,
        })
        prev_inter = inter

    scores_df = pd.DataFrame(results)
    best_k = int(scores_df.loc[scores_df['tagc_score'].idxmax(), 'k'])

    return best_k, scores_df
