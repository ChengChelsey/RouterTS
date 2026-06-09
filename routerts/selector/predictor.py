import os
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from routerts.config import CANDIDATE_MODEL_SET_20

logger = logging.getLogger(__name__)


def predict_detector_scores(
    meta_features: np.ndarray,
    model_path: str,
) -> np.ndarray:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"SATzilla model not found: {model_file}")

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    feature_cols = [f'val_{i}' for i in range(meta_features.shape[1])]
    meta_mat = pd.DataFrame(meta_features, columns=feature_cols)
    meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)

    preds = model.predict(meta_mat)
    if isinstance(preds, pd.DataFrame):
        preds = preds.to_numpy()
    return preds


def gap_weighted_voting(
    preds: np.ndarray,
    candidate_models: list = None,
    gap_threshold: float = 0.05,
) -> tuple:
    # Gap-based adaptive top-k retention + probability-weighted voting
    if candidate_models is None:
        candidate_models = CANDIDATE_MODEL_SET_20

    n_detectors = len(candidate_models)
    weighted_scores = np.zeros(n_detectors)
    total_weight = 0.0

    if preds.ndim != 2:
        raise ValueError(f"Expected 2D preds, got {preds.ndim}D")
    if preds.shape[1] != n_detectors:
        raise ValueError(f"preds has {preds.shape[1]} cols but {n_detectors} detectors")

    for row in preds:
        sorted_indices = np.argsort(row)[::-1]
        sorted_probs = row[sorted_indices]

        k = 1
        if len(sorted_probs) >= 2:
            gap_1_2 = sorted_probs[0] - sorted_probs[1]
            if gap_1_2 <= gap_threshold:
                k = 2
                if len(sorted_probs) >= 3:
                    gap_2_3 = sorted_probs[1] - sorted_probs[2]
                    if gap_2_3 <= gap_threshold:
                        k = 3

        for i in range(k):
            model_idx = sorted_indices[i]
            weight = sorted_probs[i]
            weighted_scores[model_idx] += weight
            total_weight += weight

    if total_weight == 0:
        logger.warning("No valid votes — all predictions are zero.")
        probs_dict = {m: 0.0 for m in candidate_models}
        return candidate_models[0], probs_dict, weighted_scores

    probs_dict = {m: 0.0 for m in candidate_models}
    for idx, w_score in enumerate(weighted_scores):
        if w_score > 0:
            model_name = candidate_models[idx]
            probs_dict[model_name] = w_score / total_weight

    selected_idx = int(np.argmax(weighted_scores))
    selected_model = candidate_models[selected_idx]

    return selected_model, probs_dict, weighted_scores


def load_anomaly_score(
    filename: str,
    selected_model: str,
    score_dir: str,
    expected_length: int = None,
) -> np.ndarray:

    score_name = filename.split('.')[0]
    score_path = os.path.join(score_dir, selected_model, f'{score_name}.npy')

    if os.path.exists(score_path):
        score = np.load(score_path)
    else:
        logger.warning(f"Score file not found: {score_path}")
        if expected_length is not None:
            return np.zeros(expected_length)
        return np.array([])

    if expected_length is not None:
        if len(score) < expected_length:
            pad_length = expected_length - len(score)
            if len(score) > 0:
                score = np.pad(score, (0, pad_length), mode='constant',
                               constant_values=(0, score[-1]))
            else:
                score = np.zeros(expected_length)
        elif len(score) > expected_length:
            score = score[:expected_length]
        score = np.nan_to_num(score)

    return score
