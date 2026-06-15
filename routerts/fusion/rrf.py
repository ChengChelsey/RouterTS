import numpy as np

def align_score_to_label(score: np.ndarray, label_len: int) -> np.ndarray:
    # Pad or truncate score to label_len, replace NaN/Inf with 0
    if score is None:
        return None

    score = np.asarray(score).reshape(-1)

    if len(score) < label_len:
        pad_len = label_len - len(score)
        if len(score) > 0:
            score = np.pad(score, (0, pad_len), 'constant',
                           constant_values=(0, score[-1]))
        else:
            score = np.zeros(label_len)
    elif len(score) > label_len:
        score = score[:label_len]

    return np.nan_to_num(score)


def fusion_rrf_topk(scores_list: list, k0: int = 60) -> np.ndarray:
    if not scores_list:
        return None

    lengths = [len(score) for score, _ in scores_list if score is not None]
    if not lengths:
        return None

    uniq_lengths = sorted(set(lengths))
    if len(uniq_lengths) != 1:
        raise ValueError(f"Inconsistent lengths after filtering Nones: {uniq_lengths}")

    n = uniq_lengths[0]
    fused = np.zeros(n)

    total_weight = sum(w for _, w in scores_list)
    if total_weight <= 0:
        return None

    for score, weight in scores_list:
        if score is None:
            continue
        normalized_weight = weight / total_weight
        order = np.argsort(-score, kind="mergesort")
        ranks = np.empty(n, dtype=np.int32)
        ranks[order] = np.arange(1, n + 1)
        fused += normalized_weight * (1.0 / (k0 + ranks))

    return fused
