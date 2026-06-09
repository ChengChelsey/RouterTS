from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from .feature_config import CANDIDATE_MODELS


@dataclass
class AmbiguityMetrics:
    raw_count: int
    active_count: int
    n_eff: float
    top1_weight: float
    top3_cum_weight: float
    coverage_ratio: float
    case_type: str
    retained_set: List[str] = field(repr=False)
    recommended_set: List[str] = field(repr=False)
    sorted_weights: List[Tuple[str, float]] = field(repr=False)


def compute_ambiguity_metrics(
    weights: Dict[str, float],
    coverage_threshold: float = 0.9,
    epsilon: float = 1e-8,
) -> AmbiguityMetrics:
    # Compute ambiguity metrics from normalized selector weights
    sorted_weights = sorted(
        [(det, w) for det, w in weights.items() if w > epsilon],
        key=lambda x: -x[1],
    )

    retained_set = [det for det, _ in sorted_weights]
    raw_count = len(retained_set)

    w_arr = np.array([w for _, w in sorted_weights])
    w_sq_sum = np.sum(w_arr ** 2)
    n_eff = 1.0 / w_sq_sum if w_sq_sum > 1e-15 else float(raw_count)

    top1_weight = sorted_weights[0][1] if sorted_weights else 0.0
    top3_cum_weight = sum(w for _, w in sorted_weights[:3])

    cumsum = 0.0
    recommended_set = []
    coverage_ratio = 0.0
    for det, w in sorted_weights:
        recommended_set.append(det)
        cumsum += w
        if cumsum >= coverage_threshold:
            coverage_ratio = cumsum
            break
    else:
        coverage_ratio = cumsum

    active_count = len(recommended_set)

    if top1_weight >= 0.8:
        case_type = 'A_high_conf'
    elif active_count <= 3:
        case_type = 'B_focused'
    else:
        case_type = 'C_ambiguous'

    return AmbiguityMetrics(
        raw_count=raw_count,
        active_count=active_count,
        n_eff=n_eff,
        top1_weight=top1_weight,
        top3_cum_weight=top3_cum_weight,
        coverage_ratio=coverage_ratio,
        case_type=case_type,
        retained_set=retained_set,
        recommended_set=recommended_set,
        sorted_weights=sorted_weights,
    )
