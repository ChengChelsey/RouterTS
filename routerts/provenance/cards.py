from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .ambiguity import AmbiguityMetrics


@dataclass
class DetectorSupport:
    detector: str
    support_count: int
    support_mass: float
    avg_score: float
    top1_count: int
    top2_count: int
    top3_count: int
    mean_gap_to_winner: float
    final_weight: float
    in_retained_set: bool
    in_recommended_set: bool
    preservation_mode: str       # 'leader' | 'near_tie' | 'specialist' | 'follower'
    representative_windows: List[int]


@dataclass
class RoutingCard:
    # Layer 1: Cluster routing explanation
    cluster_id: int
    distances: Dict[int, float]  # cluster_id → scaled distance
    margin: float
    top_routing_features: List[Tuple[str, float, float, float]]


@dataclass
class PairwiseTrace:
    detector_a: str
    detector_b: str
    score_diff: float
    base_diff: float
    feature_contributions: List[Dict]


@dataclass
class CandidatePreservationTrace:
    # Layer 2A: Window-level gap retention trace
    retained_set: List[str]
    recommended_set: List[str]
    coverage_ratio: float
    detector_supports: List[DetectorSupport]


@dataclass
class SelectionCard:
    # Layer 2B: RF selection provenance via TreeSHAP
    ambiguity: AmbiguityMetrics
    candidate_preservation: CandidatePreservationTrace
    active_set_explanations: Dict[str, Dict]
    pairwise: Optional[PairwiseTrace]


@dataclass
class FusionAudit:
    # Layer 2C: RRF fusion audit
    recommended_detectors: List[str]
    final_weight: Dict[str, float]
    rrf_weighted_rank_term: Dict[str, float]
    normalized_contribution: Dict[str, float]


@dataclass
class DecisionTrace:
    # Complete four-layer decision provenance card
    sample_id: str
    routing: RoutingCard
    selection: SelectionCard
    fusion: FusionAudit
