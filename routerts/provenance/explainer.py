import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import shap

from .feature_config import (
    CATCH22_NAMES, CANDIDATE_MODELS,
    CATCH22_SEMANTIC, CATCH22_INTERPRETATION,
    N_DETECTORS, N_FEATURES,
)
from .ambiguity import compute_ambiguity_metrics
from .cards import (
    DetectorSupport, RoutingCard, PairwiseTrace,
    CandidatePreservationTrace, SelectionCard, FusionAudit,
    DecisionTrace,
)

logger = logging.getLogger(__name__)


class ClusterDecisionTracer:

    def __init__(
        self,
        classifier_path: str,
        model_dir: str,
        meta_feature_path: str,
        config: str = 'agg_raw_k13',
        domain: str = 'ID',
        background_sample_size: int = 200,
        random_state: int = 2024,
    ):
        self.config = config
        self.domain = domain
        self.classifier_path = Path(classifier_path)
        self.model_dir = Path(model_dir)
        self.meta_feature_path = Path(meta_feature_path)
        self.background_sample_size = background_sample_size
        self.random_state = random_state

        with open(self.classifier_path, 'rb') as f:
            self._clf = pickle.load(f)
        self.scaler = self._clf['scaler']
        self.centers = self._clf['centers']
        self.n_clusters = self._clf['n_clusters']

        self._models: Dict[int, object] = {}
        self._explainers: Dict[int, object] = {}
        self._background_data: Dict[int, np.ndarray] = {}

        self._load_training_data()

    def _load_training_data(self):
        # Load training data, group by cluster, create TreeExplainer
        rng = np.random.RandomState(self.random_state)
        df = pd.read_csv(self.meta_feature_path, index_col=0)
        feature_cols = [f'val_{i}' for i in range(N_FEATURES)]
        X_all = df[feature_cols].values
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
        X_all = pd.DataFrame(X_all, columns=feature_cols)

        dataset_names = df.index.to_series().apply(lambda s: s.rsplit('_', 1)[0])
        train_dataset_names = list(self._clf['train_dataset_names'])
        train_labels = np.array(self._clf['train_labels'])

        ds_to_cluster = dict(zip(train_dataset_names, train_labels))
        window_clusters = dataset_names.map(ds_to_cluster)

        for cid in range(self.n_clusters):
            model_path = self.model_dir / self.config / f'SATzilla_Cluster_C{cid}' / f'{self.domain}.pkl'
            if not model_path.exists():
                logger.warning(f"RF model not found: {model_path}")
                continue
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self._models[cid] = model

            mask = window_clusters == cid
            X_cluster = X_all.loc[mask.values]

            if len(X_cluster) == 0:
                logger.warning(f"Cluster {cid}: no training windows found")
                continue

            n_bg = min(self.background_sample_size, len(X_cluster))
            idx = rng.choice(len(X_cluster), n_bg, replace=False)
            self._background_data[cid] = X_cluster.iloc[idx]

            try:
                self._explainers[cid] = shap.TreeExplainer(
                    model, data=self._background_data[cid]
                )
            except Exception as e:
                logger.warning(f"Failed to create TreeExplainer for cluster {cid}: {e}")

        logger.info(f"Loaded {len(self._models)} cluster RFs, "
                     f"{len(self._explainers)} SHAP explainers")

    def explain_routing(self, catch22_features: np.ndarray) -> RoutingCard:
        x = np.asarray(catch22_features).reshape(1, -1)

        x_scaled = self.scaler.transform(x)

        diffs = x_scaled - self.centers
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))

        sorted_ids = np.argsort(distances)
        nearest_id = sorted_ids[0]
        second_id = sorted_ids[1]

        margin = distances[second_id] - distances[nearest_id]

        # Distance decomposition: Δ_j > 0 means feature j supports nearest cluster
        delta = diffs[second_id] ** 2 - diffs[nearest_id] ** 2

        x_mean = self.scaler.mean_
        x_std = np.sqrt(self.scaler.var_)
        z_scores = (x[0] - x_mean) / (x_std + 1e-8)

        top_feat_indices = np.argsort(np.abs(delta))[::-1][:5]
        top_routing_features = [
            (CATCH22_NAMES[j], float(x[0, j]), float(z_scores[j]),
             float(self.scaler.inverse_transform(
                 self.centers[nearest_id].reshape(1, -1))[0, j]))
            for j in top_feat_indices
        ]

        dist_dict = {int(i): float(distances[i]) for i in range(self.n_clusters)}

        return RoutingCard(
            cluster_id=int(nearest_id),
            distances=dist_dict,
            margin=float(margin),
            top_routing_features=top_routing_features,
        )

    def explain_candidate_preservation(
        self,
        window_scores_matrix: np.ndarray,
        gap_threshold: float = 0.05,
        coverage_threshold: float = 0.9,
    ) -> CandidatePreservationTrace:
        # Analyze window-level gap retention mechanism
        n_windows = window_scores_matrix.shape[0]
        support_count = np.zeros(N_DETECTORS, dtype=int)
        support_mass = np.zeros(N_DETECTORS, dtype=float)
        top1_count = np.zeros(N_DETECTORS, dtype=int)
        top2_count = np.zeros(N_DETECTORS, dtype=int)
        top3_count = np.zeros(N_DETECTORS, dtype=int)
        gap_to_winner_sum = np.zeros(N_DETECTORS, dtype=float)

        retained_windows: List[List[int]] = [[] for _ in range(N_DETECTORS)]
        retained_scores: List[List[float]] = [[] for _ in range(N_DETECTORS)]

        for w_idx in range(n_windows):
            row = window_scores_matrix[w_idx]
            sorted_indices = np.argsort(row)[::-1]
            sorted_scores = row[sorted_indices]

            k = 1
            if len(sorted_scores) >= 2:
                gap_1_2 = sorted_scores[0] - sorted_scores[1]
                if gap_1_2 <= gap_threshold:
                    k = 2
                    if len(sorted_scores) >= 3:
                        gap_2_3 = sorted_scores[1] - sorted_scores[2]
                        if gap_2_3 <= gap_threshold:
                            k = 3

            retained_set_w = sorted_indices[:k]
            winner_score = sorted_scores[0]

            for rank, det_idx in enumerate(retained_set_w):
                support_count[det_idx] += 1
                support_mass[det_idx] += row[det_idx]
                retained_windows[det_idx].append(w_idx)
                retained_scores[det_idx].append(row[det_idx])
                gap_to_winner_sum[det_idx] += winner_score - row[det_idx]
                if rank == 0:
                    top1_count[det_idx] += 1
                elif rank == 1:
                    top2_count[det_idx] += 1
                elif rank == 2:
                    top3_count[det_idx] += 1

        weighted_scores = support_mass.copy()
        total = weighted_scores.sum()
        final_weights = weighted_scores / total if total > 0 else weighted_scores

        weights_dict = {CANDIDATE_MODELS[i]: float(final_weights[i])
                        for i in range(N_DETECTORS) if final_weights[i] > 1e-8}

        sorted_by_weight = sorted(weights_dict.items(), key=lambda x: -x[1])
        retained_set = [det for det, _ in sorted_by_weight]

        cumsum = 0.0
        recommended_set = []
        coverage_ratio = 0.0
        for det, w in sorted_by_weight:
            recommended_set.append(det)
            cumsum += w
            if cumsum >= coverage_threshold:
                coverage_ratio = cumsum
                break
        else:
            coverage_ratio = cumsum

        recommended_set_names = set(recommended_set)

        detector_supports = []
        for i in range(N_DETECTORS):
            if support_count[i] == 0:
                continue

            avg_score = support_mass[i] / support_count[i]
            mean_gap = gap_to_winner_sum[i] / support_count[i]

            mode = self._classify_preservation_mode(
                support_count[i], n_windows, top1_count[i],
                mean_gap, gap_threshold, avg_score
            )

            rep_wins = self._pick_representative_windows(
                i, retained_windows[i], retained_scores[i],
                window_scores_matrix, gap_threshold
            )

            det_name = CANDIDATE_MODELS[i]
            detector_supports.append(DetectorSupport(
                detector=det_name,
                support_count=int(support_count[i]),
                support_mass=float(support_mass[i]),
                avg_score=float(avg_score),
                top1_count=int(top1_count[i]),
                top2_count=int(top2_count[i]),
                top3_count=int(top3_count[i]),
                mean_gap_to_winner=float(mean_gap),
                final_weight=float(final_weights[i]),
                in_retained_set=det_name in set(retained_set),
                in_recommended_set=det_name in recommended_set_names,
                preservation_mode=mode,
                representative_windows=rep_wins,
            ))

        detector_supports.sort(key=lambda d: -d.final_weight)

        return CandidatePreservationTrace(
            retained_set=retained_set,
            recommended_set=recommended_set,
            coverage_ratio=coverage_ratio,
            detector_supports=detector_supports,
        )

    @staticmethod
    def _classify_preservation_mode(
        support_count, n_windows, top1_count, mean_gap, gap_threshold,
        avg_score,
    ):
        # Classify as 'leader', 'near_tie', 'specialist', or 'follower'
        if support_count == 0:
            return 'follower'
        support_ratio = support_count / n_windows if n_windows > 0 else 0
        if top1_count / support_count > 0.5:
            return 'leader'
        if mean_gap < gap_threshold:
            return 'near_tie'
        if support_ratio < 0.2 and avg_score > 0:
            return 'specialist'
        return 'follower'

    @staticmethod
    def _pick_representative_windows(
        det_idx, retained_wins, retained_scores,
        window_scores_matrix, gap_threshold
    ):
        # Select 1-3 representative windows for a detector
        if not retained_wins:
            return []

        wins = np.array(retained_wins)
        scores = np.array(retained_scores)

        # Leader: top-1 with largest margin
        leader_win = None
        leader_margin = -1.0
        for w, s in zip(retained_wins, retained_scores):
            row = window_scores_matrix[w]
            sorted_s = np.sort(row)[::-1]
            if row[det_idx] == sorted_s[0]:
                margin = sorted_s[0] - sorted_s[1] if len(sorted_s) > 1 else 1.0
                if margin > leader_margin:
                    leader_margin = margin
                    leader_win = w

        # Boundary: score closest to gap threshold
        boundary_win = None
        best_boundary_dist = float('inf')
        best_boundary_score = -float('inf')
        best_boundary_is_winner = True
        for w, s in zip(retained_wins, retained_scores):
            winner = np.max(window_scores_matrix[w])
            gap = winner - s
            is_winner = (gap < 1e-10)
            dist_to_boundary = abs(gap - gap_threshold)
            better = False
            if dist_to_boundary < best_boundary_dist:
                better = True
            elif dist_to_boundary == best_boundary_dist:
                if not is_winner and best_boundary_is_winner:
                    better = True
                elif is_winner == best_boundary_is_winner and s > best_boundary_score:
                    better = True
            if better:
                best_boundary_dist = dist_to_boundary
                best_boundary_score = s
                best_boundary_is_winner = is_winner
                boundary_win = w

        # Specialist: highest-score retained window
        specialist_win = wins[np.argmax(scores)]

        result = []
        seen = set()
        for w in [leader_win, boundary_win, specialist_win]:
            if w is not None and w not in seen:
                result.append(int(w))
                seen.add(w)

        return result

    def explain_selection(
        self,
        catch22_windows: np.ndarray,
        cluster_id: int,
        detector_indices: List[int],
        representative_window_map: Optional[Dict[int, int]] = None,
    ) -> Tuple[Dict[str, Dict], Optional[PairwiseTrace]]:
        # Run TreeSHAP on representative windows to explain RF scoring
        if cluster_id not in self._explainers:
            logger.warning(f"No SHAP explainer for cluster {cluster_id}")
            return {}, None

        explainer = self._explainers[cluster_id]
        model = self._models[cluster_id]

        active_set_explanations = {}

        for det_idx in detector_indices:
            if representative_window_map and det_idx in representative_window_map:
                w_idx = representative_window_map[det_idx]
            else:
                w_idx = 0

            X = catch22_windows[w_idx:w_idx+1]
            feature_cols = [f'val_{i}' for i in range(N_FEATURES)]
            X_df = pd.DataFrame(X, columns=feature_cols)

            try:
                shap_values = explainer.shap_values(X_df)
            except Exception as e:
                logger.warning(f"SHAP failed for detector {det_idx}: {e}")
                continue

            if isinstance(shap_values, list):
                shap_det = shap_values[det_idx][0]
                base_val = explainer.expected_value[det_idx]
            else:
                shap_det = shap_values[0, :, det_idx]
                base_val = explainer.expected_value[det_idx]

            pred_score = float(model.predict(X_df)[0, det_idx])

            top_feat_idx = np.argsort(np.abs(shap_det))[::-1][:5]
            top_features = [
                {
                    'name': CATCH22_NAMES[j],
                    'semantic': CATCH22_SEMANTIC[j],
                    'value': float(X[0, j]),
                    'shap': float(shap_det[j]),
                    'direction': 'increases score' if shap_det[j] > 0 else 'decreases score',
                    'interpretation': (CATCH22_INTERPRETATION[j]['high']
                                       if X[0, j] > np.median(X[0])
                                       else CATCH22_INTERPRETATION[j]['low']),
                }
                for j in top_feat_idx
            ]

            active_set_explanations[CANDIDATE_MODELS[det_idx]] = {
                'base_value': float(base_val),
                'pred_score': pred_score,
                'top_features': top_features,
                'window_idx': w_idx,
            }

        pairwise = self._pairwise_shap(
            explainer, model, catch22_windows,
            detector_indices,
            representative_window_map,
        )

        return active_set_explanations, pairwise

    def _pairwise_shap(
        self, explainer, model, catch22_windows,
        detector_indices,
        representative_window_map,
    ) -> Optional[PairwiseTrace]:
        if len(detector_indices) < 1:
            return None

        feature_cols = [f'val_{i}' for i in range(N_FEATURES)]

        top1_idx = detector_indices[0]

        if len(detector_indices) >= 2:
            top2_idx = detector_indices[1]
        else:
            w_idx = representative_window_map.get(top1_idx, 0) if representative_window_map else 0
            X = catch22_windows[w_idx:w_idx+1]
            X_df = pd.DataFrame(X, columns=feature_cols)
            scores = model.predict(X_df)[0]
            scores[top1_idx] = -np.inf
            top2_idx = int(np.argmax(scores))

        w_idx = representative_window_map.get(top1_idx, 0) if representative_window_map else 0
        X = catch22_windows[w_idx:w_idx+1]
        X_df = pd.DataFrame(X, columns=feature_cols)

        try:
            shap_values = explainer.shap_values(X_df)
        except Exception:
            return None

        if isinstance(shap_values, list):
            shap_a = shap_values[top1_idx][0]
            shap_b = shap_values[top2_idx][0]
            base_a = explainer.expected_value[top1_idx]
            base_b = explainer.expected_value[top2_idx]
        else:
            shap_a = shap_values[0, :, top1_idx]
            shap_b = shap_values[0, :, top2_idx]
            base_a = explainer.expected_value[top1_idx]
            base_b = explainer.expected_value[top2_idx]

        pred_a = float(model.predict(X_df)[0, top1_idx])
        pred_b = float(model.predict(X_df)[0, top2_idx])
        score_diff = pred_a - pred_b
        base_diff = float(base_a - base_b)

        shap_diff = shap_a - shap_b

        top_feat_idx = np.argsort(np.abs(shap_diff))[::-1][:5]
        feature_contributions = []
        for j in top_feat_idx:
            direction = 'supports_A' if shap_diff[j] > 0 else 'supports_B'
            feature_contributions.append({
                'name': CATCH22_NAMES[j],
                'semantic': CATCH22_SEMANTIC[j],
                'shap_diff': float(shap_diff[j]),
                'direction': direction,
                'meaning': (f"Higher {CATCH22_SEMANTIC[j]} favors "
                            f"{CANDIDATE_MODELS[top1_idx] if direction == 'supports_A' else CANDIDATE_MODELS[top2_idx]}"),
            })

        return PairwiseTrace(
            detector_a=CANDIDATE_MODELS[top1_idx],
            detector_b=CANDIDATE_MODELS[top2_idx],
            score_diff=score_diff,
            base_diff=base_diff,
            feature_contributions=feature_contributions,
        )

    def explain_fusion(
        self,
        recommended_set: List[str],
        weights: Dict[str, float],
        scores_dict: Optional[Dict[str, np.ndarray]] = None,
        k0: int = 60,
    ) -> FusionAudit:
        # Audit the RRF fusion process
        final_weight = {det: weights.get(det, 0.0) for det in recommended_set}

        if scores_dict:
            det_score_arrays = []
            for det in recommended_set:
                if det in scores_dict and scores_dict[det] is not None:
                    det_score_arrays.append((det, scores_dict[det]))

            if det_score_arrays:
                min_len = min(len(s) for _, s in det_score_arrays)
                n = min_len
                rank_contrib = np.zeros(n)
                total_weight = sum(weights.get(det, 0) for det, _ in det_score_arrays)

                if total_weight > 0:
                    per_det_contrib = {}
                    for det, score in det_score_arrays:
                        score = score[:n]
                        w = weights.get(det, 0) / total_weight
                        order = np.argsort(-score, kind="mergesort")
                        ranks = np.empty(n, dtype=np.int32)
                        ranks[order] = np.arange(1, n + 1)
                        term = w * (1.0 / (k0 + ranks))
                        per_det_contrib[det] = float(term.sum())
                        rank_contrib += term

                    total_contrib = rank_contrib.sum()
                    if total_contrib > 0:
                        normalized_contribution = {
                            det: contrib / total_contrib
                            for det, contrib in per_det_contrib.items()
                        }
                    else:
                        normalized_contribution = {det: 0.0 for det in per_det_contrib}
                else:
                    per_det_contrib = {}
                    normalized_contribution = {}

                rrf_weighted_rank_term = per_det_contrib
            else:
                rrf_weighted_rank_term = {}
                normalized_contribution = {}
        else:
            rrf_weighted_rank_term = {}
            normalized_contribution = {}

        return FusionAudit(
            recommended_detectors=recommended_set,
            final_weight=final_weight,
            rrf_weighted_rank_term=rrf_weighted_rank_term,
            normalized_contribution=normalized_contribution,
        )

    def explain_full(
        self,
        sample_id: str,
        catch22_windows: np.ndarray,
        window_scores_matrix: np.ndarray,
        gap_threshold: float = 0.05,
        coverage_threshold: float = 0.9,
        scores_dict: Optional[Dict[str, np.ndarray]] = None,
        cluster_id: Optional[int] = None,
        window_scores_source_cluster_id: Optional[int] = None,
        score_source_policy: str = 'recompute',
    ) -> DecisionTrace:
        # End-to-end: routing → candidate_preservation → selection → fusion
        _VALID_POLICIES = ('strict', 'recompute', 'trust_input')
        if score_source_policy not in _VALID_POLICIES:
            raise ValueError(
                f"score_source_policy='{score_source_policy}' not in {_VALID_POLICIES}"
            )

        catch22_windows = np.asarray(catch22_windows, dtype=float)
        window_scores_matrix = np.asarray(window_scores_matrix, dtype=float)

        if catch22_windows.ndim != 2 or catch22_windows.shape[1] != N_FEATURES:
            raise ValueError(
                f"catch22_windows shape {catch22_windows.shape}, expected (*, {N_FEATURES})"
            )
        if window_scores_matrix.ndim != 2 or window_scores_matrix.shape[1] != N_DETECTORS:
            raise ValueError(
                f"window_scores_matrix shape {window_scores_matrix.shape}, expected (*, {N_DETECTORS})"
            )
        if catch22_windows.shape[0] != window_scores_matrix.shape[0]:
            raise ValueError(
                f"row count mismatch: catch22={catch22_windows.shape[0]} vs scores={window_scores_matrix.shape[0]}"
            )

        # Routing via mean catch22
        mean_catch22 = catch22_windows.mean(axis=0)
        routing = self.explain_routing(mean_catch22)
        used_cluster_id = routing.cluster_id

        source_cluster = window_scores_source_cluster_id if window_scores_source_cluster_id is not None else cluster_id
        mismatch = source_cluster is not None and source_cluster != used_cluster_id

        if mismatch:
            if score_source_policy == 'strict':
                raise ValueError(
                    f"score_source_policy='strict': window_scores from cluster {source_cluster} "
                    f"but routing says cluster {used_cluster_id}. Aborting."
                )
            elif score_source_policy == 'trust_input':
                logger.warning(
                    f"Routing cluster {used_cluster_id} != source cluster {source_cluster}. "
                    f"trust_input: using supplied scores as-is."
                )
            else:
                logger.warning(
                    f"Routing cluster {used_cluster_id} != source cluster {source_cluster}. "
                    f"Recomputing window_scores with cluster {used_cluster_id} RF."
                )

        cluster_id = used_cluster_id

        if score_source_policy == 'recompute':
            if cluster_id in self._models:
                feature_cols = [f'val_{i}' for i in range(N_FEATURES)]
                meta_mat = pd.DataFrame(catch22_windows, columns=feature_cols)
                meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)
                window_scores_matrix = self._models[cluster_id].predict(meta_mat)
            else:
                logger.warning(
                    f"recompute policy but cluster {cluster_id} model missing. "
                    f"Falling back to supplied window_scores_matrix."
                )

        # Candidate Preservation
        preservation = self.explain_candidate_preservation(
            window_scores_matrix, gap_threshold, coverage_threshold,
        )

        weights = {ds.detector: ds.final_weight for ds in preservation.detector_supports}

        # Ambiguity
        ambiguity = compute_ambiguity_metrics(weights, coverage_threshold)

        # Selection (SHAP)
        recommended_names = set(preservation.recommended_set)
        active_indices = [
            CANDIDATE_MODELS.index(det) for det in preservation.recommended_set
        ]

        rep_map = {}
        for ds in preservation.detector_supports:
            if ds.detector in recommended_names and ds.representative_windows:
                det_idx = CANDIDATE_MODELS.index(ds.detector)
                rep_map[det_idx] = ds.representative_windows[0]

        active_explanations, pairwise = self.explain_selection(
            catch22_windows, cluster_id, active_indices, rep_map,
        )

        selection = SelectionCard(
            ambiguity=ambiguity,
            candidate_preservation=preservation,
            active_set_explanations=active_explanations,
            pairwise=pairwise,
        )

        # Fusion Audit
        fusion = self.explain_fusion(
            preservation.recommended_set, weights, scores_dict,
        )

        return DecisionTrace(
            sample_id=sample_id,
            routing=routing,
            selection=selection,
            fusion=fusion,
        )
