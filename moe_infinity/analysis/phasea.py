from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from moe_infinity.utils.prefetch_plan import (
    PrefetchCandidate,
    rank_prefetch_candidates,
    unique_candidate_experts,
)


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    i = 0
    while i < len(array):
        j = i
        while j + 1 < len(array) and array[order[j + 1]] == array[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return 0.0
    rank_x = _rankdata(values_x)
    rank_y = _rankdata(values_y)
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return 0.0
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def _candidate_unique_score_ratio(
    candidate_pairs: Sequence[Mapping[str, Any]],
) -> float:
    top_pairs = list(candidate_pairs[:8])
    if not top_pairs:
        return 0.0
    rounded = {round(float(item["score"]), 8) for item in top_pairs}
    return float(len(rounded) / len(top_pairs))


def _candidate_topk_entropy(
    candidate_pairs: Sequence[Mapping[str, Any]],
) -> float:
    top_pairs = list(candidate_pairs[:8])
    if not top_pairs:
        return 0.0
    weights = np.asarray(
        [max(float(item["score"]), 0.0) for item in top_pairs],
        dtype=np.float64,
    )
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    probs = weights / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _candidate_all_equal_topk(
    candidate_pairs: Sequence[Mapping[str, Any]],
) -> bool:
    top_pairs = list(candidate_pairs[:8])
    if len(top_pairs) < 2:
        return False
    rounded = {round(float(item["score"]), 8) for item in top_pairs}
    return len(rounded) == 1


def _candidate_contiguous_topk(
    candidate_pairs: Sequence[Mapping[str, Any]],
) -> bool:
    top_pairs = list(candidate_pairs[:8])
    if len(top_pairs) < 2:
        return False
    layer_ids = {int(item["layer_idx"]) for item in top_pairs}
    if len(layer_ids) != 1:
        return False
    expert_ids = [int(item["expert_idx"]) for item in top_pairs]
    if len(set(expert_ids)) != len(expert_ids):
        return False
    ordered = sorted(expert_ids)
    return all(curr - prev == 1 for prev, curr in zip(ordered, ordered[1:]))


def _candidate_signature(
    candidate_pairs: Sequence[Mapping[str, Any]],
    *,
    budget: int = 32,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(item["layer_idx"]), int(item["expert_idx"]))
        for item in candidate_pairs[:budget]
    )


def _future_pair_weights(
    future_events: Sequence[Mapping[str, Any]],
) -> Counter[tuple[int, int]]:
    weights: Counter[tuple[int, int]] = Counter()
    for event in future_events:
        layer_idx = int(event["layer_idx"])
        counts = event.get("actual_expert_counts", {})
        for expert_str, count in counts.items():
            weights[(layer_idx, int(expert_str))] += int(count)
    return weights


def _top_budget_weight(
    pair_weights: Mapping[tuple[int, int], int],
    *,
    budget: int,
) -> int:
    if budget <= 0 or not pair_weights:
        return 0
    weights = sorted((int(weight) for weight in pair_weights.values()), reverse=True)
    return int(sum(weights[:budget]))


@dataclass
class _SequenceState:
    request_id: str
    trace_name: str
    tag: str
    variant: str
    is_warmup: bool = False
    step_index: int = 0


class PhaseAObservationRecorder:
    def __init__(
        self,
        output_path: str | Path,
        *,
        max_ranked_candidates: int = 32,
        analysis_max_ranked_candidates: int | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_ranked_candidates = max(int(max_ranked_candidates), 1)
        if analysis_max_ranked_candidates is None:
            analysis_max_ranked_candidates = max_ranked_candidates
        self.analysis_max_ranked_candidates = max(
            int(analysis_max_ranked_candidates),
            1,
        )
        self._handle = self.output_path.open("w", encoding="utf-8")
        self._current_context: Dict[str, Any] = {}
        self._seq_states: Dict[str, _SequenceState] = {}

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.flush()
        self._handle.close()

    def set_request_context(
        self,
        *,
        request_id: str,
        trace_name: str,
        tag: str,
        variant: str,
        is_warmup: bool = False,
    ) -> None:
        self._current_context = {
            "request_id": request_id,
            "trace_name": trace_name,
            "tag": tag,
            "variant": variant,
            "is_warmup": bool(is_warmup),
        }

    def clear_request_context(self) -> None:
        self._current_context = {}

    def record_sequence_start(self, seq_ids: Sequence[str]) -> None:
        for seq_id in seq_ids:
            state = _SequenceState(
                request_id=self._current_context.get("request_id", ""),
                trace_name=self._current_context.get("trace_name", ""),
                tag=self._current_context.get("tag", ""),
                variant=self._current_context.get("variant", ""),
                is_warmup=bool(self._current_context.get("is_warmup", False)),
            )
            self._seq_states[seq_id] = state
            self._write_event(
                {
                    "event_type": "sequence_start",
                    "seq_id": seq_id,
                    "request_id": state.request_id,
                    "trace_name": state.trace_name,
                    "tag": state.tag,
                    "variant": state.variant,
                    "is_warmup": state.is_warmup,
                    "step_index": state.step_index,
                }
            )

    def record_sequence_finish(self, seq_id: str) -> None:
        state = self._seq_states.pop(seq_id, None)
        if state is None:
            return
        self._write_event(
            {
                "event_type": "sequence_finish",
                "seq_id": seq_id,
                "request_id": state.request_id,
                "trace_name": state.trace_name,
                "tag": state.tag,
                "variant": state.variant,
                "is_warmup": state.is_warmup,
                "step_index": state.step_index,
            }
        )

    def record_policy_event(
        self,
        *,
        seq_id: str,
        layer_idx: int,
        actual_expert_array: np.ndarray,
        policy_name: str,
        prefetch_enabled: bool,
        score_only: bool,
        decision_latency_us: int,
        expert_matrix: np.ndarray | None,
        num_layers: int | None,
        prefetch_future_layers: int,
        prefetch_max_candidates: int,
        analysis_expert_matrix: np.ndarray | None = None,
        analysis_future_layers: int | None = None,
        analysis_max_candidates: int | None = None,
        prefetch_candidate_min_score: float = 1e-6,
        candidate_source: str = "empty",
        fallback_reason: str = "none",
        library_match_count: int = 0,
        local_match_count: int = 0,
        raw_positive_count: int = 0,
        kept_candidate_count: int = 0,
        all_scores_below_threshold: bool = False,
        backbone_mode: str = "raw_topk",
        backbone_support_count: int = 0,
        backbone_history_match_count: int = 0,
        backbone_projection_topk: int = 0,
        backbone_fallback_used: bool = False,
        consensus_min_votes: int = 0,
        consensus_support_count: int = 0,
        consensus_retained_count: int = 0,
        consensus_fallback_used: bool = False,
        consensus_fallback_reason: str = "none",
        historical_reuse_object_mode: str = "sequence_matrix",
        local_continuation_key_layers: int = 4,
        local_continuation_future_layers: int = 4,
        historical_similarity_mode: str = "prefix_mean",
        historical_recent_window: int = 4,
    ) -> None:
        state = self._seq_states.setdefault(
            seq_id,
            _SequenceState(
                request_id=self._current_context.get("request_id", ""),
                trace_name=self._current_context.get("trace_name", ""),
                tag=self._current_context.get("tag", ""),
                variant=self._current_context.get("variant", ""),
                is_warmup=bool(self._current_context.get("is_warmup", False)),
            ),
        )
        actual_array = np.asarray(actual_expert_array, dtype=np.int64)
        actual_counts = Counter(actual_array.reshape(-1).tolist())
        actual_experts = [int(x) for x in sorted(actual_counts)]

        if analysis_expert_matrix is None:
            analysis_expert_matrix = expert_matrix
        if analysis_future_layers is None:
            analysis_future_layers = prefetch_future_layers
        if analysis_max_candidates is None:
            analysis_max_candidates = self.analysis_max_ranked_candidates

        runtime_ranked_candidates: List[PrefetchCandidate] = []
        if expert_matrix is not None:
            runtime_ranked_candidates = rank_prefetch_candidates(
                layer_id=layer_idx,
                expert_matrix=expert_matrix,
                future_layers=prefetch_future_layers,
                max_candidates=self.max_ranked_candidates,
                min_score=prefetch_candidate_min_score,
            )
        analysis_ranked_candidates: List[PrefetchCandidate] = []
        if analysis_expert_matrix is not None:
            analysis_ranked_candidates = rank_prefetch_candidates(
                layer_id=layer_idx,
                expert_matrix=analysis_expert_matrix,
                future_layers=int(analysis_future_layers),
                max_candidates=int(analysis_max_candidates),
                min_score=prefetch_candidate_min_score,
            )
        runtime_candidate_pairs = [
            {
                "layer_idx": int(candidate.layer_idx),
                "expert_idx": int(candidate.expert_idx),
                "score": float(candidate.score),
            }
            for candidate in runtime_ranked_candidates
        ]
        analysis_candidate_pairs = [
            {
                "layer_idx": int(candidate.layer_idx),
                "expert_idx": int(candidate.expert_idx),
                "score": float(candidate.score),
            }
            for candidate in analysis_ranked_candidates
        ]
        candidate_unique_score_ratio = _candidate_unique_score_ratio(
            runtime_candidate_pairs
        )
        candidate_topk_entropy = _candidate_topk_entropy(runtime_candidate_pairs)
        candidate_all_equal_top8 = _candidate_all_equal_topk(
            runtime_candidate_pairs
        )
        candidate_is_contiguous_top8 = _candidate_contiguous_topk(
            runtime_candidate_pairs
        )
        event = {
            "event_type": "policy_decision",
            "seq_id": seq_id,
            "request_id": state.request_id,
            "trace_name": state.trace_name,
            "tag": state.tag,
            "variant": state.variant,
            "is_warmup": state.is_warmup,
            "step_index": state.step_index,
            "layer_idx": int(layer_idx),
            "policy_name": policy_name,
            "prefetch_enabled": bool(prefetch_enabled),
            "score_only": bool(score_only),
            "decision_latency_us": int(decision_latency_us),
            "actual_experts": actual_experts,
            "actual_expert_counts": {
                str(expert_idx): int(count)
                for expert_idx, count in sorted(actual_counts.items())
            },
            "candidate_pairs": runtime_candidate_pairs,
            "candidate_experts": unique_candidate_experts(
                runtime_ranked_candidates
            ),
            "candidate_budget_max": int(prefetch_max_candidates),
            "prefetch_future_layers": int(prefetch_future_layers),
            "runtime_candidate_pairs": runtime_candidate_pairs,
            "runtime_candidate_experts": unique_candidate_experts(
                runtime_ranked_candidates
            ),
            "runtime_candidate_budget_max": int(prefetch_max_candidates),
            "runtime_prefetch_future_layers": int(prefetch_future_layers),
            "analysis_candidate_pairs": analysis_candidate_pairs,
            "analysis_candidate_experts": unique_candidate_experts(
                analysis_ranked_candidates
            ),
            "analysis_candidate_budget_max": int(analysis_max_candidates),
            "analysis_future_layers": int(analysis_future_layers),
            "candidate_source": str(candidate_source),
            "fallback_reason": str(fallback_reason),
            "library_match_count": int(library_match_count),
            "local_match_count": int(local_match_count),
            "raw_positive_count": int(raw_positive_count),
            "kept_candidate_count": int(kept_candidate_count),
            "all_scores_below_threshold": bool(all_scores_below_threshold),
            "candidate_unique_score_ratio": float(candidate_unique_score_ratio),
            "candidate_topk_entropy": float(candidate_topk_entropy),
            "candidate_all_equal_top8": bool(candidate_all_equal_top8),
            "candidate_is_contiguous_top8": bool(candidate_is_contiguous_top8),
            "backbone_mode": str(backbone_mode),
            "backbone_support_count": int(backbone_support_count),
            "backbone_history_match_count": int(backbone_history_match_count),
            "backbone_projection_topk": int(backbone_projection_topk),
            "backbone_fallback_used": bool(backbone_fallback_used),
            "consensus_min_votes": int(consensus_min_votes),
            "consensus_support_count": int(consensus_support_count),
            "consensus_retained_count": int(consensus_retained_count),
            "consensus_fallback_used": bool(consensus_fallback_used),
            "consensus_fallback_reason": str(consensus_fallback_reason),
            "historical_reuse_object_mode": str(historical_reuse_object_mode),
            "local_continuation_key_layers": int(local_continuation_key_layers),
            "local_continuation_future_layers": int(local_continuation_future_layers),
            "historical_similarity_mode": str(historical_similarity_mode),
            "historical_recent_window": int(historical_recent_window),
        }
        self._write_event(event)
        last_layer_idx = None
        if num_layers is not None and int(num_layers) > 0:
            last_layer_idx = int(num_layers) - 1
        elif expert_matrix is not None:
            last_layer_idx = int(expert_matrix.shape[0] - 1)
        if last_layer_idx is not None and int(layer_idx) == last_layer_idx:
            state.step_index += 1

    def _write_event(self, payload: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()


def load_phasea_events(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        events.append(json.loads(stripped))
    return events


def _policy_events_by_sequence(
    events: Sequence[Mapping[str, Any]],
    *,
    exclude_warmup_events: bool = False,
) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("event_type") != "policy_decision":
            continue
        if exclude_warmup_events and bool(event.get("is_warmup", False)):
            continue
        grouped[str(event["seq_id"])].append(event)
    for seq_id in grouped:
        grouped[seq_id].sort(
            key=lambda item: (int(item["step_index"]), int(item["layer_idx"]))
        )
    return grouped


def _future_events(
    sequence_events: Sequence[Mapping[str, Any]],
    *,
    current_index: int,
    horizon_steps: int,
) -> List[Mapping[str, Any]]:
    current_event = sequence_events[current_index]
    current_step = int(current_event["step_index"])
    current_layer = int(current_event["layer_idx"])
    future: List[Mapping[str, Any]] = []
    max_step = current_step + max(int(horizon_steps), 0)
    for next_event in sequence_events[current_index + 1 :]:
        next_step = int(next_event["step_index"])
        if next_step > max_step:
            break
        if next_step == current_step and int(next_event["layer_idx"]) <= current_layer:
            continue
        future.append(next_event)
    return future


def _same_step_window_future_events(
    sequence_events: Sequence[Mapping[str, Any]],
    *,
    current_index: int,
    future_layers: int,
) -> List[Mapping[str, Any]]:
    current_event = sequence_events[current_index]
    current_step = int(current_event["step_index"])
    current_layer = int(current_event["layer_idx"])
    upper_layer = None
    if int(future_layers) > 0:
        upper_layer = current_layer + int(future_layers)

    future: List[Mapping[str, Any]] = []
    for next_event in sequence_events[current_index + 1 :]:
        next_step = int(next_event["step_index"])
        if next_step > current_step:
            break
        next_layer = int(next_event["layer_idx"])
        if next_layer <= current_layer:
            continue
        if upper_layer is not None and next_layer > upper_layer:
            continue
        future.append(next_event)
    return future


def _runtime_candidate_pairs(event: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    pairs = event.get("runtime_candidate_pairs")
    if pairs is None:
        pairs = event.get("candidate_pairs", [])
    return list(pairs)


def _analysis_candidate_pairs(event: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    pairs = event.get("analysis_candidate_pairs")
    if pairs is None:
        pairs = event.get("candidate_pairs", [])
    return list(pairs)


def analyze_phasea_event_file(
    path: str | Path,
    *,
    horizons: Sequence[int] = (3, 5, 8),
    budgets: Sequence[int] = (4, 8, 16, 32),
    exclude_warmup_events: bool = False,
) -> Dict[str, Any]:
    events = load_phasea_events(path)
    unfiltered_policy_events = [
        event for event in events if event.get("event_type") == "policy_decision"
    ]
    warmup_policy_event_count = sum(
        1 for event in unfiltered_policy_events if bool(event.get("is_warmup", False))
    )
    grouped = _policy_events_by_sequence(
        events,
        exclude_warmup_events=exclude_warmup_events,
    )
    all_policy_events = [event for events_ in grouped.values() for event in events_]

    candidate_stats: Dict[str, Dict[str, float | int]] = {}
    utility_stats: Dict[str, Dict[str, float | int]] = {}
    gap_stats: Dict[str, Dict[str, float | int]] = {}
    same_step_stats: Dict[str, Dict[str, float | int]] = {}
    same_step_gap_stats: Dict[str, Dict[str, float | int]] = {}

    for horizon in horizons:
        per_budget_candidate: Dict[int, List[float]] = {int(b): [] for b in budgets}
        per_budget_expert_only_candidate: Dict[int, List[float]] = {
            int(b): [] for b in budgets
        }
        per_budget_oracle_gap: Dict[int, List[float]] = {int(b): [] for b in budgets}
        per_budget_restricted_gap: Dict[int, List[float]] = {int(b): [] for b in budgets}
        per_budget_correlations: Dict[int, List[float]] = {int(b): [] for b in budgets}

        for sequence_events in grouped.values():
            for idx, event in enumerate(sequence_events):
                future = _future_events(
                    sequence_events,
                    current_index=idx,
                    horizon_steps=int(horizon),
                )
                future_weights = _future_pair_weights(future)
                if not future_weights:
                    continue
                future_expert_weights: Counter[int] = Counter()
                for (_layer_idx, expert_idx), weight in future_weights.items():
                    future_expert_weights[int(expert_idx)] += int(weight)

                ranked_pairs = _analysis_candidate_pairs(event)
                for budget in budgets:
                    top_pairs = ranked_pairs[: int(budget)]
                    candidate_weights = {
                        (int(item["layer_idx"]), int(item["expert_idx"])): int(
                            future_weights.get(
                                (int(item["layer_idx"]), int(item["expert_idx"])), 0
                            )
                        )
                        for item in top_pairs
                    }
                    covered_weight = int(sum(candidate_weights.values()))
                    oracle_weight = _top_budget_weight(future_weights, budget=int(budget))
                    restricted_weight = _top_budget_weight(
                        {
                            pair: weight
                            for pair, weight in future_weights.items()
                            if pair in {
                                (int(item["layer_idx"]), int(item["expert_idx"]))
                                for item in ranked_pairs
                            }
                        },
                        budget=int(budget),
                    )

                    recall = (
                        covered_weight / max(int(sum(future_weights.values())), 1)
                    )
                    per_budget_candidate[int(budget)].append(float(recall))
                    candidate_expert_ids = {
                        int(item["expert_idx"]) for item in top_pairs
                    }
                    covered_expert_weight = int(
                        sum(
                            int(weight)
                            for expert_idx, weight in future_expert_weights.items()
                            if int(expert_idx) in candidate_expert_ids
                        )
                    )
                    expert_only_recall = covered_expert_weight / max(
                        int(sum(future_expert_weights.values())),
                        1,
                    )
                    per_budget_expert_only_candidate[int(budget)].append(
                        float(expert_only_recall)
                    )

                    if oracle_weight > 0:
                        omission_gap = max(oracle_weight - restricted_weight, 0) / oracle_weight
                        restricted_gap = max(restricted_weight - covered_weight, 0) / oracle_weight
                    else:
                        omission_gap = 0.0
                        restricted_gap = 0.0
                    per_budget_oracle_gap[int(budget)].append(float(omission_gap))
                    per_budget_restricted_gap[int(budget)].append(float(restricted_gap))

                    if top_pairs:
                        scores = [float(item["score"]) for item in top_pairs]
                        realized = [
                            float(
                                future_weights.get(
                                    (int(item["layer_idx"]), int(item["expert_idx"])),
                                    0,
                                )
                            )
                            for item in top_pairs
                        ]
                        per_budget_correlations[int(budget)].append(
                            _spearman(scores, realized)
                        )

        for budget in budgets:
            key = f"h{int(horizon)}_M{int(budget)}"
            candidate_stats[key] = {
                "mean_recall": _safe_mean(per_budget_candidate[int(budget)]),
                "mean_pair_recall": _safe_mean(per_budget_candidate[int(budget)]),
                "mean_expert_only_recall": _safe_mean(
                    per_budget_expert_only_candidate[int(budget)]
                ),
                "num_events": len(per_budget_candidate[int(budget)]),
            }
            gap_stats[key] = {
                "candidate_omission_gap_mean": _safe_mean(
                    per_budget_oracle_gap[int(budget)]
                ),
                "restricted_oracle_gap_mean": _safe_mean(
                    per_budget_restricted_gap[int(budget)]
                ),
            }
            utility_stats[key] = {
                "utility_benefit_spearman_mean": _safe_mean(
                    per_budget_correlations[int(budget)]
                ),
                "num_events": len(per_budget_correlations[int(budget)]),
            }

    per_budget_same_step_candidate: Dict[int, List[float]] = {
        int(b): [] for b in budgets
    }
    per_budget_same_step_expert_only: Dict[int, List[float]] = {
        int(b): [] for b in budgets
    }
    per_budget_same_step_omission: Dict[int, List[float]] = {
        int(b): [] for b in budgets
    }
    per_budget_same_step_restricted: Dict[int, List[float]] = {
        int(b): [] for b in budgets
    }
    for sequence_events in grouped.values():
        for idx, event in enumerate(sequence_events):
            future_layers = int(
                event.get(
                    "runtime_prefetch_future_layers",
                    event.get("prefetch_future_layers", 0),
                )
            )
            future = _same_step_window_future_events(
                sequence_events,
                current_index=idx,
                future_layers=future_layers,
            )
            future_weights = _future_pair_weights(future)
            if not future_weights:
                continue
            future_expert_weights: Counter[int] = Counter()
            for (_layer_idx, expert_idx), weight in future_weights.items():
                future_expert_weights[int(expert_idx)] += int(weight)
            ranked_pairs = _analysis_candidate_pairs(event)
            ranked_pair_set = {
                (int(item["layer_idx"]), int(item["expert_idx"]))
                for item in ranked_pairs
            }
            total_future_pair_weight = max(int(sum(future_weights.values())), 1)
            total_future_expert_weight = max(int(sum(future_expert_weights.values())), 1)
            for budget in budgets:
                top_pairs = ranked_pairs[: int(budget)]
                covered_weight = int(
                    sum(
                        int(
                            future_weights.get(
                                (int(item["layer_idx"]), int(item["expert_idx"])),
                                0,
                            )
                        )
                        for item in top_pairs
                    )
                )
                oracle_weight = _top_budget_weight(future_weights, budget=int(budget))
                restricted_weight = _top_budget_weight(
                    {
                        pair: weight
                        for pair, weight in future_weights.items()
                        if pair in ranked_pair_set
                    },
                    budget=int(budget),
                )
                per_budget_same_step_candidate[int(budget)].append(
                    covered_weight / total_future_pair_weight
                )
                candidate_expert_ids = {
                    int(item["expert_idx"]) for item in top_pairs
                }
                covered_expert_weight = int(
                    sum(
                        int(weight)
                        for expert_idx, weight in future_expert_weights.items()
                        if int(expert_idx) in candidate_expert_ids
                    )
                )
                per_budget_same_step_expert_only[int(budget)].append(
                    covered_expert_weight / total_future_expert_weight
                )
                if oracle_weight > 0:
                    omission_gap = max(oracle_weight - restricted_weight, 0) / oracle_weight
                    restricted_gap = max(restricted_weight - covered_weight, 0) / oracle_weight
                else:
                    omission_gap = 0.0
                    restricted_gap = 0.0
                per_budget_same_step_omission[int(budget)].append(float(omission_gap))
                per_budget_same_step_restricted[int(budget)].append(float(restricted_gap))

    for budget in budgets:
        key = f"M{int(budget)}"
        same_step_stats[key] = {
            "mean_pair_recall": _safe_mean(per_budget_same_step_candidate[int(budget)]),
            "mean_expert_only_recall": _safe_mean(
                per_budget_same_step_expert_only[int(budget)]
            ),
            "num_events": len(per_budget_same_step_candidate[int(budget)]),
        }
        same_step_gap_stats[key] = {
            "candidate_omission_gap_mean": _safe_mean(
                per_budget_same_step_omission[int(budget)]
            ),
            "restricted_oracle_gap_mean": _safe_mean(
                per_budget_same_step_restricted[int(budget)]
            ),
        }

    decision_latencies = [
        float(event.get("decision_latency_us", 0.0))
        for event in all_policy_events
    ]
    traces = sorted(
        {
            str(event.get("trace_name", ""))
            for event in all_policy_events
            if event.get("trace_name")
        }
    )
    backbone_modes = Counter(
        str(event.get("backbone_mode", "disabled")) for event in all_policy_events
    )
    candidate_source_counts = Counter(
        str(event.get("candidate_source", "empty")) for event in all_policy_events
    )
    fallback_reason_counts = Counter(
        str(event.get("fallback_reason", "none")) for event in all_policy_events
    )
    backbone_support_counts = [
        int(event.get("backbone_support_count", 0)) for event in all_policy_events
    ]
    backbone_history_match_counts = [
        int(event.get("backbone_history_match_count", 0))
        for event in all_policy_events
    ]
    backbone_fallback_count = sum(
        1 for event in all_policy_events if bool(event.get("backbone_fallback_used", False))
    )
    consensus_support_counts = [
        int(event.get("consensus_support_count", 0)) for event in all_policy_events
    ]
    consensus_retained_counts = [
        int(event.get("consensus_retained_count", 0)) for event in all_policy_events
    ]
    consensus_fallback_reason_counts = Counter(
        str(event.get("consensus_fallback_reason", "none"))
        for event in all_policy_events
    )
    historical_similarity_mode_counts = Counter(
        str(event.get("historical_similarity_mode", "prefix_mean"))
        for event in all_policy_events
    )
    historical_reuse_object_mode_counts = Counter(
        str(event.get("historical_reuse_object_mode", "sequence_matrix"))
        for event in all_policy_events
    )
    historical_recent_window_counts = Counter(
        int(event.get("historical_recent_window", 4)) for event in all_policy_events
    )
    local_match_counts = [
        int(event.get("local_match_count", 0)) for event in all_policy_events
    ]
    variants = sorted(
        {
            str(event.get("variant", ""))
            for event in all_policy_events
            if event.get("variant")
        }
    )
    nonempty_policy_events = [
        event for event in all_policy_events if _runtime_candidate_pairs(event)
    ]
    candidate_signatures = [
        _candidate_signature(_runtime_candidate_pairs(event), budget=32)
        for event in nonempty_policy_events
    ]
    signature_counts = Counter(candidate_signatures)
    fixed_signature_ratio = (
        max(signature_counts.values()) / len(candidate_signatures)
        if candidate_signatures
        else 0.0
    )
    candidate_health = {
        "candidate_nonempty_rate": (
            len(nonempty_policy_events) / len(all_policy_events)
            if all_policy_events
            else 0.0
        ),
        "candidate_empty_rate": (
            (len(all_policy_events) - len(nonempty_policy_events)) / len(all_policy_events)
            if all_policy_events
            else 0.0
        ),
        "candidate_fixed_signature_ratio": float(fixed_signature_ratio),
        "candidate_unique_score_ratio_mean": _safe_mean(
            [
                float(event.get("candidate_unique_score_ratio", 0.0))
                for event in nonempty_policy_events
            ]
        ),
        "candidate_topk_entropy_mean": _safe_mean(
            [
                float(event.get("candidate_topk_entropy", 0.0))
                for event in nonempty_policy_events
            ]
        ),
        "candidate_all_equal_top8_rate": _safe_mean(
            [
                1.0 if bool(event.get("candidate_all_equal_top8", False)) else 0.0
                for event in nonempty_policy_events
            ]
        ),
        "candidate_contiguous_top8_rate": _safe_mean(
            [
                1.0
                if bool(event.get("candidate_is_contiguous_top8", False))
                else 0.0
                for event in nonempty_policy_events
            ]
        ),
        "candidate_source_counts": dict(candidate_source_counts),
        "fallback_reason_counts": dict(fallback_reason_counts),
    }
    return {
        "event_path": str(path),
        "trace_names": traces,
        "variants": variants,
        "instrumentation_validation": {
            "policy_event_count": len(all_policy_events),
            "unfiltered_policy_event_count": len(unfiltered_policy_events),
            "warmup_policy_event_count": warmup_policy_event_count,
            "excluded_warmup_event_count": (
                warmup_policy_event_count if exclude_warmup_events else 0
            ),
            "sequence_event_count": len(events) - len(unfiltered_policy_events),
            "decision_latency_mean_us": _safe_mean(decision_latencies),
            "decision_latency_p95_us": _safe_percentile(decision_latencies, 95),
            "backbone_mode_counts": dict(backbone_modes),
            "backbone_support_count_mean": _safe_mean(backbone_support_counts),
            "backbone_history_match_count_mean": _safe_mean(
                backbone_history_match_counts
            ),
            "backbone_fallback_count": backbone_fallback_count,
            "consensus_support_count_mean": _safe_mean(consensus_support_counts),
            "consensus_retained_count_mean": _safe_mean(consensus_retained_counts),
            "consensus_fallback_reason_counts": dict(
                consensus_fallback_reason_counts
            ),
            "historical_similarity_mode_counts": dict(
                historical_similarity_mode_counts
            ),
            "historical_reuse_object_mode_counts": dict(
                historical_reuse_object_mode_counts
            ),
            "historical_recent_window_counts": dict(
                historical_recent_window_counts
            ),
            "local_match_count_mean": _safe_mean(local_match_counts),
        },
        "same_step_window_recall": same_step_stats,
        "same_step_window_gap_decomposition": same_step_gap_stats,
        "candidate_recall": candidate_stats,
        "candidate_health": candidate_health,
        "utility_correlation": utility_stats,
        "gap_decomposition": gap_stats,
        "controller_latency": {
            "decision_latency_mean_us": _safe_mean(decision_latencies),
            "decision_latency_p95_us": _safe_percentile(decision_latencies, 95),
        },
    }


def render_phasea_markdown_summary(
    *,
    analyses: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Phase-A Observation Summary",
        "",
    ]
    for analysis in analyses:
        lines.extend(
            [
                f"## `{analysis['event_path']}`",
                "",
                f"- Traces: `{', '.join(analysis.get('trace_names', []))}`",
                f"- Variants: `{', '.join(analysis.get('variants', []))}`",
                f"- Policy events: `{analysis['instrumentation_validation']['policy_event_count']}`",
                f"- Mean decision latency: `{analysis['controller_latency']['decision_latency_mean_us']:.2f} us`",
                f"- P95 decision latency: `{analysis['controller_latency']['decision_latency_p95_us']:.2f} us`",
            ]
        )
        mode_counts = analysis["instrumentation_validation"].get(
            "backbone_mode_counts", {}
        )
        if mode_counts:
            lines.append(f"- Backbone mode counts: `{mode_counts}`")
        consensus_counts = analysis["instrumentation_validation"].get(
            "consensus_fallback_reason_counts", {}
        )
        if consensus_counts:
            lines.append(f"- Consensus fallback counts: `{consensus_counts}`")
        similarity_counts = analysis["instrumentation_validation"].get(
            "historical_similarity_mode_counts", {}
        )
        if similarity_counts:
            lines.append(f"- Historical similarity mode counts: `{similarity_counts}`")
        object_mode_counts = analysis["instrumentation_validation"].get(
            "historical_reuse_object_mode_counts", {}
        )
        if object_mode_counts:
            lines.append(
                f"- Historical object mode counts: `{object_mode_counts}`"
            )
        recent_window_counts = analysis["instrumentation_validation"].get(
            "historical_recent_window_counts", {}
        )
        if recent_window_counts:
            lines.append(f"- Historical recent-window counts: `{recent_window_counts}`")
        overhead = analysis["instrumentation_validation"].get("latency_overhead_ratio")
        if overhead is not None:
            lines.append(
                f"- Latency overhead vs reference: `{float(overhead) * 100.0:.2f}%`"
            )
        candidate_health = analysis.get("candidate_health", {})
        if candidate_health:
            lines.extend(
                [
                    f"- Candidate source counts: `{candidate_health.get('candidate_source_counts', {})}`",
                    f"- Fallback reason counts: `{candidate_health.get('fallback_reason_counts', {})}`",
                    f"- Candidate non-empty rate: `{float(candidate_health.get('candidate_nonempty_rate', 0.0)):.4f}`",
                    f"- Candidate fixed-signature ratio: `{float(candidate_health.get('candidate_fixed_signature_ratio', 0.0)):.4f}`",
                    f"- Candidate unique-score ratio mean: `{float(candidate_health.get('candidate_unique_score_ratio_mean', 0.0)):.4f}`",
                    f"- Candidate top-k entropy mean: `{float(candidate_health.get('candidate_topk_entropy_mean', 0.0)):.4f}`",
                    f"- Candidate all-equal-top8 rate: `{float(candidate_health.get('candidate_all_equal_top8_rate', 0.0)):.4f}`",
                    f"- Candidate contiguous-top8 rate: `{float(candidate_health.get('candidate_contiguous_top8_rate', 0.0)):.4f}`",
                ]
            )
        same_step = analysis.get("same_step_window_recall", {})
        if same_step:
            lines.extend(
                [
                    "",
                    "| Same-step budget | Pair recall | Expert-only recall | Omission gap | Restricted gap |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for key in sorted(same_step):
                recall = analysis["same_step_window_recall"][key]["mean_pair_recall"]
                expert_only = analysis["same_step_window_recall"][key]["mean_expert_only_recall"]
                omission = analysis["same_step_window_gap_decomposition"][key]["candidate_omission_gap_mean"]
                restricted = analysis["same_step_window_gap_decomposition"][key]["restricted_oracle_gap_mean"]
                lines.append(
                    f"| {key} | {recall:.4f} | {expert_only:.4f} | {omission:.4f} | {restricted:.4f} |"
                )
        lines.extend(
            [
                "",
                "| Horizon/Budget | Pair recall | Expert-only recall | Omission gap | Restricted gap | Utility Spearman |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key in sorted(analysis["candidate_recall"]):
            recall = analysis["candidate_recall"][key]["mean_pair_recall"]
            expert_only = analysis["candidate_recall"][key]["mean_expert_only_recall"]
            omission = analysis["gap_decomposition"][key]["candidate_omission_gap_mean"]
            restricted = analysis["gap_decomposition"][key]["restricted_oracle_gap_mean"]
            corr = analysis["utility_correlation"][key]["utility_benefit_spearman_mean"]
            lines.append(
                f"| {key} | {recall:.4f} | {expert_only:.4f} | {omission:.4f} | {restricted:.4f} | {corr:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)
