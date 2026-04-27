from __future__ import annotations

import numpy as np

from .base import BackboneSupport, OffloadingPolicy
from .historical_library import HistoricalExpertLibrary
from .local_continuation_library import LocalContinuationLibrary


class FineGrainedHistoryReusePolicy(OffloadingPolicy):
    def __init__(
        self,
        *,
        config,
        tracer,
        predictor,
        library: HistoricalExpertLibrary,
        local_library: LocalContinuationLibrary | None = None,
        model_tag: str = "",
    ):
        super().__init__(config=config, tracer=tracer, predictor=predictor, model_tag=model_tag)
        self.library = library
        self.local_library = local_library
        self._cached_matches = {}
        self._cached_local_matches = {}

    def _object_mode(self) -> str:
        return str(
            getattr(self.config, "historical_reuse_object_mode", "sequence_matrix")
        )

    def _backbone_history_topk(self) -> int:
        return max(int(getattr(self.config, "prefetch_backbone_history_topk", 1)), 1)

    def _candidate_mode(self) -> str:
        return str(getattr(self.config, "historical_reuse_candidate_mode", "top1"))

    def _candidate_history_topk(self) -> int:
        return max(int(getattr(self.config, "historical_reuse_match_topk", 1)), 1)

    def _candidate_history_min_required(self) -> int:
        return max(
            int(getattr(self.config, "historical_reuse_match_min_required", 1)),
            1,
        )

    def _local_key_layers(self) -> int:
        return max(
            int(getattr(self.config, "local_continuation_key_layers", 4)),
            1,
        )

    def _local_future_layers(self) -> int:
        return max(
            int(getattr(self.config, "local_continuation_future_layers", 4)),
            1,
        )

    def _local_match_topk(self) -> int:
        return max(
            int(getattr(self.config, "local_continuation_match_topk", 4)),
            1,
        )

    def _local_match_min_required(self) -> int:
        return max(
            int(getattr(self.config, "local_continuation_match_min_required", 2)),
            1,
        )

    def _consensus_min_votes(self) -> int:
        return max(
            int(getattr(self.config, "historical_reuse_consensus_min_votes", 1)),
            1,
        )

    def _lookup_matches(self, seq_id: str, layer_idx: int):
        current_entry = self.tracer.get_entry(seq_id)
        matches = self.library.topk(
            current_entry.matrix,
            layer_idx=layer_idx,
            model_tag=self.model_tag,
            k=max(self._backbone_history_topk(), self._candidate_history_topk()),
        )
        self._cached_matches[(seq_id, layer_idx)] = matches
        return matches

    def _build_local_query_key(self, seq_id: str, layer_idx: int) -> np.ndarray:
        step_matrix = self.tracer.get_current_step_matrix(seq_id)
        return LocalContinuationLibrary.build_key_slice(
            step_matrix,
            anchor_layer_idx=layer_idx,
            key_layers=self._local_key_layers(),
        )

    def _lookup_local_matches(self, seq_id: str, layer_idx: int):
        if self.local_library is None:
            return []
        key_slice = self._build_local_query_key(seq_id, layer_idx)
        matches = self.local_library.topk(
            key_slice,
            anchor_layer_idx=layer_idx,
            model_tag=self.model_tag,
            k=self._local_match_topk(),
        )
        self._cached_local_matches[(seq_id, layer_idx)] = matches
        return matches

    @staticmethod
    def _normalize_match_weights(
        matches,
        *,
        metric: str,
    ) -> np.ndarray:
        if not matches:
            return np.zeros((0,), dtype=np.float32)
        if metric == "cosine":
            weights = np.asarray(
                [max(0.0, min(float(similarity), 1.0)) for _entry, similarity in matches],
                dtype=np.float32,
            )
        else:
            weights = np.ones((len(matches),), dtype=np.float32)
        total = float(np.sum(weights))
        if total <= 0.0:
            weights = np.ones((len(matches),), dtype=np.float32) / float(len(matches))
        else:
            weights = weights / total
        return weights

    def _aggregate_history_matches(self, matches, *, layer_idx: int) -> np.ndarray:
        metric = str(getattr(self.config, "historical_library_metric", "cosine"))
        weights = self._normalize_match_weights(matches, metric=metric)
        support_matrices = []
        for entry, _similarity in matches:
            entry.access_count += 1
            support_matrices.append(
                self.predictor.build_prefetch_matrix(entry.matrix, layer_idx)
            )
        stacked = np.stack(
            [np.asarray(matrix, dtype=np.float32) for matrix in support_matrices],
            axis=0,
        )
        return np.tensordot(weights, stacked, axes=(0, 0)).astype(np.float32)

    def _aggregate_local_matches(self, matches) -> np.ndarray:
        metric = str(getattr(self.config, "historical_library_metric", "cosine"))
        weights = self._normalize_match_weights(matches, metric=metric)
        future_layers = self._local_future_layers()
        local_buffer = np.zeros(
            (future_layers, int(self.predictor.num_experts)),
            dtype=np.float32,
        )
        for weight, (entry, _similarity) in zip(weights.tolist(), matches):
            entry.access_count += 1
            if weight <= 0.0:
                continue
            for future_offset, expert_idx, score in entry.value_pairs:
                if future_offset >= future_layers:
                    continue
                local_buffer[future_offset, expert_idx] += float(weight) * float(score)
        return local_buffer

    def _expand_local_value_slice(
        self,
        value_slice: np.ndarray,
        *,
        layer_idx: int,
    ) -> np.ndarray:
        num_layers = int(self.predictor.num_layers)
        num_experts = int(self.predictor.num_experts)
        expanded = np.zeros((num_layers, num_experts), dtype=np.float32)
        start = int(layer_idx) + 1
        end = min(num_layers, start + value_slice.shape[0])
        if start < end:
            expanded[start:end] = np.asarray(value_slice[: end - start], dtype=np.float32)
        return expanded

    def _score_from_sequence_history(
        self,
        seq_id: str,
        layer_idx: int,
        *,
        fallback_candidate_source: str = "history_fallback_current_trace",
        fallback_reason: str = "library_miss",
        local_match_count: int = 0,
        matched_candidate_source: str | None = None,
        matched_fallback_reason: str = "none",
    ) -> np.ndarray:
        matches = self._lookup_matches(seq_id, layer_idx)
        if not matches:
            self.set_score_metadata(
                candidate_source=fallback_candidate_source,
                fallback_reason=fallback_reason,
                library_match_count=0,
                local_match_count=int(local_match_count),
            )
            return self.predictor.predict_from_current_trace(seq_id, layer_idx)
        candidate_mode = self._candidate_mode()
        if candidate_mode != "topk":
            matched_entry, _ = matches[0]
            matched_entry.access_count += 1
            self.set_score_metadata(
                candidate_source=matched_candidate_source or "history_match",
                fallback_reason=matched_fallback_reason,
                library_match_count=len(matches[:1]),
                local_match_count=int(local_match_count),
            )
            return self.predictor.build_prefetch_matrix(
                matched_entry.matrix,
                layer_idx,
            )

        topk_matches = matches[: self._candidate_history_topk()]
        min_required = self._candidate_history_min_required()
        if len(topk_matches) < min_required:
            matched_entry, _ = topk_matches[0]
            matched_entry.access_count += 1
            self.set_score_metadata(
                candidate_source=matched_candidate_source or "history_match_top1_fallback",
                fallback_reason=(
                    matched_fallback_reason
                    if matched_candidate_source is not None
                    else "history_match_insufficient"
                ),
                library_match_count=len(topk_matches),
                local_match_count=int(local_match_count),
            )
            return self.predictor.build_prefetch_matrix(
                matched_entry.matrix,
                layer_idx,
            )

        aggregated = self._aggregate_history_matches(
            topk_matches,
            layer_idx=layer_idx,
        )
        self.set_score_metadata(
            candidate_source=matched_candidate_source or "history_match_topk",
            fallback_reason=matched_fallback_reason,
            library_match_count=len(topk_matches),
            local_match_count=int(local_match_count),
        )
        return aggregated

    def _score_from_local_history(self, seq_id: str, layer_idx: int) -> np.ndarray:
        matches = self._lookup_local_matches(seq_id, layer_idx)
        if not matches:
            return self._score_from_sequence_history(
                seq_id,
                layer_idx,
                fallback_candidate_source="local_continuation_fallback_sequence",
                fallback_reason="local_library_miss",
                local_match_count=0,
                matched_candidate_source="local_continuation_fallback_sequence",
                matched_fallback_reason="local_library_miss",
            )
        min_required = self._local_match_min_required()
        if len(matches) < min_required:
            return self._score_from_sequence_history(
                seq_id,
                layer_idx,
                fallback_candidate_source="local_continuation_fallback_sequence",
                fallback_reason="local_match_insufficient",
                local_match_count=len(matches),
                matched_candidate_source="local_continuation_fallback_sequence",
                matched_fallback_reason="local_match_insufficient",
            )

        aggregated = self._aggregate_local_matches(matches)
        expanded = self._expand_local_value_slice(
            aggregated,
            layer_idx=layer_idx,
        )
        self.set_score_metadata(
            candidate_source="local_continuation_match",
            fallback_reason="none",
            library_match_count=len(matches),
            local_match_count=len(matches),
        )
        return expanded

    def consensus_backbone_mask(
        self,
        seq_id: str,
        layer_idx: int,
        *,
        min_score: float,
    ) -> tuple[np.ndarray | None, dict]:
        matches = self._cached_matches.get((seq_id, layer_idx))
        if matches is None:
            matches = self._lookup_matches(seq_id, layer_idx)

        topk_matches = matches[: self._candidate_history_topk()]
        metadata = {
            "library_match_count": len(topk_matches),
            "consensus_support_count": len(topk_matches),
            "consensus_min_votes": self._consensus_min_votes(),
            "consensus_retained_count": 0,
            "consensus_fallback_used": False,
            "consensus_fallback_reason": "none",
        }
        if not topk_matches:
            metadata["consensus_fallback_used"] = True
            metadata["consensus_fallback_reason"] = "library_miss"
            return None, metadata
        if len(topk_matches) < self._candidate_history_min_required():
            metadata["consensus_fallback_used"] = True
            metadata["consensus_fallback_reason"] = "history_match_insufficient"
            return None, metadata

        support_matrices = [
            np.asarray(
                self.predictor.build_prefetch_matrix(entry.matrix, layer_idx),
                dtype=np.float32,
            )
            for entry, _similarity in topk_matches
        ]
        votes = np.zeros_like(support_matrices[0], dtype=np.int32)
        threshold = float(min_score)
        for matrix in support_matrices:
            votes += (matrix > threshold).astype(np.int32)
        mask = votes >= int(metadata["consensus_min_votes"])
        metadata["consensus_retained_count"] = int(np.count_nonzero(mask))
        return mask.astype(np.bool_), metadata

    def score_for_prefetch(self, seq_id: str, layer_idx: int) -> np.ndarray:
        if self._object_mode() == "local_continuation":
            return self._score_from_local_history(seq_id, layer_idx)
        return self._score_from_sequence_history(seq_id, layer_idx)

    def backbone_supports(
        self,
        seq_id: str,
        layer_idx: int,
        base_matrix: np.ndarray,
    ) -> list[BackboneSupport]:
        supports = super().backbone_supports(
            seq_id=seq_id,
            layer_idx=layer_idx,
            base_matrix=base_matrix,
        )
        min_matches = max(
            int(getattr(self.config, "prefetch_backbone_min_matches", 0)),
            0,
        )
        matches = self._cached_matches.get((seq_id, layer_idx))
        if matches is None:
            matches = self._lookup_matches(seq_id, layer_idx)
        history_matches = matches[: self._backbone_history_topk()]
        if len(history_matches) < min_matches:
            return supports

        metric = str(getattr(self.config, "historical_library_metric", "cosine"))
        for idx, (entry, similarity) in enumerate(history_matches):
            if metric == "cosine":
                weight = max(0.0, min(float(similarity), 1.0))
            else:
                weight = 1.0
            support_matrix = self.predictor.build_prefetch_matrix(
                entry.matrix,
                layer_idx,
            )
            supports.append(
                BackboneSupport(
                    matrix=support_matrix,
                    weight=weight,
                    source=f"history_match_{idx}",
                )
            )
        return supports

    def finish_sequence(self, seq_id: str) -> None:
        entry = self.tracer.get_entry(seq_id)
        self.library.admit(
            seq_id,
            entry.matrix,
            model_tag=self.model_tag,
            metadata={"num_new_tokens": entry.num_new_tokens},
        )
        if self.local_library is not None:
            for step_index, step_matrix in enumerate(
                self.tracer.get_completed_step_matrices(seq_id)
            ):
                for anchor_layer_idx in range(max(step_matrix.shape[0] - 1, 0)):
                    key_slice = LocalContinuationLibrary.build_key_slice(
                        step_matrix,
                        anchor_layer_idx=anchor_layer_idx,
                        key_layers=self._local_key_layers(),
                    )
                    value_slice = LocalContinuationLibrary.build_value_slice(
                        step_matrix,
                        anchor_layer_idx=anchor_layer_idx,
                        future_layers=self._local_future_layers(),
                    )
                    if not np.any(value_slice > 0):
                        continue
                    self.local_library.admit(
                        seq_id=seq_id,
                        step_index=step_index,
                        anchor_layer_idx=anchor_layer_idx,
                        key_slice=key_slice,
                        value_slice=value_slice,
                        model_tag=self.model_tag,
                        metadata={"num_new_tokens": entry.num_new_tokens},
                    )
        self._cached_matches = {
            key: value for key, value in self._cached_matches.items() if key[0] != seq_id
        }
        self._cached_local_matches = {
            key: value
            for key, value in self._cached_local_matches.items()
            if key[0] != seq_id
        }
