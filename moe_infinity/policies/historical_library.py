from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class HistoricalLibraryEntry:
    seq_id: str
    matrix: np.ndarray
    model_tag: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 1
    insertion_index: int = 0


class HistoricalExpertLibrary:
    def __init__(
        self,
        capacity: int,
        *,
        metric: str = "cosine",
        similarity_mode: str = "prefix_mean",
        recent_window: int = 4,
        recent_weight: float = 0.5,
        admission_policy: str = "diversity_aware",
        dedup_threshold: float = 0.995,
    ) -> None:
        self.capacity = max(int(capacity), 1)
        self.metric = metric
        self.similarity_mode = str(similarity_mode)
        self.recent_window = max(int(recent_window), 1)
        self.recent_weight = min(max(float(recent_weight), 0.0), 1.0)
        self.admission_policy = admission_policy
        self.dedup_threshold = float(dedup_threshold)
        self.entries: List[HistoricalLibraryEntry] = []
        self._insertion_counter = 0
        self.query_count = 0
        self.hit_count = 0
        self.admit_count = 0
        self.duplicate_update_count = 0

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        matrix = matrix.astype(np.float32, copy=True)
        denom = np.sum(matrix, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return matrix / denom

    def _cosine_similarity(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        *,
        start_layer: Optional[int] = None,
        upto_layer: Optional[int] = None,
    ) -> float:
        if upto_layer is not None:
            lhs = lhs[: upto_layer + 1]
            rhs = rhs[: upto_layer + 1]
        if start_layer is not None and start_layer > 0:
            lhs = lhs[start_layer:]
            rhs = rhs[start_layer:]
        if lhs.size == 0 or rhs.size == 0:
            return 0.0
        lhs = self._normalize_rows(lhs)
        rhs = self._normalize_rows(rhs)
        numer = np.sum(lhs * rhs, axis=1)
        lhs_norm = np.linalg.norm(lhs, axis=1)
        rhs_norm = np.linalg.norm(rhs, axis=1)
        denom = lhs_norm * rhs_norm
        denom[denom == 0] = 1.0
        score = numer / denom
        return float(np.mean(score))

    def _similarity(self, lhs: np.ndarray, rhs: np.ndarray, upto_layer: Optional[int] = None) -> float:
        if upto_layer is not None:
            lhs = lhs[: upto_layer + 1]
            rhs = rhs[: upto_layer + 1]
        if self.metric == "cosine":
            return self._cosine_similarity(lhs, rhs)
        if self.metric == "l2":
            lhs = self._normalize_rows(lhs)
            rhs = self._normalize_rows(rhs)
            return float(-np.mean(np.linalg.norm(lhs - rhs, axis=1)))
        if self.metric == "l1":
            lhs = self._normalize_rows(lhs)
            rhs = self._normalize_rows(rhs)
            return float(-np.mean(np.sum(np.abs(lhs - rhs), axis=1)))
        raise ValueError(f"Unsupported library metric: {self.metric}")

    def _retrieval_similarity(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        *,
        layer_idx: int,
    ) -> float:
        prefix_score = self._similarity(lhs, rhs, upto_layer=layer_idx)
        if self.metric != "cosine" or self.similarity_mode != "prefix_recent_hybrid":
            return prefix_score
        recent_start = max(int(layer_idx) - self.recent_window + 1, 0)
        recent_score = self._cosine_similarity(
            lhs,
            rhs,
            start_layer=recent_start,
            upto_layer=layer_idx,
        )
        alpha = self.recent_weight
        return float((1.0 - alpha) * prefix_score + alpha * recent_score)

    def _find_duplicate(self, matrix: np.ndarray, model_tag: str) -> Optional[int]:
        for idx, entry in enumerate(self.entries):
            if entry.model_tag != model_tag:
                continue
            if self._similarity(matrix, entry.matrix) >= self.dedup_threshold:
                return idx
        return None

    def _evict_index(self) -> int:
        if self.admission_policy == "recent_only":
            return min(
                range(len(self.entries)),
                key=lambda idx: self.entries[idx].insertion_index,
            )
        if self.admission_policy == "frequency_weighted":
            return min(
                range(len(self.entries)),
                key=lambda idx: (
                    self.entries[idx].access_count,
                    self.entries[idx].insertion_index,
                ),
            )
        if self.admission_policy == "diversity_aware":
            best_idx = 0
            best_score = None
            for idx, entry in enumerate(self.entries):
                if len(self.entries) == 1:
                    redundancy = 1.0
                else:
                    redundancy = max(
                        self._similarity(entry.matrix, other.matrix)
                        for j, other in enumerate(self.entries)
                        if j != idx
                    )
                score = (round(redundancy, 6), -entry.access_count, entry.insertion_index)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
            return best_idx
        raise ValueError(f"Unsupported admission policy: {self.admission_policy}")

    def admit(self, seq_id: str, matrix: np.ndarray, *, model_tag: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        metadata = metadata or {}
        self.admit_count += 1
        duplicate_idx = self._find_duplicate(matrix, model_tag)
        if duplicate_idx is not None:
            entry = self.entries[duplicate_idx]
            entry.access_count += 1
            entry.matrix = matrix.astype(np.float32, copy=True)
            entry.metadata.update(metadata)
            self.duplicate_update_count += 1
            return

        entry = HistoricalLibraryEntry(
            seq_id=seq_id,
            matrix=matrix.astype(np.float32, copy=True),
            model_tag=model_tag,
            metadata=dict(metadata),
            access_count=1,
            insertion_index=self._insertion_counter,
        )
        self._insertion_counter += 1

        if len(self.entries) >= self.capacity:
            self.entries.pop(self._evict_index())
        self.entries.append(entry)

    def topk(
        self,
        current_matrix: np.ndarray,
        *,
        layer_idx: int,
        model_tag: str = "",
        k: int = 1,
    ) -> List[Tuple[HistoricalLibraryEntry, float]]:
        self.query_count += 1
        candidates = [
            entry for entry in self.entries if not model_tag or entry.model_tag == model_tag
        ]
        if not candidates:
            return []
        scored = [
            (
                entry,
                self._retrieval_similarity(
                    current_matrix,
                    entry.matrix,
                    layer_idx=layer_idx,
                ),
            )
            for entry in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        result = scored[:k]
        if result:
            self.hit_count += 1
        return result

    def __len__(self) -> int:
        return len(self.entries)

    def stats(self) -> Dict[str, int]:
        return {
            "size": len(self.entries),
            "query_count": self.query_count,
            "hit_count": self.hit_count,
            "admit_count": self.admit_count,
            "duplicate_update_count": self.duplicate_update_count,
        }
