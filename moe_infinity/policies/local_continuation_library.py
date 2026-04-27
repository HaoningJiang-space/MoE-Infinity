from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LocalContinuationEntry:
    seq_id: str
    step_index: int
    anchor_layer_idx: int
    key_slice: np.ndarray
    value_slice: np.ndarray
    key_rows_normalized: np.ndarray
    key_norm_flat: np.ndarray
    value_pairs: List[Tuple[int, int, float]]
    model_tag: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 1
    insertion_index: int = 0


@dataclass
class _LocalContinuationBucket:
    entries: List[LocalContinuationEntry] = field(default_factory=list)
    key_bank: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )


class LocalContinuationLibrary:
    def __init__(
        self,
        capacity: int,
        *,
        key_layers: int,
        future_layers: int,
        metric: str = "cosine",
        admission_policy: str = "diversity_aware",
        dedup_threshold: float = 0.995,
        value_threshold: float = 1e-6,
    ) -> None:
        self.capacity = max(int(capacity), 1)
        self.key_layers = max(int(key_layers), 1)
        self.future_layers = max(int(future_layers), 1)
        self.metric = metric
        self.admission_policy = admission_policy
        self.dedup_threshold = float(dedup_threshold)
        self.value_threshold = float(value_threshold)
        self._buckets: Dict[Tuple[int, str], _LocalContinuationBucket] = {}
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

    @staticmethod
    def _row_l2_normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = matrix.astype(np.float32, copy=True)
        denom = np.linalg.norm(matrix, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return matrix / denom

    @property
    def entries(self) -> List[LocalContinuationEntry]:
        flattened: List[LocalContinuationEntry] = []
        for bucket in self._buckets.values():
            flattened.extend(bucket.entries)
        return flattened

    def _bucket_key(self, anchor_layer_idx: int, model_tag: str) -> Tuple[int, str]:
        return (int(anchor_layer_idx), str(model_tag or ""))

    def _get_bucket(
        self,
        *,
        anchor_layer_idx: int,
        model_tag: str,
        create: bool = False,
    ) -> _LocalContinuationBucket | None:
        key = self._bucket_key(anchor_layer_idx, model_tag)
        bucket = self._buckets.get(key)
        if bucket is None and create:
            bucket = _LocalContinuationBucket()
            self._buckets[key] = bucket
        return bucket

    @classmethod
    def _prepare_key_arrays(
        cls,
        key_slice: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key_rows_normalized = cls._normalize_rows(key_slice)
        cosine_rows = cls._row_l2_normalize(key_rows_normalized)
        scale = np.sqrt(float(max(cosine_rows.shape[0], 1)))
        key_norm_flat = (cosine_rows.reshape(-1) / scale).astype(np.float32, copy=False)
        return key_rows_normalized.astype(np.float32, copy=False), key_norm_flat

    def _rebuild_bucket_key_bank(self, bucket: _LocalContinuationBucket) -> None:
        if not bucket.entries:
            bucket.key_bank = np.zeros((0, 0), dtype=np.float32)
            return
        bucket.key_bank = np.stack(
            [entry.key_norm_flat for entry in bucket.entries],
            axis=0,
        ).astype(np.float32, copy=False)

    @staticmethod
    def build_value_pairs(
        value_slice: np.ndarray,
        *,
        min_score: float,
    ) -> List[Tuple[int, int, float]]:
        matrix = np.asarray(value_slice, dtype=np.float32)
        positions = np.argwhere(matrix > float(min_score))
        return [
            (int(future_offset), int(expert_idx), float(matrix[future_offset, expert_idx]))
            for future_offset, expert_idx in positions
        ]

    def _similarity(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        lhs_rows, lhs_flat = self._prepare_key_arrays(lhs)
        rhs_rows, rhs_flat = self._prepare_key_arrays(rhs)
        if self.metric == "cosine":
            return float(np.dot(lhs_flat, rhs_flat))
        if self.metric == "l2":
            return float(-np.mean(np.linalg.norm(lhs_rows - rhs_rows, axis=1)))
        if self.metric == "l1":
            return float(-np.mean(np.sum(np.abs(lhs_rows - rhs_rows), axis=1)))
        raise ValueError(f"Unsupported library metric: {self.metric}")

    def _score_bucket(
        self,
        query_rows_normalized: np.ndarray,
        query_norm_flat: np.ndarray,
        bucket: _LocalContinuationBucket,
    ) -> np.ndarray:
        if not bucket.entries:
            return np.zeros((0,), dtype=np.float32)
        if self.metric == "cosine":
            return bucket.key_bank @ query_norm_flat
        scores = []
        for entry in bucket.entries:
            if self.metric == "l2":
                score = -np.mean(
                    np.linalg.norm(query_rows_normalized - entry.key_rows_normalized, axis=1)
                )
            elif self.metric == "l1":
                score = -np.mean(
                    np.sum(np.abs(query_rows_normalized - entry.key_rows_normalized), axis=1)
                )
            else:
                raise ValueError(f"Unsupported library metric: {self.metric}")
            scores.append(float(score))
        return np.asarray(scores, dtype=np.float32)

    def _find_duplicate(
        self,
        key_slice: np.ndarray,
        *,
        anchor_layer_idx: int,
        model_tag: str,
    ) -> Tuple[_LocalContinuationBucket | None, Optional[int]]:
        bucket = self._get_bucket(
            anchor_layer_idx=anchor_layer_idx,
            model_tag=model_tag,
            create=False,
        )
        if bucket is None or not bucket.entries:
            return bucket, None
        query_rows_normalized, query_norm_flat = self._prepare_key_arrays(key_slice)
        scores = self._score_bucket(query_rows_normalized, query_norm_flat, bucket)
        if scores.size == 0:
            return bucket, None
        duplicate_idx = int(np.argmax(scores))
        if float(scores[duplicate_idx]) >= self.dedup_threshold:
            return bucket, duplicate_idx
        return bucket, None

    def _iter_entry_refs(self):
        for bucket_key, bucket in self._buckets.items():
            for idx, entry in enumerate(bucket.entries):
                yield bucket_key, idx, entry

    def _remove_from_bucket(self, bucket_key: Tuple[int, str], entry_idx: int) -> None:
        bucket = self._buckets[bucket_key]
        bucket.entries.pop(entry_idx)
        if bucket.entries:
            self._rebuild_bucket_key_bank(bucket)
        else:
            self._buckets.pop(bucket_key, None)

    def _bucket_redundancy(
        self,
        bucket: _LocalContinuationBucket,
        entry_idx: int,
    ) -> float:
        if len(bucket.entries) <= 1:
            return 1.0
        if self.metric == "cosine":
            scores = bucket.key_bank @ bucket.entries[entry_idx].key_norm_flat
            if scores.size <= 1:
                return 1.0
            scores = scores.copy()
            scores[entry_idx] = -np.inf
            return float(np.max(scores))
        entry = bucket.entries[entry_idx]
        others = [
            other for idx, other in enumerate(bucket.entries) if idx != entry_idx
        ]
        if not others:
            return 1.0
        return max(
            self._similarity(entry.key_slice, other.key_slice)
            for other in others
        )

    def _evict_ref(self) -> Tuple[Tuple[int, str], int]:
        refs = list(self._iter_entry_refs())
        if not refs:
            raise ValueError("Cannot evict from an empty local continuation library.")
        if self.admission_policy == "recent_only":
            bucket_key, entry_idx, _entry = min(
                refs,
                key=lambda item: item[2].insertion_index,
            )
            return bucket_key, entry_idx
        if self.admission_policy == "frequency_weighted":
            bucket_key, entry_idx, _entry = min(
                refs,
                key=lambda item: (
                    item[2].access_count,
                    item[2].insertion_index,
                ),
            )
            return bucket_key, entry_idx
        if self.admission_policy == "diversity_aware":
            best_ref = refs[0][:2]
            best_score = None
            for bucket_key, entry_idx, entry in refs:
                bucket = self._buckets[bucket_key]
                redundancy = self._bucket_redundancy(bucket, entry_idx)
                score = (round(redundancy, 6), -entry.access_count, entry.insertion_index)
                if best_score is None or score > best_score:
                    best_score = score
                    best_ref = (bucket_key, entry_idx)
            return best_ref
        raise ValueError(f"Unsupported admission policy: {self.admission_policy}")

    def _evict_index(self) -> int:
        bucket_key, entry_idx = self._evict_ref()
        flattened_index = 0
        for current_bucket_key, bucket in self._buckets.items():
            if current_bucket_key == bucket_key:
                return flattened_index + entry_idx
            flattened_index += len(bucket.entries)
        return None

    def admit(
        self,
        *,
        seq_id: str,
        step_index: int,
        anchor_layer_idx: int,
        key_slice: np.ndarray,
        value_slice: np.ndarray,
        model_tag: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = metadata or {}
        self.admit_count += 1
        key_rows_normalized, key_norm_flat = self._prepare_key_arrays(key_slice)
        value_pairs = self.build_value_pairs(
            value_slice,
            min_score=self.value_threshold,
        )
        bucket, duplicate_idx = self._find_duplicate(
            key_slice,
            anchor_layer_idx=anchor_layer_idx,
            model_tag=model_tag,
        )
        if duplicate_idx is not None:
            entry = bucket.entries[duplicate_idx]
            entry.access_count += 1
            entry.key_slice = key_slice.astype(np.float32, copy=True)
            entry.value_slice = value_slice.astype(np.float32, copy=True)
            entry.key_rows_normalized = key_rows_normalized
            entry.key_norm_flat = key_norm_flat
            entry.value_pairs = list(value_pairs)
            entry.metadata.update(metadata)
            self._rebuild_bucket_key_bank(bucket)
            self.duplicate_update_count += 1
            return

        entry = LocalContinuationEntry(
            seq_id=seq_id,
            step_index=int(step_index),
            anchor_layer_idx=int(anchor_layer_idx),
            key_slice=key_slice.astype(np.float32, copy=True),
            value_slice=value_slice.astype(np.float32, copy=True),
            key_rows_normalized=key_rows_normalized,
            key_norm_flat=key_norm_flat,
            value_pairs=list(value_pairs),
            model_tag=model_tag,
            metadata=dict(metadata),
            access_count=1,
            insertion_index=self._insertion_counter,
        )
        self._insertion_counter += 1
        if len(self) >= self.capacity:
            evict_bucket_key, evict_entry_idx = self._evict_ref()
            self._remove_from_bucket(evict_bucket_key, evict_entry_idx)
        bucket = self._get_bucket(
            anchor_layer_idx=anchor_layer_idx,
            model_tag=model_tag,
            create=True,
        )
        bucket.entries.append(entry)
        self._rebuild_bucket_key_bank(bucket)

    def topk(
        self,
        key_slice: np.ndarray,
        *,
        anchor_layer_idx: int,
        model_tag: str = "",
        k: int = 1,
    ) -> List[Tuple[LocalContinuationEntry, float]]:
        self.query_count += 1
        if k <= 0:
            return []
        query_rows_normalized, query_norm_flat = self._prepare_key_arrays(key_slice)
        if model_tag:
            candidate_buckets = [
                self._get_bucket(
                    anchor_layer_idx=anchor_layer_idx,
                    model_tag=model_tag,
                    create=False,
                )
            ]
        else:
            candidate_buckets = [
                bucket
                for (bucket_anchor, _bucket_tag), bucket in self._buckets.items()
                if bucket_anchor == int(anchor_layer_idx)
            ]

        scored: List[Tuple[LocalContinuationEntry, float]] = []
        for bucket in candidate_buckets:
            if bucket is None or not bucket.entries:
                continue
            scores = self._score_bucket(query_rows_normalized, query_norm_flat, bucket)
            if scores.size == 0:
                continue
            topk_count = min(int(k), int(scores.shape[0]))
            if topk_count <= 0:
                continue
            if topk_count == scores.shape[0]:
                local_indices = np.argsort(scores)[::-1]
            else:
                partial = np.argpartition(scores, -topk_count)[-topk_count:]
                local_indices = partial[np.argsort(scores[partial])[::-1]]
            for local_idx in local_indices.tolist():
                scored.append((bucket.entries[int(local_idx)], float(scores[int(local_idx)])))

        if not scored:
            return []
        scored.sort(key=lambda item: item[1], reverse=True)
        result = scored[: int(k)]
        if result:
            self.hit_count += 1
        return result

    def __len__(self) -> int:
        return sum(len(bucket.entries) for bucket in self._buckets.values())

    def stats(self) -> Dict[str, int]:
        return {
            "size": len(self.entries),
            "query_count": self.query_count,
            "hit_count": self.hit_count,
            "admit_count": self.admit_count,
            "duplicate_update_count": self.duplicate_update_count,
        }

    @staticmethod
    def build_key_slice(
        step_matrix: np.ndarray,
        *,
        anchor_layer_idx: int,
        key_layers: int,
    ) -> np.ndarray:
        key_layers = max(int(key_layers), 1)
        key_slice = np.zeros((key_layers, step_matrix.shape[1]), dtype=np.float32)
        start = max(int(anchor_layer_idx) - key_layers + 1, 0)
        source = np.asarray(step_matrix[start : int(anchor_layer_idx) + 1], dtype=np.float32)
        key_slice[-source.shape[0] :] = source
        return key_slice

    @staticmethod
    def build_value_slice(
        step_matrix: np.ndarray,
        *,
        anchor_layer_idx: int,
        future_layers: int,
    ) -> np.ndarray:
        future_layers = max(int(future_layers), 1)
        value_slice = np.zeros((future_layers, step_matrix.shape[1]), dtype=np.float32)
        start = int(anchor_layer_idx) + 1
        end = min(step_matrix.shape[0], start + future_layers)
        if start < end:
            source = np.asarray(step_matrix[start:end], dtype=np.float32)
            value_slice[: source.shape[0]] = source
        return value_slice
