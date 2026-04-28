from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .base import OffloadingPolicy


class StaticHotPrefetchPolicy(OffloadingPolicy):
    """No-sync diagnostic policy that emits a static per-layer expert plan."""

    def __init__(self, *, config, tracer, predictor, model_tag: str = ""):
        super().__init__(
            config=config,
            tracer=tracer,
            predictor=predictor,
            model_tag=model_tag,
        )
        self.layer_plan = self._load_layer_plan()

    def update(self, seq_id: str, expert_list, layer_idx: int) -> None:
        return None

    def _default_experts(self) -> List[int]:
        topk = max(int(getattr(self.config, "static_prefetch_default_topk", 8)), 0)
        topk = min(topk, int(self.predictor.num_experts))
        return list(range(topk))

    def _normalize_layer_plan(self, payload) -> Dict[int, List[int]]:
        if isinstance(payload, dict) and "layers" in payload:
            payload = payload["layers"]
        if isinstance(payload, list):
            payload = {str(index): value for index, value in enumerate(payload)}
        if not isinstance(payload, dict):
            raise ValueError("static prefetch plan must be a dict or a list")

        plan: Dict[int, List[int]] = {}
        for raw_layer, raw_experts in payload.items():
            layer_idx = int(raw_layer)
            if layer_idx < 0 or layer_idx >= int(self.predictor.num_layers):
                continue
            if not isinstance(raw_experts, list):
                raise ValueError(f"static plan layer {layer_idx} must be a list")
            experts: List[int] = []
            seen = set()
            for raw_expert in raw_experts:
                expert_idx = int(raw_expert)
                if expert_idx < 0 or expert_idx >= int(self.predictor.num_experts):
                    continue
                if expert_idx in seen:
                    continue
                seen.add(expert_idx)
                experts.append(expert_idx)
            plan[layer_idx] = experts
        return plan

    def _load_layer_plan(self) -> Dict[int, List[int]]:
        plan_path = str(getattr(self.config, "static_prefetch_plan_path", "") or "")
        if plan_path:
            payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            plan = self._normalize_layer_plan(payload)
            if plan:
                return plan
        default = self._default_experts()
        return {
            layer_idx: list(default)
            for layer_idx in range(int(self.predictor.num_layers))
        }

    def score_for_prefetch(self, seq_id: str, layer_idx: int) -> np.ndarray:
        matrix = np.zeros(
            (self.predictor.num_layers, self.predictor.num_experts),
            dtype=np.float32,
        )
        for target_layer, experts in self.layer_plan.items():
            score = float(len(experts))
            for expert_idx in experts:
                matrix[target_layer, expert_idx] = max(score, 1.0)
                score -= 1.0
        return matrix
