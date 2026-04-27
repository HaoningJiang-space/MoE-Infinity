# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team


import numpy as np
from transformers import PretrainedConfig

from moe_infinity.utils.prefetch_plan import rank_prefetch_candidates
from moe_infinity.utils import parse_moe_param


class ExpertPrefetcher(object):
    cache_file_rd = None
    first_k_dense_replace: int = 0

    def __init__(self, config: PretrainedConfig):
        print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )

    def set_archer_engine(self, archer_engine):
        global _expert_prefetcher
        _expert_prefetcher = archer_engine
        self.archer_engine = archer_engine

    def prefetch_experts_list(self, layer_id, expert_list):
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)

    def fetch_experts_lock_cache(self, layer_id, expert_list):
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        self.archer_engine.replace_cache_candidates(tensor_ids)

    def prefetch_experts(self, layer_id, expert_matrix):
        future_layers = int(getattr(self, "prefetch_future_layers", 0))
        max_candidates = int(getattr(self, "prefetch_max_candidates", 0))
        min_score = float(getattr(self, "prefetch_candidate_min_score", 1e-6))
        ranked_candidates = rank_prefetch_candidates(
            layer_id=layer_id,
            expert_matrix=expert_matrix,
            future_layers=future_layers,
            max_candidates=max_candidates,
            min_score=min_score,
        )
        tensor_ids = [
            self.expert_tensor_map[(candidate.layer_idx, candidate.expert_idx)]
            for candidate in ranked_candidates
        ]
        assert len(np.unique(tensor_ids)) == len(tensor_ids)
        self.archer_engine.replace_cache_candidates(tensor_ids)
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)
