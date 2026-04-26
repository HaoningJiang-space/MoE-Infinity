# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team


import numpy as np
from transformers import PretrainedConfig

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
        expert_list = []
        future_layers = int(getattr(self, "prefetch_future_layers", 0))
        max_candidates = int(getattr(self, "prefetch_max_candidates", 0))
        max_layer = self.num_layers
        if future_layers > 0:
            max_layer = min(self.num_layers, layer_id + 1 + future_layers)
        # print("expert_tensor_map", self.expert_tensor_map)
        # Current-layer experts are handled by the real dispatch path.
        # Prefetch should only schedule future layers, otherwise we can
        # race the active execution enqueue on the same expert node.
        for i in range(layer_id + 1, max_layer):
            for j in range(self.num_experts):
                if expert_matrix[i, j] > 0:
                    expert_list.append(
                        (self.expert_tensor_map[(i, j)], expert_matrix[i, j])
                    )
        ordered_expert_list = sorted(
            expert_list, key=lambda x: x[1], reverse=True
        )
        if max_candidates > 0:
            ordered_expert_list = ordered_expert_list[:max_candidates]
        tensor_ids = [x[0] for x in ordered_expert_list]
        assert len(np.unique(tensor_ids)) == len(tensor_ids)
        self.archer_engine.replace_cache_candidates(tensor_ids)
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)
