# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import HfArgumentParser


@dataclass
class ArcherConfig:
    offload_path: str = field(
        default="", metadata={"help": "Path to parameter storage"}
    )
    trace_capacity: int = field(
        default=1000, metadata={"help": "Capacity of trace"}
    )
    trace_path: Optional[os.PathLike] = field(
        default=None, metadata={"help": "Path to trace file"}
    )
    # master_addr: str = field(
    #     default="127.0.0.1",
    #     metadata={"help": "Hosts for running archer"},
    # )
    # master_port: str = field(
    #     default=29500,
    #     metadata={"help": "Port for running archer"},
    # )
    # device_per_node: int = field(
    #     default=1,
    #     metadata={"help": "Number of devices per node"},
    # )
    prefetch: bool = field(
        default=False, metadata={"help": "Enable prefetching"}
    )
    policy_score_only: bool = field(
        default=False,
        metadata={
            "help": "Evaluate offloading policy scoring without issuing prefetch work"
        },
    )
    device_memory_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of device memory to use"},
    )
    num_threads: int = field(
        default=1, metadata={"help": "Number of threads for each GPU exec"}
    )
    host_memory_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of host memory to use"},
    )
    offloading_policy: str = field(
        default="baseline_trace_similarity",
        metadata={"help": "Offloading policy name"},
    )
    historical_library_capacity: int = field(
        default=256,
        metadata={"help": "Capacity of reusable historical expert-state library"},
    )
    historical_library_metric: str = field(
        default="cosine",
        metadata={"help": "Similarity metric for historical library retrieval"},
    )
    historical_library_similarity_mode: str = field(
        default="prefix_mean",
        metadata={"help": "History-retrieval similarity mode: prefix_mean or prefix_recent_hybrid"},
    )
    historical_library_recent_window: int = field(
        default=4,
        metadata={"help": "Recent-layer window size used when historical_library_similarity_mode=prefix_recent_hybrid"},
    )
    historical_library_recent_weight: float = field(
        default=0.5,
        metadata={"help": "Weight on recent-layer similarity when historical_library_similarity_mode=prefix_recent_hybrid"},
    )
    historical_library_admission: str = field(
        default="diversity_aware",
        metadata={"help": "Admission/eviction policy for the historical library"},
    )
    historical_library_dedup_threshold: float = field(
        default=0.995,
        metadata={"help": "Deduplication similarity threshold for the historical library"},
    )
    historical_reuse_object_mode: str = field(
        default="sequence_matrix",
        metadata={"help": "History-reuse retrieval object: sequence_matrix or local_continuation"},
    )
    historical_reuse_candidate_mode: str = field(
        default="top1",
        metadata={"help": "History-reuse candidate construction mode: top1 or topk"},
    )
    historical_reuse_match_topk: int = field(
        default=4,
        metadata={"help": "Number of historical matches to aggregate when historical_reuse_candidate_mode=topk"},
    )
    historical_reuse_match_min_required: int = field(
        default=2,
        metadata={"help": "Minimum number of historical matches required before using top-k history aggregation"},
    )
    historical_reuse_consensus_min_votes: int = field(
        default=2,
        metadata={"help": "Minimum number of top-k historical supports that must agree before an expert enters the consensus backbone mask"},
    )
    local_continuation_library_capacity: int = field(
        default=4096,
        metadata={"help": "Capacity of the step/layer-local continuation library"},
    )
    local_continuation_key_layers: int = field(
        default=4,
        metadata={"help": "Number of recent layers to keep in each local-continuation retrieval key"},
    )
    local_continuation_future_layers: int = field(
        default=4,
        metadata={"help": "Number of future layers stored in each local-continuation value slice"},
    )
    local_continuation_match_topk: int = field(
        default=4,
        metadata={"help": "Number of local-continuation matches to aggregate"},
    )
    local_continuation_match_min_required: int = field(
        default=2,
        metadata={"help": "Minimum number of local-continuation matches required before using local aggregation"},
    )
    prefetch_backbone_topk: int = field(
        default=0,
        metadata={"help": "Optional per-layer top-k backbone projection for prefetch scores"},
    )
    prefetch_backbone_mode: str = field(
        default="raw_topk",
        metadata={"help": "Backbone projection mode: raw_topk, history_score, or consensus_mask"},
    )
    prefetch_backbone_history_topk: int = field(
        default=4,
        metadata={"help": "Number of historical matches to consider when building runtime backbone supports"},
    )
    prefetch_backbone_lambda: float = field(
        default=0.5,
        metadata={"help": "Instability penalty weight for runtime history-backed backbone scores"},
    )
    prefetch_backbone_min_matches: int = field(
        default=2,
        metadata={"help": "Minimum number of historical matches required before using history-backed backbone scoring"},
    )
    prefetch_future_layers: int = field(
        default=0,
        metadata={"help": "Maximum number of future layers to consider for expert prefetch; 0 means unbounded"},
    )
    prefetch_max_candidates: int = field(
        default=0,
        metadata={"help": "Maximum number of prefetched expert candidates per policy step; 0 means unbounded"},
    )
    prefetch_candidate_min_score: float = field(
        default=1e-6,
        metadata={"help": "Minimum candidate score required to keep an expert in the prefetch candidate set"},
    )
    prefetch_admission_enabled: bool = field(
        default=False,
        metadata={"help": "Enable evictability-aware admission for speculative expert prefetch"},
    )
    prefetch_admission_demand_reserve: int = field(
        default=2,
        metadata={"help": "Minimum evictable expert slots reserved for demand fetches before admitting prefetch"},
    )
    prefetch_admission_locked_ratio_threshold: float = field(
        default=0.8,
        metadata={"help": "Locked/cached sparse expert ratio that triggers prefetch throttling"},
    )
    prefetch_admission_max_under_pressure: int = field(
        default=4,
        metadata={"help": "Maximum admitted prefetch candidates per GPU when lock pressure is high"},
    )
    prefetch_admission_max_per_plan: int = field(
        default=-1,
        metadata={"help": "Hard cap on admitted prefetch candidates per GPU per policy step; -1 disables the cap"},
    )
    prefetch_credit_gated_enabled: bool = field(
        default=False,
        metadata={"help": "Enable upstream credit-gated speculative prefetch generation"},
    )
    prefetch_credit_count: int = field(
        default=-1,
        metadata={"help": "Speculative prefetch candidates allowed per policy step when credit gating is enabled; 0 skips generation"},
    )
    prefetch_credit_zero_action: str = field(
        default="update_only",
        metadata={"help": "Action when credit-gated prefetch has zero credit: update_only or skip_policy_update"},
    )
    prefetch_policy_disabled: bool = field(
        default=False,
        metadata={"help": "Disable Python expert-policy drive while keeping prefetch wiring enabled"},
    )
    prefetch_execution_mode: str = field(
        default="replace_and_enqueue",
        metadata={"help": "Prefetch execution mode: replace_and_enqueue, replace_only, enqueue_only, or disabled"},
    )
    static_prefetch_plan_path: str = field(
        default="",
        metadata={"help": "Optional JSON layer->expert plan for static no-sync prefetch diagnostics"},
    )
    static_prefetch_default_topk: int = field(
        default=8,
        metadata={"help": "Fallback number of low-index experts per layer for static no-sync prefetch diagnostics"},
    )
    phasea_analysis_future_layers: int = field(
        default=0,
        metadata={"help": "Analysis-only candidate future-layer window for Phase-A observation; 0 means unbounded"},
    )
    phasea_analysis_max_ranked_candidates: int = field(
        default=128,
        metadata={"help": "Analysis-only maximum ranked candidates to retain for Phase-A observation"},
    )

    @classmethod
    def load_from_file(self, config_path):
        parser = HfArgumentParser(self)
        self = parser.parse_json_file(json_file=config_path)[0]
        return self

    @classmethod
    def load_from_json(self, config_json):
        parser = HfArgumentParser(self)
        self = parser.parse_dict(config_json)[0]
        return self

    def __post_init__(self):
        self.perfect_cache_file = os.path.join(
            self.offload_path, "perfect_cache"
        )

        self.device_per_node = (
            torch.cuda.device_count()
        )  # always run on heterogeneous nodes

        if self.trace_path is not None:
            self.trace_path = os.path.abspath(self.trace_path)
            if os.path.isdir(self.trace_path):
                raise ValueError(
                    "The trace path should be a file, not a directory."
                )
