from moe_infinity.utils.qwen_smoke import (
    build_qwen_smoke_config,
    dispatcher_stats_dict,
)


def test_build_qwen_smoke_config_baseline_disables_prefetch():
    config = build_qwen_smoke_config(
        phase="baseline",
        offload_path="/tmp/offload",
        device_memory_ratio=0.5,
        num_threads=1,
        library_capacity=8,
        library_metric="cosine",
        library_admission="diversity_aware",
        backbone_topk=8,
    )

    assert config["offloading_policy"] == "baseline_trace_similarity"
    assert config["prefetch"] is False
    assert config["policy_score_only"] is False
    assert config["prefetch_backbone_topk"] == 0
    assert config["prefetch_future_layers"] == 0
    assert config["prefetch_max_candidates"] == 0


def test_build_qwen_smoke_config_history_and_backbone():
    history = build_qwen_smoke_config(
        phase="history",
        offload_path="/tmp/offload",
        device_memory_ratio=0.5,
        num_threads=1,
        library_capacity=8,
        library_metric="cosine",
        library_admission="diversity_aware",
        backbone_topk=8,
    )
    backbone = build_qwen_smoke_config(
        phase="backbone",
        offload_path="/tmp/offload",
        device_memory_ratio=0.5,
        num_threads=1,
        library_capacity=8,
        library_metric="cosine",
        library_admission="diversity_aware",
        backbone_topk=8,
    )

    assert history["offloading_policy"] == "finegrained_history_reuse"
    assert history["prefetch"] is False
    assert history["policy_score_only"] is True
    assert history["prefetch_backbone_topk"] == 0
    assert history["prefetch_future_layers"] == 0
    assert history["prefetch_max_candidates"] == 0
    assert backbone["prefetch"] is True
    assert backbone["policy_score_only"] is False
    assert backbone["prefetch_backbone_topk"] == 8
    assert backbone["prefetch_future_layers"] == 4
    assert backbone["prefetch_max_candidates"] == 32


def test_dispatcher_stats_dict_names_payload():
    stats = dispatcher_stats_dict([12, 3, 400, 250])

    assert stats == {
        "enqueue_count": 12,
        "busy_wait_count": 3,
        "busy_wait_total_wait_us": 400,
        "busy_wait_max_wait_us": 250,
    }
