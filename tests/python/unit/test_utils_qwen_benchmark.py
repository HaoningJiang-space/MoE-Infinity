import json

import torch

from moe_infinity.utils.qwen_benchmark import (
    aggregate_request_records,
    build_qwen_benchmark_config,
    load_chat_trace,
    percentile,
    summarize_hit_rate_tensor,
)
from moe_infinity.utils.qwen_smoke import dispatcher_stats_dict


def test_build_qwen_benchmark_config_variants():
    on_demand = build_qwen_benchmark_config(
        variant="on_demand",
        offload_path="/tmp/offload",
        device_memory_ratio=0.6,
        num_threads=1,
        library_capacity=32,
        library_metric="cosine",
        library_admission="diversity_aware",
    )
    assert on_demand["prefetch"] is False
    assert on_demand["offloading_policy"] == "baseline_trace_similarity"
    assert on_demand["prefetch_backbone_topk"] == 0

    static_hot = build_qwen_benchmark_config(
        variant="static_hot_prefetch",
        offload_path="/tmp/offload",
        device_memory_ratio=0.6,
        num_threads=1,
        library_capacity=32,
        library_metric="cosine",
        library_admission="diversity_aware",
        static_prefetch_default_topk=4,
    )
    assert static_hot["prefetch"] is True
    assert static_hot["offloading_policy"] == "static_hot_prefetch"
    assert static_hot["static_prefetch_default_topk"] == 4
    assert static_hot["prefetch_execution_mode"] == "replace_and_enqueue"

    trace_prefetch = build_qwen_benchmark_config(
        variant="trace_similarity_prefetch",
        offload_path="/tmp/offload",
        device_memory_ratio=0.6,
        num_threads=1,
        library_capacity=32,
        library_metric="cosine",
        library_admission="diversity_aware",
    )
    assert trace_prefetch["prefetch"] is True
    assert trace_prefetch["offloading_policy"] == "baseline_trace_similarity"
    assert trace_prefetch["prefetch_backbone_topk"] == 0

    history_backbone = build_qwen_benchmark_config(
        variant="history_reuse_backbone",
        offload_path="/tmp/offload",
        device_memory_ratio=0.6,
        num_threads=1,
        library_capacity=32,
        library_metric="cosine",
        library_admission="diversity_aware",
        backbone_topk=8,
    )
    assert history_backbone["prefetch"] is True
    assert history_backbone["offloading_policy"] == "finegrained_history_reuse"
    assert history_backbone["prefetch_backbone_topk"] == 8
    assert history_backbone["prefetch_future_layers"] == 4
    assert history_backbone["prefetch_max_candidates"] == 32
    assert history_backbone["prefetch_admission_enabled"] is False
    assert history_backbone["prefetch_admission_max_per_plan"] == -1
    assert history_backbone["prefetch_credit_gated_enabled"] is False
    assert history_backbone["prefetch_credit_count"] == -1
    assert history_backbone["prefetch_credit_zero_action"] == "update_only"
    assert history_backbone["prefetch_policy_disabled"] is False

    credit_gated = build_qwen_benchmark_config(
        variant="history_reuse_local_backbone",
        offload_path="/tmp/offload",
        device_memory_ratio=0.6,
        num_threads=1,
        library_capacity=32,
        library_metric="cosine",
        library_admission="diversity_aware",
        prefetch_credit_gated_enabled=True,
        prefetch_credit_count=8,
        prefetch_credit_zero_action="skip_policy_update",
        prefetch_policy_disabled=True,
        prefetch_execution_mode="replace_only",
    )
    assert credit_gated["historical_reuse_object_mode"] == "local_continuation"
    assert credit_gated["prefetch_credit_gated_enabled"] is True
    assert credit_gated["prefetch_credit_count"] == 8
    assert credit_gated["prefetch_credit_zero_action"] == "skip_policy_update"
    assert credit_gated["prefetch_policy_disabled"] is True
    assert credit_gated["prefetch_execution_mode"] == "replace_only"


def test_load_chat_trace_validates_schema(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "req-1",
                        "messages": [
                            {"role": "user", "content": "hello"},
                        ],
                        "tag": "x",
                    }
                ),
                json.dumps(
                    {
                        "request_id": "req-2",
                        "messages": [
                            {"role": "system", "content": "s"},
                            {"role": "user", "content": "u"},
                        ],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    requests = load_chat_trace(path)
    assert [request.request_id for request in requests] == ["req-1", "req-2"]
    assert requests[0].tag == "x"
    assert requests[1].messages[1]["content"] == "u"


def test_load_chat_trace_rejects_bad_schema(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"request_id": "req-1", "messages": "bad"}),
        encoding="utf-8",
    )
    try:
        load_chat_trace(path)
    except ValueError as exc:
        assert "messages" in str(exc)
    else:
        raise AssertionError("Expected schema validation to fail")


def test_percentile_and_aggregate_metrics():
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0

    records = [
        {
            "latency_s": 1.0,
            "generated_tokens": 10,
            "dispatcher_stats": {
                "enqueue_count": 100,
                "busy_wait_count": 0,
                "busy_wait_total_wait_us": 0,
                "busy_wait_max_wait_us": 0,
                "cache_hit_fetch_count": 90,
                "cache_miss_fetch_count": 10,
                "eviction_count": 3,
                "all_locked_event_count": 0,
                "no_victim_wait_count": 0,
                "no_victim_wait_total_us": 0,
                "no_victim_wait_max_us": 0,
                "prefetch_resident_hit_count": 2,
                "late_prefetch_demand_miss_count": 1,
                "demand_candidate_protect_skip_count": 5,
                "demand_candidate_protect_fallback_count": 1,
                "candidate_resident_hit_count": 7,
                "candidate_demand_miss_count": 2,
            },
            "library_stats_delta": {
                "query_count": 2,
                "hit_count": 1,
                "admit_count": 1,
            },
            "cache_hit_rate_delta": {"overall_hit_rate": 0.5},
            "prefetcher_stats": {
                "prefetch_candidate_count": 10,
                "prefetch_admitted_count": 8,
                "prefetch_enqueue_count": 8,
                "prefetch_drop_count": 2,
                "prefetch_drop_cap_count": 1,
                "prefetch_drop_pressure_count": 1,
                "prefetch_drop_no_evictable_count": 1,
                "prefetch_under_pressure_count": 2,
                "demand_prefetch_conflict_count": 1,
                "pressure_sample_count": 1,
                "pressure_locked_max": 3,
                "pressure_evictable_min": 2,
                "prefetch_credit_skip_count": 0,
                "prefetch_credit_issued_total": 8,
                "prefetch_credit_limited_count": 1,
                "prefetch_credit_materialized_count": 8,
                "prefetch_credit_skip_policy_update_count": 0,
                "prefetch_plan_replace_count": 2,
                "prefetch_plan_empty_replace_count": 1,
                "prefetch_plan_candidate_count": 8,
                "prefetch_plan_cleared_candidate_count": 4,
                "prefetch_runtime_enqueue_count": 8,
                "prefetch_runtime_queue_push_count": 6,
                "prefetch_runtime_same_device_skip_count": 2,
                "prefetch_runtime_dequeue_count": 7,
                "prefetch_runtime_complete_count": 6,
                "prefetch_runtime_queue_cleared_task_count": 3,
            },
        },
        {
            "latency_s": 2.0,
            "generated_tokens": 20,
            "dispatcher_stats": {
                "enqueue_count": 200,
                "busy_wait_count": 1,
                "busy_wait_total_wait_us": 50,
                "busy_wait_max_wait_us": 50,
                "cache_hit_fetch_count": 150,
                "cache_miss_fetch_count": 50,
                "eviction_count": 10,
                "all_locked_event_count": 2,
                "no_victim_wait_count": 2,
                "no_victim_wait_total_us": 80,
                "no_victim_wait_max_us": 60,
                "prefetch_resident_hit_count": 3,
                "late_prefetch_demand_miss_count": 4,
                "demand_candidate_protect_skip_count": 9,
                "demand_candidate_protect_fallback_count": 2,
                "candidate_resident_hit_count": 11,
                "candidate_demand_miss_count": 3,
            },
            "library_stats_delta": {
                "query_count": 4,
                "hit_count": 3,
                "admit_count": 1,
            },
            "cache_hit_rate_delta": {
                "overall_hit_rate": 0.75,
                "prefetch_count": 3,
            },
            "prefetcher_stats": {
                "prefetch_candidate_count": 20,
                "prefetch_admitted_count": 15,
                "prefetch_enqueue_count": 15,
                "prefetch_drop_count": 5,
                "prefetch_drop_cap_count": 3,
                "prefetch_drop_pressure_count": 2,
                "prefetch_drop_no_evictable_count": 4,
                "prefetch_under_pressure_count": 5,
                "demand_prefetch_conflict_count": 2,
                "pressure_sample_count": 1,
                "pressure_locked_max": 7,
                "pressure_evictable_min": 1,
                "prefetch_credit_skip_count": 4,
                "prefetch_credit_issued_total": 8,
                "prefetch_credit_limited_count": 1,
                "prefetch_credit_materialized_count": 7,
                "prefetch_credit_skip_policy_update_count": 5,
                "prefetch_plan_replace_count": 3,
                "prefetch_plan_empty_replace_count": 0,
                "prefetch_plan_candidate_count": 15,
                "prefetch_plan_cleared_candidate_count": 8,
                "prefetch_runtime_enqueue_count": 15,
                "prefetch_runtime_queue_push_count": 11,
                "prefetch_runtime_same_device_skip_count": 4,
                "prefetch_runtime_dequeue_count": 12,
                "prefetch_runtime_complete_count": 10,
                "prefetch_runtime_queue_cleared_task_count": 5,
            },
        },
    ]
    aggregate = aggregate_request_records(records)
    assert aggregate["request_count"] == 2
    assert aggregate["generated_tokens_total"] == 30
    assert aggregate["generated_tokens_per_second"] == 10.0
    assert aggregate["library_query_count_total"] == 6
    assert aggregate["library_hit_count_total"] == 4
    assert aggregate["mean_dispatcher_enqueue_count"] == 150.0
    assert aggregate["dispatcher_eviction_count_total"] == 13
    assert aggregate["mean_dispatcher_eviction_count"] == 6.5
    assert aggregate["dispatcher_all_locked_event_count_total"] == 2
    assert aggregate["mean_dispatcher_no_victim_wait_total_us"] == 40.0
    assert aggregate["p95_dispatcher_no_victim_wait_max_us"] > 0.0
    assert aggregate["mean_cache_hit_rate"] == 0.625
    assert aggregate["prefetch_candidate_count_total"] == 30
    assert aggregate["prefetch_drop_count_total"] == 7
    assert aggregate["prefetch_drop_cap_count_total"] == 4
    assert aggregate["prefetch_drop_pressure_count_total"] == 3
    assert aggregate["prefetch_under_pressure_count_total"] == 7
    assert aggregate["prefetch_admit_rate"] == 23 / 30
    assert aggregate["prefetch_pressure_drop_rate"] == 3 / 30
    assert aggregate["demand_prefetch_conflict_count_total"] == 3
    assert aggregate["pressure_locked_max"] == 7
    assert aggregate["pressure_evictable_min"] == 1
    assert aggregate["prefetch_credit_skip_count_total"] == 4
    assert aggregate["prefetch_credit_issued_total"] == 16
    assert aggregate["prefetch_credit_limited_count_total"] == 2
    assert aggregate["prefetch_credit_materialized_count_total"] == 15
    assert aggregate["prefetch_credit_skip_policy_update_count_total"] == 5
    assert aggregate["prefetch_plan_replace_count_total"] == 5
    assert aggregate["prefetch_plan_empty_replace_count_total"] == 1
    assert aggregate["prefetch_plan_candidate_count_total"] == 23
    assert aggregate["prefetch_plan_cleared_candidate_count_total"] == 12
    assert aggregate["prefetch_runtime_enqueue_count_total"] == 23
    assert aggregate["prefetch_runtime_queue_push_count_total"] == 17
    assert aggregate["prefetch_runtime_same_device_skip_count_total"] == 6
    assert aggregate["prefetch_runtime_dequeue_count_total"] == 19
    assert aggregate["prefetch_runtime_complete_count_total"] == 16
    assert aggregate["prefetch_runtime_queue_cleared_task_count_total"] == 8
    assert aggregate["dispatcher_prefetch_resident_hit_count_total"] == 5
    assert aggregate["dispatcher_late_prefetch_demand_miss_count_total"] == 5
    assert aggregate["dispatcher_demand_candidate_protect_skip_count_total"] == 14
    assert aggregate["dispatcher_demand_candidate_protect_fallback_count_total"] == 3
    assert aggregate["dispatcher_candidate_resident_hit_count_total"] == 18
    assert aggregate["dispatcher_candidate_demand_miss_count_total"] == 5
    assert aggregate["cache_prefetch_count_total"] == 3


def test_dispatcher_stats_dict_accepts_legacy_and_extended_payloads():
    legacy = dispatcher_stats_dict([10, 1, 20, 30])
    assert legacy == {
        "enqueue_count": 10,
        "busy_wait_count": 1,
        "busy_wait_total_wait_us": 20,
        "busy_wait_max_wait_us": 30,
    }

    extended = dispatcher_stats_dict([10, 1, 20, 30, 7, 3, 2, 1, 1, 50, 50])
    assert extended["cache_hit_fetch_count"] == 7
    assert extended["cache_miss_fetch_count"] == 3
    assert extended["eviction_count"] == 2
    assert extended["all_locked_event_count"] == 1
    assert extended["no_victim_wait_total_us"] == 50

    lifecycle = dispatcher_stats_dict(
        [10, 1, 20, 30, 7, 3, 2, 1, 1, 50, 50, 4, 5, 6, 7, 8, 9, 0, 11, 12]
    )
    assert lifecycle["pending_stall_count"] == 0
    assert lifecycle["prefetch_resident_hit_count"] == 11
    assert lifecycle["late_prefetch_demand_miss_count"] == 12

    candidate_lifecycle = dispatcher_stats_dict(
        [
            10,
            1,
            20,
            30,
            7,
            3,
            2,
            1,
            1,
            50,
            50,
            4,
            5,
            6,
            7,
            8,
            9,
            0,
            11,
            12,
            13,
            14,
            15,
            16,
        ]
    )
    assert candidate_lifecycle["demand_candidate_protect_skip_count"] == 13
    assert candidate_lifecycle["demand_candidate_protect_fallback_count"] == 14
    assert candidate_lifecycle["candidate_resident_hit_count"] == 15
    assert candidate_lifecycle["candidate_demand_miss_count"] == 16


def test_summarize_hit_rate_tensor():
    raw = torch.tensor(
        [
            [10, 8, 2, 6, 5, 1, 3, 4, 0, 0, 1],
            [20, 12, 8, 10, 7, 3, 2, 5, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    summary = summarize_hit_rate_tensor(raw)
    assert summary["visit_count"] == 30
    assert summary["hit_count"] == 16
    assert abs(summary["overall_hit_rate"] - (16 / 30)) < 1e-9
    assert summary["prefetch_count"] == 9
