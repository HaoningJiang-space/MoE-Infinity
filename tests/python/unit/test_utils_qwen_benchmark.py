import json

import torch

from moe_infinity.utils.qwen_benchmark import (
    aggregate_request_records,
    build_qwen_benchmark_config,
    load_chat_trace,
    percentile,
    summarize_hit_rate_tensor,
)


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
            },
            "library_stats_delta": {
                "query_count": 2,
                "hit_count": 1,
                "admit_count": 1,
            },
            "cache_hit_rate_delta": {"overall_hit_rate": 0.5},
        },
        {
            "latency_s": 2.0,
            "generated_tokens": 20,
            "dispatcher_stats": {
                "enqueue_count": 200,
                "busy_wait_count": 1,
                "busy_wait_total_wait_us": 50,
                "busy_wait_max_wait_us": 50,
            },
            "library_stats_delta": {
                "query_count": 4,
                "hit_count": 3,
                "admit_count": 1,
            },
            "cache_hit_rate_delta": {"overall_hit_rate": 0.75},
        },
    ]
    aggregate = aggregate_request_records(records)
    assert aggregate["request_count"] == 2
    assert aggregate["generated_tokens_total"] == 30
    assert aggregate["generated_tokens_per_second"] == 10.0
    assert aggregate["library_query_count_total"] == 6
    assert aggregate["library_hit_count_total"] == 4
    assert aggregate["mean_dispatcher_enqueue_count"] == 150.0
    assert aggregate["mean_cache_hit_rate"] == 0.625


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
