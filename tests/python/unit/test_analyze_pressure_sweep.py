import json

from benchmarks.analyze_pressure_sweep import build_summary


def _write_raw(
    path,
    *,
    trace,
    variant,
    tok_s,
    hit_rate,
    evictions=0,
    include_progress_counters=True,
):
    dispatcher_stats = {
        "enqueue_count": 10,
        "busy_wait_count": 0,
        "busy_wait_total_wait_us": 0,
        "busy_wait_max_wait_us": 0,
    }
    if include_progress_counters:
        dispatcher_stats.update(
            {
                "cache_hit_fetch_count": 10,
                "cache_miss_fetch_count": 0,
                "eviction_count": evictions,
                "all_locked_event_count": 0,
                "no_victim_wait_count": 0,
                "no_victim_wait_total_us": 0,
                "no_victim_wait_max_us": 0,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "trace_name": trace,
                "variant": variant,
                "fixed_new_tokens": True,
                "max_new_tokens": 16,
                "aggregate": {
                    "request_count": 2,
                    "success_count": 2,
                    "failure_count": 0,
                    "generated_tokens_per_second": tok_s,
                    "latency_per_generated_token_mean_ms": 1000.0 / tok_s,
                    "latency_p95_s": 2.0,
                    "mean_cache_hit_rate": hit_rate,
                    "mean_dispatcher_busy_wait_count": 0.0,
                    "dispatcher_eviction_count_total": evictions,
                    "mean_dispatcher_eviction_count": float(evictions),
                    "mean_dispatcher_all_locked_event_count": 0.0,
                    "mean_dispatcher_no_victim_wait_count": 0.0,
                    "mean_dispatcher_no_victim_wait_total_us": 0.0,
                },
                "records": [{"dispatcher_stats": dispatcher_stats}],
            }
        ),
        encoding="utf-8",
    )


def test_build_summary_handles_partial_sweep_and_local_comparison(tmp_path):
    ratio = tmp_path / "ratio_060"
    _write_raw(
        ratio / "raw" / "mixed__history_reuse_consensus_backbone.json",
        trace="mixed",
        variant="history_reuse_consensus_backbone",
        tok_s=10.0,
        hit_rate=1.0,
        include_progress_counters=False,
    )
    _write_raw(
        ratio / "raw" / "mixed__history_reuse_local_backbone.json",
        trace="mixed",
        variant="history_reuse_local_backbone",
        tok_s=12.0,
        hit_rate=1.0,
        include_progress_counters=False,
    )
    running_log = ratio / "logs" / "mixed__on_demand.log"
    running_log.parent.mkdir(parents=True, exist_ok=True)
    running_log.write_text("Creating model from scratch ...\n", encoding="utf-8")

    summary = build_summary(
        benchmark_root=tmp_path,
        traces=["mixed"],
        variants=[
            "on_demand",
            "history_reuse_consensus_backbone",
            "history_reuse_local_backbone",
        ],
        hit_threshold=0.99,
    )

    coverage = summary["coverage"][0]
    assert coverage["complete"] == 2
    assert coverage["partial"] == 1
    assert coverage["non_pressure"] == 2
    assert coverage["unknown_pressure"] == 1
    assert summary["cases"][0]["has_progress_counters"] is False
    assert summary["comparisons"][0]["tokens_per_second_delta_pct"] == 20.0


def test_build_summary_marks_eviction_cases_as_pressure(tmp_path):
    ratio = tmp_path / "ratio_035"
    _write_raw(
        ratio / "raw" / "mixed__history_reuse_local_backbone.json",
        trace="mixed",
        variant="history_reuse_local_backbone",
        tok_s=4.0,
        hit_rate=0.95,
        evictions=3,
    )

    summary = build_summary(
        benchmark_root=tmp_path,
        traces=["mixed"],
        variants=["history_reuse_local_backbone"],
        hit_threshold=0.99,
    )

    assert summary["cases"][0]["pressure_label"] == "pressure"
    assert summary["cases"][0]["has_progress_counters"] is True
