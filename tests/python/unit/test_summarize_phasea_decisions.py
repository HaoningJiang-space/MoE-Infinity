from benchmarks.summarize_phasea_decisions import (
    _build_comparisons,
    _fixed_token_status,
    _render_markdown,
)


def test_fixed_token_status_checks_measured_records_only():
    raw = {
        "max_new_tokens": 16,
        "records": [
            {"is_warmup": True, "generated_tokens": 3},
            {"is_warmup": False, "generated_tokens": 16},
            {"is_warmup": False, "generated_tokens": 16},
        ],
    }

    status = _fixed_token_status(raw)

    assert status["all_measured_fixed_length"] is True
    assert status["observed_generated_tokens"] == [16, 16]


def test_build_comparisons_compares_local_against_sequence_variants():
    cases = [
        {
            "trace_name": "mixed",
            "variant": "history_reuse_local_backbone",
            "aggregate": {
                "latency_per_generated_token_mean_ms": 90.0,
                "generated_tokens_per_second": 11.0,
                "latency_p95_s": 1.8,
            },
            "phasea_same_step": {
                "pair_recall": 0.30,
                "expert_only_recall": 0.50,
                "candidate_omission_gap": 0.60,
                "restricted_gap": 0.0,
            },
        },
        {
            "trace_name": "mixed",
            "variant": "history_reuse_backbone",
            "aggregate": {
                "latency_per_generated_token_mean_ms": 100.0,
                "generated_tokens_per_second": 10.0,
                "latency_p95_s": 2.0,
            },
            "phasea_same_step": {
                "pair_recall": 0.20,
                "expert_only_recall": 0.40,
                "candidate_omission_gap": 0.75,
                "restricted_gap": 0.05,
            },
        },
    ]

    comparisons = _build_comparisons(cases)

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["rhs_variant"] == "history_reuse_backbone"
    assert comparison["latency_per_token_delta_pct"] == -10.0
    assert comparison["tokens_per_second_delta_pct"] == 10.0
    assert round(comparison["pair_recall_delta"], 6) == 0.1
    assert round(comparison["omission_gap_delta"], 6) == -0.15


def test_render_markdown_includes_progress_counter_columns():
    summary = {
        "benchmark_root": "/tmp/run",
        "exclude_warmup_events": True,
        "same_step_budget": 32,
        "cases": [
            {
                "trace_name": "mixed",
                "variant": "history_reuse_local_backbone",
                "object_label": "local_continuation_simple",
                "fixed_tokens": {"all_measured_fixed_length": True},
                "aggregate": {
                    "generated_tokens_per_second": 10.0,
                    "latency_per_generated_token_mean_ms": 100.0,
                    "latency_p95_s": 2.0,
                    "mean_dispatcher_busy_wait_count": 1.0,
                    "mean_dispatcher_eviction_count": 4.0,
                    "mean_dispatcher_all_locked_event_count": 0.5,
                    "mean_dispatcher_no_victim_wait_total_us": 25.0,
                },
                "controller_latency": {"decision_latency_mean_us": 500.0},
                "phasea_same_step": {
                    "pair_recall": 0.4,
                    "expert_only_recall": 0.6,
                    "candidate_omission_gap": 0.5,
                    "restricted_gap": 0.0,
                },
            }
        ],
        "comparisons": [],
    }

    rendered = _render_markdown(summary)

    assert "all-locked" in rendered
    assert "no-victim us" in rendered
    assert "| mixed | history_reuse_local_backbone" in rendered
