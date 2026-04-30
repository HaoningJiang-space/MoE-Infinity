# Phase-A Experiment Audit and Forward Protocol

Date: 2026-04-29

## Current Diagnosis

The recent experiments should be interpreted as a mechanism audit, not as a
sequence of small policy improvements. The main finding is that the current
runtime path mixes several different mechanisms under the word `prefetch`:

- retrieval-object quality;
- candidate-set replacement / eviction protection;
- actual H2D prefetch transfer;
- synchronous policy update overhead;
- admission / progress control;
- benchmark cache-state effects.

These must be separated before any HPCA/System claim is defensible.

## Evidence Grading

| Result group | Trust level | Use in paper | Reason |
| --- | --- | --- | --- |
| v9 score-only object-vs-controller | High | Yes, characterization | Fixed-token score-only comparison. Local continuation improves M32 pair recall from about `0.18-0.23` to `0.47-0.48`, and lowers omission gap from about `0.68-0.74` to `0.49-0.50`. |
| v15-v21 pressure/progress boundary | Medium-high | Yes, robustness characterization | Shows unbounded speculative traffic can violate demand progress, and bounded admission restores progress. Use as progress evidence, not final performance evidence. |
| v24 policy-overhead decomposition | High | Yes, overhead diagnosis | Shows synchronous policy update dominates: update-only paths drop to about `0.93 tok/s`, while skip-policy/no-policy paths stay around `9.2-9.3 tok/s`. |
| v26-v28 static/retention ablations | Medium-low | Debug only until rerun | These runs exposed the need to split retention from H2D transfer, but the no-protect/protect comparison was invalid because C++ always skipped cache candidates during eviction. |
| v47 forward real-prefetch | Medium | Negative runtime evidence | Fixed forward bracket shows local real prefetch is slower than on-demand and has low useful conversion. Good negative result, not a method win. |
| v50-v51 lifecycle/opportunity counters | High | Yes, lifecycle diagnosis | Shows most local candidates are already resident: `opportunity/candidate ~= 1.6%`, `skip/enqueue ~= 98.3%`. Real opportunities are queued and often useful, but too rare. |
| v52 execution-mode ablation | Low-medium | Debug only | Partial run suggests `replace_only`, `enqueue_only`, and `replace_and_enqueue` differ sharply. It also exposed that mode semantics were not clean enough before fixing candidate eviction protection. |
| v53 isolated full-copy ablation | Stop | No | Full 27GB per-case copies are too slow and disk-heavy on `/data`; do not use as the standard protocol. |
| v54 retention-flag canary | Medium | Yes, as bug-fix validation | After fixing candidate protection, `replace_only_no_protect` no longer improves over on-demand, while `replace_only_with_protect` slightly lowers misses. `enqueue_only` performs real queue pushes but has very low resident-hit conversion and is slower. |
| v55 corrected retention/transfer | Medium | Yes, mechanism diagnostic | Confirms true H2D enqueue has low resident-hit conversion and hurts latency in this setup. The best static protected case improves speed but has zero transfer opportunity, so it is retention/protection, not H2D prefetch. Eviction count is zero, so this is not final pressure-performance evidence. |

## Corrected Mechanism Interpretation

`replace_cache_candidates()` is not a pure prefetch operation. It changes the
candidate set used by the eviction path. Before the fix in
`core/prefetch/task_scheduler.cpp`, sparse eviction skipped candidates
unconditionally, even when `prefetch_retention_protect_demand_eviction=false`.

Therefore:

- previous `replace_only_no_protect` cases were not actually no-protect;
- miss reductions in static cases are mostly retention / candidate protection,
  not H2D prefetch;
- true H2D prefetch evidence must require nonzero `queue_push`, `complete`, and
  `dispatcher_prefetch_resident_hit_count`;
- speedup without transfer opportunity is not prefetch acceleration.

## Forward Experiment Protocol

All future runtime evidence must follow this protocol:

1. Use fixed-input `benchmark_mode=forward` for mechanism comparisons.
2. Use bracketed on-demand baselines: `baseline_pre -> mechanism -> baseline_post`.
3. Report normalized-to-bracket tok/s, but do not claim speedup unless lifecycle counters agree.
4. For each mechanism, report:
   - candidates;
   - transfer opportunities;
   - queue push;
   - completion;
   - resident hit;
   - late miss;
   - `opportunity/candidate`;
   - `push/opportunity`;
   - `hit/opportunity`;
   - `skip/enqueue`.
5. Treat `opportunity/candidate < 5%` as a no-real-prefetch-opportunity regime.
6. Treat `queue_push == 0` and `miss reduction > 0` as candidate retention, not H2D prefetch.
7. Do not use full-copy isolated templates as the default. Use fresh offload stores only for small validation runs, or explicitly mark shared-template results as lifecycle diagnostics.

## Immediate Next Experiments

Run a small corrected ablation after rebuilding the C++ extension:

- `on_demand`;
- `static_top4_replace_only_no_protect`;
- `static_top4_replace_only_with_protect`;
- `static_top4_enqueue_only`;
- `static_top4_replace_and_enqueue_no_protect`;
- `static_top4_replace_and_enqueue_with_protect`;
- `local_enqueue_only`;
- `local_replace_and_enqueue`.

This rerun replaces v27/v28/v52 for retention-vs-transfer conclusions. The
expected decision points are:

- If no-protect no longer reduces misses, previous static gains were eviction protection.
- If enqueue-only completes but has low resident hits, H2D prefetch timing is poor.
- If with-protect reduces misses without transfer, the immediate mechanism is
  residency protection, not prefetch.
- If local still has low `opportunity/candidate`, the next bottleneck is creating
  real transfer opportunity or using stronger memory pressure, not improving ranking.

The initial v54 canary already supports this direction:

| case | tok/s | miss | transfer opp | queue push | prefetch resident hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| on_demand | `163.746` | `768` | `0` | `0` | `0` |
| replace_only_no_protect | `159.588` | `912` | `224` | `0` | `0` |
| replace_only_with_protect | `164.021` | `702` | `160` | `0` | `0` |
| enqueue_only | `142.286` | `1403` | `351` | `351` | `15` |

This is not yet a final performance table, but it validates the corrected
semantics: candidate-set retention/protection and H2D prefetch transfer are
different mechanisms and must not be reported under one `prefetch` number.

The follow-up v55 corrected ablation should replace v27/v28/v52 for
retention-vs-transfer interpretation. Its bracket baseline is about
`152.9 tok/s`. The important outcomes are:

- `static_replace_enqueue_with_protect` is about `1.09x` bracket baseline, but
  has zero transfer opportunity and zero queue push, so the observed effect is
  retention/protection only.
- `static_enqueue_only` and `local_enqueue_only` perform real queue pushes and
  completions, but resident-hit conversion is low and latency is worse.
- `local_replace_enqueue_with_protect` has useful hits only when opportunities
  become rare; it remains slower than on-demand.
- all v55 cases have `evict=0`, so v55 remains a mechanism diagnostic rather
  than a strong memory-pressure performance result.

## Research Direction After Audit

The clean story is not "better expert predictor". The clean story is:

> MoE expert paging needs explicit semantics separating speculative hints,
> retention/protection, actual transfer, and demand progress.

Continuation retrieval remains useful as an object-mismatch observation, but the
runtime contribution should be framed around expert paging semantics:

- demand faults are blocking;
- speculative prefetch is cancelable and best-effort;
- candidate retention is distinct from transfer;
- admission must be opportunity-aware and progress-aware;
- control-plane work must be credit-gated before synchronous trace update.

## Named Experiment Protocol

Stop adding opaque `vXX` runs for the next stage. Use named roots:

- `original_workflow_parity_v1`: checks whether upstream open-source
  MoE-Infinity actually enables the prefetch path for the target model.
  Current DeepSeek upstream-compatible smoke:
  `/data/ziheng/moe_infinity_fgo_runs/original_workflow_parity_v1_upstream_env_check`.
  Both README-default and `prefetch=true` cases succeed under `forward_fallback`
  and report zero `ExpertPrefetcher` calls, so they are workflow evidence only,
  not performance baselines.
- `pressure_validated_retention_transfer_v1`: reruns the retention/transfer
  mechanism split with explicit pressure labels.
- `prefetch_lifecycle_timeliness_v1`: measures enqueue, dequeue, completion,
  resident hit, and late miss as primary evidence.
- `local_continuation_runtime_v1`: only runs after a validated pressure point
  exists.
