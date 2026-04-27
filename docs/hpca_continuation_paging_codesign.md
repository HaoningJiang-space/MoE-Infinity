# HPCA Direction: Continuation-Aware Expert Paging Co-Design

Date: 2026-04-27

## Purpose

This note reframes the current `continuation cache` line for an HPCA-style hardware/software co-design paper.

The key shift is:

- not `better predictor`
- not `better paging heuristic`
- but `metadata / lookup / scheduling co-design for decode-time MoE expert paging`

---

## One-Sentence Thesis

The dominant hidden failure mode in decode-time MoE expert paging is **retrieval-object mismatch**:

> request-level or sequence-level traces are too coarse for layer-local prefetch deadlines.

Once the retrieval object is corrected to a `local continuation` object, the bottleneck shifts from candidate omission to:

- metadata lookup latency
- continuation aggregation overhead
- transfer scheduling and deadline management
- runtime progress under no-victim / all-locked cache pressure

This opens a hardware/software co-design opportunity.

---

## What We Are Not Writing

This should **not** be framed as:

- a better expert predictor
- a better utility estimator
- a better cache heuristic
- a semantic expert specialization paper

Those framings are too crowded and too incremental for HPCA.

---

## Current Empirical Position

The current software line has already established several facts:

1. `candidate omission gap` is consistently larger than `restricted gap`
   - the dominant loss happens before ranking/scheduling
   - controller refinement is secondary

2. `local continuation object` improves same-step candidate quality substantially
   - sequence-level retrieval asks the wrong question
   - layer-local continuation retrieval better matches decode-time paging deadlines

3. the first local implementation was too slow, but acceleration fixed the main software bottleneck
   - local retrieval semantics are viable
   - remaining overhead is implementation-path dependent, not evidence that the object is wrong

4. current runtime validation is moving toward pressure-sensitive fixed-length evaluation
   - this is the correct bridge from observation to architecture

In short:

> the software evidence already supports object mismatch as the primary failure mode.

What remains is to show that:

> once the object is fixed, the next bottleneck is the metadata path itself, and that bottleneck benefits from co-design.

---

## HPCA Problem Statement

### The real systems problem

Decode-time MoE paging must answer a highly local question:

> given the current token and current layer, which experts are likely needed in the next few layers, and can they be transferred before their deadlines?

Sequence-level trace abstractions are poorly aligned with this question.

### The architecture consequence

Once retrieval is made local and continuation-centric, software pays a new cost:

- constructing local keys
- searching continuation metadata
- aggregating continuation values
- turning retrieved continuation into prefetch descriptors
- scheduling transfers under deadlines and queue contention

This is no longer just a `prediction` problem.

It becomes a **metadata-path and transfer-scheduling problem**, which is architecture-relevant.

---

## Proposed HPCA Claim

The paper should claim something like:

> Existing MoE paging systems retrieve history at the wrong granularity, causing oracle experts to be omitted before scheduling begins.  
> We show that layer-local continuation retrieval fixes this dominant miss source, but shifts the bottleneck to metadata lookup and deadline-aware transfer scheduling.  
> We co-design a continuation-aware paging substrate that makes local continuation retrieval and expert transfer hardware-efficient.

This is stronger than:

- “we have a better predictor”
- “we have a better scheduler”

because it identifies:

1. the hidden failure mode
2. the correct runtime object
3. the new post-fix bottleneck
4. the co-design mechanism

---

## Architecture Insight

### Before object correction

The main problem is:

- correct experts never enter the candidate set

This appears as:

- high omission gap
- low same-step recall

### After object correction

The main problem becomes:

- software metadata handling cost
- retrieval lookup latency
- transfer scheduling under deadlines

This is exactly the point where HPCA becomes natural:

> the software path has identified the right object, but that object exposes a new microarchitectural bottleneck.

---

## New Runtime-Progress Evidence From v10b

The strong-pressure v10b run exposed a separate issue from prediction quality:

- configuration: `device_memory_ratio=0.30`, `prefetch_future_layers=4`, `prefetch_max_candidates=32`
- trace/variant at failure: `mixed`, `history_reuse_local_backbone`
- failure mode: `ExpertDispatcher::GPUFetchFunc: evict_node is nullptr`
- preceding symptom: `All cached expert locked, waiting for cache to be available`

This is not a CUDA OOM result. It is a runtime progress failure:

> a demand fetch needs cache space, but the currently evictable candidate set is temporarily empty; the runtime waits once and then treats the condition as fatal.

This matters because it separates two problems:

1. `retrieval-object mismatch`: the original candidate-generation failure, measured by omission gap.
2. `runtime progress under pressure`: the post-retrieval execution failure, exposed only when prefetch and demand traffic contend for a small expert cache.

For paper framing, v10b should not be used as a normal performance point. It should be treated as a robustness boundary:

- normal performance sweep: use stable pressure points such as `0.40/0.35`, or use conservative `0.30` with smaller prefetch windows.
- robustness stress case: keep `0.30 + future_layers=4 + max_candidates=32` to show where the current runtime loses progress.

The main lesson is:

> hardware cannot replace the software progress fix, but the crash exposes exactly the pressure signals and queue conflicts that an HPCA-style paging substrate should manage.

---

## Required Software Progress Semantics

Before claiming hardware help, the software runtime needs a correct progress contract.

### 1. Demand fetch must make progress

If no victim is immediately available:

- wait and recheck in a loop
- tolerate spurious condition-variable wakeups
- emit bounded diagnostics if the wait is long
- do not fatal just because the evictable set is temporarily empty

### 2. Prefetch is best-effort

Prefetch should not compete with demand fetch as a hard requirement. Under no-victim pressure:

- drop prefetch
- defer prefetch
- or throttle prefetch admission

The runtime must never let speculative prefetch make demand fetch lose progress.

### 3. Victim selection needs ownership or revalidation

`FindExpertEvict()` should not merely observe a candidate and release its lock. The runtime needs one of:

- hold the victim lock until eviction completes
- or revalidate atomically before eviction

Otherwise victim selection has a TOCTOU window under concurrent execution and prefetch.

---

## Hardware-Relevant Pressure Mechanisms

The v10b failure suggests four concrete co-design mechanisms.

### 1. Demand / prefetch queue separation

Demand fetches need priority over prefetch traffic. A continuation-aware paging substrate should expose separate queues or priorities so prefetch cannot starve demand.

### 2. Reserved demand capacity

Keep a small reserve for demand fetches. Prefetch should use opportunistic capacity, not all capacity.

### 3. Hardware-visible pressure counters

Useful signals include:

- cache occupancy
- in-flight transfer count
- locked-node count
- evictable-node count
- no-victim wait time

These counters let the runtime throttle prefetch before reaching the fatal boundary.

### 4. Fast evictable-set metadata

The runtime currently has to discover eviction candidates by scanning software metadata and trying locks. A paging substrate could maintain a low-cost evictable set or victim queue.

This does not replace the software correctness fix. It reduces the frequency and duration of all-locked states.

---

## Minimal Hardware/Software Co-Design

The co-design should stay small and disciplined.

Do **not** jump to a large custom accelerator.

Instead propose a minimal **Continuation Paging Engine (CPE)** with three responsibilities.

### 1. Continuation key buffer

Maintain a small hardware-friendly buffer for the current local continuation key:

- current decode step
- current anchor layer
- recent local routing prefix

Goal:

- avoid repeated software-side key reconstruction
- keep key metadata close to the execution path

### 2. Bucketed continuation lookup

Support exact or near-exact lookup over continuation metadata buckets:

- bucketed by anchor layer
- compact normalized keys
- top-k lookup support

Goal:

- reduce lookup latency
- avoid repeated Python/runtime-side scanning and aggregation overhead

### 3. Deadline-aware transfer queue

Convert retrieved continuation values into transfer descriptors and schedule them by:

- estimated usefulness
- arrival deadline
- queue occupancy
- demand/prefetch priority
- evictable-cache pressure

Goal:

- reduce late prefetches
- reduce wasted bandwidth
- better overlap transfer with expert execution
- avoid prefetch-induced no-victim states

---

## Why This Is HPCA-Like

This direction naturally emphasizes:

- memory hierarchy stress
- metadata organization
- queueing and scheduling under deadlines
- interaction between retrieval granularity and bandwidth efficiency
- architecture-visible latency breakdown

It is therefore closer to:

- paging substrate design
- metadata-path acceleration
- transfer-scheduling co-design

than to:

- systems-only benchmark engineering
- ML-only prediction improvement

---

## What the Software Prototype Must Provide

The current codebase should now be treated as a **bottleneck finder**.

The immediate goal is not to perfect the software implementation indefinitely.

The immediate goal is to extract the right architecture-facing evidence.

The prototype should provide:

1. fixed-length runtime results
2. pressure sweeps across device memory ratio
3. latency breakdowns for:
   - key construction
   - lookup
   - aggregation
   - prefetch enqueue
   - transfer wait / deadline miss
4. metadata footprint breakdowns:
   - per-entry storage
   - total library footprint
   - working-set size at query time

Without this breakdown, the paper remains a systems story.

With this breakdown, it becomes an architecture story.

---

## Immediate Experimental Plan

### Stage 1: finish runtime pressure evidence

Complete and summarize the fixed-length pressure sweep:

- traces:
  - `mixed`
  - `recurrence_heavy`
  - `stationary`
- variants:
  - `on_demand`
  - `history_reuse_backbone`
  - `history_reuse_consensus_backbone`
  - `history_reuse_local_backbone`
- pressure:
  - multiple `device_memory_ratio` settings

The objective is to confirm:

- local continuation reduces runtime stall under pressure
- gains are not an artifact of variable-length generation

### Stage 2: add bottleneck timing instrumentation

Instrument:

- local key construction time
- local lookup time
- continuation aggregation time
- enqueue time
- transfer wait time
- all-locked event count
- no-victim wait time
- prefetch drop/defer count
- demand-vs-prefetch conflict count

This is the most important step for HPCA framing.

### Stage 3: build a trace-driven co-design model

Before RTL, build a compact performance model that estimates the impact of:

- lower metadata lookup latency
- on-chip continuation metadata residency
- deadline-aware transfer scheduling

This lets us compare:

- software local continuation
- software + CPE model

without overcommitting to a hardware implementation too early.

---

## Evaluation Structure

### Main software baselines

- `on_demand`
- `sequence_object_backbone`
- `sequence_object_consensus`
- `local_continuation`

### Main architecture comparison

Compare:

1. sequence object + software scheduling
2. local continuation + software scheduling
3. local continuation + continuation paging engine model

This is critical.

It proves:

- object correction matters more than controller refinement
- co-design matters after the object is corrected

### Primary metrics

- p50 / p95 / p99 latency
- ms/token
- deadline miss rate
- wasted prefetch bytes
- expert miss stall time
- metadata lookup time
- transfer queue occupancy
- no-victim wait count/time
- all-locked cache events
- prefetch drop/defer rate

### Supporting metrics

- same-step recall
- omission gap
- restricted gap
- cache hit rate

Supporting metrics justify the object story.

Primary metrics justify the architecture story.

---

## What Not To Do Next

Do not prioritize:

- confidence-aware paging as the main paper contribution
- router calibration as the current mainline
- geometry-aware retrieval before the current line is closed
- new predictor heuristics

These can all exist later, but they weaken the current HPCA story if introduced too early.

---

## Strongest Current Narrative

The strongest HPCA-style story available now is:

1. Existing decode-time MoE paging uses the wrong retrieval granularity.
2. This creates a candidate omission bottleneck before scheduling begins.
3. A continuation cache corrects the retrieval object and exposes a new metadata-path bottleneck.
4. A continuation-aware paging substrate is needed to make the corrected object hardware-efficient.

That is much sharper than:

- “our predictor is better”
- “our utility estimator is better”
- “our scheduler is smarter”

---

## Recommendation

If the target is HPCA, continue on the current path, but change the center of gravity:

- from software retrieval method
- to metadata-path and transfer-scheduling co-design

In practical terms:

1. finish `v10`
2. add fine-grained timing/metadata instrumentation
3. write down the minimal CPE abstraction
4. evaluate software vs software+co-design model

One sentence:

> Treat `continuation cache` as the architecture entry point, not the final paper endpoint.
