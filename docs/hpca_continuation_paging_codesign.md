# HPCA 方向：面向内存超配 MoE 推理的 Progress-Guaranteed Expert Paging

日期：2026-04-27

## 核心转向

这份文档把原来的 `continuation cache / continuation-aware paging` 方向，重构成更适合 HPCA 的硬件/软件协同故事。

新的主线不是：

- 更好的 expert predictor
- 更好的 prefetch heuristic
- 更好的 eviction policy

而是：

> 内存超配 MoE 推理缺的不是又一个预测器，而是一个有进展保证的 expert paging substrate。  
> 在强 memory pressure 下，speculative prefetch 会和 demand fetch 抢 cache、抢 victim、抢 PCIe/IO 队列。  
> 如果 runtime 不区分 blocking demand fault 和 speculative prefetch fault，就会进入 no-victim / all-locked 状态，甚至直接 fatal。

换成大白话：

- demand fetch 是“现在不用这个 expert 就走不下去”。
- prefetch 是“猜测未来可能要用，提前搬一下”。
- 这两类请求不能平权。
- prefetch 可以失败、丢弃、延后；demand fetch 必须保证能继续前进。

因此，HPCA 版本的论文不应叫 `Continuation-Aware Expert Paging`。更合适的标题方向是：

> Progress-Guaranteed Expert Paging for Memory-Oversubscribed MoE Inference

`continuation cache` 仍然重要，但它应该降级为上层 hint source / retrieval abstraction；真正的 HPCA 核心贡献应放到底层 paging 语义和 memory hierarchy co-design。

---

## 为什么不能主打 continuation predictor

如果论文写成：

> continuation cache + 更好的 expert prefetch

很容易被打成 incremental。

原因是相邻工作已经非常拥挤：

- MoE-Infinity 已经做 request-level trace-guided expert cache/prefetch。
- ProMoE 已经做 proactive caching。
- FineMoE / fMoE 已经做 fine-grained pattern 和 semantic hints。
- DuoServe-MoE 已经在 decode 阶段做 lightweight layer-level predictor。
- LayerScope / PreScope 已经讲 layer-aware predictor、cross-layer scheduling、PCIe bandwidth competition 和 AsyncIO。
- ActiveEvict 已经讲 eviction/loading critical path 解耦。
- SpecMD 已经把 cache policy、eviction policy 和硬件约束放进 benchmark。
- MoE-SpeQ 已经把 speculative execution 和 expert offloading 做 co-design。

所以非 incremental 的边界必须非常清楚：

> 我们不是提出一个更准的 predictor，也不是一个更聪明的 eviction policy。  
> 我们提出 MoE expert paging 的体系结构语义：speculative prefetch 不能破坏 demand progress；expert cache 必须向 runtime 暴露 evictability、lock、deadline 和 pressure 状态。

---

## 当前证据链怎么放

当前软件实验已经证明了第一层问题：retrieval object mismatch。

### 1. object mismatch 是一级瓶颈

v6/v7/v8/v9 的共同结论是：

- `candidate omission gap` 明显大于 `restricted gap`。
- 正确 expert 经常在排序之前就没进候选集合。
- 因此，controller/ranking 再复杂也救不回来。
- sequence/request-level trace 与 layer-local decode-time deadline 不对齐。
- local continuation object 更接近 runtime 真正要执行的动作。

v9 是目前最干净的 object-vs-controller 证据：

- local continuation + simple ranking 的 same-step M32 pair recall 约 `0.466-0.480`。
- sequence object + simple/topk/consensus/recent retrieval 只有约 `0.176-0.226`。
- local continuation 的 omission gap 约 `0.487-0.503`。
- sequence object 的 omission gap 约 `0.680-0.737`。

这说明：

> 先换 retrieval object，比继续调 controller 更重要。

### 2. v10 目前说明 ratio_060 是 control 点

v10 fixed-length pressure sweep 正在跑。当前已完成的 `ratio_060` 部分显示：

- cache hit rate 基本都是 `1.0`。
- busy wait 是 `0`。
- 已完成 case 都被标记为 `non-pressure/control`。

这说明：

> `ratio_060` 不是强 paging-pressure 点。  
> 它适合当 control，不适合证明 paging bottleneck 或 prefetch 在压力下有效。

当前 partial 分析报告：

- `/data/ziheng/moe_infinity_fgo_runs/phasea_v10_runtime_fixedlen_qwen_pressure_sweep/analysis/pressure_sweep_summary.md`

### 3. v10b 暴露了第二层问题：runtime progress

v10b strong-pressure run 的关键现象：

- 配置：`device_memory_ratio=0.30`
- prefetch：`future_layers=4`, `max_candidates=32`
- trace/variant：`mixed / history_reuse_local_backbone`
- failure：`ExpertDispatcher::GPUFetchFunc: evict_node is nullptr`
- 前置症状：`All cached expert locked, waiting for cache to be available`

这不是 CUDA OOM。

它说明：

> demand fetch 需要 cache slot，但当前 evictable candidate set 暂时为空。  
> 当前 runtime 只等一次，然后把“暂时没有 victim”当成不可恢复错误。

这正是 HPCA 切入口：

- 当前系统把 expert cache 当成普通软件缓存。
- 但 MoE decode 里的 expert 更像 deadline-bearing page。
- demand fetch 是 blocking page fault。
- prefetch 是 speculative page fault。
- 两者不能共享同一套无优先级、无 reserve、无 progress contract 的路径。

---

## 新的 HPCA problem statement

内存超配 MoE 推理中的 expert paging 需要同时满足三件事：

1. **候选正确性**：未来要用的 expert 不能在 candidate generation 阶段被漏掉。
2. **deadline 可达性**：prefetch 不是只要猜对就行，还要在对应 layer 执行前到达。
3. **demand progress**：无论 prefetch 多激进，blocking demand fetch 不能被 speculative traffic 卡死。

现有很多工作主要优化第 1 点或第 2 点：

- predictor 更准
- cache policy 更好
- eviction 更聪明
- prefetch/scheduling 更激进

我们的 HPCA 版本要主打第 3 点，并把前两点连接起来：

> 当 retrieval object 从 sequence-level 改成 local continuation 后，candidate omission 被缓解，但更激进、更局部的 prefetch 会把系统推向新的瓶颈：evictability metadata、queue priority、reserved capacity 和 no-victim progress。  
> 因此，需要一个 progress-guaranteed expert paging substrate。

---

## 论文主张草案

可以这样写：

> Existing MoE offloading systems treat speculative expert prefetch and blocking expert demand fetch as ordinary cache traffic. Under strong memory pressure, this breaks progress: prefetch can occupy cache capacity, lock victim candidates, and contend for transfer queues, leaving demand fetch with no evictable victim. We propose a progress-guaranteed expert paging substrate that separates demand and prefetch semantics, exposes evictability metadata, reserves demand capacity, and performs deadline-aware admission. Continuation retrieval is used as an upper-layer hint source, while the core contribution is the paging substrate that preserves demand progress under speculative MoE expert traffic.

中文版本：

> 现有 MoE offloading 系统通常把 speculative expert prefetch 和 blocking expert demand fetch 都当成普通缓存请求。  
> 在强 memory pressure 下，这会破坏进展：prefetch 可能占住 cache、锁住 victim、挤占传输队列，导致 demand fetch 找不到可驱逐对象。  
> 我们提出一个有进展保证的 expert paging substrate：区分 demand/prefetch 语义，暴露 evictability metadata，保留 demand capacity，并做 deadline-aware admission。  
> continuation retrieval 只是上层 hint source；核心贡献是让 speculative MoE expert traffic 不破坏 demand progress。

---

## 目标贡献

### 贡献 1：failure characterization

证明强 pressure 下的关键失败不是预测准确率，而是 progress failure：

- no-victim wait
- all-locked expert cache
- demand/prefetch conflict
- prefetch 占用 cache slot
- prefetch 与 demand 抢 transfer queue
- software victim scan / try-lock 路径过慢或不稳定

v10b 的 fatal log 是 motivation，但不够。需要把它扩展成可量化 characterization。

### 贡献 2：progress-guaranteed paging semantics

定义 expert paging 的最小 runtime contract：

- demand fetch 必须有进展保证。
- prefetch 是 best-effort。
- prefetch admission 必须受 pressure 控制。
- victim selection 必须有 ownership 或 revalidation。
- no-victim 不能直接 fatal，只能 wait/recheck/drop/defer/diagnose。

### 贡献 3：硬件/软件协同 substrate

提出一个小而克制的 paging substrate，不做大 accelerator：

- demand/prefetch queue isolation
- reserved demand cache capacity
- hardware-visible pressure counters
- fast evictable-set metadata
- deadline-aware prefetch admission

### 贡献 4：continuation hint 的作用边界

continuation cache 的定位是：

- 提供更对齐 layer-local deadline 的 prefetch hint。
- 证明 sequence-level retrieval object 会造成 candidate omission。
- 触发更真实的 speculative expert traffic。

但它不是 HPCA 论文的唯一主贡献。

---

## 必须满足的软件 progress 语义

### 1. demand fetch 必须保证继续前进

如果当前没有 victim：

- 不能直接 fatal。
- 不能只 wait 一次。
- 必须 `while no victim -> wait/recheck`。
- 要允许 condition variable 的虚假唤醒。
- 长时间等待时输出 bounded diagnostics。

这修的是 correctness/progress bug，不是性能优化。

### 2. prefetch 必须是 best-effort

prefetch 不应该和 demand fetch 平权。

在 no-victim 或 cache pressure 高时，prefetch 应该：

- drop
- defer
- throttle
- 或只允许进入 opportunistic capacity

原则是：

> prefetch 可以损失命中率，但不能让 demand fetch 丢 progress。

### 3. victim selection 要有 ownership 或 revalidation

`FindExpertEvict()` 不能只是看一眼谁能 `try_lock`。

选中 victim 后必须满足其中一种：

- 持有 victim lock 直到 eviction 完成。
- 或 eviction 前立即重新原子校验。

否则就有 TOCTOU：看见能驱逐和真正驱逐之间，状态可能已经变了。

---

## 硬件相关机制

### 1. demand / prefetch 队列隔离

demand fetch 和 prefetch 应该进入不同 priority queue。

目标：

- demand 不被 speculative traffic 阻塞。
- prefetch 可以被降级、合并、取消。
- deadline 近的请求优先级更高。

### 2. reserved demand capacity

cache 里保留少量 demand-only slots。

目标：

- prefetch 不能占满全部 cache。
- 即使 aggressive prefetch 失控，demand 仍有最小进展空间。

### 3. hardware-visible pressure counters

runtime 需要看到这些状态：

- cache occupancy
- in-flight transfer count
- locked-node count
- evictable-node count
- no-victim wait time
- prefetch drop/defer count
- demand/prefetch conflict count

这些 counter 的作用不是为了好看，而是为了 admission control：

> 在到达 fatal 边界之前，就把 prefetch 降速、丢弃或延后。

### 4. fast evictable-set metadata

当前软件路径需要扫描 metadata、尝试锁、再决定 victim。

可以考虑维护：

- evictable set
- victim queue
- per-layer priority metadata
- lock/evictable bitmap

目标：

- 降低 victim selection latency。
- 缩短 no-victim 状态持续时间。
- 减少 all-locked 状态出现概率。

---

## Minimal CPE：Continuation Paging Engine

如果需要一个硬件抽象，可以称为 CPE：Continuation Paging Engine。

注意：CPE 不应该被写成“大型 MoE accelerator”。它只是 expert paging substrate。

### 1. continuation hint buffer

保存当前局部 hint：

- 当前 decode step
- 当前 layer
- 最近几层 local routing prefix
- continuation cache 返回的候选 expert 和 deadline

作用：

- 让 runtime 不必反复重建 key。
- 让 prefetch hint 更接近 transfer scheduling path。

### 2. deadline-aware admission

对每个 prefetch candidate 判断：

- deadline 是否够近
- 当前 transfer queue 是否拥塞
- evictable set 是否足够
- demand reserve 是否被侵占
- candidate usefulness 是否足够高

不满足条件则：

- drop
- defer
- 或降低优先级

### 3. evictable metadata path

维护低开销 metadata：

- 哪些 expert 当前可驱逐
- 哪些 expert 被执行路径锁住
- 哪些 expert 是 demand reserve
- 哪些 prefetch candidate 可以取消

作用：

- 避免纯软件扫描和 try-lock 造成的长尾。
- 给 demand fetch 提供更稳定的 victim discovery。

---

## 实验路线

### Stage 1：完成正常 runtime pressure sweep

继续完成 v10：

- traces: `mixed`, `recurrence_heavy`, `stationary`
- variants:
  - `on_demand`
  - `history_reuse_backbone`
  - `history_reuse_consensus_backbone`
  - `history_reuse_local_backbone`
- ratios:
  - `0.60`
  - `0.45`
  - `0.35`

解释方式：

- `0.60` 如果全是 hit rate 1.0，就是 control。
- 真正看 pressure 的是 `0.45/0.35`。
- aggressive `0.30` 不混入正常性能图，应单独当 robustness boundary。

### Stage 2：补 progress counters

必须记录：

- `no_victim_wait_count`
- `no_victim_wait_total_us`
- `no_victim_wait_max_us`
- `all_locked_event_count`
- `prefetch_drop_count`
- `prefetch_defer_count`
- `demand_prefetch_conflict_count`
- eviction count
- cache hit/miss fetch count

当前已经加了一部分 dispatcher counters，但 v10 运行时没有使用 inplace rebuild，所以 v10 不能拿这些 counter 当证据。后续要重新 build/install 后跑 canary。

### Stage 3：修 progress bug

软件修复顺序：

1. no-victim wait 从 one-shot wait 改为 loop wait/recheck。
2. demand fetch 不再因为 temporarily no victim fatal。
3. prefetch 在 no-victim/high-pressure 下 drop/defer。
4. victim selection 做 lock ownership 或 revalidation。
5. 所有 drop/defer/wait/conflict 都进 raw JSON 和 summary。

验收标准：

- v10b 同类强压力配置不再 fatal。
- aggressive prefetch 不一定变快，但 demand progress 保住。
- prefetch drop/defer 随 pressure 上升而上升。
- no-victim wait 从 fatal log 变成可量化指标。

### Stage 4：跑 post-fix 三类实验

正常性能：

- `0.60/0.45/0.35`

保守强压力：

- `0.30`
- `future_layers=2`
- `max_candidates=16`

robustness boundary：

- `0.30`
- `future_layers=4`
- `max_candidates=32`

预期不是 aggressive `0.30` 一定最快。

预期是：

- 不 fatal。
- demand progress preserved。
- prefetch drop/defer 明显增加。
- no-victim wait 变成可分析曲线。

### Stage 5：trace-driven CPE model

在 RTL 之前，先做 trace-driven model。

比较：

- 纯软件 local continuation
- 软件 + demand/prefetch queue priority
- 软件 + reserved demand slots
- 软件 + fast evictable metadata
- 软件 + deadline-aware admission

模型输入来自真实 counter：

- candidate deadline
- transfer queue occupancy
- no-victim wait
- evictable count
- drop/defer rate
- demand/prefetch conflict

这样可以先证明 architecture substrate 的价值，而不是过早承诺硬件实现。

---

## 代码 backlog

### P0：保持证据干净

- v10 继续跑完。
- `ratio_060` 标为 control。
- `ratio_030 + future_layers=4 + max_candidates=32` 标为 robustness boundary。
- 不把 crash 配置混入正常 performance 图。

### P1：补全 counter

已有：

- cache hit/miss fetch count
- eviction count
- all-locked count
- no-victim wait count/time

还需要：

- prefetch drop count
- prefetch defer count
- demand/prefetch conflict count
- evictable-node count snapshot
- locked-node count snapshot

### P2：补 timing breakdown

policy 层：

- key construction
- lookup
- aggregation
- scoring/ranking
- enqueue

runtime 层：

- fetch queue wait
- H2D transfer wait
- execution wait
- no-victim wait
- deadline miss

### P3：实现 progress semantics

demand：

- loop on no-victim wait
- recheck after wakeup
- no fatal on temporarily empty evictable set

prefetch：

- best-effort
- drop/defer/throttle under pressure
- 不占 demand reserve

victim：

- ownership 或 revalidation
- 避免 TOCTOU

### P4：实验与报告

- 重新 inplace build 后跑 counter canary。
- 跑 post-fix robustness boundary。
- 更新 `pressure_sweep_summary.md`。
- 把 progress counter 加进 decision summary。
- 形成 failure characterization 图：
  - all-locked count vs ratio
  - no-victim wait vs ratio
  - prefetch drop/defer vs ratio
  - demand latency tail vs ratio

---

## 最终判断

当前版本如果写成：

> continuation cache improves expert prefetch

不够 HPCA，也不够 non-incremental。

更稳的 HPCA 版本是：

> MoE expert paging needs progress-guaranteed memory hierarchy semantics.  
> Continuation cache shows how to generate better local hints, but strong memory pressure exposes a deeper substrate problem: speculative prefetch must not break demand progress.  
> We co-design demand/prefetch priority, reserved demand capacity, evictable metadata, and deadline-aware admission to make expert paging robust and efficient.

中文一句话：

> 这篇不要讲“我预测 expert 更准”，要讲“MoE expert paging 需要一种新的内存层级语义：prefetch 可以投机，但 demand 必须有进展保证”。
