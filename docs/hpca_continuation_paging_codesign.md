# HPCA 方向：面向内存超配 MoE 推理的 MoE-Specific Expert Paging

日期：2026-04-27

## 核心转向

这份文档把原来的 `continuation cache / continuation-aware paging` 方向，重构成更适合 HPCA 的硬件/软件协同故事。

新的主线不是：

- 更好的 expert predictor
- 更好的 prefetch heuristic
- 更好的 eviction policy

更准确的新主线是：

> 内存超配 MoE 推理缺的不是又一个预测器，也不只是普通的 prefetch throttling。  
> MoE expert 是一种特殊 page：大粒度、layer-deadline、执行期带锁、候选来自 speculative routing，而且 demand miss 会阻塞 decode critical path。  
> 因此，普通 UVM / cache prefetch 语义不足，需要 MoE-specific expert paging semantics。

换成大白话：

- demand fetch 是“现在不用这个 expert 就走不下去”。
- prefetch 是“猜测未来可能要用，提前搬一下”。
- 这两类请求不能平权。
- prefetch 可以失败、丢弃、延后；demand fetch 必须保证能继续前进。

因此，HPCA 版本的论文不应叫 `Continuation-Aware Expert Paging`。更合适的标题方向是：

> Progress-Guaranteed Expert Paging for Memory-Oversubscribed MoE Inference

但要注意：

> `progress-guaranteed` 不是自动 non-incremental。  
> 真正的新意不是“prefetch 会污染 cache”或“demand 应该优先”，而是 MoE expert page fault 的语义和通用 UVM/page prefetch 不一样。

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
> 我们提出 MoE expert paging 的体系结构语义：expert 是 deadline-bearing、lock-constrained、large-object page；speculative expert traffic 的 admission 必须同时考虑 evictability、lock state、layer deadline 和 demand reserve。

---

## 哪些不能当 novelty

下面这些点在传统体系结构、GPU UVM、cache/prefetch 文献里都不是新问题。可以作为背景压力，但不能当论文核心新意：

- demand 优先于 prefetch
- prefetch 可以 drop / defer / throttle
- cache pollution
- bandwidth waste
- reserved free buffer
- prefetch accuracy / timeliness feedback
- prefetch 和 eviction coordination
- proactive eviction
- page migration / oversubscription 管理

如果只写这些，审稿人很容易说：

> 这是把已有 UVM/prefetch/paging 技术套到 MoE expert cache 上。

必须把 novelty 收紧到 MoE expert 的特殊 fault model：

- expert 是大粒度对象，不是 4KB/2MB 普通 page。
- expert 有 layer deadline，晚到就等于 miss。
- expert 执行期会被锁住，锁状态直接影响 evictability。
- expert candidate 来自 routing speculation，不是 regular stride 或普通 locality。
- demand expert miss 会阻塞 autoregressive decode critical path。
- 多个 future-layer prefetch 会制造 cancelable speculative traffic。

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

### 3. v10b/v13 暴露了第二层问题：runtime progress

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

v13 用同类长运行配置重跑，并带上了新的 dispatcher counters：

- root：`/data/ziheng/moe_infinity_fgo_runs/phasea_v13_boundary_repro_counters`
- 配置：`ratio=0.30`, `future_layers=4`, `max_candidates=32`, `warmup=2`, `measured=32`
- `on_demand` 完成：`8.932 tok/s`, miss `5868`, eviction `2418`
- `history_reuse_consensus_backbone` 完成但极慢：`1.145 tok/s`, miss `9381`, eviction `3910`
- `history_reuse_local_backbone` 没有复现 fatal，但进入 progress stall，被人工 `SIGTERM` 结束
- stall 位置：events 停在 `mixed-026 step=11 layer=0`
- gdb 堆栈：主线程在 `ExpertDispatcher::WaitHiddenStates()`，`GPUFetchFunc0`/`GPUExecFunc0` 都在 condition wait

v13 使结论更克制，也更强：

> 强 pressure 下的问题不只是 fatal。  
> speculative expert traffic 会显著增加 miss/evict，并可能把 decode 推入 non-progress 状态；表现可以是 v10b 的 fatal，也可以是 v13 的长期 stall。

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
3. **MoE-specific progress**：无论 prefetch 多激进，blocking expert demand fault 不能被 speculative routing traffic 推入 no-victim / all-locked 状态。

现有很多工作主要优化第 1 点或第 2 点：

- predictor 更准
- cache policy 更好
- eviction 更聪明
- prefetch/scheduling 更激进

我们的 HPCA 版本要主打第 3 点，并把前两点连接起来：

> 当 retrieval object 从 sequence-level 改成 local continuation 后，candidate omission 被缓解，但更激进、更局部的 prefetch 会把系统推向新的瓶颈：expert evictability、execution lock state、layer deadline、demand reserve 和 no-victim progress。  
> 因此，需要 MoE-specific expert paging semantics，而不是简单复用普通 UVM/prefetch 语义。

---

## MoE expert paging fault model

这应该成为 HPCA 论文的核心抽象。

一个 expert miss 不是普通 cache miss，而是：

> 一个大粒度、deadline-bearing、lock-constrained、speculation-fed page fault。

具体包含五个字段：

- `object_size`: expert weight 很大，迁移成本高，不能按普通 cache line/page 思维处理。
- `layer_deadline`: expert 必须在目标 MoE layer 执行前到达，晚到就是 blocking stall。
- `lock_state`: expert 执行期间不可驱逐，lock state 决定 evictable set。
- `fault_type`: demand fault 是 blocking；prefetch fault 是 speculative/cancelable。
- `routing_confidence/source`: candidate 来自 routing continuation 或其他 predictor，存在错和晚的风险。

这组语义才是和通用 prefetch/UVM 区分开的地方。

---

## 真正可能新的机制点

### 1. evictability-aware admission

prefetch admission 不能只看预测分数、带宽、cache occupancy。

它还必须看：

- evictable-set size
- locked expert count
- demand reserve 是否被占用
- candidate layer deadline
- prefetch 是否仍可取消
- demand queue 是否已经积压

这比普通 prefetch throttling 更 MoE-specific，因为 expert 的 evictability 被执行锁和 layer-local deadline 共同决定。

### 2. lock-aware expert page metadata

CPE 维护的不是普通 resident bit，而是 expert-level metadata：

- resident
- locked
- evictable
- deadline
- demand-reserved
- cancelable-prefetch

这样 runtime 可以知道：

- 哪些 expert 真的能驱逐
- 哪些 prefetch 可以取消
- 哪些 demand fault 必须保底

### 3. deadline-bearing prefetch descriptor

prefetch descriptor 需要带：

- target layer
- latest-arrival deadline
- candidate source
- usefulness score
- cancelability

这样 prefetch 不只是“提前搬”，而是“在 deadline 前搬；过期则取消或降级”。

### 4. demand-reserved expert capacity

reserved capacity 本身不新。

MoE-specific 的地方是：

- reserve 是给 blocking expert demand fault 的。
- reserve admission 要看 expert lock/evictability。
- reserve 被 speculative expert traffic 侵占时会直接破坏 decode progress。

### 5. object-level evidence bridge

v9 证明 sequence/request-level retrieval object 会造成 candidate omission。

v10b 证明更 aggressive 的 local hint 会暴露底层 paging progress 问题。

这两个证据要连起来：

> 上层 retrieval object 修对之后，系统不是结束了，而是把 bottleneck 推到底层 expert paging substrate。  
> 这就是为什么本文不是 predictor paper，而是 memory hierarchy contract paper。

---

## 论文主张草案

更锋利的版本应该这样写：

> Existing MoE offloading systems optimize expert prediction and scheduling, but lack a paging contract for speculative expert traffic. We show that under memory pressure, even correct speculation can break progress because MoE experts are large, locked during execution, and constrained by layer deadlines. We propose a MoE-specific expert paging substrate with evictability-aware admission, lock-aware expert metadata, cancelable deadline-bearing prefetch descriptors, and demand-reserved capacity.

中文版本：

> 现有 MoE offloading 系统主要优化 expert prediction 和 scheduling，但缺少 speculative expert traffic 的 paging contract。  
> 我们证明，在强 memory pressure 下，即使预测是对的，也可能破坏 progress，因为 MoE expert 是大对象、执行期带锁、并受 layer deadline 约束。  
> 我们提出 MoE-specific expert paging substrate：evictability-aware admission、lock-aware expert metadata、可取消的 deadline-bearing prefetch descriptor，以及 demand-reserved capacity。

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

### 贡献 2：MoE expert paging fault model

定义 expert miss 为什么不是普通 page/cache miss：

- large-object
- layer-deadline
- lock-constrained
- speculation-fed
- demand-blocking

这部分是 non-incremental 边界，必须比“prefetch 会污染 cache”更靠前。

### 贡献 3：progress-guaranteed paging semantics

定义 expert paging 的最小 runtime contract：

- demand fetch 必须有进展保证。
- prefetch 是 best-effort。
- prefetch admission 必须受 pressure 控制。
- victim selection 必须有 ownership 或 revalidation。
- no-victim 不能直接 fatal，只能 wait/recheck/drop/defer/diagnose。

### 贡献 4：硬件/软件协同 substrate

提出一个小而克制的 paging substrate，不做大 accelerator：

- evictability-aware admission
- lock-aware expert metadata
- cancelable deadline-bearing prefetch descriptors
- demand-reserved expert capacity
- hardware-visible expert pressure counters

### 贡献 5：continuation hint 的作用边界

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

下面机制要写成 MoE-specific expert paging semantics，而不是泛泛的 prefetch queue 管理。

### 1. fault-type aware queue

demand expert fault 和 speculative expert prefetch fault 应该进入不同语义队列。

目标：

- blocking demand fault 不被 speculative routing traffic 阻塞。
- prefetch descriptor 可以根据 layer deadline 被取消、降级、合并。
- deadline 已经过期的 prefetch 不继续占 transfer queue。

注意：队列优先级本身不是新意。MoE-specific 的地方是 queue entry 带有 expert deadline、lock/evictable state 和 cancelability。

### 2. demand-reserved expert capacity

cache 里保留少量 demand-only expert slots。

目标：

- speculative expert traffic 不能占满全部 expert cache。
- demand reserve 的释放和 expert lock state / evictable state 绑定。
- 即使 aggressive local continuation prefetch 失控，blocking demand fault 仍有最小进展空间。

注意：reserved buffer 本身不是新意。MoE-specific 的地方是 reserve 面向 blocking expert fault，而 expert 的可驱逐性由执行锁决定。

### 3. expert-state pressure counters

runtime 需要看到这些状态：

- expert cache occupancy
- in-flight transfer count
- locked expert count
- evictable expert count
- no-victim wait time
- prefetch drop/defer count
- demand/prefetch conflict count
- expired prefetch count
- cancelable prefetch count

这些 counter 的作用不是为了好看，而是为了 admission control：

> 在到达 no-victim/all-locked 边界之前，就把 speculative expert prefetch 降速、丢弃、取消或延后。

### 4. lock-aware evictable-set metadata

当前软件路径需要扫描 metadata、尝试锁、再决定 victim。

可以考虑维护：

- evictable set
- victim queue
- per-layer priority metadata
- lock/evictable bitmap
- deadline-indexed resident expert list
- cancelable prefetch descriptor list

目标：

- 降低 victim selection latency。
- 缩短 no-victim 状态持续时间。
- 减少 all-locked 状态出现概率。

---

## Minimal CPE：Continuation Paging Engine

如果需要一个硬件抽象，可以称为 CPE：Continuation Paging Engine。

注意：CPE 不应该被写成“大型 MoE accelerator”。它只是 expert paging substrate。

### 1. expert fault descriptor buffer

保存当前 expert fault / prefetch descriptor：

- fault type: demand 或 speculative prefetch
- expert object id
- object size
- target layer deadline
- lock/evictable state
- cancelability
- hint source: continuation cache 或其他 predictor

作用：

- 让 paging substrate 直接看到 MoE-specific fault semantics。
- 让 prefetch 不再只是普通异步拷贝，而是带 deadline 和 cancelability 的 expert object fault。

### 2. evictability-aware admission

对每个 prefetch candidate 判断：

- deadline 是否够近
- 当前 transfer queue 是否拥塞
- evictable set 是否足够
- locked expert 是否过多
- demand reserve 是否被侵占
- candidate usefulness 是否足够高

不满足条件则：

- drop
- defer
- 或降低优先级

### 3. lock-aware metadata path

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
- locked expert count snapshot
- evictable expert count snapshot
- expired prefetch count
- cancelable prefetch count

当前已经加了一部分 dispatcher counters。v10 运行时没有使用 inplace rebuild，所以 v10 不能拿这些 counter 当证据；v12/v13 已经在 inplace rebuild 后重跑，可以作为 counter 证据。

当前 counter 证据：

- v12 短 canary：所有 case 完成，没有 no-victim/all-locked，但 miss/evict 非零，说明它是中等 pressure sanity point。
- v13 长 boundary：on-demand 完成，consensus 完成但 miss/evict 和 latency 明显上升，local 进入 progress stall。

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

这一步不能只写成“修了一个 bug”。论文里要把它解释为：

> demand expert fault 的 progress contract 被明确化；speculative expert prefetch 被降级为 cancelable/best-effort traffic。

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
- 软件 + fault-type aware queue
- 软件 + demand-reserved expert capacity
- 软件 + lock-aware evictable metadata
- 软件 + evictability-aware admission
- 软件 + cancelable deadline-bearing prefetch descriptors

模型输入来自真实 counter：

- candidate deadline
- transfer queue occupancy
- no-victim wait
- evictable count
- locked expert count
- drop/defer rate
- demand/prefetch conflict
- expired/canceled prefetch count

这样可以先证明 architecture substrate 的价值，而不是过早承诺硬件实现。

关键图不应该只是 “CPE 更快”。应拆成：

- queue isolation 单独带来多少 tail latency 改善
- demand reserve 单独减少多少 no-victim event
- lock-aware metadata 单独降低多少 victim discovery latency
- deadline admission 单独减少多少 expired / late prefetch
- 组合后是否同时保持 candidate recall 和 demand progress

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
- expired prefetch count
- canceled prefetch count
- late prefetch count

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
- victim discovery latency
- lock wait latency

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

> MoE expert paging needs MoE-specific memory hierarchy semantics.  
> An expert miss is a large-object, layer-deadline, lock-constrained, speculation-fed page fault.  
> Continuation cache shows how to generate better local hints, but strong memory pressure exposes a deeper substrate problem: speculative expert traffic must be admitted based on evictability, lock state, deadline, and demand reserve.

中文一句话：

> 这篇不要讲“我预测 expert 更准”，也不要只讲“prefetch 要节流”；要讲“MoE expert 是一种特殊 page，所以 expert paging 需要不同于普通 UVM/prefetch 的内存层级语义”。
