# HPCA 方向：面向内存超配 MoE 推理的 MoE-Specific Expert Paging

日期：2026-04-28

## 2026-04-29 Runtime Guardrail

v43-v47 之后，HPCA 方向需要更克制：

- 不能再把当前 local-continuation runtime prefetch 当作正向加速机制。
- v47 固定 forward 输入下，`history_reuse_local_backbone` 发出 `11008` 个 candidate/admit/enqueue，但只有约 `100` 个 prefetch resident hit，吞吐约为 baseline 的 `0.885x`。
- `trace_similarity_prefetch` 当前 candidate/admit 为 `0`，不能作为有效 prefetch baseline。
- `generate` 模式只用于功能 smoke；正式机制比较必须使用 fixed-input forward benchmark、bracketed baseline 和 lifecycle counters。

当前可保留的 HPCA 主张是：

> MoE expert speculation 的关键问题不是“这个 predictor 稍微更准”，而是 speculative expert traffic 的生命周期转化率、deadline、credit/admission 和 demand progress contract。当前 local-continuation prefetch 实现是一个负例：它证明无约束同步 speculation 会制造大量无效 traffic，而不是证明 local continuation 已经能加速。

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

### 2. v10 说明 ratio_045/060 都是 control 点

v10 fixed-length pressure sweep 的已完成部分显示：

- `ratio_045` 和 `ratio_060` 都完成。
- cache hit rate 基本都是 `1.0`。
- busy wait 是 `0`。
- 已完成 case 都被标记为 `non-pressure/control`。
- 这些 case 使用的是旧 rebuild，dispatcher pressure counters 不完整，因此只能当 control，不适合作为 pressure 证据。

这说明：

> `ratio_045/060` 不是强 paging-pressure 点。
> 它们适合当 control，不适合证明 paging bottleneck 或 prefetch 在压力下有效。

当前 partial 分析报告：

- `/data/ziheng/moe_infinity_fgo_runs/phasea_v10_runtime_fixedlen_qwen_pressure_sweep/analysis/pressure_sweep_summary.md`

### 3. v10b/v13/v15/v17/v18/v19/v20 暴露并开始闭环第二层问题：runtime progress

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

v15 在 progress loop patch 后重跑同类 aggressive boundary：

- root：`/data/ziheng/moe_infinity_fgo_runs/phasea_v15_progress_patch_boundary`
- 配置：`ratio=0.30`, `future_layers=4`, `max_candidates=32`, `warmup=2`, `measured=32`
- `on_demand` 完成：`9.150 tok/s`, miss `4366`, eviction `1915`
- `history_reuse_consensus_backbone` 完成但极慢：`1.150 tok/s`, miss `9395`, eviction `3949`
- `history_reuse_local_backbone` 进入 all-locked / no-victim 循环，被人工 `SIGTERM` 结束
- local log 中 `All cached expert locked` 出现 `3499` 次

v17 修正 pending-stall guard 后，用 60s timeout targeted 复现 local boundary：

- root：`/data/ziheng/moe_infinity_fgo_runs/phasea_v17_progress_guard_local_timeout`
- 配置：`mixed / history_reuse_local_backbone`, `ratio=0.30`, `future_layers=4`, `max_candidates=32`
- 结果：exit code `1`，不再需要人工 kill
- 错误：`ExpertDispatcher::WaitHiddenStates progress stall`
- 关键诊断：`pending=1`, `enqueue=2137`, `fetch_dequeue=230`, `exec_dequeue=2136`, `output=2136`, `eviction=55`, `no_victim_wait=613`, `idle_us=60003517`

v17 的含义要克制解释：

> 这不是最终修复。
> 它只说明 runtime 已经从 fatal / silent stall 前进到可诊断的 progress failure。
> 真正的下一步是 prefetch admission / demand reserve / drop-defer，而不是继续调 predictor。

v18 在同一类 targeted boundary 上打开第一版 prefetch admission，但仍然失败：

- root：`/data/ziheng/moe_infinity_fgo_runs/phasea_v18_prefetch_admission_local_timeout`
- 配置：`mixed / history_reuse_local_backbone`, `ratio=0.30`, `future_layers=4`, `max_candidates=32`
- admission：enabled, `demand_reserve=2`, `locked_ratio_threshold=0.8`, `max_under_pressure=4`
- 结果：exit code `1`
- 关键诊断：`pending=1`, `enqueue=1560`, `fetch_dequeue=195`, `exec_dequeue=1559`, `output=1559`, `eviction=19`, `no_victim_wait=613`, `idle_us=60003469`
- summary 中 `all-locked=639`, `progress-stall=1`

v18 的含义是：

> 只在“看起来已经锁很多”时才限流太晚。
> admission 必须更像 contract，而不是事后补救；speculative prefetch 的进入量本身要被 bounded。

v19 是 strict admission control：

- root：`/data/ziheng/moe_infinity_fgo_runs/phasea_v19_prefetch_admission_strict_local`
- 配置：同样 `ratio=0.30`, `future_layers=4`, `max_candidates=32`, `warmup=2`, `measured=32`
- admission：enabled, `demand_reserve=64`, `locked_ratio_threshold=0.0`, `max_under_pressure=0`
- 结果：完成，exit code `0`
- miss `5959`, eviction `2621`，说明仍然是真实 paging pressure
- `no-victim=0`, `all-locked=0`, `pending-stall=0`
- prefetch candidate `349208`, admitted `0`, drop `349208`

v19 的含义是：

> `ratio=0.30` 本身不是必崩点。
> 当 speculative prefetch 全部被 admission 掉，demand-only 路径可以保持 progress。
> 这把 failure 从“内存比例太低”收紧成“aggressive speculative expert traffic 破坏 progress”。

v20 是 bounded speculation：

- root：`/data/ziheng/moe_infinity_fgo_runs/phasea_v20_prefetch_admission_cap4_local`
- 配置：同样 `ratio=0.30`, `future_layers=4`, `max_candidates=32`, `warmup=2`, `measured=32`
- admission：enabled, `demand_reserve=64`, `locked_ratio_threshold=0.0`, `max_under_pressure=4`
- 结果：完成，exit code `0`
- miss `8908`, eviction `3389`，仍然有明显 paging pressure
- `no-victim=0`, `all-locked=0`, `pending-stall=0`
- prefetch candidate `349926`, admitted/enqueued `47104`, drop `302822`

v20 的含义比 v19 更重要：

> 不需要完全关闭 prefetch。
> 只要把 speculative expert traffic 做 bounded admission，系统就能在强 pressure 下保住 demand progress。
> 这支持 HPCA 主张：关键不是再调 predictor，而是给 speculative expert traffic 加 MoE-specific paging contract。

v20 还有一个限制：

> 它用 `locked_ratio_threshold=0.0` 间接实现每个 plan 最多放行 4 个 candidate。
> 这能证明 bounded speculation 有效，但机制表达不够干净。

v21 已经把这个 hack 变成显式机制：

- `no_admission`: 不启用 admission，保留真正 unbounded speculation failure/control baseline。
- `prefetch_admission_max_per_plan = -1`: 不启用 hard cap。
- `prefetch_admission_max_per_plan = 0`: 全部 speculative prefetch drop。
- `prefetch_admission_max_per_plan = 4/8/16/32`: 每个 policy step、每个 GPU 最多放行对应数量的 speculative expert。
- summary 需要区分 `cap drop` 和 `pressure drop`，避免把“主动限流”和“已经无 victim 的压力 drop”混在一起。
- derived metrics 需要记录 `admit_rate` 和 `pressure_drop_rate`，用于画 speculation intensity 与 progress boundary 的关系。

v21 的核心目标：

> Identify the largest safe speculation window under strong memory pressure.

安全窗口定义为：

- `progress_stall = 0`
- `all_locked_event` 低或为 0
- `no_victim_wait` bounded
- `tok/s` 不差于 `cap0`

v21 cap sweep 结果：

- roots: `/data/ziheng/moe_infinity_fgo_runs/phasea_v21_cap_sweep_*`
- trace/variant: `mixed / history_reuse_local_backbone`
- pressure: `device_memory_ratio=0.30`, `future_layers=4`, `max_candidates=32`
- `no_admission`: failed, `admit_rate=1.0000`, `no_victim=714`, `all_locked=714`, `progress_stall=1`
- `cap0`: complete, `admit_rate=0.0000`, `no_victim=0`, `all_locked=0`, `progress_stall=0`
- `cap4`: complete, `admit_rate=0.1347`, `no_victim=0`, `all_locked=0`, `progress_stall=0`
- `cap8`: complete, `admit_rate=0.2688`, `no_victim=0`, `all_locked=0`, `progress_stall=0`
- `cap16`: failed, `admit_rate=0.5278`, `no_victim=654`, `all_locked=654`, `progress_stall=1`
- `cap32`: failed, `admit_rate=1.0000`, `no_victim=735`, `all_locked=735`, `progress_stall=1`

v21 的结论：

> 当前配置下最大安全 speculation window 是 `cap8`；`cap16` 已经越过 progress boundary。
> 这说明 per-plan cap 不是随便的 throttle，而是在寻找 bounded speculation 的安全窗口。

注意：v21 使用双 GPU 并行跑 cap sweep，因此 tok/s 只能作为粗略参考；progress / stall / admission counters 才是这批结果的主证据。

v22 用单 GPU sequential rerun 把 `cap8` 从 robustness 证据推进到性能 tradeoff 证据：

- root: `/data/ziheng/moe_infinity_fgo_runs/phasea_v22_sequential_cap_perf`
- trace/variant: `mixed / history_reuse_local_backbone`
- pressure: `device_memory_ratio=0.30`, `future_layers=4`, `max_candidates=32`
- `cap0`: complete, `1.096 tok/s`, `admit_rate=0`, `drop_cap=348892`, `no_victim=0`, `all_locked=0`
- `cap4`: complete, `1.046 tok/s`, `admit_rate=0.1348`, `no_victim=0`, `all_locked=0`
- `cap8`: complete, `1.188 tok/s`, `admit_rate=0.2694`, `no_victim=0`, `all_locked=0`
- `cap12`: complete, `1.171 tok/s`, `admit_rate=0.3988`, `no_victim=0`, `all_locked=0`
- `cap16_probe`: failed, `admit_rate=0.5271`, `no_victim=3157`, `all_locked=3157`, `progress_stall=1`

v22 的直接结论：

> `cap8` 是当前最干净的 safe-window point，但相对 `cap0` 只有约 8% 吞吐提升。
> 这说明 hard cap 可以恢复 progress，但还不是最终机制。

更重要的是，v22 暴露了一个新的控制面问题：

> `cap0` 不是 no-prefetch baseline。
> 它仍然在 decode critical path 上做 local continuation lookup、candidate generation、ranking、admission、counter/logging，然后把所有 candidates drop。

因此，`drop prefetch != free`。当前实现是：

> generate speculative candidates first, then drop/admit.

更合理的机制应该是：

> paging substrate 先给 speculation credit；没有 credit 就不生成 optional speculative work。

这把主线从“调 cap”推进到 `Credit-Gated Transactional Expert Paging`：

- credit-gated: 没有 evictability/bandwidth/deadline credit，就不 materialize prefetch candidates。
- transactional: prefetched expert 不能直接变成 committed cache state；它应该先处于可撤销 speculative state，只有 demand 使用后才 commit。

当前已实现 v23 的第一步：

- `prefetch_credit_gated_enabled`
- `prefetch_credit_count`
- `credit=0` 时只做 `update_only`，跳过 `update_and_score`、ranking 和 admission。
- `credit>0` 时只 materialize credit 数量以内的 prefetch candidates。
- 新 counter: `prefetch_credit_skip_count`, `prefetch_credit_issued_total`, `prefetch_credit_materialized_count`。

v23 的目标不是先证明最终速度，而是做 overhead decomposition：

- `on_demand`
- `cap0_generate_then_drop`
- `cap0_skip_generation`
- `cap8_generate_then_drop`
- `cap8_credit_gated_generation`

如果 `cap0_skip_generation` 明显快于 `cap0_generate_then_drop`，就能证明：

> dropped speculation still consumes critical-path control resources.

如果 `cap8_credit_gated_generation` 快于 `cap8_generate_then_drop`，就能证明：

> speculation throttling must move upstream before candidate materialization.

v24 已经把这个归因进一步钉住：

- root: `/data/ziheng/moe_infinity_fgo_runs/phasea_v24_policy_overhead_decomposition`
- trace/pressure: `mixed`, `ratio=0.30`, fixed 16 new tokens
- `on_demand`: `6.130 tok/s`, `163.14 ms/token`
- `prefetch_enabled_no_policy`: `9.341 tok/s`, `107.05 ms/token`
- `local_credit0_skip_policy`: `9.244 tok/s`, `108.18 ms/token`
- `sequence_credit0_update_only`: `0.930 tok/s`, `1075.83 ms/token`
- `local_credit0_update_only`: `0.926 tok/s`, `1080.09 ms/token`
- `cap8_credit_gated_generation`: `0.873 tok/s`, `1145.56 ms/token`

v24 的关键含义不是“local continuation 太重”，而是：

> 同步 `policy.update_only` / route capture 放在 decode critical path 上，本身就足以把吞吐从约 `9 tok/s` 打到约 `0.9 tok/s`。

这个结论非常重要，因为它修正了前面对 cap8 的解释：

- `cap0` / `cap8` 的差异不是最终机制收益，只是在一个很重的同步 control path 内部比较。
- `local_credit0_update_only` 和 `sequence_credit0_update_only` 几乎一样慢，说明问题不是 local-continuation object 独有，而是同步 expert trace capture / update path。
- `prefetch_enabled_no_policy` 和 `local_credit0_skip_policy` 都接近 `9 tok/s`，说明底层 prefetch wiring 本身不是主要慢点。
- `cap8_credit_gated_generation` 更慢，说明“生成候选、更新历史、再限流”仍然违反 optional-work 原则。

所以当前 HPCA 主张要再收紧一层：

> MoE speculation 不仅要 progress-isolated，还要 control-plane-isolated。
> optional prefetch work 不能在没有 credit 的情况下做同步 GPU->CPU route capture、candidate generation 和 ranking。

换句话说，`Credit-Gated Transactional Expert Paging` 不能只是末端 admission cap。更正确的 contract 是：

1. paging substrate 先发 credit。
2. 没有 credit 时，predictor 不做同步 trace capture，也不生成 candidates。
3. 有 credit 时，只 materialize credit 范围内的 schedulable speculation。
4. prefetched expert 先进入可撤销 speculative state，不能直接污染 committed expert cache。

v24 之后，下一步实验不应该继续泛泛调 cap，而应该先验证两个问题：

- **prefetch lifecycle 是否健康**：issued / dequeued / completed / canceled / used / late 各是多少。
- **无同步 trace capture 的 prefetch 是否还能工作**：用 static/no-sync prefetch baseline 检查数据面是否比 `local_sync_cap8` 更接近 on-demand。

v26 已经给出第一版答案：

- root: `/data/ziheng/moe_infinity_fgo_runs/phasea_v26_static_prefetch_sanity`
- trace/pressure: `mixed`, `ratio=0.30`, fixed 16 new tokens
- `on_demand`: `8.636 tok/s`, miss `3032`, eviction `1309`
- `prefetch_enabled_no_policy`: `9.259 tok/s`, miss `3074`, eviction `1339`
- `static_hot_top4`: `9.071 tok/s`, miss `1480`, eviction `620`
- `static_hot_top8`: `8.977 tok/s`, miss `1875`, eviction `821`
- `local_sync_cap8`: `1.866 tok/s`, miss `4663`, eviction `1883`

v26 最关键的 lifecycle 结果：

- `static_hot_top4/top8` 都有约 `46K-47K` 次 Python/C++ enqueue 调用。
- 但它们的 runtime dequeue / complete / resident-hit / cache-prefetch 都是 `0`。
- `local_sync_cap8` 有 `47075` 次 enqueue、`1132` 次 runtime dequeue/complete，但 resident-hit 仍是 `0`。

这说明当前必须把三个概念分开：

1. **candidate-set retention**：`replace_cache_candidates()` 会保护 candidate set，使这些 expert 不容易被 demand eviction 踢掉。
2. **prefetch task admission**：Python/C++ 调用了 `enqueue_prefetch()`，但这不等于 task 真正进入 worker queue。
3. **real H2D prefetch utility**：只有 runtime dequeue/complete 后又被 demand 命中，才算真正 useful prefetch。

v26 的解释应非常克制：

> static/no-sync cases 接近 on-demand，说明去掉同步 trace capture 后 control-plane tax 消失。
> static cases 的 miss/evict 下降，主要像是 candidate-set retention / eviction side effect，而不是 prefetch worker 提前搬运后的命中。
> local_sync_cap8 慢，说明同步 policy update/query 仍是主瓶颈；它完成的 prefetch 也没有形成 resident-hit。

因此，下一步代码和实验要先修 counter 语义：

- `prefetch_runtime_enqueue_count` 表示 API enqueue attempt。
- 新增 `prefetch_runtime_queue_push_count`，表示真正放入 worker queue。
- 新增 `prefetch_runtime_same_device_skip_count`，表示因为 source/destination 已相同而没有入队。
- 后续图里不能把 enqueue attempt 当作 issued prefetch。

### 4. prefetch lifecycle 语义还没有完全证明

代码路径还暴露了一个必须单独记录的问题：

```text
drive_expert_policy
-> update_and_score
-> prefetch_experts
-> rank_prefetch_candidates
-> _admit_prefetch_tensor_ids
-> replace_cache_candidates(admitted)
-> enqueue_prefetch(admitted)
```

`replace_cache_candidates` 在 C++ 里的语义不是普通 append：

- 清空 `candidates_`
- 插入新的 admitted candidates
- 清空所有 priority>0 的 prefetch queue

因此：

> `cap0` / `cap8` 的当前语义不是单纯 prefetch cap，而是 bounded admission + plan replacement/cancellation。

这会影响对结果的解释：

- `cap0_generate_then_drop` 每层都会 `replace_cache_candidates([])`，也就是清空旧 prefetch plan。
- `cap8_generate_then_drop` 每层会用新的 admitted set 替换旧 plan，旧 queue 可能来不及完成就被清掉。
- 这可能解释为什么 prefetch 吞吐收益不明显：plan replacement 太频繁，prefetch lifecycle 很短。

因此后续 summary 必须额外记录：

- `prefetch_plan_replace_count`
- `prefetch_plan_empty_replace_count`
- `prefetch_plan_candidate_count`
- `prefetch_plan_cleared_candidate_count`
- `cache_prefetch_count_total`

这些 counter 的含义：

- plan replace: runtime 收到多少次新 prefetch plan。
- empty replace: 多少次用空 plan 清掉旧 prefetch queue。
- plan cleared: replacement 之前还有多少 plan candidates 被覆盖。
- completed prefetch: runtime hit-rate tensor 里完成过的 prefetch 次数，是 prefetch lifecycle 的低成本下界信号。

v25 开始补更强的 lifecycle counter：

- runtime prefetch enqueue / dequeue / complete
- replacement 清掉的 queued prefetch task
- candidate set replacement 前被覆盖的 candidate 数
- demand 命中已经 resident 的 prefetched expert
- demand miss 时同一个 expert 是否还在 pending prefetch 队列里，也就是 late prefetch signal
- try-lock / eviction failure on prefetch task

仍然不能过度声称：

> cap8 的收益来自 useful prefetch。

因为 `prefetch unused / expired` 和真正的 deadline hit 还没有完整状态机；但 v25 已经足够回答更基础的问题：

- prefetch task 是否真的进入底层 runtime。
- 它们是否被 plan replacement 清掉。
- 它们是否完成。
- demand 是否真的撞上了 prefetched resident。
- 需求到达时 prefetch 是否还没来得及完成。

更稳的论文说法是：

> cap8 恢复了 progress，并在当前实现下给出有限性能收益；下一步要分离 control-plane tax、plan cancellation 和 true prefetch utility。

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
- per-plan speculative traffic cap

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
- v15 长 boundary：on-demand 和 consensus 完成，local aggressive 进入 no-victim/all-locked 循环并被人工终止。
- v17 targeted boundary：local aggressive 自动抛出 `WaitHiddenStates progress stall`，把 silent/fatal failure 转成结构化诊断。
- v18 targeted admission-v1：默认压力门控仍然太晚，local aggressive 继续触发 progress stall。
- v19 strict admission：drop 全部 speculative prefetch 后，同样 pressure 下 demand progress 恢复。
- v20 cap4 admission：每个 prefetch plan 放行少量 speculative experts，其余 drop；仍然完成，且 miss/evict 非零。
- v21 cap sweep：用显式 `prefetch_admission_max_per_plan` 跑 no_admission/cap0/cap4/cap8/cap16/cap32；当前最大安全 speculation window 是 cap8。

### Stage 3：修 progress bug

软件修复顺序：

1. no-victim wait 从 one-shot wait 改为 loop wait/recheck。
2. demand fetch 不再因为 temporarily no victim fatal。
3. prefetch 在 no-victim/high-pressure 下 drop/defer。
4. victim selection 做 lock ownership 或 revalidation。
5. 所有 drop/defer/wait/conflict 都进 raw JSON 和 summary。

当前状态：

- 第 1/2/4 步已经有第一版实现，并在 v17 中证明能诊断 progress stall。
- 第 3/5 步已经有第一版 admission + counter 实现：v19/v20 证明 drop / bounded admission 可以恢复 progress。
- 新增显式 per-plan cap 后，v21 将不再依赖 `locked_ratio_threshold=0.0` 这种实验 hack。
- 现在还不能把它说成最终硬件机制，只能说是 software proof-of-concept：prefetch 从 hard traffic 被降级成 best-effort / bounded speculative traffic。

验收标准：

- v10b 同类强压力配置不再 fatal：v19/v20 已初步满足。
- aggressive prefetch 不一定变快，但 demand progress 保住：v19/v20 已初步满足。
- prefetch drop/defer 随 pressure 上升而上升：v19/v20 已有 drop counter，defer 还未实现。
- no-victim wait 从 fatal log 变成可量化指标：v17/v18 已经能结构化暴露，v19/v20 在 admission 后降为 0。

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

- prefetch defer count
- expired prefetch count
- canceled prefetch count
- late prefetch count

已补第一版：

- prefetch candidate/admitted/enqueue/drop count
- demand/prefetch conflict count
- evictable-node count snapshot
- locked-node count snapshot
- failure raw artifact：即使 request 中途 `RuntimeError`，也会写出 partial raw 和 failure snapshot

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
- 支持 per-plan hard cap，避免单个 policy step 注入过多 speculative expert traffic

当前实现状态：

- 已支持 drop / bounded admission。
- 还没有真正 defer queue。
- v19/v20 的 cap 已经证明方向有效。
- v21 的显式 `prefetch_admission_max_per_plan` 会把 cap 从实验 hack 变成正式机制。

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
