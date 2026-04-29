# Continuation Cache Research Positioning

Date: 2026-04-28

## 2026-04-29 Runtime Guardrail

v43-v47 之后，本文档里所有早期 runtime acceleration 说法都必须降级解释：

- `generate` 模式不能作为机制比较主证据；跨进程输出路径不稳定，会污染 TPOT、miss 和 eviction。
- 正式 runtime 对比必须优先使用 `benchmark_mode=forward` 的固定输入路径，并配合 bracketed baseline。
- `history_reuse_local_backbone` 不能再写成当前加速机制；v47 中它 admitted `11008` 个 candidates，但只有约 `100` 个 used，吞吐只有 baseline 的约 `0.885x`。
- `trace_similarity_prefetch` 当前 real run 中 candidate/admit 为 `0`，不能作为有效 MoE-Infinity prefetch baseline，除非先证明 tracebase 和 candidate 生成非空。

因此当前可信主张是：

> local continuation 仍可作为 retrieval-object mismatch 的观察证据；但当前同步 runtime prefetch 实现是负结果，只能作为“speculative expert traffic/lifecycle 转化率差”的诊断样本，不能作为主方法。

## 一句话主张

当前最强主线不是“更好的 expert predictor”，而是：

> decode-time MoE expert paging 的隐藏主瓶颈，是 retrieval object mismatch。  
> sequence/request-level trace 和 layer-local prefetch deadline 不对齐，导致正确 expert 没进候选集合。  
> continuation cache 把检索对象改成当前 step/layer 的 local continuation，从源头降低 candidate omission gap。

换成大白话：

- 以前很多方法在问：“这整条请求像不像以前某条请求？”
- 但 prefetch 真正需要回答：“当前 token、当前 layer，接下来几层马上要用哪些 expert？”
- 这两个问题不是一回事。
- 如果候选集合一开始就漏掉正确 expert，后面 controller/ranking 再聪明也救不回来。

## 为什么不能写成 better predictor

expert prediction / prefetch / cache / paging 这条线已经很挤。相关工作包括：

- MoE-Infinity: request-level tracing + prefetch/caching.
- ProMoE: proactive caching, 用中间结果预测后续 expert.
- fMoE: fine-grained expert pattern + semantic hints.
- Fate: cross-layer gate input 用于 expert prefetch.
- DuoServe-MoE: decode 阶段 lightweight layer-level predictor.
- Pre-Attention Expert Prediction: 同层 pre-attention activation 预测 expert.
- SpecMD / Speculating Experts / MoE-SpAc: speculative prefetch、eviction、utility estimator、benchmarking.

所以如果论文写成：

> 我们提出一个更好的 prefetch predictor。

很容易被审稿人认为是 incremental。

更稳的写法是：

> 我们发现已有 sequence/request-level retrieval object 与 decode-time layer-local paging action 不匹配。  
> 我们用 omission-gap decomposition 证明主要损失来自 candidate omission，而不是 ranking。  
> 我们提出 continuation cache 作为新的 runtime retrieval abstraction。

## 当前核心证据

已有 v6/v7/v8/v9 结果支持这个方向：

- v6 score-only: local continuation 明显提高 same-step recall，降低 omission gap，但最初 decision latency 偏高。
- v7 acceleration: local path decision latency 从约 2.5ms 降到约 0.6ms，说明工程瓶颈主要是实现方式，不是 object 本身错。
- v8 runtime: 早期 `generate` 模式结果只能说明 local continuation 路径可运行；v43-v47 后不再把它作为性能收益证据。
- v9 score-only object-vs-controller: local continuation + simple ranking 稳定打过 sequence object + 更复杂 controller/retrieval。

v9 是目前最干净的 object-vs-controller 证据：

- 结果目录：`/data/ziheng/moe_infinity_fgo_runs/phasea_v9_scoreonly_object_vs_controller_qwen`
- 汇总文件：`analysis/decision_summary.md`
- 固定输出长度：所有 case 都是 `fixed_new_tokens=true`，避免 EOS early stop 污染 score-only 对比。
- 对比对象：
  - sequence object + simple: `history_reuse_backbone`
  - sequence object + topk controller: `history_reuse_topk_backbone`
  - sequence object + consensus controller: `history_reuse_consensus_backbone`
  - sequence object + recent retrieval: `history_reuse_consensus_backbone_retrieval`
  - local continuation object + simple ranking: `history_reuse_local_backbone`
- 三个 trace 上，local continuation 的 same-step M32 pair recall 稳定在约 `0.466-0.480`。
- sequence object 的几个 controller/retrieval 变体，same-step M32 pair recall 只有约 `0.176-0.226`。
- local continuation 的 same-step M32 omission gap 约 `0.487-0.503`。
- sequence object 的 omission gap 约 `0.680-0.737`。
- local continuation 的 decision latency mean 约 `497-509us`，低于 sequence object simple 的约 `1105-1135us`，也明显低于 consensus/recent retrieval 的约 `1489-2195us`。

v9 的解释很关键：

- 把 sequence object 的 controller 做得更复杂，只能带来小幅 recall 波动，无法根治 omission。
- 把 retrieval object 换成 local continuation，即使用 simple ranking，也能把 oracle expert 更频繁放进候选集合。
- 所以主张应写成“object mismatch 是一级瓶颈”，而不是“我们有一个更强 predictor/controller”。

最关键的诊断是：

- candidate omission gap 长期明显大于 restricted gap。
- 这说明主要问题不是“候选内部排序不好”，而是“正确 expert 根本没进候选集合”。
- 因此 controller refinement 是二级问题，retrieval object 才是一级问题。

## 和 Fate / cross-layer predictor 的区别

Fate 等工作已经使用 adjacent-layer gate input 做 expert prefetch，所以不能只说：

> 我们使用 layer-local 信息预测 expert。

这会撞车。

必须强调：

- 我们不是直接提出又一个 cross-layer predictor。
- 我们先提出 gap decomposition，证明 sequence-level retrieval 的失败来自 object mismatch。
- continuation cache 是一个 runtime retrieval object：key 是当前 decode step/layer 附近的 local routing prefix，value 是同一步未来几层的 expert continuation。

换句话说，重点不是“预测器更 clever”，而是“检索对象和 runtime actuation 对齐”。

## 相关理论文献如何帮助定位

几篇理论/解释性文献让方向更清楚：

- Local Routing Consistency: 说明不是所有 MoE 都同样适合 offloading，offloadability 本身是模型属性。
- Myth of Expert Specialization: 说明 routing similarity 更接近 hidden-state geometry，不应过度讲 semantic specialization。
- Equifinality in MoE: 说明 routing topology 不一定决定模型质量，不应把故事写成“更 clever 的 router topology”。
- Router Calibration: 说明 deployment-time router-expert mismatch 是真实问题，但它更适合作为后续方向。

这些文献共同支持一个更大的 umbrella：

> deployment-oriented routing / runtime-oriented retrieval abstraction.

但当前论文不要铺太大。当前最稳的落点仍是：

> retrieval-object mismatch + continuation cache.

## 当前不建议主打的方向

### Confidence-aware paging

可以作为后续 controller 层优化，但不应作为主线。

原因：

- 这条线也拥挤。
- 你自己的实验显示一级瓶颈是 candidate omission。
- 如果 object 还没对齐，先做 confidence-aware scheduler 会冲淡最强故事。

### Semantic expert specialization

不建议主打。

原因：

- routing similarity 未必来自语义 specialization。
- 更稳的解释框架是 local routing geometry / continuation pattern。

### 现在就跳 geometry-aware retrieval

有潜力，但不建议现在跳。

原因：

- 实现复杂。
- 容易变成 representation learning paper。
- 当前 continuation cache 已经足够支撑一个更清晰的 systems story。

## 推荐路线

### 第一阶段：当前主线

Continuation Cache for MoE Expert Paging

目标：

- 证明 retrieval object mismatch 是 decode-time expert paging 的主失败模式。
- 用 omission-gap decomposition 区分 candidate generation 和 ranking/controller。
- 用 continuation cache 降低 omission gap；runtime 加速必须由后续 fixed-input forward benchmark 和 lifecycle counters 单独证明。

当前实验状态：

- v9 score-only object-vs-controller 已完成:
  - 对比 sequence object + simple/topk/consensus/recent-retrieval controller。
  - 对比 local continuation + simple ranking。
  - 结论是 object change 比 controller refinement 更关键。
- v10 runtime fixed-length pressure sweep 已完成 `ratio_045/060`:
  - 固定生成长度，消除 EOS early stop 干扰。
  - `ratio_045/060` 全部是 `non-pressure/control`：hit rate 约 `1.0`，busy wait 为 `0`。
  - 这批结果适合做 control，不适合证明 paging pressure。
- v10b runtime strong pressure sweep 暴露 robustness boundary:
  - 在 GPU0 上补 `device_memory_ratio=0.30/0.25`。
  - 只保留 `on_demand`、`history_reuse_consensus_backbone`、`history_reuse_local_backbone`。
  - `ratio_030 + future_layers=4 + max_candidates=32` 在 `mixed / history_reuse_local_backbone` 触发 no-victim / all-locked fatal。
  - 这个结果不能作为正常性能点，但可以作为强 memory pressure 下 runtime progress bug 的 robustness evidence。
- v15/v17 进一步把 robustness boundary 固化成 progress failure 证据:
  - v15：同类 aggressive boundary 下，`on_demand` 和 `consensus` 完成，`local` 进入 all-locked/no-victim 循环，被人工 `SIGTERM`。
  - v17：修正 pending-stall guard 后，`local` 自动抛出 `WaitHiddenStates progress stall`。
  - 关键诊断是 `pending=1`, `no_victim_wait=613`, `idle_us=60003517`。
  - 这说明当前还不是最终修复，而是把 fatal/silent stall 推进成可诊断 failure。
- v18/v19/v20 开始把 progress failure 推向 mitigation:
  - v18：第一版 admission-v1 仍失败，说明只在 locked-ratio 高时限流太晚。
  - v19：strict admission 把 speculative prefetch 全部 drop，同样 `ratio_030/local/aggressive` 完成；miss `5959`、evict `2621`，但 no-victim/all-locked/pending-stall 都是 `0`。
  - v20：bounded admission 每个 plan 放行少量 prefetch，admitted/enqueued `47104`、drop `302822`，同样完成；miss `8908`、evict `3389`，no-victim/all-locked/pending-stall 仍是 `0`。
  - 这说明 `ratio_030` 不是必然不可跑；问题来自 unbounded speculative expert traffic，bounded admission 可以恢复 demand progress。
- v21 把 v20 的手工 cap 固化成正式机制:
  - 使用显式 `prefetch_admission_max_per_plan`，而不是 `locked_ratio_threshold=0.0` 的间接 hack。
  - 新增 `cap drop` 和 `pressure drop` counter，区分主动 bounded admission 与真正 pressure-triggered drop。
  - 新增 `admit_rate` 和 `pressure_drop_rate`，把 speculation 强度量化。
  - 已跑 `no_admission/cap0/cap4/cap8/cap16/cap32` 小矩阵，找最大安全 speculation window。
  - 当前 `mixed/ratio030/aggressive` 下，`cap0/cap4/cap8` 完成，`no_admission/cap16/cap32` 触发 progress stall。
  - 初步最大安全窗口是 `cap8`；`cap16` 已越过 progress boundary。
  - 这批是双 GPU 并行 robustness sweep，不能直接包装成最终性能图；如果要比较 tok/s，需要对 `cap0/cap4/cap8` 做单独 sequential rerun。
- v22/v24 改变了对“cap8 性能收益”的解释:
  - v22 sequential rerun 显示 `cap8` 相对 `cap0` 只有约 8% 吞吐提升，说明 hard cap 可以恢复 progress，但不是最终机制。
  - v24 overhead decomposition 显示 `prefetch_enabled_no_policy` 和 `local_credit0_skip_policy` 都在约 `9.2-9.3 tok/s`，而 `sequence_credit0_update_only` 和 `local_credit0_update_only` 都只有约 `0.93 tok/s`。
  - 这说明当前最大性能问题不是 local continuation object 独有，而是同步 `policy.update_only` / expert trace capture 被放进 decode critical path。
  - 因此，下一步不能只继续调 cap；要证明 optional speculation 必须在源头被 credit-gated，且不能先同步生成再 drop。
- v26 进一步确认了 prefetch 语义需要拆开:
  - `static_hot_top4/top8` 接近 on-demand，说明无同步 trace capture 的路径没有 v24 那种 10x control-plane tax。
  - 但 static cases 虽然有约 `46K-47K` 次 enqueue attempt，runtime dequeue/complete/resident-hit 都是 `0`；miss/evict 下降更像 candidate-set retention / eviction side effect，而不是 real H2D prefetch hit。
  - `local_sync_cap8` 只有 `1.866 tok/s`，且虽然完成 `1132` 次 prefetch，resident-hit 仍是 `0`。
  - 所以后续必须区分 candidate-set retention、prefetch task queueing、真正 H2D prefetch completion 和 demand hit，不能把它们都叫 prefetch 收益。

这组结果对主线的影响：

- continuation cache 仍然是 retrieval-object mismatch 的核心证据。
- 但 HPCA 方向不应写成“local continuation 更会 prefetch”。
- 更强的说法是：local continuation 让上层 hint 更局部、更激进，从而暴露了底层 expert paging contract 缺失；需要把 speculative expert traffic 降级成 best-effort / bounded traffic，保证 demand progress。
- v24 后还要再加一句：speculative control-plane 也必须被隔离；没有 credit 时不应同步捕获 route trace、生成 candidates、排序再丢弃。

### 第二阶段：增强方向

Offloadability-aware router calibration

目标：

- 不改变 expert，只做 lightweight router-only calibration。
- 让 routing 更稳定、更 cacheable、更适合 offloading。
- 这可以作为 continuation cache 后的一篇，或作为当前工作的增强版。

风险：

- 容易被看成 training/model adaptation paper。
- 必须限制成 post-training、unlabeled calibration、router-only small update，并且明确目标是 offloading latency/cacheability。

### 第三阶段：远期方向

Geometry-aware continuation retrieval

目标：

- 把 continuation cache 的 key 从 local routing pattern 升级为 local hidden-state geometry。
- 更直接利用 routing geometry，而不是语义标签。

风险：

- 系统边界更难控制。
- 不建议作为当前第一篇主线。

## 当前论文 claim 草案

可以写成：

> Existing MoE expert paging systems often retrieve history at the wrong granularity.  
> Request-level traces are poorly aligned with layer-local decode-time prefetch deadlines, causing the oracle experts to be omitted before ranking or scheduling begins.  
> We expose this failure mode with omission-gap decomposition and introduce a continuation cache that retrieves local future expert continuations conditioned on recent layer-local routing prefixes.

中文版本：

> 现有 MoE expert paging 系统经常在错误粒度上检索历史。  
> request-level trace 和 decode-time layer-local prefetch deadline 不对齐，导致正确 expert 在排序之前就被漏掉。  
> 我们用 omission-gap decomposition 暴露这个失败模式，并提出 continuation cache，用当前 step/layer 的 local routing prefix 检索未来几层的 expert continuation。

## 需要补强的证据

已经补到第一版的证据：

- v9 固定输出长度 score-only object-vs-controller。
- 三个 Qwen trace/workload：`mixed`、`recurrence_heavy`、`stationary`。
- local object + simple ranking 优于 sequence object + stronger ranking/controller。

仍必须补：

- 固定输出长度的 runtime，避免 generated token 数不同污染 latency。
- 更强 memory pressure，证明真的解决 paging/stall，而不是低压力下的 benchmark noise。
- 正常 performance sweep 和 robustness boundary 必须分开，避免把 runtime correctness bug 混进性能结论。

## 具体补充路线

现在不要继续盲目加 predictor。后续补充应该按下面三个轨道推进。

### A. 正常性能证据

目标是证明 continuation cache 的 candidate 优势能在真实 runtime 下转成 latency / stall 收益。

优先补：

- 把 `v10 ratio_045/060` 明确写成 control，不再当 pressure 证据。
- 后续正常 performance sweep 只使用不崩溃、可重复的 pressure 点。
- 汇总稳定 pressure 点下的 `ms/token`、p95 latency、busy wait、cache hit、Phase-A same-step M32 recall / omission gap。
- 如果 `ratio_035` 稳定，再补一个 conservative strong-pressure 点：
  - `device_memory_ratio=0.30`
  - `prefetch_future_layers=2`
  - `prefetch_max_candidates=16`
- 正常性能图表只放不崩溃、可重复的配置。

预期图表：

- Figure 1: object-vs-controller score-only recall / omission gap, 使用 v9。
- Figure 2: fixed-length runtime under memory pressure, 使用 v10。
- Figure 3: local vs sequence object latency breakdown / busy-wait trend。

### B. Runtime progress 修复与 robustness 证据

目标是把 v10b 暴露的问题收成 runtime progress guarantee，而不是把 crash 混进性能结论。

软件侧需要补：

- demand fetch: 已从 `wait once then fatal` 推进到 `while no victim -> wait/recheck`，并能触发 bounded progress-stall 诊断。
- prefetch: 仍需实现 no-victim 或 cache-pressure 高时 drop/defer，不能和 demand fetch 平权抢 cache。
- victim selection: 已加入 victim lock ownership 的第一版修复，但还要继续用实验验证。
- diagnostics: 已有 all-locked/no-victim/pending-stall，仍缺 prefetch drop/defer、demand-vs-prefetch conflict、locked/evictable snapshot。

实验侧需要分开：

- normal performance sweep: `0.60/0.45/0.35`，或者 conservative `0.30`。
- robustness boundary: `0.30 + future_layers=4 + max_candidates=32`。

robustness boundary 的用途：

- 证明强 pressure 下当前 runtime 会进入 all-locked / no-victim 区间。
- 证明 progress guard 后不再 silent/fatal，而是结构化 `progress stall`。
- 后续 admission 修复要证明它进一步变成可完成或可控 drop/defer。
- 给 HPCA 方向提供 demand/prefetch priority、reserved slots、evictable metadata 的动机。

### C. HPCA / co-design 补充

目标是把 continuation cache 从 software method 提升成 architecture entry point。

需要补的 measurement：

- local key construction time。
- continuation lookup time。
- aggregation time。
- prefetch enqueue time。
- transfer wait / deadline miss。
- metadata footprint: per-entry bytes、library total bytes、query working set。
- runtime pressure counters: evictable-node count、locked-node count、queue occupancy。

需要补的模型：

- software local continuation baseline。
- local continuation + reduced metadata lookup latency。
- local continuation + demand/prefetch priority queue。
- local continuation + reserved demand slots。
- local continuation + fast evictable-set metadata。

HPCA 论文里不要声称硬件“修 bug”。正确说法是：

> 软件先提供 progress guarantee；硬件/architecture substrate 让 all-locked 状态更少、更短、更可控。

后续最好补：

- 第二个模型，验证不是 Qwen 特例。
- capacity sensitivity，尤其 local continuation library capacity 对 recall、latency、memory metadata overhead 的影响。
- failure cases，说明哪些 workload 下 continuation cache 不占优。

## 当前结论

当前最稳判断：

> retrieval-object mismatch + continuation cache 是现在最接近可投系统论文的故事。  
> 不要退回 better predictor / better paging heuristic。  
> router calibration 和 geometry-aware retrieval 可以作为后续方向，但现在不应抢主线。
