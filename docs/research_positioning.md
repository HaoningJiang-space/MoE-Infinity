# Continuation Cache Research Positioning

Date: 2026-04-27

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
- v8 runtime: `history_reuse_local_backbone` 在真实 runtime 下已经能和 consensus/backbone 对比，recurrence-heavy 上收益明显。
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
- 用 continuation cache 降低 omission gap，并转化为 runtime latency / stall 改善。

当前实验状态：

- v9 score-only object-vs-controller 已完成:
  - 对比 sequence object + simple/topk/consensus/recent-retrieval controller。
  - 对比 local continuation + simple ranking。
  - 结论是 object change 比 controller refinement 更关键。
- v10 runtime fixed-length pressure sweep 正在跑:
  - 固定生成长度，消除 EOS early stop 干扰。
  - 扫 device memory ratio，制造更强 paging pressure。
  - 目标是证明 continuation cache 不只是 observation artifact，而能改善 runtime。
- v10b runtime strong pressure sweep 暴露 robustness boundary:
  - 在 GPU0 上补 `device_memory_ratio=0.30/0.25`。
  - 只保留 `on_demand`、`history_reuse_consensus_backbone`、`history_reuse_local_backbone`。
  - `ratio_030 + future_layers=4 + max_candidates=32` 在 `mixed / history_reuse_local_backbone` 触发 no-victim / all-locked fatal。
  - 这个结果不能作为正常性能点，但可以作为强 memory pressure 下 runtime progress bug 的 robustness evidence。

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

后续最好补：

- 第二个模型，验证不是 Qwen 特例。
- capacity sensitivity，尤其 local continuation library capacity 对 recall、latency、memory metadata overhead 的影响。
- failure cases，说明哪些 workload 下 continuation cache 不占优。

## 当前结论

当前最稳判断：

> retrieval-object mismatch + continuation cache 是现在最接近可投系统论文的故事。  
> 不要退回 better predictor / better paging heuristic。  
> router calibration 和 geometry-aware retrieval 可以作为后续方向，但现在不应抢主线。
