# Deployment-Oriented Routing Directions for MoE Inference

Date: 2026-04-27

## 目的

这份文档回答一个具体问题：

> 如果目标仍然是推理加速，如何避开拥挤的 “better predictor / better paging heuristic” 赛道，把问题提升成更有新意的 deployment-oriented routing？

结论先说：

- 当前最强主线仍然是 `retrieval-object mismatch -> continuation cache abstraction`
- 最值得保留的下一篇/增强版方向是 `offloadability-aware router calibration`
- `geometry-aware continuation retrieval` 是有潜力的远期升级版，但不适合现在先跳

---

## 一句话判断

最不拥挤、又还能落到推理加速的方向，不是“再做一个 predictor”，而是：

> 把系统问题提升成 `deployment-oriented routing`。  
> 重点不是谁预测未来 expert 更准，而是 MoE 的 routing structure 是否和 deployment/runtime abstraction 对齐。

---

## 为什么不能再写成 better predictor

`expert prediction / prefetch / cache / paging` 这条线已经很挤。代表工作包括：

- [MoE-Infinity](https://arxiv.org/abs/2401.14361)
- [ProMoE](https://arxiv.org/abs/2410.22134)
- [fMoE](https://arxiv.org/abs/2502.05370)
- [Fate](https://arxiv.org/abs/2502.12224)
- [DuoServe-MoE](https://arxiv.org/abs/2509.07379)
- [Pre-Attention Expert Prediction](https://arxiv.org/abs/2511.10676)
- [SpecMD](https://arxiv.org/abs/2602.03921)
- [Speculating Experts](https://arxiv.org/abs/2603.19289)
- [MoE-SpAc](https://arxiv.org/abs/2603.09983)

如果论文写成：

> 我们提出一个更好的 prefetch predictor / utility estimator。

大概率会被压成 incremental。

---

## 理论与解释性文献给出的方向收缩

下面几类工作对研究定位很重要：

- [Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models](https://arxiv.org/abs/2505.16056)
  - 说明 `offloadability` 是模型属性
  - 不是所有 MoE 都同样适合 offloading
  - local routing consistency 跨模型差异很大

- [The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily Domain Expertise](https://arxiv.org/abs/2604.09780)
  - 说明 routing similarity 更接近 hidden-state geometry
  - 不宜把故事写成 semantic expert specialization

- [Equifinality in Mixture of Experts: Routing Topology Does Not Determine Language Modeling Quality](https://arxiv.org/abs/2604.14419)
  - 说明 routing topology 不一定决定模型质量
  - 不宜把故事写成“更 clever 的 router topology”

- [Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression](https://arxiv.org/abs/2603.02217)
  - 虽然是 compression，但点出了更一般的 deployment-time mismatch
  - 对 `router-expert mismatch` 和 deployment co-design 有启发

这些文献共同支持一个更大的 umbrella：

> `deployment-oriented routing / runtime-oriented retrieval abstraction`

---

## 方向 A：Retrieval-Object Mismatch + Continuation Cache

这是当前最强、也最接近系统论文的主线。

### 核心 claim

- 现有很多方法已经利用 locality
- 但 retrieval object 往往仍是 `sequence/request-level`
- decode-time prefetch 真正需要的是 `layer-local continuation`
- 主要损失先出在 `candidate omission gap`，不是 ranking

### 为什么这条线没那么拥挤

- 大家都在做 prediction / prefetch / caching
- 但很少人把问题明确诊断成 `retrieval object mismatch`
- 更少人把它提升成 `continuation cache abstraction`

### 和现有工作的边界

这条线不能写成：

> 我们有一个更好的 local predictor。

更稳的写法是：

> 现有 sequence/request-level retrieval object 与 decode-time layer-local paging action 不匹配。  
> 我们用 omission-gap decomposition 证明主要损失来自 candidate omission，而不是 controller ranking。  
> continuation cache 是一个新的 runtime retrieval abstraction。

### 当前已有证据

现有代码和实验已经支持这条线：

- `v9 score-only object-vs-controller`
  - local continuation 在 `mixed / recurrence_heavy / stationary` 上都显著优于 sequence-level controller 变体
  - same-step `M32 pair recall` 约 `0.47–0.48`
  - sequence-level 变体只有约 `0.18–0.23`
  - omission gap 约从 `0.68–0.74` 降到 `0.49–0.50`

- `v10 runtime pressure sweep`
  - 当前正在验证 local continuation object 的 candidate 优势能否在真实 paging 压力下转成 runtime 收益

- `v10b strong-pressure boundary`
  - `ratio_030 + future_layers=4 + max_candidates=32` 暴露了 expert cache eviction progress bug
  - failure mode 是 no-victim / all-locked 状态下 runtime fatal，不是 CUDA OOM
  - 这类结果应作为 robustness boundary，不应混入正常性能图表

### 风险

- 很容易被说成 “localized MoE-Infinity”
- 所以 novelty 必须放在：
  - `omission-gap diagnosis`
  - `retrieval object mismatch`
  - `continuation cache abstraction`
- 不能放在工程名字，例如 `history_reuse_local_backbone`

### 现有代码可复用部分

可以直接复用：

- [research_positioning.md](research_positioning.md)
- [local_continuation_library.py](../moe_infinity/policies/local_continuation_library.py)
- [history_reuse.py](../moe_infinity/policies/history_reuse.py)
- [phasea.py](../moe_infinity/analysis/phasea.py)
- [phasea_v9_scoreonly_object_vs_controller_qwen.sh](../benchmarks/phasea_v9_scoreonly_object_vs_controller_qwen.sh)
- [phasea_v10_runtime_fixedlen_qwen_pressure_sweep.sh](../benchmarks/phasea_v10_runtime_fixedlen_qwen_pressure_sweep.sh)

### 还差哪些证据

必须补：

- `v10` 跑完并给出 fixed-length runtime 结论
- 更强 memory pressure 下的 stall / latency 证据
- 把正常 performance sweep 和 robustness boundary 分开报告
- 第二个模型
- 明确 failure cases

### 当前判断

这是第一优先级主线。

---

## 方向 B：Offloadability-Aware Router Calibration

这是最值得保留的第二主线，也是最像“下一篇”或“增强版”的方向。

### 核心 claim

- 不同模型、甚至同一模型不同部署条件下，offloadability 差异很大
- 原始 router 并不是为 offloading runtime 训练的
- 可以只做 `lightweight router-only calibration`
- 目标是提升 local routing consistency / continuation stability，从而改善 cacheability 与 prefetchability

### 为什么它可能更不拥挤

- `Local Routing Consistency` 已经说明 offloadability 是模型属性
- `Router Calibration` 已经说明 deployment-time mismatch 是真实问题
- 但“为 offloading 友好性做 router-only calibration”还不算拥挤

### 为什么它比 predictor race 更安全

- 它不是“再发明一个 predictor”
- 它是在做 `deployment co-design`
- 审稿视角更像 systems + model co-design

### 风险

- 容易被看成 training / model adaptation paper
- 必须克制成：
  - post-training
  - unlabeled calibration data
  - router-only small update
  - 明确目标是 offloading latency / cacheability

### 和当前代码的关系

这条线目前没有直接落地代码，但和当前 continuation-cache 方向兼容：

- continuation cache 先解决 retrieval object mismatch
- router calibration 再提升 routing stability / cacheability

### 当前判断

这是最值得保留的第二主线，但不应该抢当前主线。

---

## 方向 C：Geometry-Aware Expert Paging

这是理论文献给出的自然延伸，但不适合现在先做。

### 核心 claim

- 如果 routing similarity 本质上是 hidden-state geometry 决定的
- 那 expert paging 的 retrieval key 应该建在 `local hidden-state geometry`
- 而不是 sequence-level expert-id counts，甚至不是 prompt semantics

### 为什么它有潜力

- 很多系统工作还停留在 expert-id / gate output / historical route
- 用 hidden-state geometry 做 paging retrieval 还不算 crowded

### 和当前工作的关系

- continuation cache 是这条线的前一层
- 当前 key 还是 local routing pattern
- 再往前走一步，就是 geometry-aware continuation retrieval

### 风险

- 实现复杂
- 容易滑向 representation learning
- 容易超出 systems paper 的边界

### 当前判断

值得作为中长期分叉，但不适合现在先跳。

---

## 方向 D：Which MoEs Are Deployable/Offloadable, and Why?

这条更像 characterization paper，而不是主机制 paper。

### 核心 claim

- 并不是所有 MoE 都适合 offloading
- offloadability 跟 shared experts、routing consistency、depth-wise structure 等有关
- 可以先做 deployability diagnosis，再做 runtime policy selection

### 风险

- 和 `Local Routing Consistency` 很近
- 如果单独做，很容易撞车

### 何时值得做

如果你能把它和 runtime policy selection 绑定起来，例如：

- diagnosis predicts which paging strategy to use
- model-specific offloading strategy selection

那它会更有价值。

### 当前判断

适合做 characterization side branch，不适合抢当前主机制主线。

---

## 当前不建议主打的方向

### Confidence-Aware Paging

不建议现在作为主线。

原因：

- 这条线也不空
- 当前实验已经显示一级瓶颈先是 candidate omission
- retrieval object 还没讲透时，先上 confidence-aware scheduler 会冲淡 strongest story

### Semantic Expert Specialization

不建议现在主打。

原因：

- 语义 specialization 不是最稳的解释框架
- 几何 / local continuation 更稳

---

## 推荐优先级

### 第一优先级：当前主线

`retrieval-object mismatch -> continuation cache abstraction`

需要继续补：

- 跨模型
- 强 memory pressure
- 强 baseline
- fixed-length runtime

### 第二优先级：下一篇/增强版

`offloadability-aware router calibration`

### 远期升级版

`geometry-aware continuation retrieval`

---

## 建议的 paper framing

### 当前最稳的一句话 claim

> Existing MoE expert paging systems often retrieve history at the wrong granularity.  
> Request-level traces are poorly aligned with layer-local decode-time prefetch deadlines, causing oracle experts to be omitted before ranking begins.  
> We expose this hidden failure mode with omission-gap decomposition and introduce a continuation cache that retrieves local future expert continuations conditioned on recent layer-local routing prefixes.

### 中文版本

> 现有 MoE expert paging 系统经常在错误粒度上检索历史。  
> request-level trace 和 decode-time layer-local prefetch deadline 不对齐，导致正确 expert 在排序之前就被漏掉。  
> 我们用 omission-gap decomposition 暴露这个失败模式，并提出 continuation cache，用当前 step/layer 的 local routing prefix 检索未来几层的 expert continuation。

---

## 当前最终建议

如果目标还是推理加速，又想尽量避开最拥挤的 predictor race，那么优先级应该是：

1. 主线继续做 `retrieval-object mismatch + continuation cache`
2. 备选保留 `offloadability-aware router calibration`
3. 远期再考虑 `geometry-aware continuation retrieval`

一句话收住：

> 更有机会的方向，不是再做一个 predictor，而是把问题提升成 `deployment-oriented routing`。  
> 现在最值得打透的是 `continuation cache`；最值得保留的下一篇方向是 `router calibration`。
