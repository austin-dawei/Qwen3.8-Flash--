# 文档总览

这套文档解释 Qwen3.8-Flash-Next Explorer 的数据来源、架构映射与容量公式。页面的目标是回答三个问题：checkpoint 中有哪些权重、长上下文推理需要多少持久状态、不同量化与可选组件如何改变容量。

> 默认场景不是性能基准。它是一个可复算的静态容量模型：Weight 加增量解码 State / KV，不包含 activation、kernel workspace、通信和碎片。

## 推荐阅读顺序

1. [快速开始](quick-start.md)：启动服务、操作页面、保存和复算场景。
2. [架构映射](model-architecture-mapping.md)：理解 36 层 GDN、12 层 QSA、MoE、N-gram、Vision 与 MTP。
3. [Gated Residual 与 GDN](gated-residual-and-gdn.md)：理解四路 residual、Read/Write gate、低秩 320 维控制空间与 SiLU。
4. [一级与二级算子索引](operator-reference.md)：按 Web 图层级查找各模块及展开节点。
5. [Sparse MoE](sparse-moe-operators.md)、[N-gram PLE](ngram-ple-operators.md)、[Qwen Sparse Attention](qwen-sparse-attention-operators.md) 与 [MTP](mtp-operators.md)：深入理解四个专题模块。
6. [Hybrid State 与 QSA Cache](hybrid-state-and-qsa.md)：理解为什么长上下文容量主要来自少数 QSA 层。
7. [计算方法](calculation-methodology.md)：逐项查看 Weight、KV 与 recurrent state 公式。
8. [量化指南](quantization-guide.md)：区分全局 Weight、组件覆盖和 State / KV 量化。
9. [数据格式](data-format.md)：开发或批量生成场景时参考 JSON 字段。
10. [数据来源与验证](validation-and-provenance.md)：核对固定 revision、SHA256 与复现命令。
11. [假设与限制](assumptions-and-limitations.md)：确认结果适用边界。

## 默认基线

默认场景启用 N-gram、Vision 和 MTP，Weight 使用官方 BF16 checkpoint，State / KV 使用 BF16，Context 为 262,144，Batch 为 1。

- checkpoint 存储元素：`179,999,981,459`
- checkpoint Weight：`359,999,963,128 bytes`
- State / KV：`7,968,116,752 bytes`
- Weight + State / KV：`367,968,079,880 bytes`

这里的“存储元素”包含 checkpoint 内独立保存的 MTP tensor，并按实际 tensor shape 汇总；它不等同于论文中常用的激活参数量或去重后的模型规模。

## 页面与代码的对应关系

- Explorer：`index.html` 与 `app.js`，负责交互式容量分析。
- Document：`documents.html` 与 `documents.js`，直接读取 `docs/*.md`。
- 语义规格：`data/model/qwen3.8-flash-next.model-spec.json`。
- 量化配置：`data/quantization/`。
- 离线复算：`scripts/analysis_math.py` 与 `scripts/calculate_analysis.py`。
- 数据校验：`scripts/validate_data.py`。

## 结果应如何解读

优先比较同一 Context、Batch 与组件集合下的场景。Weight 变化反映常驻模型数据变化；State / KV 变化反映增量解码期间的持久状态变化。二者都不能直接换算为吞吐或首 token 延迟。
