# 量化配置指南

Explorer 将量化分为 Weight 与 State / KV 两条互不混用的路径。Weight 决定模型参数容量；State / KV 决定增量解码的持久状态容量。

## 全局 Weight 配置

- `bf16-checkpoint`：严格复现官方 checkpoint；大多数权重 16 BPW，N-gram 哈希元数据保持 INT64。
- `hybrid-w4-fp8`：路由专家按 4.25 BPW，其余主要 Linear 按 FP8 block-128；Embedding、N-gram 与 LM Head 保持 BF16。
- `fp8`：可量化大矩阵按 FP8 block-128 加 scale 估算，小型固定精度 tensor 保持原格式。
- `w4-effective-425`：可量化矩阵统一按有效 4.25 BPW，包含抽象化的量化元数据预算。
- `w4-raw`：只计算 4-bit 权重本体，是不含 scale、zero-point 与 packing 对齐的理论下限。

## 组件独立覆盖

N-gram Embedding、Vision Tower 与 MTP 各有独立下拉框：

- `inherit`：采用全局 Weight 配置对该参数组的策略。
- `bf16`、`fp8`、`w4-effective-425`、`w4-raw`：覆盖该组件中的可量化矩阵。

覆盖不会改变 `fixed_bf16`、`fixed_fp32`、`fixed_int32` 或 `fixed_int64` tensor。尤其是 N-gram 的哈希元数据仍按 INT64 计算，Vision/MTP 的 Norm 与 Bias 仍保持规格中声明的固定精度。

组件开关关闭时，下拉框会禁用，而且该组件既不贡献逻辑参数量，也不贡献 Weight；重新打开后会保留原选择。

## State / KV 配置

- `bf16`：QSA K/V、RoPE K、Indexer key 与 GDN conv state 使用 BF16；GDN recurrent state 固定 FP32。
- `fp8-mixed-rope`：非 RoPE K/V 与 Indexer key 按 FP8，RoPE K 保持 BF16。
- `int8-mixed-rope`：容量上与当前 FP8 配置相同，但表达部署数据类型语义不同。
- `int4-mixed-rope`：非 RoPE K/V 与 Indexer key 按 INT4，RoPE K 保持 BF16，GDN conv state 按 8 BPW。

State / KV 的 INT4 配置未计 scale、zero-point、页表或块对齐，因此也是容量估算，而非运行时显存承诺。

## 选择建议

若要核对官方文件，选择 BF16 checkpoint 并让三个组件跟随全局。若要估算单机或多卡部署，先确定哪些组件常驻 GPU、CPU 或被关闭，再分别设置量化。比较方案时保持 Context 和 Batch 一致，并为运行时额外预留 activation、workspace 与 allocator 空间。

## 不应从容量推导的结论

量化后的 bytes 不能单独证明某种 kernel 可用，也不能推导精度、吞吐或延迟。硬件支持、量化 group size、反量化融合、专家并行、N-gram Host Memory 带宽和 Vision 输入形状都会改变真实部署结果。
