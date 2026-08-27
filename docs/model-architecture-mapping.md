# Qwen3.8-Flash-Next 架构映射

映射基于 Hugging Face 模型 revision `f5d08274bafd880402bd16f5e3e6c514136ec06c`，以及 Transformers revision `dabae5fcb924a8eece0e727b627ca5f050b40d40` 的 Qwen4-Exp 实现。

模型类为 `Qwen4ExpForConditionalGeneration`，文本 hidden size 2560，词表 248,320，原生最大 Context 262,144，并配置 1M YaRN 扩展。

## 顶层结构

```text
Token IDs / image-video patches
  ├─ Token Embedding ───────────────────────────────┐
  └─ 27-layer Vision Tower → merger (2560 dims) ───┤
                                                    ↓
                     expand to 4 residual streams
                                ↓
  12 × [3 × (Gated DeltaNet → MoE) + 1 × (QSA → MoE)]
                                ↓
                     final Gated Residual mixer
                                ↓
                             LM Head

Optional training/inference helper: 1 × QSA + MoE MTP block
```

## Gated Residual

更完整的逐节点 shape、Read gate 低秩投影、SiLU 与 Write gate 公式见 [Gated Residual 与 GDN 数据流](gated-residual-and-gdn.md)。

每个 decoder block 有两个 `Qwen4ExpTextGatedResidual`，分别包裹 sequence mixer 与 MoE。输入是 `4 × 2560 = 10240` 维：

- `input_mix_weight_down`: `10240 → 320`
- `input_mix_weight_up`: `320 → 10240`
- element-wise sigmoid read gate 混合四个分支
- `block_inject_weight`: `10240 → 4`，为四个残差分支生成写入强度

因此它不是传统单路 residual add，也不同于 DeepSeek V4 Flash 的 Sinkhorn Hyper-Connection。

## Gated DeltaNet

48 层中的 36 层使用 GDN：

- 16 个 Q/K heads、48 个 V heads，head dimension 均为 128
- fused QKV projection 输出 10240 维
- kernel size 4 的 depthwise causal convolution
- 每层 recurrent state 为 `48 × 128 × 128`，官方 `mamba_ssm_dtype=float32`
- recurrent state 不随 Context 增长，是长上下文低容量路径

## Qwen Sparse Attention

Indexer、micro-block、Top-K、output gate 与 cache 的逐节点说明见 [Qwen Sparse Attention 一级与二级算子](qwen-sparse-attention-operators.md)。

每第 4 层使用 QSA，共 12 层；MTP 也使用 1 层 QSA：

- 24 个 Q heads、2 个 KV heads、head dimension 256
- Q projection 同时生成逐元素 output gate
- RoPE 只覆盖每 head 64 维
- Indexer 为 4 个 query heads + 1 个 shared key head，head dimension 128
- 每 4 tokens 形成一个 micro-block，选取最多 512 blocks，即 2048 tokens 的稀疏注意力预算

注意：2048 是每次 attention 的选择预算，不是 KV cache 长度。开源实现仍缓存完整上下文的 2-head K/V 与 Indexer raw key。

## Sparse MoE

Router、routed/shared expert 与 combine 的逐节点说明见 [Sparse MoE 一级与二级算子](sparse-moe-operators.md)。

每层都有相同 MoE：512 个 routed experts、每 token Top-10、1 个 shared expert。单专家 SwiGLU 中间维度为 640。路由专家权重是模型常驻容量的主体，但每 token 只激活其中 10 个。

## N-gram PLE

哈希 lookup、stream gating 与 dilated convolution 的逐节点说明见 [N-gram PLE 一级与二级算子](ngram-ple-operators.md)。

配置 `ple_layer_ids=[2]` 使用 one-indexed 编号，因此 PLE 位于网页的 Layer 1。它将 bigram / trigram 哈希到 16 个 embedding heads，每 head 160 维。checkpoint 将概念上的 N-gram table 拆成 128 份，共 `51,200,245,760` 个 BF16 权重元素。设计允许将表放在 Host Memory 并异步预取。

## Vision 与 MTP

Vision Tower 为 27 层、hidden 1152、16 heads、MLP 4304，最终通过 patch merger 投影到文本 hidden 2560。MTP checkpoint 独立新增 `2,607,150,848` 个元素；它复用 token embedding 与 LM head，若按其完整执行路径归因，接近官方所称的额外 4B。

MTP 独有投影、辅助 QSA/MoE 层、final mixer 与共享权重归属见 [MTP 一级与二级算子](mtp-operators.md)。

## 参数归属

规格将 tensor 分为全局参数、48 个主 decoder 层与 1 个辅助 MTP 层。全局参数包括 token embedding、LM head、final mixer、Vision 和顶层 Norm；N-gram 参数归到启用 PLE 的 Layer 1；MTP 只计算 checkpoint 中独立保存的 tensor，不重复 token embedding 与 LM head。

这种归属方式用于回答“文件中实际存了多少”，不等同于训练论文中的共享参数记账方式。页面顶部的存储元素总数以 safetensors index 为最终校验基准。

## 页面交互映射

Layer 地图中的 GDN/QSA 标签来自每层 `attention_type`。算子图一级节点表示 PLE、两个 Gated Residual、sequence mixer 与 MoE；双击模块后展示其内部投影、门控与数据依赖。Inspector 中的参数组来自 model spec，而非浏览器运行时推断。
