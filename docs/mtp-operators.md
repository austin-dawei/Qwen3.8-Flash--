# MTP 一级与二级算子

MTP（Multi-Token Prediction）是 checkpoint 中独立保存的辅助预测路径。官方 README 描述为 1 层、使用 multi-steps 训练；配置中 `mtp_num_hidden_layers=1`、`hybrid=true`，其 layer type 为 full attention，在 Qwen4-Exp 配置加载时归一化为 Qwen Sparse Attention。

## 解释边界

固定 revision 的 checkpoint index 和 config 明确给出了 MTP tensor 与结构配置，但当前开源 Transformers `Qwen4ExpForConditionalGeneration` 将 `mtp.*` 列为加载时可忽略的 unexpected keys，没有实现完整 MTP forward。

因此本文区分：

- **可确认**：checkpoint tensor 名称、shape、独立参数量、1 个 QSA + Sparse MoE 层及共享设置。
- **结构推断**：`fc_embedding`、`fc_hidden`、pre-norm 与 final mixer 在完整 MTP 执行中的精确组合顺序。
- **不计算**：多步预测带来的接受率、吞吐提升或训练 loss。

Explorer 的 MTP 开关首先是容量归属开关，不代表当前 Transformers 推理路径一定会执行 MTP。

## 一级算子：MTP 0

Layer 地图把 MTP 显示为独立辅助层 `MTP 0`。其核心 layer 与主 QSA Decoder Layer 同构：

```text
MTP prepared hidden state
  ↓
Sequence-mixer Gated Residual
  ↓
Qwen Sparse Attention
  ↓
MoE Gated Residual
  ↓
Sparse MoE
  ↓
final Gated Residual mixer
  ↓
shared LM head / multi-token prediction
```

MTP 不包含 N-gram PLE。开启后会同时增加独立 Weight 和一层 QSA persistent cache。

## MTP 独有输入投影

checkpoint 保存两条 2560 维投影：

```text
mtp.fc_embedding.weight: [2560,2560]
mtp.fc_hidden.weight:    [2560,2560]
```

以及对应 pre-norm：

```text
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
```

这些名称表明 MTP 将 token embedding 分支和主模型 hidden-state 分支分别归一化、投影后构造辅助层输入。由于当前 Transformers 快照没有 MTP forward，Explorer 只对这些 tensor 做参数和容量归属，不宣称具体是相加、拼接还是其他融合顺序。

两个投影各有：

```text
2560 × 2560 = 6,553,600 parameters
```

## 二级算子：Sequence-mixer Gated Residual

MTP layer 的第一套 Gated Residual 与主层相同：

```text
4-stream input
  ↓ Read gate: 10240 → 320 → 10240
mixed input [B,S,2560]
  ↓ QSA
QSA output [B,S,2560]
  ↓ Write gate: 10240 → 4 scalars
4-stream output
```

Read/Write gate 的详细公式见 [Gated Residual 与 GDN 数据流](gated-residual-and-gdn.md)。MTP 组件量化会覆盖这些可量化矩阵，但 Norm 保持固定 BF16。

## 二级算子：MTP QSA

MTP 配置只有一个 attention layer，checkpoint tensor 包含：

- Q + output gate projection。
- 2-head grouped K 与 V projection。
- Q/K RMSNorm。
- 4Q + 1K、128 维的 Indexer fused QK projection。
- Indexer Q/K RMSNorm。
- `6144 → 2560` output projection。

其 micro-block 4:1、Top-512 blocks 和 2048-token budget 与主 QSA 相同。详细数据流见 [Qwen Sparse Attention 算子](qwen-sparse-attention-operators.md)。

因为 MTP QSA 可能在增量解码中维护独立 layer cache，Explorer 按一整层 QSA 计入主 K/V 和 Indexer raw-key cache。关闭 MTP 后，这部分 State / KV 也一并移除。

## 二级算子：MTP Sparse MoE

MTP layer 同样保存：

- Top-10 / 512 Router。
- 512 个 `2560 → 640 → 2560` routed SwiGLU experts。
- 1 个 shared SwiGLU expert。
- shared expert sigmoid output gate。
- MoE 前的第二套 Gated Residual。

其专家数量、Top-K、shape 与主 Decoder Layer 完全相同，详见 [Sparse MoE 算子](sparse-moe-operators.md)。

MTP 的 2.607B 独立参数中，2,516,582,400 个元素来自 512 个 routed experts，因此 MTP Weight 也主要由 MoE 决定。

## Final Gated Residual mixer

checkpoint 还保存一个没有 `block_inject_weight` 的 final mixer：

```text
RMSNorm width 10240
Linear 10240 → 320
SiLU
Linear 320 → 10240
Sigmoid read weights
4-stream weighted mean → [B,S,2560]
```

它与普通 Gated Residual 的 Read gate 相同，但只把四路汇聚成单路输出，不再产生四个 write gates。这与文本主模型在 48 层之后使用 final hyper-connection mixer 的方式一致。

## 参数归属与“额外 4B”

checkpoint 中 MTP 独立保存的 tensor 合计：

```text
2,607,150,848 elements
```

配置 `mtp_use_dedicated_embeddings=false`，所以 MTP 复用主模型 Token Embedding 与 LM Head，不重复保存：

```text
Token Embedding: 248320 × 2560 = 635,699,200
LM Head:         248320 × 2560 = 635,699,200
```

若按完整预测路径把这两个共享矩阵也归因给 MTP：

```text
2,607,150,848 + 635,699,200 + 635,699,200
= 3,878,549,248
≈ 4B
```

这解释了为什么官方描述“plus 4B MTP”，而 Explorer 的 MTP 开关只新增约 2.61B 存储元素：前者是功能路径归因，后者是 checkpoint 独立存储归因，避免重复计算共享权重。

## MTP 量化

MTP 下拉框的覆盖顺序为：

```text
MTP 独立量化选择
  ├─ inherit：跟随全局 Weight profile
  └─ BF16 / FP8 / W4.25 / W4：覆盖 MTP 可量化矩阵
```

Norm 等固定精度 tensor 不受覆盖影响。MTP 量化只改变 Weight bytes，不改变逻辑参数量或 QSA State / KV。

## 对应来源

- MTP 配置：`config.json -> text_config.mtp`
- checkpoint tensor：`model.safetensors.index.json -> mtp.*`
- 官方规模说明：README 的 Model Architecture 部分
- 主 QSA/MoE 实现用于同构映射：`modeling_qwen4_exp.py`
