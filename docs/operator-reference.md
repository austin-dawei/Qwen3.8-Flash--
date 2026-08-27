# 一级与二级算子索引

Explorer 使用两层算子视图。一级算子描述一个 Decoder Layer 的模块级数据流；双击可展开模块时，二级算子展示模块内部的投影、门控、检索、专家或状态更新。

## 一级算子

主 Decoder Layer 的一级数据流为：

```text
4-stream input
  ↓
N-gram PLE                    仅 Layer 1，可选
  ↓
Gated Residual · Read/Write   sequence mixer wrapper
  ↓
Gated DeltaNet 或 Qwen Sparse Attention
  ↓
Gated Residual · Read/Write   MoE wrapper
  ↓
Sparse MoE
  ↓
4-stream output
```

- `4-stream input/output`：四条 widened residual streams，语义 shape 为 `[B,S,4,2560]`，源码中通常展平为 `[B,S,10240]`。
- `N-gram PLE`：将 bigram/trigram 哈希记忆注入四路 residual，仅配置在网页 Layer 1。
- `Gated Residual`：动态读取四路为单路 `[B,S,2560]`，并把模块输出用四个写入门注回四路。
- `Gated DeltaNet`：36 个主层使用的固定 recurrent-state 线性注意力。
- `Qwen Sparse Attention`：12 个主层与 1 个 MTP 层使用的 Indexer 引导稀疏全局注意力。
- `Sparse MoE`：512 个 routed experts 中每 token 激活 10 个，同时执行 1 个 shared expert。
- `MTP`：独立辅助路径，包含 1 个 QSA + MoE hybrid layer 和输入/输出投影。

## 二级算子

双击一级模块后，Web 端当前展示以下二级节点：

### Gated Residual

```text
4-stream input → Read gate → wrapped module → Write gate → 4-stream output
```

详细公式见 [Gated Residual 与 GDN 数据流](gated-residual-and-gdn.md)。

### Gated DeltaNet

```text
Fused QKV → Causal DWConv → Delta-rule state
          → β / decay / z gates → Output projection
```

### Qwen Sparse Attention

```text
Q + output gate ───────────────────────────────┐
Grouped K/V ──────────────────────────────────┤
Indexer QK → Micro-block mean → Top-512 blocks├→ Sparse attention → Output projection
```

详细说明见 [Qwen Sparse Attention 算子](qwen-sparse-attention-operators.md)。

### Sparse MoE

```text
                ┌→ Top-10 router → 10 / 512 routed experts ─┐
GR mixed input ─┤                                            ├→ Weighted combine
                └→ Shared expert + sigmoid output gate ─────┘
```

详细说明见 [Sparse MoE 算子](sparse-moe-operators.md)。

### N-gram PLE

```text
Token history → Hashed lookup → Stream gating
              → Dilated DWConv → 4-stream injection
```

详细说明见 [N-gram PLE 算子](ngram-ple-operators.md)。

### MTP

MTP 在 Layer 地图中作为辅助一级层显示，其 QSA、Gated Residual 和 Sparse MoE 可继续按上述二级图展开。MTP 独有的 embedding/hidden 投影和 final mixer 当前计入 Inspector 与容量，但没有单独画成二级流程图。详见 [MTP 算子与参数归属](mtp-operators.md)。

## Shape 与指标约定

- `[B,S,D]`：Batch、Sequence、hidden dimension。
- `Logical`：该节点归属 parameter groups 的 tensor 元素数，不表示每 token 实际激活量。
- `Weight`：当前全局或组件覆盖量化下的存储容量。
- `Layer`：节点 Weight 占当前层 Weight 的比例。
- `KV Load`：该节点关联的增量解码持久 State / KV，不是单步真实内存带宽。

二级图用于解释语义连接，不是 kernel trace。无权重的 reshape、mask、Top-K、pooling 与 combine 节点会显示 `Logical=0`，但运行时仍然有计算和临时 activation。
