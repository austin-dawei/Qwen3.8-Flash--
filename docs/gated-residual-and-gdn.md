# Gated Residual 与 GDN 数据流

本文解释 Explorer 中 GDN 层的展开图。图展示的是一个 Decoder Block 内，四路 Gated Residual 如何读取 residual streams、调用 GDN，再把结果分别写回四路；它不是普通的单路 residual add。

## 整体数据流

```text
Token Embedding [B,S,2560]
          │ repeat 4 次
          ▼
4-stream input [B,S,4,2560]
          │
          ▼ Read gate
mixed input [B,S,2560]
          │
          ▼ Gated DeltaNet
block output [B,S,2560]
          │
          ▼ Write gate + residual bypass
4-stream output [B,S,4,2560]
```

可以用下面的简化公式表示：

```text
M  = ReadGate(H₁,H₂,H₃,H₄)
Y  = GDN(M)
Hᵢ' = Hᵢ + WriteGateᵢ(H) × Y
```

其中 `B` 是 Batch，`S` 是序列长度，`4` 是 residual stream 数，`2560` 是每条 stream 的 hidden size。

## 为什么第一层输入是四路

模型配置为：

```text
hidden_size = 2560
hc_count = 4
hc_hidden_size = 4 × 2560 = 10240
```

第一层前，源码把 Token Embedding 沿最后一维重复四次：

```python
hidden_states = inputs_embeds
hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)
```

因此源码实际保存的 shape 是 `[B,S,10240]`。Explorer 将最后一维按语义写成 `[B,S,4,2560]`，强调它代表四条 residual stream，而不是 4 个 attention heads。

第一层开始时四路内容相同。经过每个模块不同的 write gate 注入后，四路状态逐渐分化。

## Read gate 做什么

Read gate 从四路 residual 中动态读取一个 2560 维表示，作为 GDN 的直接输入：

```text
[B,S,10240]
      │ Grouped RMSNorm
      │ Linear 10240 → 320
      │ 除以 hc_count=4
      │ SiLU
      │ Linear 320 → 10240
      │ Sigmoid
      ▼
[B,S,4,2560] element-wise read weights
```

设展平的四路输入为 `X`，计算过程为：

```text
Xn = RMSNorm(X)
U  = SiLU(Wdown Xn / 4)
G  = sigmoid(Wup U)
G  = reshape(G, [B,S,4,2560])
M  = mean(G × reshape(Xn), dim=stream)
```

最终 `M` 的 shape 为 `[B,S,2560]`。因此图中的 `[B,S,4,2560]` 是 Decoder Block 输入，GDN 本身的直接输入仍是 `[B,S,2560]`。

### Grouped RMSNorm

RMSNorm 的总宽度是 10,240，`group_size=2560`，因此四条 stream 分组归一化。归一化后的原始四路仍完整保留，供 element-wise read gate 加权。

### 为什么压缩到 320

320 是配置项 `hc_lowrank`，用于生成门控控制信号。这里压缩的不是 GDN 的主干信息；原始 10,240 维输入仍在另一条路径上参与加权与 residual write-back。

如果直接使用 `10240 → 10240`：

```text
参数量 = 10240² = 104,857,600
```

使用低秩瓶颈：

```text
Wdown: 10240 × 320 = 3,276,800
Wup:     320 × 10240 = 3,276,800
合计                     6,553,600
```

门控投影参数量缩小 16 倍。每个 Decoder Layer 分别在 sequence mixer 和 MoE 前使用一套 Gated Residual，因此低秩设计显著降低总参数量与推理开销。

`Wdown` 的每个输出都可以读取全部 10,240 个输入位置，所以 320 维不是简单截断，而是对四路、所有 hidden channels 的学习型汇总。它可以理解为“当前 token 应如何读取四路 residual”的控制空间。

### SiLU 的作用

SiLU 定义为：

```text
SiLU(z) = z × sigmoid(z)
```

如果没有 SiLU：

```text
Wup(Wdown(X)) = Weffective X
```

两层 Linear 可以合并成一个固定的 rank-320 线性映射。加入 SiLU 后，不同 token 和上下文可以激活不同的低秩控制特征，Read gate 因而成为输入相关的非线性映射。

先执行 `Wdown` 再 SiLU，意味着模型先跨 stream、跨 hidden channel 组合出 320 个控制特征，再平滑地决定哪些特征需要激活。SiLU 不是最终的门；后面的 Sigmoid 才把 10,240 个输出限制为 `(0,1)` 的读取权重。

除以 `hc_count=4` 用来控制进入 SiLU 的数值尺度，避免激活幅度随 residual stream 数增加。

### 320 如何变成 4 × 2560

`Wup` 是一个无 Bias 的全连接层：

```text
Wup shape = [10240,320]
[B,S,320] → [B,S,10240]
```

对每个 stream `i` 和 hidden channel `d`：

```text
logit[i,d] = Σ Wup[i × 2560 + d,k] × U[k]
gate[i,d]  = sigmoid(logit[i,d])
```

每个输出位置都有自己的一组 320 维投影参数。它不是把 320 维简单复制四份，而是从共同的 320 维控制状态生成 10,240 个不同的 gate logits。

随后 `unflatten(-1, (4,2560))` 只改变张量视图：

```text
[B,S,10240] → [B,S,4,2560]
```

这个 reshape 没有可学习参数，也不会产生额外计算。

## GDN 节点

Read gate 输出 `[B,S,2560]` 后才进入 Gated DeltaNet。GDN 执行线性注意力、短 causal convolution 和固定 recurrent state，输出 shape 仍为 `[B,S,2560]`。

Explorer 使用同一张展开图表示两种 sequence mixer：GDN 层显示 GDN，QSA 层显示 QSA。Qwen3.8-Flash-Next 的主层按 `GDN, GDN, GDN, QSA` 循环，因此 Layer 0 的这个节点实际是 GDN。

## Write gate 做什么

Write gate 从原始展平输入生成每个 token 的四个标量：

```text
[B,S,10240] → Linear → [B,S,4]
injection_weights = 2 × sigmoid(linear(X) / 4)
```

四个写入系数的范围约为 `(0,2)`。GDN 输出被扩展到四路并分别乘以对应系数：

```text
injection[b,s,i,d] = gate[b,s,i] × Y[b,s,d]
H'[b,s,i,d] = H[b,s,i,d] + injection[b,s,i,d]
```

图中的虚线表示原始 `4-stream input` 绕过 GDN，被保留到 Write gate 处执行 residual add。输出仍为 `[B,S,4,2560]`，继续送入 MoE 的 Gated Residual 或下一层。

Read gate 是每个 hidden channel 一个权重，而 Write gate 是每条 stream 一个标量。这使读取过程足够细粒度，同时让写回保持低开销。

## 图中参数指标

Read gate 严格拆分为：

```text
Grouped RMSNorm                   10,240
Linear 10240 → 320             3,276,800
Linear 320 → 10240             3,276,800
合计                           6,563,840
BF16 Weight                   12.52 MiB
```

Write gate 为：

```text
Linear 10240 → 4                 40,960
BF16 Weight                    80.0 KiB
```

`Logical` 表示 tensor 元素数，`Weight` 表示当前量化配置下的存储容量，`Layer` 表示该节点 Weight 占当前层 Weight 的比例。

当前 Explorer 的 Read gate 节点使用 `gr.attn.*` 前缀归集参数，因此它显示的 `6.60M / 12.6 MiB` 还包含了 40,960 个 Write gate 参数；Write gate 节点又单独显示一次。这个重叠只影响节点标签，不影响 Layer 或模型总 Weight 的汇总。

## 对应源码

- 四路初始化：`modeling_qwen4_exp.py:1415-1417`
- Gated Residual 参数定义：`modeling_qwen4_exp.py:941-950`
- Read gate、SiLU 与 reshape：`modeling_qwen4_exp.py:952-969`
- GDN 调用与 Write gate：`modeling_qwen4_exp.py:1222-1237`
