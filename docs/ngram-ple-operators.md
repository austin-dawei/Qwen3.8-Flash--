# N-gram PLE 一级与二级算子

N-gram PLE（Per-Layer Embedding）把局部 token 组合映射到一个大型哈希 Embedding，再根据四路 residual 当前内容生成注入门。它为模型提供显式的 bigram/trigram 模式记忆。

配置 `ple_layer_ids=[2]` 使用 one-indexed 编号，因此只出现在 Web 的 Layer 1，而不是 Layer 2。

## 一级算子：N-gram PLE

一级节点位于该层 sequence-mixer Gated Residual 之前：

```text
Token IDs + previous 2 token IDs
            │
            ▼
N-gram PLE
            │ output [B,S,4,2560]
            ▼
hidden_states = hidden_states + PLE output
            │
            ▼
Attention Read gate → GDN
```

PLE 不替换四路 residual，而是先产生同 shape 的注入量并做加法。其一级标签 `bigram + trigram → 4×2560` 概括了从 token history 到四路输出的过程。

## 二级数据流

```text
Token history
    ↓
Hashed lookup
    ↓
Stream gating
    ↓
Dilated DWConv
    ↓
4-stream injection [B,S,4,2560]
```

## Token history

`ngram_size=3`，所以每个位置需要当前 token 与前两个 token。增量解码时，PLE 使用额外 cache state 保存 2 个历史 token IDs。

历史构造不会跨 EOS 边界：如果向右移后落到当前 segment 之前，源码用 EOS token 填充，避免把不同对话段或样本的 n-gram 串联起来。

## Hashed lookup

PLE 同时构造 bigram 和 trigram，每种 n-gram 有 8 个 hash heads，总共：

```text
(ngram_size - 1) × heads_per_ngram
= (3 - 1) × 8
= 16 heads
```

总 embedding dim 为 2560，所以每个 head：

```text
2560 / 16 = 160 dims
```

每个 n-gram head 的 ID 大致按以下过程生成：

```text
mixed_id = token₀ × multiplier₀
mixed_id = mixed_id XOR token₁ × multiplier₁
mixed_id = mixed_id XOR token₂ × multiplier₂   # trigram 才有
head_id  = mixed_id mod head_vocab_size + head_offset
```

multiplier 由固定 seed 和 SplitMix64 生成奇数；每个 head 使用从约 20M 开始的不同质数 vocab size，以减少多个 head 产生完全相同碰撞模式的概率。

16 个 lookup 结果各为 160 维，拼接得到：

```text
ngram_embedding: [B,S,16,160]
flatten:         [B,S,2560]
```

checkpoint 中合并后的表为：

```text
[320,001,536,160]
= 51,200,245,760 elements
```

它是模型中最大的单类参数，官方 checkpoint 拆为 128 份。源码显式允许 Embedding 位于不同 device，因此设计上可以放到 Host Memory 后再把 lookup 结果传回计算设备。

哈希会产生碰撞，但 16 个 heads 使用不同表范围与 multiplier；模型学习的是碰撞条件下仍有用的分布式局部模式表示。

## Stream gating

lookup 输出 `E:[B,S,2560]` 进入两条投影：

```text
key_proj:   2560 → 10240 → reshape [B,S,4,2560]
value_proj: 2560 → 2560
```

原有四路 residual `H:[B,S,4,2560]` 经 grouped RMSNorm 得到 query。每条 stream 的门值为 key/query 的点积：

```text
scoreᵢ = sum(keyᵢ × queryᵢ) / sqrt(2560)
scoreᵢ = sign(scoreᵢ) × sqrt(abs(scoreᵢ))
gateᵢ  = sigmoid(scoreᵢ)
```

然后把同一个 2560 维 N-gram value 按四个标量分别注入：

```text
gated_valueᵢ = gateᵢ × value
```

这里的 gate 是每条 residual stream 一个标量；它根据当前 stream 内容判断该 token 的 N-gram 记忆应该注入多少。

带符号平方根压缩大点积分数的幅度，同时保留正负号；`clamp_min(1e-6)` 避免零附近数值问题。

## Dilated DWConv

四路 gated values 展平为 `[B,S,10240]`，归一化后进入 depthwise causal convolution：

```text
channels = 10240
kernel_size = 4
dilation = ngram_size = 3
groups = 10240
```

每个 channel 独立卷积，感受野对应当前及间隔 3 的历史位置，最大历史跨度：

```text
(kernel_size - 1) × dilation = 3 × 3 = 9 tokens
```

卷积输出经过 SiLU，最终：

```text
PLE output = gated_value + SiLU(dilated_depthwise_conv(norm(gated_value)))
```

增量解码需要保存 9-token convolution state，另保存 2 个 token IDs；这些状态远小于 51.2B Embedding 权重。

## 参数与量化边界

- N-gram table：51,200,245,760 个可量化 embedding 元素。
- Key projection：`[10240,2560]`。
- Value projection：`[2560,2560]`。
- Dilated DWConv：`[10240,1,4]`，当前规格固定 BF16。
- 三个 RMSNorm：固定 BF16。
- hash multiplier、head sizes 与 offsets：共 35 个 INT64 元素，组件量化不会改变它们。

关闭 N-gram 开关表示不把 PLE 权重和状态计入当前场景，不代表模型可以在语义上无损删除这一模块。

## 对应源码

- 哈希 Embedding：`modeling_qwen4_exp.py:1018-1114`
- PLE 投影与卷积定义：`modeling_qwen4_exp.py:1117-1148`
- Stream gate 与输出：`modeling_qwen4_exp.py:1169-1189`
