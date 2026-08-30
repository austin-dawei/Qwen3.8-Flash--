# N-gram PLE 一级与二级算子

N-gram PLE（Per-Layer Embedding）把局部 token 组合映射到一个大型哈希 Embedding，再根据四路 residual 当前内容生成注入门。它为模型提供显式的 bigram/trigram 模式记忆。

可以把它的核心计算概括为：根据当前位置附近的短 token 序列生成少量地址，从 51.2B 参数的大表中读取少量向量，再将结果注入主干隐藏状态。51.2B 描述的是表的总容量，并不表示每个 token 都要与全部参数做矩阵乘法。

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

例如输入为：

```text
x₁ = 我
x₂ = 喜欢
x₃ = 北京
x₄ = 大学
```

在位置 `t=4`，本实现使用的局部组合是：

```text
bigram: [北京, 大学]
trigram: [喜欢, 北京, 大学]
```

一般地，它们都结束于当前位置 `t`：

\[
g_t^{(n)}=(x_{t-n+1},\ldots,x_t),\qquad n\in\{2,3\}
\]

这是当前 checkpoint / 源码的具体配置。把 1-gram、4-gram 或更多阶数也纳入查表，是通用 N-gram embedding 的可能设计，不是这里已经验证的实现。

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

抽象地，也可以写成：

\[
a_{t,h}^{(n)}=\operatorname{Hash}_h(g_t^{(n)})\bmod V_{n,h}+o_{n,h}
\]

其中 `h` 是 hash head，`V` 是该 head 的槽位数，`o` 是它在合并表中的偏移。随后只读取对应行：

\[
e_{t,h}^{(n)}=E[a_{t,h}^{(n)}]
\]

完整枚举 token 组合并不可行。若词表有 25 万个 token，仅所有二元组合就有 `250000² = 625 亿` 种；确定性哈希把组合压入有限槽位，并保证同一组合总能访问同一地址。

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

尽管整张表包含约 51.2B 个元素，一个 token 在当前配置下只产生 16 次 lookup、读取 16 个 160 维向量。因而总容量与单 token 计算量是分离的：前者主要形成存储压力，后者更接近少量地址计算、embedding gather 与后续投影，而不是一个 51.2B 参数全连接网络的乘法。

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

因此，通用说明中常写作的

\[
m_t=\operatorname{Aggregate}(e_t^{(2)},e_t^{(3)}),\qquad
\widetilde h_t=h_t+\operatorname{Project}(m_t)
\]

在本实现中并非简单求和：16 路 lookup 先拼接，经 key/value 投影，再由四路 residual 内容生成门控，最终形成与 `[B,S,4,2560]` 隐藏状态同 shape 的注入量。

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

## Host Memory 预取与计算重叠

51.2B 个元素即使用 1 byte 表示也约为 51.2 GB；BF16 则约为 102.4 GB。因此工程上可以把完整表放在 CPU / Host Memory，只把当前 batch 实际命中的行传到 GPU。一次前向过程可概括为：

```text
CPU / Host：构造 bigram、trigram → 计算 16 路地址 → gather / 合并传输
GPU：        执行前一层主干计算 ────────────────────────┐
                                                        ▼
                                              PLE 门控、卷积与注入
```

配置把 PLE 放在 one-indexed 第 2 层，使前一层计算可用于隐藏查表与传输延迟。这也是“注入第 2 层”和“网页 Layer 1”两种说法看似不同的原因：前者按 one-indexed 层号，后者按 zero-indexed 页面编号。

实际额外成本主要来自哈希、稀疏 gather、Host-to-Device 传输、投影、门控与 depthwise convolution。能否完全隐藏传输延迟取决于 batch、命中行去重、内存带宽、互连和运行时实现，不能只由参数量判断。

## 训练时的稀疏更新

若某个 N-gram 命中表的第 `j` 行，反向传播只会为实际访问的行产生 embedding 梯度：

\[
e=E[j],\qquad
E[j]\leftarrow E[j]-\eta\frac{\partial L}{\partial E[j]}
\]

本 batch 未访问的行没有来自该次 lookup 的梯度，所以大表可按稀疏访问方式更新。不同短语也可能哈希到同一槽位并共享参数，即哈希碰撞；当前实现以 16 个 heads、不同质数表范围和不同 multiplier 分散碰撞模式，但不会彻底消除碰撞。

从计算视角看，完整链路可以压缩为：

```text
局部 token 序列
→ 16 路确定性地址
→ 从 51.2B 元素的大表读取 16×160 个元素
→ 拼接、投影与四路 stream gating
→ dilated depthwise convolution
→ 残差注入
```

它以较大的存储容量和内存带宽需求换取参数化局部模式记忆，同时避免同等参数规模的 dense 矩阵乘法。

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
