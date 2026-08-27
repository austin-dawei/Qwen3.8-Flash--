# Qwen Sparse Attention 一级与二级算子

Qwen Sparse Attention（QSA）在保留完整上下文 K/V 的同时，用一个轻量 Indexer 为每个 query 选择最多 2048 个历史 tokens，再在选中集合上执行注意力。

48 个主层按 `GDN, GDN, GDN, QSA` 重复 12 次，因此有 12 个主 QSA 层；MTP 还包含 1 个 QSA 层。

## 一级算子：Qwen Sparse Attention

一级节点接收 sequence-mixer Gated Residual 产生的：

```text
GR mixed input [B,S,2560]
```

主要配置：

```text
Q heads             24
KV heads             2
head dim            256
RoPE dim per head    64
Indexer Q heads       4
Indexer K heads       1
Indexer head dim    128
micro-block size      4 tokens
token budget       2048 tokens
```

一级输出恢复为 `[B,S,2560]`，再由 Gated Residual Write gate 注入四路 residual。

## 二级数据流

```text
GR mixed input
 ├→ Q + output gate ───────────────────────────────────┐
 ├→ Grouped K/V ──────────────────────────────────────┤
 └→ Indexer QK → Micro-block mean → Top-512 blocks ──┤
                                                     ▼
                                            Sparse attention
                                                     │
                                                     ▼
                                            Output projection
```

Indexer 只决定主 attention mask，主 Q/K/V 仍由各自投影生成。

## Q + output gate

Q projection 同时生成 query 和逐元素输出门：

```text
Linear: 2560 → 24 × 256 × 2 = 12288
```

reshape 后沿每个 head 的最后一维分成：

```text
query: [B,S,24,256]
gate:  [B,S,24,256] → flatten [B,S,6144]
```

Query 经过每 head RMSNorm，并在前 64 维应用 RoPE。注意力输出展平后先乘：

```text
sigmoid(gate)
```

再进入 output projection。因此这里的 gate 控制每个 attention output channel 的通过强度，与四路 Gated Residual 的 Read/Write gate 不是同一个门。

Q + gate 权重 shape 为 `[12288,2560]`，包含 31,457,280 个元素。

## Grouped K/V

K 与 V 分别投影为 2 个 heads：

```text
K: 2560 → 2 × 256 = 512
V: 2560 → 2 × 256 = 512
```

K 经每 head RMSNorm，并与 Q 一样只在 64 维应用 RoPE。24 个 Q heads 共享 2 个 KV heads，分组比例为：

```text
24 / 2 = 12 query heads per KV head
```

增量解码时，完整上下文的 K/V 会进入 persistent cache。每 token 的主 K/V 元素数为：

```text
2 KV heads × 256 dims × (K + V) = 1024 elements
```

## Indexer QK

Indexer 使用独立的 fused QK projection：

```text
2560 → (4 Q heads + 1 K head) × 128 = 640
```

输出拆为：

```text
index Q: [B,S,4,128]
raw key: [B,S,128]
```

Q/K 分别 RMSNorm，Q 应用当前位置 RoPE。raw Indexer key 会按完整 Context 缓存；它不是永久保存的 4:1 pooled key。

Indexer 投影包含 `640 × 2560 = 1,638,400` 个权重元素。

## Micro-block mean

对当前 query 可见的历史 token indices，每连续 4 个 token 组成一个完整 micro-block：

```text
[4,128] raw keys → FP32 mean → [128] pooled block key
```

pooled key 经过 K RMSNorm，并使用 block 第一个 token 的位置应用 RoPE。不能组成完整 4-token block 的尾部 tokens 会直接保留，不参与 block Top-K。

Micro-block mean 没有模型权重。它降低 Indexer 打分的候选数量，但 pooled keys 是运行时计算结果，不是持久 cache 格式。

## Top-512 blocks

Indexer token budget 为 2048，compress ratio 为 4：

```text
block_topk = 2048 / 4 = 512 blocks
```

每个 pooled block key 与 4 个 Indexer query heads 计算分数：

```text
score_per_head = Qindex · Kblock
block_score = sum(ReLU(score_per_head)) / sqrt(128)
```

然后选择最多 512 个 blocks，并展开回最多 2048 个 token indices。可见历史不足时选择数量相应减少；最后不足 4 个 token 的 tail 额外保留。

使用 `ReLU` 后再跨 4 个 heads 求和，意味着正相关证据会增加 block 分数，负相关分数不会互相抵消正相关证据。

## Sparse attention

选中的 token indices 被转换成 mask，并与原始 causal/padding mask 合并。主 Q/K/V 只允许访问 Indexer 选中的历史位置：

```text
Attention(Q, selected K, selected V)
```

QSA 仍然是全局检索，因为被选 token 可以位于完整历史的任意位置；“稀疏”指每个 query 实际允许访问的 token 子集。

Indexer budget 降低 attention 计算候选量，但不会把持久 K/V cache 自动限制为 2048 tokens。下一个 query 可能选中不同历史位置，因此完整 K/V 与 raw Indexer keys 仍需保留。

## Output projection

24 个 heads 拼接得到：

```text
24 × 256 = 6144
```

先乘前述 `sigmoid(output_gate)`，然后：

```text
Linear: 6144 → 2560
```

输出 shape 为 `[B,S,2560]`。

## Cache 归属

每个 QSA 层保存：

- 主 K/V：完整 Context，2 KV heads × 256 dims × K/V。
- Indexer raw keys：完整 Context，1 head × 128 dims。
- 不持久保存 pooled micro-block keys、Top-K 结果或 attention output。

在 BF16、Batch 1、256K Context 下，单个 QSA 层约占 576 MiB cache。MTP 开启时还会增加一层同规模 QSA cache。详细公式见 [Hybrid State 与 QSA Cache](hybrid-state-and-qsa.md) 和 [计算方法](calculation-methodology.md)。

## 对应源码

- Indexer 投影、pooling 与 Top-K：`modeling_qwen4_exp.py:611-717`
- 主 Q/K/V、output gate 与 attention：`modeling_qwen4_exp.py:757-839`
