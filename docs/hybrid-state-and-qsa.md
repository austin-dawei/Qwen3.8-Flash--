# Hybrid State 与 QSA Cache

48 个主层严格按 `GDN, GDN, GDN, QSA` 重复 12 次。GDN 每层只保留固定 recurrent / convolution state；QSA 保存随 Context 线性增长的全局 K/V 与 Indexer raw key。

在 BF16、Batch 1 下：

- 单个 GDN 层约 3.08 MiB state，与 Context 无关
- 单个 QSA 层在 256K context 下约 576 MiB cache
- 开启 MTP 会额外增加一个 QSA 层的 cache

因此长上下文容量几乎由 12 个主 QSA 层与可选 MTP QSA 层决定。Indexer 的 4:1 micro-block 与 2048-token budget 主要降低每次 attention 的计算，而不是把持久 cache 直接缩小到 1/4。

## 每层持久内容

GDN 层保存 FP32 recurrent matrix 与短 causal convolution state。它不需要保留完整 token 历史，因此从 32K 增长到 1M Context 时容量基本不变。

QSA 层保存两类随序列增长的数据：2 个 KV heads 的 K/V，以及 1 个 128 维 Indexer key head。Indexer 在检索时把连续 4 tokens 池化为 micro-block，再选择最多 512 个 blocks；raw key 本身仍按完整 Context 保存。

## MTP 的影响

MTP 是额外一层 QSA + MoE。启用后不仅增加约 2.61B checkpoint 元素，也增加一整层 QSA cache。因此它对长 Context 的 State / KV 影响远高于新增一个 GDN 层。

## 容量与计算量的区别

QSA 的 2048-token budget 限制单次稀疏 attention 读取的候选规模，但页面计算的是持久存储，而不是每步实际读取 bytes。真实带宽还取决于 block 命中、cache layout、并行策略和 kernel 实现。
