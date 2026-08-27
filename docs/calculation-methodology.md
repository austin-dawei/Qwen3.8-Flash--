# Qwen3.8-Flash-Next 计算方法

计算分成参数归属、Weight 编码、逐层 State / KV 和最终汇总四步。浏览器与 `scripts/analysis_math.py` 使用同一组公式，但实现相互独立，便于交叉检查。

## Weight

逻辑元素数为每个 checkpoint tensor shape 的乘积。多个同形 tensor 用 `copies` 表示：

```text
elements(group) = product(shape) × copies
```

固定 BPW 配置：

```text
bytes = elements × bpw / 8
```

FP8 block-128 估算：

```text
weight_bytes = out_features × in_features
scale_count = ceil(out_features / 128) × ceil(in_features / 128)
bytes = (weight_bytes + scale_count × scale_bits / 8) × copies
```

卷积先把除输出 channel 外的维度展平。BF16 checkpoint 配置保持所有 released weights 为 16 BPW，三个 N-gram metadata buffers 按 INT64，从而精确复现 safetensors index 的 `359,999,963,128` bytes。

N-gram Embedding、Vision Tower 与 MTP 可以分别覆盖全局 Weight 配置。覆盖只作用于可量化矩阵；Norm、Bias 及 N-gram 的 INT64 metadata buffers 仍保持模型规格中声明的固定精度。选择“跟随全局 Weight”时与原有计算完全一致。

### 有效位宽

W4.25 使用统一有效位宽近似 weight body 与量化元数据：

```text
bytes = elements × 4.25 / 8
```

W4 raw 则使用 `4 / 8`，不包含 scale、zero-point、packing padding 或对齐，因此只能作为理论下限。

## QSA KV 与 Indexer cache

每个 QSA 层有 2 个 KV heads、head dimension 256。K 和 V 合计每 token 1024 个元素，其中 K 的 `2 × 64 = 128` 个元素属于 RoPE 部分：

```text
main_kv_bytes = batch × context ×
  ((1024 - 128) × nope_bpw + 128 × rope_bpw) / 8

indexer_bytes = batch × context × 1 × 128 × index_bpw / 8
```

QSA 的 4:1 micro-block pooling 与 Top-512 block 仅降低 attention 计算量；当前开源实现缓存 raw Indexer key，因此不对 cache 长度做 4:1 除法。

## GDN recurrent state

GDN 使用固定大小 FP32 recurrent state：

```text
recurrent_bytes = batch × 48 × 128 × 128 × 4
conv_bytes = batch × 10240 × 4 × conv_bpw / 8
```

Layer 1 启用 N-gram PLE 时还包含 9-token dilated convolution state 与 2 个 token id 的历史。

## Context 与 Batch 缩放

QSA main K/V 与 Indexer cache 同时随 Context 和 Batch 线性增长。GDN recurrent/conv state、PLE 辅助状态只随 Batch 增长。Weight 与 Context、Batch 无关。

## 汇总

`Weight + KV` 为所选组件 Weight 与所有启用 decoder / MTP 层的增量解码 State / KV 之和。Vision 没有跨请求持久 KV；页面只将其权重计入，不估算一次图像前向的 activation。

所有显示容量使用 IEC 单位：KiB、MiB、GiB、TiB 的除数分别为 `1024`、`1024²`、`1024³`、`1024⁴`。JSON 中保留原始 bytes，避免格式化造成复算误差。
