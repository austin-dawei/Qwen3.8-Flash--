# 假设与限制

## 数据与公式

- 架构和 shape 来自固定 revision，而不是运行时 Python tracing；上游代码变化后需重新核对映射。
- 官方 BF16 配置与 safetensors 总字节数精确对齐；FP8、INT8、INT4、W4.25 和 W4 均为理论或部署容量估算。
- N-gram 开关表示是否将 51.2B table 计入当前常驻集合；关闭不等于模型在语义上仍可无损运行。
- Vision 开关只影响权重，不计算图像 / 视频 token activation 或 vision attention workspace。
- MTP 开关计入 checkpoint 中独立的 MTP tensors；共享 token embedding 与 LM head 不重复计数。
- QSA cache 公式描述当前 Transformers 实现：micro-block indexer 在读取时池化 raw key，而非保存永久 4:1 压缩 cache。
- GDN recurrent state 固定按 FP32；conv state 使用所选 State / KV profile 的 `conv_bpw`。
- 不计算 activation、prefill 峰值、CUDA graph、kernel workspace、量化临时缓冲、多卡通信与 allocator 碎片。
- 页面不预测吞吐或延迟；QSA 的 2048-token budget 与 N-gram host offload 会影响计算和带宽，但不能仅从参数量推导性能。

## 量化

- FP8 block-128 只计算 weight body 与每块一个 8-bit scale，没有模拟具体框架的 tensor header、padding 或额外 scale hierarchy。
- W4.25 是有效位宽预算；W4 raw 是不含任何元数据的下限。
- INT4 KV 未计 scale、zero-point、page/block 对齐和量化 kernel workspace。
- 容量下降不代表模型质量、硬件支持或 kernel 性能已经验证。

## 运行时范围

- State / KV 按增量解码持久状态计算，不覆盖 prefill 中间 activation 峰值。
- Vision 仅计算 Weight；视觉 token 数、动态分辨率和视频帧数不进入当前模型。
- 未模拟 tensor parallel、expert parallel、pipeline parallel 或 CPU/GPU 分层放置带来的复制与切分。
- 未给操作系统、驱动、CUDA context、通信库与服务框架预留空间。

## 使用建议

把页面结果视为容量账本，而不是显存验收值。规划部署时应在 Weight + State / KV 之上单独评估 activation、workspace、碎片、并行复制和安全余量，并在目标软件栈上实测。
