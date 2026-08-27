# Qwen3.8 Flash Next Explorer

一个本地优先的 Qwen3.8-Flash-Next 模型结构、权重与推理状态容量分析页面，视觉和交互方式沿用 DeepSeek V4 Flash Explorer。

## 已实现

- 固化官方 Hugging Face `config.json`、README、safetensors 索引，以及对应 Transformers Qwen4-Exp 实现
- 展示 48 个主 Block：36 层 Gated DeltaNet 与 12 层 Qwen Sparse Attention
- 展示 4 路 Gated Residual、Top-10 / 512 MoE、Layer 1 N-gram PLE、27 层 Vision Tower 和 1 层 MTP
- 单击算子查看张量与容量；双击一级模块展开内部数据流
- Weight 配置：官方 BF16 checkpoint、FP8/W4 混合估算、FP8、W4.25 与 W4 理论下限
- State / KV 配置：BF16、FP8、INT8、INT4，并区分 QSA 全局 KV、Indexer key 与 GDN 固定 recurrent state
- Context、Batch、N-gram、Vision、MTP 开关实时计算；三个可选组件均可独立选择跟随全局、BF16、FP8、W4.25 或 W4
- 场景和分析结果 JSON 导入、导出与本地保存
- 内置 Document 页面，直接浏览 `docs/` 中的架构、量化、公式与验证文档
- 默认 BF16 全组件结果与官方索引严格对齐：`179,999,981,459` 个存储元素、`359,999,963,128` bytes

## 启动

Linux / macOS：

```bash
python3 server.py
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

然后打开终端打印的地址；也可以使用 `python3 server.py --host 127.0.0.1 --port 8000` 后访问 `http://127.0.0.1:8000`。不要直接双击 `index.html`，浏览器通常不允许 `file://` 页面读取相邻 JSON。

## 重新生成与校验

```bash
python3 scripts/generate_model_spec.py
python3 scripts/calculate_analysis.py
python3 scripts/validate_data.py
```

## 重要文件

- `data/model/qwen3.8-flash-next.model-spec.json`：网页使用的模型语义规格
- `data/sources/qwen3.8-flash-next/`：官方配置、索引与实现快照
- `data/quantization/`：Weight 和 State / KV 容量配置
- `docs/model-architecture-mapping.md`：架构与源码映射
- `docs/calculation-methodology.md`：容量公式
- `docs/assumptions-and-limitations.md`：估算边界

## 计算边界

页面计算 Weight 与增量解码时的 State / KV 常驻容量，不预测吞吐、延迟、预填充峰值，也不包含 activation、kernel workspace、allocator 碎片或多卡通信。非 BF16 Weight 方案均为容量估算，不代表官方发布了对应量化 checkpoint。
