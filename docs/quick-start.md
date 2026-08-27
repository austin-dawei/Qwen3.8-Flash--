# 快速开始

## 启动本地服务

在项目根目录运行：

```bash
python3 server.py
```

终端会打印实际监听地址。通过该 HTTP 地址访问页面，不要直接双击 `index.html`；`file://` 页面通常无法读取相邻 JSON，也无法使用文档与场景 API。

如需指定地址和端口：

```bash
python3 server.py --host 127.0.0.1 --port 8000
```

## 使用 Explorer

1. 选择全局 Weight 与 State / KV 配置。
2. 设置 Context 和 Batch。
3. 决定是否计入 N-gram、Vision 与 MTP。
4. 如有需要，为三个可选组件分别覆盖 Weight 量化。
5. 在 Layer 地图中选择层，单击算子查看参数组，双击可展开模块内部结构。

所有控件都会即时重算顶部的 Weight、KV cache 和 Weight + KV。

## 场景保存与交换

“导出场景”只导出输入配置；“导出结果”导出带逐层容量的分析结果。通过“保存场景与结果”写入仓库时，会生成：

```text
data/scenarios/<name>.json
data/generated/<name>.analysis.json
```

本地保存需要通过 `server.py` 访问。导入旧版场景时，缺少的 N-gram、Vision、MTP 独立量化字段会自动按 `inherit` 处理。

## 离线复算

重新生成默认结果并校验：

```bash
python3 scripts/generate_model_spec.py
python3 scripts/calculate_analysis.py
python3 scripts/validate_data.py
```

复算其他场景：

```bash
python3 scripts/calculate_analysis.py \
  --scenario data/scenarios/default.json \
  --output data/generated/custom.analysis.json
```

## 常见问题

### 为什么 BF16 显示约 335 GiB，而不是 360 GB？

官方索引总量为 359,999,963,128 十进制 bytes。页面使用二进制 GiB，即除以 `1024³`，因此显示约 335 GiB。

### 为什么关闭 N-gram 会少很多 Weight？

N-gram table 本身包含约 51.2B 个 BF16 元素。关闭开关表示不把它计入当前常驻集合，不代表模型功能可以无损移除。

### 为什么修改 Context 不改变 GDN 层状态？

GDN 保存固定大小 recurrent state；只有 QSA 的全局 K/V 和 Indexer key 随 Context 线性增长。
