# 数据来源与验证

## 固定上游版本

架构映射基于以下不可变 revision：

- Hugging Face `Qwen/Qwen3.8-Flash-Next`：`f5d08274bafd880402bd16f5e3e6c514136ec06c`
- Transformers Qwen4-Exp 实现：`dabae5fcb924a8eece0e727b627ca5f050b40d40`

项目在 `data/sources/qwen3.8-flash-next/` 保存官方 `config.json`、README、safetensors index、源码和 `source-info.json`。后者记录文件路径与 SHA256，避免上游分支变化后无意混用数据。

## 规格生成链路

```text
官方 config / index / modeling source
                 ↓
scripts/generate_model_spec.py
                 ↓
data/model/qwen3.8-flash-next.model-spec.json
                 ↓
浏览器 app.js + Python analysis_math.py
```

生成脚本按 checkpoint tensor 名称映射全局参数组、48 个主层和 1 个 MTP 层，并为网页附加算子图和源码行号。

## 当前自动校验

`scripts/validate_data.py` 会检查：

- 官方快照文件 SHA256 与 `source-info.json` 一致。
- 主层数为 48，且恰好包含 36 个 GDN 与 12 个 QSA。
- MTP 层数为 1，PLE 层数为 1。
- 所有 parameter group 的元素数等于 `179,999,981,459`。
- 默认 BF16 Weight 等于官方 safetensors index 的 `359,999,963,128 bytes`。
- 全局与逐层 Weight share 总和为 100%。
- N-gram、Vision、MTP 分别切换 W4 后 Weight 下降，而逻辑参数量与 State / KV 不变。
- 生成的默认分析结果与实时复算 totals 一致。

运行方式：

```bash
python3 scripts/generate_model_spec.py
python3 scripts/calculate_analysis.py
python3 scripts/validate_data.py
```

## 修改数据后的检查清单

1. 固定新的上游 revision，并更新 `source-info.json` 中的路径与 SHA256。
2. 检查配置字段、tensor 命名、层类型循环和源码行号是否变化。
3. 重新生成 model spec 与默认 analysis result。
4. 运行校验脚本和 JavaScript 语法检查。
5. 启动服务，用浏览器检查 Explorer 与 Document 页面。

## 信任边界

“精确”只适用于固定 revision 下的 checkpoint 元素数与 BF16 文件字节数。非 BF16 方案、KV 量化和运行时显存均是明确写出假设的模型估算。
