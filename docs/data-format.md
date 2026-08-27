# 数据格式

## Model spec

`data/model/qwen3.8-flash-next.model-spec.json` 包含：

- `source`：Hugging Face 与 Transformers revision、快照路径与 SHA256
- `architecture`：GDN、QSA、GR、MoE、N-gram、Vision 和 MTP 的关键配置
- `global_parameter_groups`：Embedding、LM Head、final mixer 与 Vision tensors
- `layers`：48 个 decoder blocks；`attention_type` 与 `has_ple` 控制条件节点
- `auxiliary_layers`：MTP block
- `operator_graphs`：浏览器渲染的一级模块与展开节点

Parameter group 核心字段：`id`、`shape`、`copies`、`logical_params`、`storage_class`、`policy_group`、`optional_component` 与 `source`。

- `storage_class` 决定固定精度 tensor，优先级高于量化 profile。
- `policy_group` 将可量化 tensor 映射到 profile 中的格式。
- `optional_component` 目前为 `ngram` 或 `vision`；MTP 通过辅助层 `kind` 识别。
- `source` 保存上游文件与行号，供算子 Inspector 跳转。

## Scenario

场景包含 Weight / State 量化配置、Context、Batch，以及 `include_ngram`、`include_vision`、`include_mtp` 三个组件开关。`ngram_quantization`、`vision_quantization`、`mtp_quantization` 可取 `inherit`、`bf16`、`fp8`、`w4-effective-425` 或 `w4-raw`。网页导出的分析结果会固化 source revision、公式 ID、总量与逐层结果，便于复算。

最小完整示例：

```json
{
  "schema_version": "1.1.0",
  "name": "default",
  "model_id": "qwen3.8-flash-next",
  "weight_quantization": "bf16-checkpoint",
  "kv_cache_quantization": "bf16",
  "context_length": 262144,
  "batch_size": 1,
  "include_ngram": true,
  "ngram_quantization": "inherit",
  "include_vision": true,
  "vision_quantization": "inherit",
  "include_mtp": true,
  "mtp_quantization": "inherit",
  "operator_detail": "module"
}
```

## Analysis result

结果的 `totals` 包含 `logical_params`、`global_weight_bytes`、`weight_bytes`、`kv_cache_bytes` 与 `weight_plus_kv_bytes`。`layers` 分别记录主层和 MTP 的 Weight、State / KV 分项及模型 Weight 占比。

`generated_at` 仅记录生成时间；判断结果是否适用于当前数据时应核对 `source_revision`、`formula_ids` 和嵌入的 `scenario`。

## Schema 与兼容性

`data/schemas/scenario.schema.json` 描述当前场景格式。Web 导入器对缺少三个组件量化字段的旧场景补 `inherit`；后端保存接口要求提交规范化后的完整对象。
