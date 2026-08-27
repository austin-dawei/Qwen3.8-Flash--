# Sparse MoE 一级与二级算子

项目源码中的正式名称是 **Sparse MoE**。本文也覆盖有时被口头写成 “Spark MoE” 的同一模块；它与 Apache Spark 无关。

Qwen3.8-Flash-Next 的每个主 Decoder Layer 和 MTP 层都有一个 Sparse MoE：512 个 routed experts、每 token Top-10，再加 1 个始终计算的 shared expert。输入输出 hidden size 均为 2560。

## 一级算子：Sparse MoE

一级图中的 Sparse MoE 位于第二套 Gated Residual 内：

```text
4-stream residual [B,S,4,2560]
          │ MoE Read gate
          ▼
GR mixed input [B,S,2560]
          │
          ▼
Sparse MoE
          │
          ▼
MoE output [B,S,2560]
          │ MoE Write gate
          ▼
4-stream residual [B,S,4,2560]
```

一级节点的 `Top-10 / 512 + shared` 表示：routed path 只激活 512 个专家中的 10 个，shared path 对所有 token 始终启用。

## 二级数据流

```text
                              ┌→ Router → Top-10 IDs/weights
GR mixed input [B,S,2560] ────┤        → Routed experts ─────┐
                              └→ Shared expert → output gate ├→ add → [B,S,2560]
```

Web 展开图包含 `Top-10 router`、`10 / 512 experts`、`Shared expert` 与 `Weighted combine` 四个二级节点。

## Top-10 router

Router 是无 Bias 的 Linear：

```text
[N,2560] × [512,2560]ᵀ → router_logits [N,512]
N = B × S
```

随后：

```text
router_probs = softmax(router_logits, dtype=FP32)
top_values, top_indices = topk(router_probs, k=10)
top_values = top_values / sum(top_values)
```

FP32 softmax 用于降低路由概率归一化的数值误差，最终 Top-10 权重再转换回 logits dtype。`norm_topk_prob=True`，所以每个 token 的 10 个 routed weights 之和为 1。

Router 参数量：

```text
512 × 2560 = 1,310,720
```

## 10 / 512 routed experts

每个 routed expert 是中间维度 640 的 SwiGLU MLP：

```text
x [2560]
 ├→ gate projection [640] → SiLU ─┐
 └→ up projection   [640] ─────────×→ [640]
                                      │ down projection
                                      ▼
                                    [2560]
```

源码把 gate 与 up 合并存储：

```text
gate_up_proj per expert: [1280,2560]
down_proj per expert:    [2560,640]
```

单个 expert 参数量：

```text
1280 × 2560 + 2560 × 640 = 4,915,200
```

512 个 routed experts 的常驻参数：

```text
4,915,200 × 512 = 2,516,582,400
```

每 token 只激活 10 个，因此 routed expert 激活参数量为：

```text
4,915,200 × 10 = 49,152,000
```

“激活参数量”描述一次 token 路由涉及的专家权重规模，不等于实际 FLOPs 或显存读取量；batching、expert grouping、缓存与并行策略会改变真实执行。

每个专家输出先乘对应 Top-10 routing weight，再用 `index_add_` 累加回 token 位置。

## Shared expert

Shared expert 也是 `2560 → 640 → 2560` 的 SwiGLU，但只有一份，所有 token 都执行：

```text
shared = down(SiLU(gate(x)) × up(x))
```

它另外有一个无 Bias 标量输出门：

```text
shared_gate = sigmoid(Wshared_gate x)   # 2560 → 1
shared_output = shared_gate × shared
```

Shared expert 参数：

```text
gate/up/down = 3 × 2560 × 640 = 4,915,200
output gate  = 2560
```

shared path 提供始终可用的通用变换，routed path 则提供按 token 选择的高容量专门化变换。

## Weighted combine

二级图的 `Weighted combine` 没有独立权重。它表示：

```text
output = Σ routing_weightᵢ × expertᵢ(x)
       + sigmoid(shared_gate(x)) × shared_expert(x)
```

输出 shape 恢复为 `[B,S,2560]`，再由 MoE Write gate 注入四路 residual。

## 容量与计算的区别

Explorer 的 Sparse MoE `Weight` 显示所有 512 个 routed experts 的常驻权重，因为部署时通常需要保存完整专家集合。Top-10 只影响每 token 激活路径，不会把模型文件或常驻参数自动缩小到 `10/512`。

量化 profile 中的 `expert_linear` 专门控制 routed expert 的 gate/up/down 矩阵；Router、shared expert 和其他 Linear 可以采用不同策略。W4.25 Experts + FP8 Dense 就是利用这种参数组区分。

## 对应源码

- Top-K Router：`modeling_qwen4_exp.py:898-916`
- Routed Experts：`modeling_qwen4_exp.py:858-895`
- Shared expert 与 combine：`modeling_qwen4_exp.py:919-938`
