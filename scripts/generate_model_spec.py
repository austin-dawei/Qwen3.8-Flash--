#!/usr/bin/env python3
"""Generate the browser-facing Qwen3.8-Flash-Next model specification.

The adapter is intentionally model-specific. Shapes are mapped from the
official config, Hugging Face Transformers Qwen4-Exp implementation, and the
published safetensors index/header metadata captured in ``data/sources``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "sources" / "qwen3.8-flash-next"
CONFIG_PATH = SOURCE_DIR / "config.json"
SOURCE_INFO_PATH = SOURCE_DIR / "source-info.json"
OUTPUT_PATH = ROOT / "data" / "model" / "qwen3.8-flash-next.model-spec.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_group(
    group_id: str,
    name: str,
    component: str,
    shape: list[int],
    *,
    copies: int = 1,
    storage_class: str = "profile",
    policy_group: str | None = None,
    source_lines: str,
    optional_component: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    elements_per_copy = math.prod(shape)
    result: dict[str, Any] = {
        "id": group_id,
        "name": name,
        "component": component,
        "shape": shape,
        "copies": copies,
        "elements_per_copy": elements_per_copy,
        "logical_params": elements_per_copy * copies,
        "storage_class": storage_class,
        "source": {
            "file": "data/sources/qwen3.8-flash-next/modeling_qwen4_exp.py",
            "lines": source_lines,
        },
    }
    if policy_group:
        result["policy_group"] = policy_group
    if optional_component:
        result["optional_component"] = optional_component
    if notes:
        result["notes"] = notes
    return result


def build_gr_groups(prefix: str, label: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    text = config["text_config"]
    hidden = text["hidden_size"]
    streams = text["hc_count"]
    hc_hidden = hidden * streams
    rank = text["hc_lowrank"]
    return [
        parameter_group(f"gr.{prefix}.norm", f"{label} GR RMSNorm", "Gated Residual", [hc_hidden], storage_class="fixed_bf16", source_lines="947-947"),
        parameter_group(f"gr.{prefix}.down", f"{label} read gate down", "Gated Residual", [rank, hc_hidden], policy_group="standard_linear", source_lines="948-948"),
        parameter_group(f"gr.{prefix}.up", f"{label} read gate up", "Gated Residual", [hc_hidden, rank], policy_group="standard_linear", source_lines="949-949"),
        parameter_group(f"gr.{prefix}.inject", f"{label} write gate", "Gated Residual", [streams, hc_hidden], policy_group="standard_linear", source_lines="950-950"),
    ]


def build_moe_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    text = config["text_config"]
    hidden = text["hidden_size"]
    experts = text["num_experts"]
    intermediate = text["moe_intermediate_size"]
    shared_intermediate = text["shared_expert_intermediate_size"]
    return [
        parameter_group("moe.router", "Top-K router", "MoE", [experts, hidden], policy_group="router", source_lines="899-917"),
        parameter_group("moe.expert.gate_up", "Routed expert gate + up", "MoE", [2 * intermediate, hidden], copies=experts, policy_group="expert_linear", source_lines="859-893"),
        parameter_group("moe.expert.down", "Routed expert down", "MoE", [hidden, intermediate], copies=experts, policy_group="expert_linear", source_lines="859-893"),
        parameter_group("moe.shared.gate", "Shared expert gate", "MoE", [shared_intermediate, hidden], policy_group="shared_expert_linear", source_lines="842-855, 924-934"),
        parameter_group("moe.shared.up", "Shared expert up", "MoE", [shared_intermediate, hidden], policy_group="shared_expert_linear", source_lines="842-855, 924-934"),
        parameter_group("moe.shared.down", "Shared expert down", "MoE", [hidden, shared_intermediate], policy_group="shared_expert_linear", source_lines="842-855, 924-934"),
        parameter_group("moe.shared.output_gate", "Shared expert output gate", "MoE", [1, hidden], policy_group="router", source_lines="925-934"),
    ]


def build_gdn_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    text = config["text_config"]
    hidden = text["hidden_size"]
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_dim = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    conv_dim = 2 * key_dim + value_dim
    value_heads = text["linear_num_value_heads"]
    value_head_dim = text["linear_value_head_dim"]
    return [
        parameter_group("gdn.qkv", "GDN fused QKV projection", "Gated DeltaNet", [conv_dim, hidden], policy_group="standard_linear", source_lines="444-444"),
        parameter_group("gdn.z", "GDN output gate projection", "Gated DeltaNet", [value_dim, hidden], policy_group="standard_linear", source_lines="445-445"),
        parameter_group("gdn.beta", "GDN beta projection", "Gated DeltaNet", [value_heads, hidden], policy_group="standard_linear", source_lines="446-446"),
        parameter_group("gdn.decay", "GDN decay projection", "Gated DeltaNet", [value_heads, hidden], policy_group="standard_linear", source_lines="447-447"),
        parameter_group("gdn.conv", "Depthwise causal convolution", "Gated DeltaNet", [conv_dim, 1, text["linear_conv_kernel_dim"]], storage_class="fixed_bf16", source_lines="421-429"),
        parameter_group("gdn.A_log", "Recurrent decay base", "Gated DeltaNet", [value_heads], storage_class="fixed_bf16", source_lines="435-436"),
        parameter_group("gdn.dt_bias", "Recurrent time bias", "Gated DeltaNet", [value_heads], storage_class="fixed_bf16", source_lines="432-432"),
        parameter_group("gdn.norm", "Gated output RMSNorm", "Gated DeltaNet", [value_head_dim], storage_class="fixed_bf16", source_lines="437-439"),
        parameter_group("gdn.out", "GDN output projection", "Gated DeltaNet", [hidden, value_dim], policy_group="standard_linear", source_lines="440-440"),
    ]


def build_qsa_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    text = config["text_config"]
    hidden = text["hidden_size"]
    heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    index_heads = text["indexer_n_heads"]
    index_kv_heads = text["indexer_kv_heads"]
    index_dim = text["indexer_head_dim"]
    return [
        parameter_group("qsa.q_gate", "Q + output gate projection", "Qwen Sparse Attention", [2 * heads * head_dim, hidden], policy_group="standard_linear", source_lines="769-771"),
        parameter_group("qsa.k", "Grouped key projection", "Qwen Sparse Attention", [kv_heads * head_dim, hidden], policy_group="standard_linear", source_lines="772-774"),
        parameter_group("qsa.v", "Grouped value projection", "Qwen Sparse Attention", [kv_heads * head_dim, hidden], policy_group="standard_linear", source_lines="775-777"),
        parameter_group("qsa.out", "Attention output projection", "Qwen Sparse Attention", [hidden, heads * head_dim], policy_group="standard_linear", source_lines="778-780"),
        parameter_group("qsa.q_norm", "Q RMSNorm", "Qwen Sparse Attention", [head_dim], storage_class="fixed_bf16", source_lines="781-781"),
        parameter_group("qsa.k_norm", "K RMSNorm", "Qwen Sparse Attention", [head_dim], storage_class="fixed_bf16", source_lines="782-782"),
        parameter_group("qsa.indexer.qk", "Indexer fused QK projection", "QSA Indexer", [(index_heads + index_kv_heads) * index_dim, hidden], policy_group="standard_linear", source_lines="622-626"),
        parameter_group("qsa.indexer.q_norm", "Indexer Q RMSNorm", "QSA Indexer", [index_dim], storage_class="fixed_bf16", source_lines="627-627"),
        parameter_group("qsa.indexer.k_norm", "Indexer K RMSNorm", "QSA Indexer", [index_dim], storage_class="fixed_bf16", source_lines="628-628"),
    ]


def build_ple_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    text = config["text_config"]
    hidden = text["hidden_size"]
    hc_hidden = hidden * text["hc_count"]
    ple_dim = text["ple_embed_dim"]
    ngram_rows = 2_500_012 * text["split_ngram_parts"]
    head_dim = ple_dim // ((text["ngram_size"] - 1) * text["heads_per_ngram"])
    return [
        parameter_group("ple.ngram", "Hashed bigram + trigram embedding", "N-gram Embedding", [ngram_rows, head_dim], policy_group="ngram_embedding", source_lines="1018-1051", optional_component="ngram", notes="128 checkpoint shards; host-memory offload is supported by the design."),
        parameter_group("ple.key", "PLE key projection", "N-gram Embedding", [hc_hidden, ple_dim], policy_group="standard_linear", source_lines="1133-1133"),
        parameter_group("ple.value", "PLE value projection", "N-gram Embedding", [hidden, ple_dim], policy_group="standard_linear", source_lines="1134-1134"),
        parameter_group("ple.norm_key", "PLE key RMSNorm", "N-gram Embedding", [hc_hidden], storage_class="fixed_bf16", source_lines="1138-1138"),
        parameter_group("ple.norm_query", "PLE query RMSNorm", "N-gram Embedding", [hc_hidden], storage_class="fixed_bf16", source_lines="1139-1139"),
        parameter_group("ple.norm_conv", "PLE convolution RMSNorm", "N-gram Embedding", [hc_hidden], storage_class="fixed_bf16", source_lines="1140-1140"),
        parameter_group("ple.conv", "PLE dilated depthwise convolution", "N-gram Embedding", [hc_hidden, 1, text["ple_conv_kernel_size"]], storage_class="fixed_bf16", source_lines="1141-1148"),
        parameter_group("ple.buffers", "N-gram hash metadata", "N-gram Embedding", [35], storage_class="fixed_int64", source_lines="1043-1049"),
    ]


def build_layer_groups(config: dict[str, Any], index: int) -> list[dict[str, Any]]:
    text = config["text_config"]
    groups = [
        *build_gr_groups("attn", "Sequence mixer", config),
        *(build_gdn_groups(config) if text["layer_types"][index] == "linear_attention" else build_qsa_groups(config)),
        *build_gr_groups("mlp", "MoE", config),
        *build_moe_groups(config),
    ]
    if index + 1 in text["ple_layer_ids"]:
        groups.extend(build_ple_groups(config))
    return groups


def build_mtp_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    text = config["text_config"]
    hidden = text["hidden_size"]
    hc_hidden = hidden * text["hc_count"]
    return [
        *build_gr_groups("attn", "Sequence mixer", config),
        *build_qsa_groups(config),
        *build_gr_groups("mlp", "MoE", config),
        *build_moe_groups(config),
        parameter_group("mtp.embedding_proj", "MTP embedding projection", "MTP", [hidden, hidden], policy_group="mtp_projection", source_lines="checkpoint: mtp.fc_embedding.weight"),
        parameter_group("mtp.hidden_proj", "MTP hidden projection", "MTP", [hidden, hidden], policy_group="mtp_projection", source_lines="checkpoint: mtp.fc_hidden.weight"),
        parameter_group("mtp.embedding_norm", "MTP embedding pre-norm", "MTP", [hidden], storage_class="fixed_bf16", source_lines="checkpoint: mtp.pre_fc_norm_embedding.weight"),
        parameter_group("mtp.hidden_norm", "MTP hidden pre-norm", "MTP", [hc_hidden], storage_class="fixed_bf16", source_lines="checkpoint: mtp.pre_fc_norm_hidden.weight"),
        parameter_group("mtp.final_norm", "MTP final GR RMSNorm", "MTP", [hc_hidden], storage_class="fixed_bf16", source_lines="checkpoint: mtp.hyper_connection_mixer.hc_norm.weight"),
        parameter_group("mtp.final_down", "MTP final GR down", "MTP", [text["hc_lowrank"], hc_hidden], policy_group="mtp_projection", source_lines="checkpoint: mtp.hyper_connection_mixer.input_mix_weight_down.weight"),
        parameter_group("mtp.final_up", "MTP final GR up", "MTP", [hc_hidden, text["hc_lowrank"]], policy_group="mtp_projection", source_lines="checkpoint: mtp.hyper_connection_mixer.input_mix_weight_up.weight"),
    ]


def build_vision_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    vision = config["vision_config"]
    hidden = vision["hidden_size"]
    inter = vision["intermediate_size"]
    depth = vision["depth"]
    merged = hidden * vision["spatial_merge_size"] ** 2
    optional = "vision"
    return [
        parameter_group("vision.patch.weight", "Vision patch embedding", "Vision", [hidden, vision["in_channels"], vision["temporal_patch_size"], vision["patch_size"], vision["patch_size"]], policy_group="vision_linear", source_lines="1685-1703", optional_component=optional),
        parameter_group("vision.patch.bias", "Vision patch bias", "Vision", [hidden], storage_class="fixed_bf16", source_lines="1685-1703", optional_component=optional),
        parameter_group("vision.position", "Vision position embedding", "Vision", [vision["num_position_embeddings"], hidden], policy_group="embedding", source_lines="1866-1866", optional_component=optional),
        parameter_group("vision.attn.qkv", "Vision QKV projections", "Vision", [3 * hidden, hidden], copies=depth, policy_group="vision_linear", source_lines="1735-1745", optional_component=optional),
        parameter_group("vision.attn.qkv_bias", "Vision QKV biases", "Vision", [3 * hidden], copies=depth, storage_class="fixed_bf16", source_lines="1735-1745", optional_component=optional),
        parameter_group("vision.attn.out", "Vision attention output", "Vision", [hidden, hidden], copies=depth, policy_group="vision_linear", source_lines="1735-1745", optional_component=optional),
        parameter_group("vision.attn.out_bias", "Vision attention output biases", "Vision", [hidden], copies=depth, storage_class="fixed_bf16", source_lines="1735-1745", optional_component=optional),
        parameter_group("vision.mlp.fc1", "Vision MLP up", "Vision", [inter, hidden], copies=depth, policy_group="vision_linear", source_lines="1672-1682", optional_component=optional),
        parameter_group("vision.mlp.fc1_bias", "Vision MLP up biases", "Vision", [inter], copies=depth, storage_class="fixed_bf16", source_lines="1672-1682", optional_component=optional),
        parameter_group("vision.mlp.fc2", "Vision MLP down", "Vision", [hidden, inter], copies=depth, policy_group="vision_linear", source_lines="1672-1682", optional_component=optional),
        parameter_group("vision.mlp.fc2_bias", "Vision MLP down biases", "Vision", [hidden], copies=depth, storage_class="fixed_bf16", source_lines="1672-1682", optional_component=optional),
        parameter_group("vision.norms", "Vision LayerNorm weights + biases", "Vision", [hidden], copies=depth * 4, storage_class="fixed_bf16", source_lines="1818-1823", optional_component=optional),
        parameter_group("vision.merger.norm", "Vision merger norm weight + bias", "Vision", [hidden], copies=2, storage_class="fixed_bf16", source_lines="1705-1717", optional_component=optional),
        parameter_group("vision.merger.fc1", "Vision merger projection", "Vision", [merged, merged], policy_group="vision_linear", source_lines="1705-1717", optional_component=optional),
        parameter_group("vision.merger.fc1_bias", "Vision merger projection bias", "Vision", [merged], storage_class="fixed_bf16", source_lines="1705-1717", optional_component=optional),
        parameter_group("vision.merger.fc2", "Vision-to-text projection", "Vision", [vision["out_hidden_size"], merged], policy_group="vision_linear", source_lines="1705-1717", optional_component=optional),
        parameter_group("vision.merger.fc2_bias", "Vision-to-text projection bias", "Vision", [vision["out_hidden_size"]], storage_class="fixed_bf16", source_lines="1705-1717", optional_component=optional),
    ]


def summarize_groups(groups: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, int]:
    text = config["text_config"]
    routed_ids = {"moe.expert.gate_up", "moe.expert.down"}
    active_expert = sum(group["elements_per_copy"] * text["num_experts_per_tok"] for group in groups if group["id"] in routed_ids)
    return {
        "logical_params": sum(group["logical_params"] for group in groups),
        "routed_expert_resident_params": sum(group["logical_params"] for group in groups if group["id"] in routed_ids),
        "routed_expert_active_params_per_token": active_expert,
    }


def build_graph() -> dict[str, Any]:
    module_nodes = [
        {"id": "input", "name": "4-stream input", "group": "tensor", "shape": "[B,S,4,2560]", "description": "Four widened residual streams enter each decoder layer."},
        {"id": "ple", "name": "N-gram PLE", "group": "embedding", "shape": "bigram + trigram → 4×2560", "condition": "has_ple", "description": "Layer 1 injects hashed local-pattern memory into all four streams.", "parameter_prefixes": ["ple."]},
        {"id": "gr_attn", "name": "Gated Residual · Read", "group": "gr", "shape": "4×2560 → 2560", "description": "Element-wise read gate mixes four residual branches.", "parameter_prefixes": ["gr.attn."]},
        {"id": "gdn", "name": "Gated DeltaNet", "group": "linear_attention", "shape": "48 V / 16 QK heads × 128", "condition": "linear_attention", "description": "Fixed-state recurrent linear attention; used by 3 of every 4 layers.", "parameter_prefixes": ["gdn."]},
        {"id": "qsa", "name": "Qwen Sparse Attention", "group": "attention", "shape": "24 Q / 2 KV heads × 256", "condition": "qsa", "description": "Global attention with a micro-block indexer and 2048-token budget.", "parameter_prefixes": ["qsa."]},
        {"id": "gr_mlp", "name": "Gated Residual · Read", "group": "gr", "shape": "4×2560 → 2560", "description": "A second gated read/write wrapper around MoE.", "parameter_prefixes": ["gr.mlp."]},
        {"id": "moe", "name": "Sparse MoE", "group": "moe", "shape": "Top-10 / 512 + shared", "description": "Ten routed experts plus one always-on shared expert.", "parameter_prefixes": ["moe."]},
        {"id": "output", "name": "4-stream output", "group": "tensor", "shape": "[B,S,4,2560]", "description": "Per-branch write gates inject sequence-mixer and MoE outputs."},
    ]
    module_edges = [
        {"from": "input", "to": "ple", "condition": "has_ple"},
        {"from": "ple", "to": "gr_attn", "condition": "has_ple"},
        {"from": "input", "to": "gr_attn", "condition": "no_ple"},
        {"from": "gr_attn", "to": "gdn", "condition": "linear_attention"},
        {"from": "gr_attn", "to": "qsa", "condition": "qsa"},
        {"from": "gdn", "to": "gr_mlp", "condition": "linear_attention"},
        {"from": "qsa", "to": "gr_mlp", "condition": "qsa"},
        {"from": "gr_mlp", "to": "moe"},
        {"from": "moe", "to": "output"},
    ]
    expanded_nodes = [
        {"id": "gr_attn_read", "name": "Read gate", "group": "gr", "shape": "10240 → 320 → 10240", "parameter_prefixes": ["gr.attn."]},
        {"id": "gr_attn_write", "name": "Write gate", "group": "gr", "shape": "10240 → 4 scalars", "parameter_ids": ["gr.attn.inject"]},
        {"id": "gdn_qkv", "name": "Fused QKV", "group": "linear", "shape": "2560 → 10240", "parameter_ids": ["gdn.qkv"]},
        {"id": "gdn_conv", "name": "Causal DWConv", "group": "linear_attention", "shape": "10240 × kernel 4", "parameter_ids": ["gdn.conv"]},
        {"id": "gdn_recur", "name": "Delta-rule state", "group": "cache", "shape": "48×128×128 FP32", "parameter_ids": ["gdn.A_log", "gdn.dt_bias"]},
        {"id": "gdn_gate", "name": "β / decay / z gates", "group": "linear_attention", "shape": "48 + 48 + 6144", "parameter_ids": ["gdn.beta", "gdn.decay", "gdn.z", "gdn.norm"]},
        {"id": "gdn_out", "name": "Output projection", "group": "linear", "shape": "6144 → 2560", "parameter_ids": ["gdn.out"]},
        {"id": "qsa_q", "name": "Q + gate", "group": "linear", "shape": "2560 → 24×256×2", "parameter_ids": ["qsa.q_gate", "qsa.q_norm"]},
        {"id": "qsa_kv", "name": "Grouped KV", "group": "attention", "shape": "2 KV heads × 256", "parameter_ids": ["qsa.k", "qsa.v", "qsa.k_norm"]},
        {"id": "qsa_index", "name": "Indexer QK", "group": "index", "shape": "4 Q + 1 K heads × 128", "parameter_prefixes": ["qsa.indexer."]},
        {"id": "qsa_pool", "name": "Micro-block mean", "group": "index", "shape": "4 tokens → 1 key"},
        {"id": "qsa_topk", "name": "Top-512 blocks", "group": "index", "shape": "budget 2048 tokens"},
        {"id": "qsa_attn", "name": "Sparse attention", "group": "attention", "shape": "selected global KV"},
        {"id": "qsa_out", "name": "Output projection", "group": "linear", "shape": "6144 → 2560", "parameter_ids": ["qsa.out"]},
        {"id": "gr_mlp_read", "name": "Read gate", "group": "gr", "shape": "10240 → 320 → 10240", "parameter_prefixes": ["gr.mlp."]},
        {"id": "gr_mlp_write", "name": "Write gate", "group": "gr", "shape": "10240 → 4 scalars", "parameter_ids": ["gr.mlp.inject"]},
        {"id": "router", "name": "Top-10 router", "group": "moe", "shape": "2560 → 512", "parameter_ids": ["moe.router"]},
        {"id": "routed", "name": "10 / 512 experts", "group": "moe", "shape": "SwiGLU 2560→640→2560", "parameter_prefixes": ["moe.expert."]},
        {"id": "shared", "name": "Shared expert", "group": "moe", "shape": "SwiGLU + sigmoid gate", "parameter_prefixes": ["moe.shared."]},
        {"id": "combine", "name": "Weighted combine", "group": "moe", "shape": "routed + shared"},
        {"id": "ple_lookup", "name": "Hashed lookup", "group": "embedding", "shape": "16 heads × 160", "parameter_ids": ["ple.ngram", "ple.buffers"]},
        {"id": "ple_gate", "name": "Stream gating", "group": "embedding", "shape": "key/query → 4 gates", "parameter_ids": ["ple.key", "ple.value", "ple.norm_key", "ple.norm_query"]},
        {"id": "ple_conv", "name": "Dilated DWConv", "group": "embedding", "shape": "10240 × kernel 4", "parameter_ids": ["ple.conv", "ple.norm_conv"]},
    ]
    return {"module": {"nodes": module_nodes, "edges": module_edges}, "expanded": {"nodes": expanded_nodes, "edges": []}}


def main() -> None:
    config = load_json(CONFIG_PATH)
    source_info = load_json(SOURCE_INFO_PATH)
    text = config["text_config"]
    layers = []
    for index, layer_type in enumerate(text["layer_types"]):
        groups = build_layer_groups(config, index)
        has_ple = index + 1 in text["ple_layer_ids"]
        layers.append({
            "index": index,
            "label": f"Layer {index}",
            "kind": "decoder",
            "template_id": "gdn-moe" if layer_type == "linear_attention" else "qsa-moe",
            "attention_type": layer_type,
            "has_ple": has_ple,
            "cache_mode": "fixed recurrent state" if layer_type == "linear_attention" else "global KV + micro-block sparse selection",
            "parameter_groups": groups,
            "parameter_summary": summarize_groups(groups, config),
        })

    mtp_groups = build_mtp_groups(config)
    auxiliary_layers = [{
        "index": 48,
        "auxiliary_index": 0,
        "label": "MTP 0",
        "kind": "mtp",
        "template_id": "mtp-qsa-moe",
        "attention_type": "qwen_sparse_attention",
        "has_ple": False,
        "cache_mode": "global KV + micro-block sparse selection",
        "parameter_groups": mtp_groups,
        "parameter_summary": summarize_groups(mtp_groups, config),
    }]

    hidden = text["hidden_size"]
    vocab = text["vocab_size"]
    hc_hidden = hidden * text["hc_count"]
    global_groups = [
        parameter_group("global.embedding", "Token embedding", "Global", [vocab, hidden], policy_group="embedding", source_lines="1323-1325"),
        parameter_group("global.lm_head", "LM head", "Global", [vocab, hidden], policy_group="lm_head", source_lines="2349-2353"),
        parameter_group("global.final_gr_norm", "Final GR RMSNorm", "Global", [hc_hidden], storage_class="fixed_bf16", source_lines="1329-1330"),
        parameter_group("global.final_gr_down", "Final GR read gate down", "Global", [text["hc_lowrank"], hc_hidden], policy_group="standard_linear", source_lines="948-949, 1330-1330"),
        parameter_group("global.final_gr_up", "Final GR read gate up", "Global", [hc_hidden, text["hc_lowrank"]], policy_group="standard_linear", source_lines="948-949, 1330-1330"),
        *build_vision_groups(config),
    ]

    spec = {
        "schema_version": "1.1.0",
        "generator_version": "1.0.0",
        "model_id": "qwen3.8-flash-next",
        "display_name": "Qwen3.8-Flash-Next",
        "source": source_info,
        "architecture": {
            "class_name": config["architectures"][0],
            "model_type": config["model_type"],
            "main_layer_count": text["num_hidden_layers"],
            "gdn_layer_count": text["layer_types"].count("linear_attention"),
            "qsa_layer_count": text["layer_types"].count("full_attention"),
            "mtp_layer_count": text["mtp_num_hidden_layers"],
            "hidden_size": hidden,
            "vocab_size": vocab,
            "max_context_length": text["max_position_embeddings"],
            "extended_context_length": 1_000_000,
            "attention_heads": text["num_attention_heads"],
            "kv_heads": text["num_key_value_heads"],
            "head_dim": text["head_dim"],
            "rope_head_dim": int(text["head_dim"] * text["partial_rotary_factor"]),
            "linear_qk_heads": text["linear_num_key_heads"],
            "linear_value_heads": text["linear_num_value_heads"],
            "linear_head_dim": text["linear_key_head_dim"],
            "linear_conv_kernel": text["linear_conv_kernel_dim"],
            "routed_experts": text["num_experts"],
            "shared_experts": 1,
            "experts_per_token": text["num_experts_per_tok"],
            "expert_intermediate_size": text["moe_intermediate_size"],
            "hc_mult": text["hc_count"],
            "hc_lowrank": text["hc_lowrank"],
            "index_heads": text["indexer_n_heads"],
            "index_kv_heads": text["indexer_kv_heads"],
            "index_head_dim": text["indexer_head_dim"],
            "index_compress_ratio": text["indexer_compress_ratio"],
            "index_token_budget": text["indexer_budget"],
            "ngram_size": text["ngram_size"],
            "ngram_parameters": 51_200_245_760,
            "vision_layer_count": config["vision_config"]["depth"],
            "checkpoint_total_bytes": 359_999_963_128,
            "checkpoint_elements": 179_999_981_459,
            "features": [
                "Gated DeltaNet + Qwen Sparse Attention",
                "4-branch Gated Residual",
                "512-expert sparse MoE",
                "51.2B hashed N-gram Embedding",
                "Micro-block QSA indexer",
                "27-layer vision encoder",
                "Multi-token prediction auxiliary block",
            ],
        },
        "global_parameter_groups": global_groups,
        "global_parameter_summary": {"logical_params": sum(group["logical_params"] for group in global_groups)},
        "layers": layers,
        "auxiliary_layers": auxiliary_layers,
        "operator_graphs": build_graph(),
        "formula_ids": [
            "logical_parameter_count_v1",
            "weight_storage_fixed_bpw_v1",
            "weight_storage_fp8_block_v1",
            "qsa_kv_cache_capacity_v1",
            "gdn_recurrent_state_capacity_v1",
            "layer_weight_ratio_v1",
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    main_params = sum(layer["parameter_summary"]["logical_params"] for layer in layers)
    total_params = main_params + spec["global_parameter_summary"]["logical_params"] + summarize_groups(mtp_groups, config)["logical_params"]
    print(f"Generated {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Layers: {len(layers)} = {spec['architecture']['gdn_layer_count']} GDN + {spec['architecture']['qsa_layer_count']} QSA")
    print(f"All released checkpoint elements: {total_params:,}")


if __name__ == "__main__":
    main()
