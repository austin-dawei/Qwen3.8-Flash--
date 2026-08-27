"""Dependency-free Qwen3.8-Flash-Next capacity calculations."""

from __future__ import annotations

import math
from typing import Any


FIXED_BYTES = {"fixed_fp32": 4, "fixed_bf16": 2, "fixed_int32": 4, "fixed_int64": 8}
COMPONENT_FORMATS = {
    "bf16": {"type": "fixed_bpw", "bpw": 16},
    "fp8": {"type": "fp8_block", "block_rows": 128, "block_cols": 128, "scale_bits": 8},
    "w4-effective-425": {"type": "effective_bpw", "bpw": 4.25},
    "w4-raw": {"type": "effective_bpw", "bpw": 4},
}


def group_included(group: dict[str, Any], scenario: dict[str, Any]) -> bool:
    optional = group.get("optional_component")
    if optional == "ngram":
        return scenario["include_ngram"]
    if optional == "vision":
        return scenario["include_vision"]
    return True


def weight_group_bytes(group: dict[str, Any], profile: dict[str, Any], override: str = "inherit") -> float:
    storage_class = group["storage_class"]
    if storage_class in FIXED_BYTES:
        return group["logical_params"] * FIXED_BYTES[storage_class]
    fmt = COMPONENT_FORMATS[override] if override != "inherit" else profile["formats"][group["policy_group"]]
    if fmt["type"] in {"fixed_bpw", "effective_bpw"}:
        return group["logical_params"] * fmt["bpw"] / 8
    out_features = group["shape"][0]
    in_features = math.prod(group["shape"][1:])
    copies = group["copies"]
    if fmt["type"] == "fp8_block":
        scales = math.ceil(out_features / fmt["block_rows"]) * math.ceil(in_features / fmt["block_cols"])
        return (out_features * in_features + scales * fmt["scale_bits"] / 8) * copies
    raise ValueError(f"Unsupported weight format: {fmt['type']}")


def component_quantization(group: dict[str, Any], layer: dict[str, Any] | None, scenario: dict[str, Any]) -> str:
    if group.get("optional_component") == "ngram":
        return scenario["ngram_quantization"]
    if group.get("optional_component") == "vision":
        return scenario["vision_quantization"]
    if layer and layer.get("kind") == "mtp":
        return scenario["mtp_quantization"]
    return "inherit"


def layer_state_bytes(
    layer: dict[str, Any], architecture: dict[str, Any], kv_profile: dict[str, Any],
    context: int, batch: int, include_ngram: bool,
) -> dict[str, float]:
    main_bytes = index_bytes = recurrent_bytes = conv_bytes = auxiliary_bytes = 0.0
    if layer["attention_type"] == "linear_attention":
        recurrent_bytes = batch * architecture["linear_value_heads"] * architecture["linear_head_dim"] ** 2 * 4
        conv_dim = (
            2 * architecture["linear_qk_heads"] * architecture["linear_head_dim"]
            + architecture["linear_value_heads"] * architecture["linear_head_dim"]
        )
        conv_bytes = batch * conv_dim * architecture["linear_conv_kernel"] * kv_profile["conv_bpw"] / 8
    else:
        total_kv = 2 * architecture["kv_heads"] * architecture["head_dim"]
        rope = architecture["kv_heads"] * architecture["rope_head_dim"]
        main_bytes = batch * context * ((total_kv - rope) * kv_profile["nope_bpw"] + rope * kv_profile["rope_bpw"]) / 8
        index_bytes = batch * context * architecture["index_kv_heads"] * architecture["index_head_dim"] * kv_profile["index_bpw"] / 8
    if layer["has_ple"] and include_ngram:
        auxiliary_bytes = (
            batch * architecture["hc_mult"] * architecture["hidden_size"] * 9 * kv_profile["conv_bpw"] / 8
            + batch * (architecture["ngram_size"] - 1) * 8
        )
    return {
        "main_cache_bytes": main_bytes,
        "index_cache_bytes": index_bytes,
        "recurrent_state_bytes": recurrent_bytes,
        "conv_state_bytes": conv_bytes,
        "auxiliary_state_bytes": auxiliary_bytes,
        "total_bytes": main_bytes + index_bytes + recurrent_bytes + conv_bytes + auxiliary_bytes,
    }


def calculate(spec: dict[str, Any], weight_profile: dict[str, Any], kv_profile: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    included_layers = [("main", layer) for layer in spec["layers"]]
    if scenario["include_mtp"]:
        included_layers.extend(("aux", layer) for layer in spec["auxiliary_layers"])

    global_groups = [group for group in spec["global_parameter_groups"] if group_included(group, scenario)]
    global_weight = sum(
        weight_group_bytes(group, weight_profile, component_quantization(group, None, scenario))
        for group in global_groups
    )
    layer_results = []
    for result_kind, layer in included_layers:
        groups = [group for group in layer["parameter_groups"] if group_included(group, scenario)]
        weight_bytes = sum(
            weight_group_bytes(group, weight_profile, component_quantization(group, layer, scenario))
            for group in groups
        )
        state = layer_state_bytes(
            layer, spec["architecture"], kv_profile, scenario["context_length"],
            scenario["batch_size"], scenario["include_ngram"],
        )
        layer_results.append({
            "resultKey": f"{result_kind}:{layer['index']}",
            "index": layer["index"], "label": layer["label"], "kind": layer["kind"],
            "weight_bytes": weight_bytes, "kv_cache": state,
            "logical_params": sum(group["logical_params"] for group in groups),
        })

    total_weight = global_weight + sum(item["weight_bytes"] for item in layer_results)
    total_state = sum(item["kv_cache"]["total_bytes"] for item in layer_results)
    total_params = sum(group["logical_params"] for group in global_groups) + sum(item["logical_params"] for item in layer_results)
    for item in layer_results:
        item["weight_ratio"] = item["weight_bytes"] / total_weight
    return {
        "schema_version": "1.1.0", "model_id": spec["model_id"],
        "source_revision": spec["source"]["revision"], "scenario": scenario,
        "weight_profile": weight_profile["id"], "kv_cache_profile": kv_profile["id"],
        "formula_ids": spec["formula_ids"],
        "totals": {
            "logical_params": total_params, "global_weight_bytes": global_weight,
            "weight_bytes": total_weight, "kv_cache_bytes": total_state,
            "weight_plus_kv_bytes": total_weight + total_state,
        },
        "layers": layer_results,
    }
