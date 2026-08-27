#!/usr/bin/env python3
"""Validate official snapshots, model mapping and capacity formulas."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from analysis_math import calculate


ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    config = load("data/sources/qwen3.8-flash-next/config.json")
    source = load("data/sources/qwen3.8-flash-next/source-info.json")
    spec = load("data/model/qwen3.8-flash-next.model-spec.json")
    scenario = load("data/scenarios/default.json")
    weights = load("data/quantization/weight-profiles.json")
    kv_profiles = load("data/quantization/kv-cache-profiles.json")

    for metadata in source["files"].values():
        assert sha256(ROOT / metadata["path"]) == metadata["local_sha256"]
    text = config["text_config"]
    assert len(spec["layers"]) == text["num_hidden_layers"] == 48
    assert len(spec["auxiliary_layers"]) == text["mtp_num_hidden_layers"] == 1
    assert sum(layer["attention_type"] == "linear_attention" for layer in spec["layers"]) == 36
    assert sum(layer["attention_type"] != "linear_attention" for layer in spec["layers"]) == 12
    assert sum(layer["has_ple"] for layer in spec["layers"]) == 1
    mapped_elements = (
        spec["global_parameter_summary"]["logical_params"]
        + sum(layer["parameter_summary"]["logical_params"] for layer in spec["layers"])
        + sum(layer["parameter_summary"]["logical_params"] for layer in spec["auxiliary_layers"])
    )
    assert mapped_elements == spec["architecture"]["checkpoint_elements"] == 179_999_981_459

    weight_profile = next(item for item in weights["profiles"] if item["id"] == scenario["weight_quantization"])
    kv_profile = next(item for item in kv_profiles["profiles"] if item["id"] == scenario["kv_cache_quantization"])
    result = calculate(spec, weight_profile, kv_profile, scenario)
    assert result["totals"]["logical_params"] == 179_999_981_459
    assert result["totals"]["weight_bytes"] == spec["architecture"]["checkpoint_total_bytes"]
    assert result["totals"]["kv_cache_bytes"] > 0
    layer_share = sum(item["weight_ratio"] for item in result["layers"])
    global_share = result["totals"]["global_weight_bytes"] / result["totals"]["weight_bytes"]
    assert abs(layer_share + global_share - 1) < 1e-9

    component_fields = ("ngram_quantization", "vision_quantization", "mtp_quantization")
    for field in component_fields:
        assert scenario[field] == "inherit"
        quantized_scenario = {**scenario, field: "w4-raw"}
        quantized = calculate(spec, weight_profile, kv_profile, quantized_scenario)
        assert quantized["totals"]["logical_params"] == result["totals"]["logical_params"]
        assert quantized["totals"]["weight_bytes"] < result["totals"]["weight_bytes"]
        assert quantized["totals"]["kv_cache_bytes"] == result["totals"]["kv_cache_bytes"]

    generated = load("data/generated/analysis-result.json")
    assert generated["totals"] == result["totals"]

    docs_dir = ROOT / "docs"
    required_docs = {
        "overview.md", "quick-start.md", "model-architecture-mapping.md",
        "gated-residual-and-gdn.md", "operator-reference.md", "sparse-moe-operators.md",
        "ngram-ple-operators.md", "qwen-sparse-attention-operators.md", "mtp-operators.md",
        "hybrid-state-and-qsa.md",
        "calculation-methodology.md", "quantization-guide.md",
        "data-format.md", "validation-and-provenance.md", "assumptions-and-limitations.md",
    }
    assert required_docs <= {path.name for path in docs_dir.glob("*.md")}
    for path in docs_dir.glob("*.md"):
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# "), f"{path.name} must start with an H1"
        for target in re.findall(r"\[[^]]+\]\(([^)]+\.md)\)", content):
            assert (path.parent / target).is_file(), f"Broken documentation link: {path.name} -> {target}"
    print("Validation passed")
    print(f"Stored elements: {result['totals']['logical_params']:,}")
    print(f"BF16 checkpoint bytes: {result['totals']['weight_bytes']:,.0f}")
    print(f"State + KV bytes: {result['totals']['kv_cache_bytes']:,.0f}")


if __name__ == "__main__":
    main()
