#!/usr/bin/env python3
"""Generate a reproducible analysis-result.json for a saved scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis_math import calculate


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_profile(document, profile_id):
    for profile in document["profiles"]:
        if profile["id"] == profile_id:
            return profile
    raise KeyError(f"Profile not found: {profile_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="data/scenarios/default.json")
    parser.add_argument("--output", default="data/generated/analysis-result.json")
    args = parser.parse_args()

    spec = load(ROOT / "data/model/qwen3.8-flash-next.model-spec.json")
    scenario = load(ROOT / args.scenario)
    weights = load(ROOT / "data/quantization/weight-profiles.json")
    kv_profiles = load(ROOT / "data/quantization/kv-cache-profiles.json")
    weight_profile = find_profile(weights, scenario["weight_quantization"])
    kv_profile = find_profile(kv_profiles, scenario["kv_cache_quantization"])
    result = calculate(spec, weight_profile, kv_profile, scenario)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
