#!/usr/bin/env python3
"""Local-only static server with safe JSON scenario persistence."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "data" / "scenarios"
ANALYSIS_DIR = ROOT / "data" / "generated"
DOCS_DIR = ROOT / "docs"
SAFE_NAME = re.compile(r"^[\w.\-\u4e00-\u9fff]{1,80}$", re.UNICODE)
SAFE_DOC_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
MAX_BODY_BYTES = 5 * 1024 * 1024
DOC_ORDER = {
    "overview": 0,
    "quick-start": 1,
    "model-architecture-mapping": 2,
    "gated-residual-and-gdn": 3,
    "operator-reference": 4,
    "sparse-moe-operators": 5,
    "ngram-ple-operators": 6,
    "qwen-sparse-attention-operators": 7,
    "mtp-operators": 8,
    "hybrid-state-and-qsa": 9,
    "calculation-methodology": 10,
    "quantization-guide": 11,
    "data-format": 12,
    "validation-and-provenance": 13,
    "assumptions-and-limitations": 14,
}


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


class ExplorerHandler(SimpleHTTPRequestHandler):
    server_version = "Qwen38FlashExplorer/1.0"

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self.send_json({"status": "ok", "root": str(ROOT)})
        if path == "/api/scenarios":
            return self.list_scenarios()
        if path == "/api/docs":
            return self.list_docs()
        if path.startswith("/api/docs/"):
            return self.get_doc(unquote(path[len("/api/docs/"):]))
        if path.startswith("/api/scenarios/"):
            return self.get_scenario(unquote(path[len("/api/scenarios/"):]))
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/scenarios":
            return self.send_error(HTTPStatus.NOT_FOUND)
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            return self.send_json({"error": "request body size is invalid"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            payload = json.loads(self.rfile.read(length))
            name = str(payload["name"]).strip()
            scenario = payload["scenario"]
            analysis = payload["analysis"]
            self.validate_name(name)
            self.validate_scenario(scenario)
            if not isinstance(analysis, dict) or analysis.get("model_id") != "qwen3.8-flash-next":
                raise ValueError("analysis model_id is invalid")
            scenario_path = SCENARIO_DIR / f"{name}.json"
            analysis_path = ANALYSIS_DIR / f"{name}.analysis.json"
            atomic_write_json(scenario_path, scenario)
            atomic_write_json(analysis_path, analysis)
            return self.send_json({
                "status": "saved",
                "scenario_path": scenario_path.relative_to(ROOT).as_posix(),
                "analysis_path": analysis_path.relative_to(ROOT).as_posix(),
            }, HTTPStatus.CREATED)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def list_scenarios(self):
        SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
        scenarios = []
        for path in sorted(SCENARIO_DIR.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                scenarios.append({
                    "name": path.stem,
                    "path": path.relative_to(ROOT).as_posix(),
                    "context_length": value.get("context_length"),
                    "weight_quantization": value.get("weight_quantization"),
                    "kv_cache_quantization": value.get("kv_cache_quantization"),
                })
            except (OSError, json.JSONDecodeError):
                continue
        return self.send_json({"scenarios": scenarios})

    def get_scenario(self, name: str):
        try:
            self.validate_name(name)
        except ValueError as error:
            return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        path = SCENARIO_DIR / f"{name}.json"
        if not path.is_file():
            return self.send_json({"error": "scenario not found"}, HTTPStatus.NOT_FOUND)
        try:
            scenario = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.send_json({"error": "scenario file is invalid"}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return self.send_json({"name": name, "scenario": scenario})

    def list_docs(self):
        documents = []
        paths = sorted(DOCS_DIR.glob("*.md"), key=lambda path: (DOC_ORDER.get(path.stem, 999), path.name))
        for path in paths:
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            title = next(
                (line[2:].strip() for line in content.splitlines() if line.startswith("# ")),
                path.stem,
            )
            documents.append({
                "slug": path.stem,
                "title": title,
                "path": path.relative_to(ROOT).as_posix(),
            })
        return self.send_json({"documents": documents})

    def get_doc(self, name: str):
        if not SAFE_DOC_NAME.fullmatch(name):
            return self.send_json({"error": "document name is invalid"}, HTTPStatus.BAD_REQUEST)
        path = DOCS_DIR / f"{name}.md"
        if not path.is_file():
            return self.send_json({"error": "document not found"}, HTTPStatus.NOT_FOUND)
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return self.send_json({"error": "document cannot be read"}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return self.send_json({"slug": name, "path": path.relative_to(ROOT).as_posix(), "content": content})

    @staticmethod
    def validate_name(name: str):
        if not SAFE_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError("name may only contain letters, numbers, Chinese characters, dot, dash and underscore")

    @staticmethod
    def validate_scenario(value):
        if not isinstance(value, dict):
            raise ValueError("scenario must be an object")
        required = {
            "schema_version", "name", "model_id", "weight_quantization", "kv_cache_quantization",
            "context_length", "batch_size", "include_ngram", "ngram_quantization",
            "include_vision", "vision_quantization", "include_mtp", "mtp_quantization", "operator_detail"
        }
        missing = required - value.keys()
        if missing:
            raise ValueError(f"scenario is missing: {', '.join(sorted(missing))}")
        if value["model_id"] != "qwen3.8-flash-next":
            raise ValueError("scenario model_id is invalid")
        if not isinstance(value["context_length"], int) or not 1 <= value["context_length"] <= 1000000:
            raise ValueError("context_length is outside 1..1000000")
        if not isinstance(value["batch_size"], int) or not 1 <= value["batch_size"] <= 1024:
            raise ValueError("batch_size is outside 1..1024")
        if value["operator_detail"] not in {"module", "expanded"}:
            raise ValueError("operator_detail is invalid")
        component_profiles = {"inherit", "bf16", "fp8", "w4-effective-425", "w4-raw"}
        for key in ("ngram_quantization", "vision_quantization", "mtp_quantization"):
            if value[key] not in component_profiles:
                raise ValueError(f"{key} is invalid")

    def send_json(self, value, status=HTTPStatus.OK):
        body = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        print(f"[{self.log_date_time_string()}] {format_string % args}")


def main():
    parser = argparse.ArgumentParser(description="Serve Qwen3.8 Flash Next Explorer locally")
    parser.add_argument("--host", default="192.168.64.60")
    parser.add_argument("--port", type=int, default=20026)
    args = parser.parse_args()
    handler = partial(ExplorerHandler, directory=str(ROOT))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Qwen3.8 Flash Next Explorer: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
