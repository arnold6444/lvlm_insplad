#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.validation import read_jsonl, validate_qa_rows


def load_set(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = list(value.values()) if all(isinstance(v, str) for v in value.values()) else list(value.keys())
    return set(map(str, value))


def load_index(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["relative_path"]: row for row in csv.DictReader(f)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("qa", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--defects", type=Path)
    parser.add_argument("--image-index", type=Path, help="CSV columns: relative_path, asset_category, split, label, sha256")
    parser.add_argument("--report", type=Path, default=Path("reports/qa_validation.md"))
    args = parser.parse_args()

    try:
        rows = read_jsonl(args.qa)
    except ValueError as exc:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"# QA Validation\n\n- Status: FAIL\n- Error: `{exc}`\n", encoding="utf-8")
        print(exc, file=sys.stderr)
        return 1

    result = validate_qa_rows(rows, load_set(args.assets), load_set(args.defects), load_index(args.image_index))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# QA Validation", "", f"- Status: {'PASS' if result.ok else 'FAIL'}", f"- Rows checked: {result.checked}", f"- Errors: {len(result.errors)}", f"- Warnings: {len(result.warnings)}", ""]
    if result.errors:
        lines += ["## Errors", *[f"- {e}" for e in result.errors], ""]
    if result.warnings:
        lines += ["## Warnings", *[f"- {w}" for w in result.warnings], ""]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    return 0 if result.ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
