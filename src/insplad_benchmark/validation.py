from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class ValidationResult:
    checked: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(value)
    return rows


def validate_qa_rows(rows: Iterable[dict], known_assets: set[str] | None = None, known_defects: set[str] | None = None, image_meta: dict[str, dict] | None = None) -> ValidationResult:
    result = ValidationResult()
    seen_ids: set[str] = set()
    image_meta = image_meta or {}

    for row in rows:
        result.checked += 1
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            result.errors.append(f"row {result.checked}: missing id")
        elif row_id in seen_ids:
            result.errors.append(f"duplicate id: {row_id}")
        seen_ids.add(row_id)

        image = str(row.get("image", "")).strip()
        asset = str(row.get("asset_category", "")).strip()
        if not image:
            result.errors.append(f"{row_id or result.checked}: missing image")
        if known_assets is not None and asset not in known_assets:
            result.errors.append(f"{row_id or result.checked}: unknown asset: {asset}")

        if image_meta and image:
            meta = image_meta.get(image)
            if meta is None:
                result.errors.append(f"{row_id or result.checked}: image not found in index: {image}")
            elif asset and meta.get("asset_category") and meta.get("asset_category") != asset:
                result.errors.append(f"{row_id or result.checked}: image asset mismatch")

        task = row.get("task")
        if task == "structured_explainability":
            answer = row.get("answer") or {}
            anomaly = answer.get("anomaly")
            defect = answer.get("defect_class")
            if known_defects is not None and defect not in known_defects:
                result.errors.append(f"{row_id or result.checked}: unknown defect: {defect}")
            expected_anomaly = "normal" if defect == "normal" else "anomalous"
            if defect and anomaly != expected_anomaly:
                result.errors.append(f"{row_id or result.checked}: GT inconsistency ({anomaly}, {defect})")

        reference = str(row.get("reference_image", "")).strip()
        if reference:
            if reference == image:
                result.errors.append(f"{row_id or result.checked}: reference equals test image")
            ref_meta = image_meta.get(reference) if image_meta else None
            test_meta = image_meta.get(image) if image_meta else None
            if image_meta and ref_meta is None:
                result.errors.append(f"{row_id or result.checked}: reference not found in index: {reference}")
            if ref_meta:
                if ref_meta.get("label") != "normal":
                    result.errors.append(f"{row_id or result.checked}: reference is not normal")
                if asset and ref_meta.get("asset_category") and ref_meta.get("asset_category") != asset:
                    result.errors.append(f"{row_id or result.checked}: reference category mismatch")
                if str(ref_meta.get("split", "")).lower() == "test":
                    result.errors.append(f"{row_id or result.checked}: test split used as reference")
            if ref_meta and test_meta and ref_meta.get("sha256") and ref_meta.get("sha256") == test_meta.get("sha256"):
                result.errors.append(f"{row_id or result.checked}: duplicate hash leakage")
    return result
