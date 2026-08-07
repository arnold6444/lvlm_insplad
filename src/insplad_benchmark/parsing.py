from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedAnomaly:
    anomaly: str | None
    valid: bool


@dataclass(frozen=True)
class ParsedStructured:
    anomaly: str | None
    defect_class: str | None
    valid: bool


def _json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def parse_anomaly_output(text: str, valid_labels: tuple[str, str] = ("normal", "anomalous")) -> ParsedAnomaly:
    normalized = text.strip().lower()
    if normalized in valid_labels:
        return ParsedAnomaly(normalized, True)
    obj = _json_object(text)
    if obj is not None:
        value = str(obj.get("anomaly", "")).strip().lower()
        if value in valid_labels:
            return ParsedAnomaly(value, True)
    return ParsedAnomaly(None, False)


def parse_structured_output(text: str, valid_defect_classes: set[str]) -> ParsedStructured:
    obj = _json_object(text)
    if obj is None:
        return ParsedStructured(None, None, False)
    anomaly = str(obj.get("anomaly", "")).strip().lower()
    defect = str(obj.get("defect_class", "")).strip()
    valid = anomaly in {"normal", "anomalous"} and defect in valid_defect_classes
    return ParsedStructured(anomaly if anomaly else None, defect if defect else None, valid)
