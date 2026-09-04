from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class AgentDecision:
    anomaly: str | None
    confidence: float | None
    request_tool: str | None
    valid: bool


def _json_object(text: str) -> dict | None:
    text = str(text).strip()
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_agent_decision(text: str, allowed_tools: Iterable[str]) -> AgentDecision:
    obj = _json_object(text)
    if obj is None:
        return AgentDecision(None, None, None, False)
    anomaly = str(obj.get("anomaly", "")).strip().lower()
    tool = str(obj.get("request_tool", "")).strip().lower()
    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    allowed = set(allowed_tools)
    valid_confidence = confidence is not None and math.isfinite(confidence) and 0.0 <= confidence <= 1.0
    valid = anomaly in {"normal", "anomalous"} and tool in allowed and valid_confidence
    return AgentDecision(
        anomaly if anomaly in {"normal", "anomalous"} else None,
        confidence if valid_confidence else None,
        tool if tool in allowed else None,
        valid,
    )


def strong_patchcore_conflict(
    anomaly: str | None,
    percentile: float,
    *,
    low: float = 10.0,
    high: float = 90.0,
) -> bool:
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("expected 0 <= low < high <= 100")
    if anomaly == "normal":
        return percentile >= high
    if anomaly == "anomalous":
        return percentile <= low
    return True


def should_route(
    decision: AgentDecision,
    *,
    expected_tool: str,
    patchcore_percentile: float,
    min_confidence: float = 0.70,
    conflict_low: float = 10.0,
    conflict_high: float = 90.0,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not decision.valid:
        reasons.append("invalid_output")
    else:
        if decision.request_tool == expected_tool:
            reasons.append("model_requested_tool")
        if decision.confidence is not None and decision.confidence < min_confidence:
            reasons.append("low_confidence")
        if strong_patchcore_conflict(
            decision.anomaly,
            patchcore_percentile,
            low=conflict_low,
            high=conflict_high,
        ):
            reasons.append("patchcore_conflict")
    return bool(reasons), reasons


def classification_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    tp = fp = tn = fn = invalid = 0
    for row in rows:
        gt = str(row.get("answer", ""))
        pred = str(row.get("final_prediction", ""))
        if pred not in {"normal", "anomalous"}:
            invalid += 1
            continue
        if gt == "anomalous" and pred == "anomalous": tp += 1
        elif gt == "normal" and pred == "anomalous": fp += 1
        elif gt == "normal" and pred == "normal": tn += 1
        elif gt == "anomalous" and pred == "normal": fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bacc = 0.5 * (recall + specificity)
    n = len(rows)
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "invalid_n": invalid,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1, "balanced_accuracy": bacc,
        "invalid_output_rate": invalid / n if n else 0.0,
    }
