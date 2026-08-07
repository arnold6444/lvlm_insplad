from __future__ import annotations

import random
import re
from typing import Iterable, Mapping, Sequence


def normalize_label(value: str) -> str:
    """Normalize a label for benchmark keys while preserving source labels elsewhere."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def category_intersection(left: Iterable[str], right: Iterable[str]) -> list[str]:
    return sorted(set(left) & set(right))


def anomaly_from_defect(defect_class: str, normal_label: str = "normal") -> str:
    return "normal" if defect_class == normal_label else "anomalous"


def choose_normal_reference(test_sample: Mapping[str, str], candidates: Sequence[Mapping[str, str]], *, seed: int = 42, normal_label: str = "normal") -> Mapping[str, str]:
    """Choose a deterministic same-asset normal, non-test, non-duplicate reference."""
    eligible = []
    test_path = test_sample.get("relative_path", "")
    test_hash = test_sample.get("sha256", "")
    for candidate in candidates:
        if candidate.get("asset_category") != test_sample.get("asset_category"):
            continue
        if candidate.get("label") != normal_label:
            continue
        if str(candidate.get("split", "")).lower() == "test":
            continue
        if candidate.get("relative_path") == test_path:
            continue
        if test_hash and candidate.get("sha256") == test_hash:
            continue
        eligible.append(candidate)
    if not eligible:
        raise ValueError(f"no eligible normal reference for asset {test_sample.get('asset_category')!r}")
    eligible.sort(key=lambda row: str(row.get("relative_path", "")))
    return random.Random(seed).choice(eligible)
