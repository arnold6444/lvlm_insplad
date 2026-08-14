from collections import Counter

import pytest

from insplad_benchmark.sampling import (
    balanced_binary_sample,
    balanced_structured_sample,
    matched_one_shot_sample,
)


def _binary_rows():
    rows = []
    for asset in ("a", "b", "c"):
        for label in ("normal", "anomalous"):
            for i in range(30):
                rows.append({
                    "id": f"{asset}-{label}-{i}",
                    "asset_category": asset,
                    "answer": label,
                })
    return rows


def test_balanced_binary_sample_is_exactly_balanced_and_deterministic():
    first = balanced_binary_sample(_binary_rows(), 100, seed=42)
    second = balanced_binary_sample(_binary_rows(), 100, seed=42)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert Counter(r["answer"] for r in first) == {"normal": 50, "anomalous": 50}
    assets = Counter(r["asset_category"] for r in first)
    assert max(assets.values()) - min(assets.values()) <= 2


def test_balanced_binary_sample_rejects_insufficient_class():
    rows = [
        {"id": f"n-{i}", "asset_category": "a", "answer": "normal"}
        for i in range(20)
    ] + [
        {"id": f"a-{i}", "asset_category": "a", "answer": "anomalous"}
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="not enough anomalous"):
        balanced_binary_sample(rows, 10)


def test_matched_one_shot_preserves_zero_shot_order():
    zero = [
        {"id": "b", "asset_category": "a", "answer": "normal"},
        {"id": "a", "asset_category": "a", "answer": "anomalous"},
    ]
    one = [
        {"id": "a", "reference_image": "ref.jpg"},
        {"id": "b", "reference_image": "ref.jpg"},
    ]
    assert [r["id"] for r in matched_one_shot_sample(zero, one)] == ["b", "a"]


def test_balanced_structured_sample_spreads_assets_and_defects():
    rows = []
    labels_by_asset = {
        "a": ("normal", "rust", "bird-nest"),
        "b": ("normal", "crack"),
        "c": ("normal", "rust"),
    }
    for asset, labels in labels_by_asset.items():
        for label in labels:
            for i in range(30):
                rows.append({
                    "id": f"{asset}-{label}-{i}",
                    "asset_category": asset,
                    "answer": {
                        "anomaly": "normal" if label == "normal" else "anomalous",
                        "defect_class": label,
                    },
                })
    sample = balanced_structured_sample(rows, 90, seed=42)
    assert Counter(r["asset_category"] for r in sample) == {"a": 30, "b": 30, "c": 30}
    for asset in labels_by_asset:
        seen = {
            r["answer"]["defect_class"]
            for r in sample
            if r["asset_category"] == asset
        }
        assert seen == set(labels_by_asset[asset])
