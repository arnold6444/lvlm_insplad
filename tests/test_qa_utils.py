import pytest

from insplad_benchmark.qa_utils import anomaly_from_defect, category_intersection, choose_normal_reference, normalize_label


def test_label_normalization():
    assert normalize_label("  Polymer Insulator ") == "polymer_insulator"
    assert normalize_label("Defect-Type/A") == "defect_type_a"


def test_category_intersection():
    assert category_intersection(["a", "b", "c"], ["b", "c", "d"]) == ["b", "c"]


def test_supervised_defect_to_binary_anomaly():
    assert anomaly_from_defect("normal") == "normal"
    assert anomaly_from_defect("corrosion") == "anomalous"


def test_normal_reference_selection_is_deterministic_and_safe():
    test = {"relative_path": "test/x.jpg", "asset_category": "tower", "split": "test", "label": "rust", "sha256": "same"}
    candidates = [{"relative_path": "train/a.jpg", "asset_category": "tower", "split": "train", "label": "normal", "sha256": "a"}, {"relative_path": "train/b.jpg", "asset_category": "tower", "split": "train", "label": "normal", "sha256": "b"}, {"relative_path": "train/dup.jpg", "asset_category": "tower", "split": "train", "label": "normal", "sha256": "same"}, {"relative_path": "test/c.jpg", "asset_category": "tower", "split": "test", "label": "normal", "sha256": "c"}, {"relative_path": "train/d.jpg", "asset_category": "other", "split": "train", "label": "normal", "sha256": "d"}]
    first = choose_normal_reference(test, candidates, seed=42); second = choose_normal_reference(test, candidates, seed=42)
    assert first == second
    assert first["relative_path"] in {"train/a.jpg", "train/b.jpg"}


def test_reference_selection_fails_without_eligible_normal():
    test = {"relative_path": "test/x.jpg", "asset_category": "tower", "split": "test", "label": "rust", "sha256": "x"}
    with pytest.raises(ValueError):
        choose_normal_reference(test, [], seed=42)
