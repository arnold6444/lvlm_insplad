from insplad_benchmark.validation import validate_qa_rows


def test_reference_rules_and_gt_consistency():
    rows = [{"id": "x1", "task": "structured_explainability", "image": "test/a.jpg", "asset_category": "asset-a", "reference_image": "train/ref.jpg", "answer": {"anomaly": "anomalous", "defect_class": "rust"}}]
    meta = {"test/a.jpg": {"asset_category": "asset-a", "split": "test", "label": "rust", "sha256": "1"}, "train/ref.jpg": {"asset_category": "asset-a", "split": "train", "label": "normal", "sha256": "2"}}
    result = validate_qa_rows(rows, {"asset-a"}, {"normal", "rust"}, meta)
    assert result.ok


def test_duplicate_reference_and_hash_leakage_fail():
    rows = [{"id": "x", "task": "anomaly_detection", "image": "same.jpg", "reference_image": "same.jpg", "asset_category": "a", "answer": "normal"}]
    meta = {"same.jpg": {"asset_category": "a", "split": "test", "label": "normal", "sha256": "abc"}}
    result = validate_qa_rows(rows, {"a"}, None, meta)
    assert not result.ok
    assert any("reference equals test image" in e for e in result.errors)
    assert any("test split used as reference" in e for e in result.errors)
