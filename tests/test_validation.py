from insplad_benchmark.validation import validate_aligned_anomaly_sets, validate_qa_rows


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
    assert any("reference is not from train split" in e for e in result.errors)


def test_retrieved_reference_requires_train_provenance_and_finite_cosine():
    rows = [{
        "id": "x", "task": "anomaly_detection", "image": "a/test/good/x.jpg",
        "asset_category": "a", "answer": "normal",
        "reference_image": "a/val/good/ref.jpg", "reference_split": "val",
        "reference_source_label_raw": "good", "retrieval_method": "dinov2_cosine_top1",
        "reference_similarity": "nan",
    }]
    meta = {
        "a/test/good/x.jpg": {"asset_category": "a", "split": "test", "label": "normal", "content_fingerprint": "1:x"},
        "a/val/good/ref.jpg": {"asset_category": "a", "split": "val", "label": "normal", "source_label_raw": "good", "content_fingerprint": "2:y"},
    }
    result = validate_qa_rows(rows, {"a"}, None, meta)
    assert not result.ok
    assert any("not from train split" in error for error in result.errors)
    assert any("reference_split must be train" in error for error in result.errors)
    assert any("outside cosine range" in error for error in result.errors)


def test_retrieved_reference_metadata_is_validated_against_manifest():
    rows = [{
        "id": "x", "task": "anomaly_detection", "image": "a/test/good/x.jpg",
        "asset_category": "a", "answer": "normal",
        "reference_image": "a/train/good/ref.jpg", "reference_split": "train",
        "reference_source_label_raw": "normal", "retrieval_method": "dinov2_cosine_top1",
        "reference_similarity": 0.75,
    }]
    meta = {
        "a/test/good/x.jpg": {"asset_category": "a", "split": "test", "label": "normal", "content_fingerprint": "1:x"},
        "a/train/good/ref.jpg": {"asset_category": "a", "split": "train", "label": "normal", "source_label_raw": "good", "content_fingerprint": "2:y"},
    }
    result = validate_qa_rows(rows, {"a"}, None, meta)
    assert not result.ok
    assert any("reference_source_label_raw metadata mismatch" in error for error in result.errors)


def test_retrieved_reference_rejects_global_test_fingerprint_and_bad_query_contract():
    rows = [{
        "id": "x", "task": "anomaly_detection", "image": "a/train/good/x.jpg",
        "asset_category": "a", "answer": "anomalous",
        "reference_image": "a/train/good/ref.jpg", "reference_split": "train",
        "reference_source_label_raw": "good", "retrieval_method": "dinov2_cosine_top1",
        "retrieval_model_id": "facebook/dinov2-base",
        "retrieval_model_revision": "immutable-commit",
        "retrieval_embedding": "last_hidden_state_cls_l2",
        "reference_similarity": 0.75,
    }]
    meta = {
        "a/train/good/x.jpg": {"asset_category": "a", "split": "train", "label": "normal", "content_fingerprint": "1:x"},
        "a/train/good/ref.jpg": {"asset_category": "a", "split": "train", "label": "normal", "source_label_raw": "good", "content_fingerprint": "2:cross"},
        "b/test/good/copy.jpg": {"asset_category": "b", "split": "test", "label": "normal", "content_fingerprint": "2:cross"},
    }

    result = validate_qa_rows(rows, {"a"}, None, meta)

    assert any("query is not from test split" in error for error in result.errors)
    assert any("GT differs from manifest" in error for error in result.errors)
    assert any("reference fingerprint occurs in test split" in error for error in result.errors)


def test_anomaly_alignment_requires_exact_order():
    first = [
        {"id": "a", "image": "a.jpg", "asset_category": "asset", "answer": "normal"},
        {"id": "b", "image": "b.jpg", "asset_category": "asset", "answer": "anomalous"},
    ]

    assert validate_aligned_anomaly_sets(first, list(reversed(first)))
