from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


LBA = _load("loc_box_annotation_t", "loc_box_annotation.py")
BUILD = _load("build_gold_v2_loc_mmad_t", "build_gold_v2_loc_mmad.py")


def _rows() -> list[dict]:
    rows = []
    spec = [("c-normal", "vari-grip", "normal"), ("c-seg", "lightning-rod-suspension", "rust"),
            ("c-a", "vari-grip", "rust"), ("c-b", "glass-insulator", "missing-cap"),
            ("c-c", "yoke-suspension", "bird-nest")]
    for core, asset, defect in spec:
        base = {"core_id": core, "asset_category": asset, "image": f"sup/{core}.png",
                "unsupervised_image": f"unsup/{core}.png", "normal_reference_image": "unsup/good/0.png",
                "anomaly_gt": "normal" if defect == "normal" else "anomalous", "defect_gt": defect}
        rows.append({**base, "id": f"{core}-q1", "question_type": "anomaly_detection"})
        rows.append({**base, "id": f"{core}-q2", "question_type": "defect_classification"})
    return rows


LOC = [
    {"core_id": "c-seg", "localization_gt": {"empty_mask": False, "normalized_bboxes_xywh": [[0.4, 0.4, 0.2, 0.2]]}},
    {"core_id": "c-normal", "localization_gt": {"empty_mask": True, "normalized_bboxes_xywh": []}},
]


def test_targets_skip_normal_and_seg_labelled_images():
    targets = LBA.annotation_targets(_rows(), LOC)
    assert [t["core_id"] for t in targets] == ["c-b", "c-a", "c-c"]   # sorted by asset; no normal, no seg
    review = LBA.annotation_targets(_rows(), LOC, include_seg=True)
    assert any(t["core_id"] == "c-seg" and t["seg_boxes"] for t in review)


def test_validate_boxes_clamps_and_drops_clicks():
    assert LBA.validate_boxes([[0.9, 0.9, 0.5, 0.5], [0.1, 0.1, 0.0005, 0.3]]) == [[0.9, 0.9, 0.1, 0.1]]
    with pytest.raises(ValueError):
        LBA.validate_boxes([[0.1, 0.2, 0.3]])


def _png(path: Path, size=(300, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 90, 90)).save(path)
    return path


def test_session_item_save_resume(tmp_path):
    targets = LBA.annotation_targets(_rows(), LOC)
    paths = {t["image"]: _png(tmp_path / t["image"]) for t in targets}
    store = LBA.AnnotationStore(tmp_path / "ann" / "annotations.json", annotator="tester")
    session = LBA.AnnotationSession(targets, paths, store, drafts={"c-a": [[0.1, 0.1, 0.2, 0.2]]})
    item = session.item(0)
    assert item["image_url"].startswith("data:image/jpeg;base64,") and item["width"] == 300
    assert item["first_todo"] == 0 and item["done"] == 0
    with pytest.raises(ValueError):
        session.save(0, [], "boxed")                       # boxed needs a box
    r = session.save(0, [[0.1, 0.2, 0.3, 0.4]], "boxed", "edge of cap")
    assert r["done"] == 1 and r["first_todo"] == 1
    session.save(1, [], "unlocatable", "blurry")
    # resume from disk
    again = LBA.AnnotationSession(targets, paths, LBA.AnnotationStore(store.path), {})
    assert again.progress() == {"total": 3, "done": 2, "first_todo": 2}
    assert again.item(0)["boxes"] == [[0.1, 0.2, 0.3, 0.4]]
    assert session.item(1)["draft"] == [[0.1, 0.1, 0.2, 0.2]]   # index 1 is c-a after sorting
    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert data["c-b"]["annotator"] == "tester" and data["c-a"]["status"] == "unlocatable"


def test_draft_boxes_from_manifest():
    rows = [{"expert": "patchcore", "query_id": "c-a", "roi_left": "30", "roi_top": "20",
             "roi_right": "90", "roi_bottom": "80"}, {"expert": "dinomaly", "query_id": "c-a"}]
    drafts = LBA.draft_boxes_from_manifest(rows, "patchcore", {"c-a": (300, 200)})
    assert drafts == {"c-a": [[0.1, 0.1, 0.2, 0.3]]}


def test_build_rows_mmad_style():
    ann = {"c-a": {"status": "boxed", "boxes_xywh": [[0.75, 0.05, 0.2, 0.2]], "annotator": "t"},
           "c-b": {"status": "unlocatable", "note": "cap not visible"},
           "c-c": {"status": "boxed", "boxes_xywh": [[0.0, 0.7, 0.2, 0.25]]}}
    out, summary = BUILD.build_rows(_rows(), LOC, ann)
    by_core = {r["core_id"]: r for r in out}
    assert set(by_core) == {"c-seg", "c-a", "c-c"}          # normal excluded, unlocatable excluded
    assert summary["excluded_unlocatable"][0]["core_id"] == "c-b"
    for r in out:
        assert r["question"].startswith("There is a defect") and len(r["options"]) == 4
        assert "No localized defect region is present." not in r["options"].values()
        assert r["options"][r["answer"]] == r["answer_text"]
        assert r["id"].endswith("-q3m") and r["normal_reference_image"]
    assert by_core["c-seg"]["localization_gt"]["basis"] == "insplad_seg_instance_annotation"
    assert by_core["c-seg"]["answer_text"] == "Center region."
    assert by_core["c-a"]["answer_text"] == "Upper-right region."
    assert by_core["c-c"]["answer_text"] == "Lower-left region."
    again, _ = BUILD.build_rows(_rows(), LOC, ann)
    assert again == out                                        # deterministic


def test_build_rows_requires_all_boxes_unless_partial():
    with pytest.raises(RuntimeError):
        BUILD.build_rows(_rows(), LOC, {})
    out, summary = BUILD.build_rows(_rows(), LOC, {}, allow_partial=True)
    assert [r["core_id"] for r in out] == ["c-seg"] and len(summary["missing_annotation"]) == 3


def test_annotation_notebook_is_cpu_only_and_publishes_only_frozen():
    nb = json.loads((ROOT / "notebooks" / "lvlm_gold_v2_loc_box_annotation_colab.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"loc_nb:{i}", "exec")
    code = "\n".join(cells)
    assert "lba.register_colab(session)" in code and "build_gold_v2_loc_mmad.py" in code
    assert "import torch" not in code and '"pip"' not in code
    assert 'summary.get("quality_status") == "FROZEN"' in code
    assert '"--prefix", "gold_v2_loc_mmad"' in code
