from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import dev_part_crop_annotation as ann
import run_dev_part_crop_diagnostic as runner
import test_gold_v2_verify_agent as fixture


def workspace(tmp_path):
    rows = fixture._rows()
    first, io = fixture._stage(rows)
    _, qa = fixture._inputs(tmp_path, rows, first, io)
    out = tmp_path / "gold_v2_part_crop_diagnostic_test"
    selected, pairs = ann.prepare(qa, tmp_path / "retrieved_refs.csv", tmp_path / "sup.zip", tmp_path / "unsup.zip", out, 4)
    return selected, pairs, out


def test_annotation_is_blind_validated_persistent_and_resumable(tmp_path):
    rows, pairs, root = workspace(tmp_path)
    session = ann.PartSession(pairs, root / "annotations.json")
    item = session.item(0)
    assert set(item) == {"index", "total", "done", "first_todo", "asset_category", "status", "note", "target", "reference"}
    assert set(item["target"]) == {"image_url", "box"}
    with pytest.raises(ValueError, match="finish"):
        session.completed()
    for bad in (None, [0, 0, 0, 1], [-.1, 0, .5, .5], [0, 0, float("nan"), .5], [0, 0, 2, 1]):
        with pytest.raises(ValueError):
            session.save(0, [.1, .1, .8, .8], bad)
    assert not (root / "annotations.json").exists()
    session.save(0, [0, 0, 1, 1], [.25, .25, .5, .5])
    reopened = ann.PartSession(pairs, root / "annotations.json")
    assert reopened.progress()["first_todo"] == 1 and reopened.progress()["done"] == 1
    assert reopened.item(0)["target"]["box"] == [.25, .25, .5, .5]
    with pytest.raises(ValueError, match="reason"):
        reopened.save(1, None, None, "unavailable")
    for i in range(1, len(pairs)):
        reopened.save(i, None, None, "unavailable", "matching part not visible")
    good, excluded = reopened.completed()
    assert len(good) == 1 and len(excluded) == 3
    changed = [dict(p) for p in pairs]
    changed[0]["pair_identity"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        ann.PartSession(changed, root / "annotations.json")
    sentinel = root / "annotations.json"
    before = sentinel.read_bytes()
    with pytest.raises(ValueError, match="different dev subset"):
        ann.prepare(tmp_path / "dev_master_qa.jsonl", tmp_path / "retrieved_refs.csv", tmp_path / "sup.zip", tmp_path / "unsup.zip", root, 4, 99)
    assert sentinel.read_bytes() == before


def test_crop_geometry_and_source_change_are_checked(tmp_path):
    _, pairs, root = workspace(tmp_path)
    crop = runner.crop_pair(pairs[0], {"reference_box": [0, 0, 1, 1], "target_box": [.25, .2, .5, .6]}, root / "crops")
    with Image.open(crop[0]) as im:
        assert im.size == (40, 30)
    with Image.open(crop[1]) as im:
        assert im.size == (20, 18)
    Image.new("RGB", (40, 30), "red").save(pairs[0]["target"]["path"])
    with pytest.raises(ValueError, match="changed"):
        runner.crop_pair(pairs[0], {"reference_box": [0, 0, 1, 1], "target_box": [.25, .2, .5, .6]}, root / "crops")


def test_paired_inference_error_retry_annotation_guard_and_preservation(tmp_path):
    rows, pairs, root = workspace(tmp_path)
    session = ann.PartSession(pairs, root / "annotations.json")
    session.save(0, None, None, "unavailable", "occluded")
    for i in range(1, len(pairs)):
        session.save(i, [.1, .1, .8, .8], [.25, .2, .5, .6])
    legacy = tmp_path / "gold_v2_verify_agent_dev" / "frozen_config.json"
    legacy.parent.mkdir()
    legacy.write_text("preserve")
    argv = ["--workspace", str(root), "--stage1-dir", str(tmp_path / "stage1"), "--models", fixture.MODEL, "--n-boot", "30"]
    failed, calls = False, []
    fake = fixture._fake_infer(rows, [])

    def infer(adapter, images, prompt):
        nonlocal failed
        calls.append((images, prompt))
        if not failed:
            failed = True
            raise RuntimeError("test inference failure")
        return fake(adapter, images, prompt)

    assert runner.main(argv, loader=lambda _: {}, infer=infer) == 1
    out = root / "inference"
    statuses = json.loads((out / "run_status.json").read_text())
    assert sum(r["row_errors"] for r in statuses) == 1
    assert len(calls) == 12
    assert [p for _, p in calls[:6]] == [p for _, p in calls[6:]]
    assert all("previous answer" not in p and "second look" not in p for _, p in calls)
    assert all(Path(paths[0]).name == "other.png" for paths, _ in calls[:6])
    assert all(Path(paths[0]).suffix == ".png" and Path(paths[0]).name != "other.png" for paths, _ in calls[6:])
    coverage = json.loads((out / "coverage.json").read_text())
    assert coverage["included_images"] == 3 and coverage["excluded_images"] == 1
    assert {r["condition"] for r in csv.DictReader((out / "leaderboard.csv").open())} == {"part_crop"}
    calls.clear()
    assert runner.main(argv, loader=lambda _: {}, infer=infer) == 0 and len(calls) == 1
    calls.clear()
    assert runner.main(argv, loader=lambda _: {}, infer=infer) == 0 and calls == []
    assert legacy.read_text() == "preserve" and not (out / "frozen_config.json").exists()
    previous_meta = (out / "run_metadata.json").read_bytes()
    session.save(1, [.1, .1, .8, .8], [.1, .1, .8, .8])
    with pytest.raises(ValueError, match="changed after inference"):
        runner.main(argv, loader=lambda _: {}, infer=infer)
    assert previous_meta == (out / "run_metadata.json").read_bytes()


def test_notebook_compiles_and_has_annotation_gate():
    nb = json.loads((ROOT / "notebooks/lvlm_gold_v2_part_crop_diagnostic_colab.ipynb").read_text(encoding="utf-8"))
    code = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            compile(src, f"cell{i}", "exec")
            code.append(src)
    text = "\n".join(code)
    assert "register_colab" in text and "session.completed()" in text
    assert "run_dev_part_crop_diagnostic.py" in text
    assert "publish_results_to_github" not in text and "make_frozen_config" not in text
