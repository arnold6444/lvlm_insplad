import importlib.util
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_missing_component_probe.py"
spec = importlib.util.spec_from_file_location("probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def png(color, note=""):
    b = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("note", note)
    Image.new("RGB", (8, 12), color).save(b, format="PNG", pnginfo=metadata)
    return b.getvalue()


def inputs(tmp_path):
    sup, unsup = tmp_path / "sup.zip", tmp_path / "unsup.zip"
    dev, gold, train = [], [], []
    with ZipFile(sup, "w") as z:
        for day in range(1, 7):
            for y, cls in [(0, "good"), (1, "missing-cap")]:
                for i in range(2):
                    n = f"defect_supervised/glass-insulator/train/{cls}/Fotos {day:02}-12-2020_{y}_{i}.png"
                    data = png((30 + 160*y, day*10, i*10))
                    z.writestr(n, data)
                    train.append((n, y, data))
        for y, cls in [(0, "good"), (1, "missing-cap")]:
            for i in range(2):
                n = f"defect_supervised/glass-insulator/train/{cls}/Fotos 07-12-2020_dev{y}_{i}.png"
                z.writestr(n, png((30+160*y, 80, i*10)))
                row = dict(image=n, asset_category="glass-insulator", defect_gt="missing-cap" if y else "normal",
                           anomaly_gt="anomalous" if y else "normal")
                dev += [dict(row, question_type=q) for q in ["anomaly_detection", "defect_classification"]]
            n = f"defect_supervised/glass-insulator/val/{cls}/gold{y}.png"
            z.writestr(n, png((30+160*y, 90, 0)))
            gold.append(dict(image=n, quality_status="approve"))
    with ZipFile(unsup, "w") as z:
        z.writestr("glass-insulator/test/good/official.png", png((50, 100, 0)))
    qa, csv = tmp_path / "dev.jsonl", tmp_path / "gold.csv"
    qa.write_text("\n".join(json.dumps(x) for x in dev), encoding="utf-8")
    probe.write_csv(csv, gold)
    return SimpleNamespace(sup_zip=sup, unsup_zip=unsup, dev_qa=qa, gold_csv=csv,
                           seed=42, minimum_per_class=2, out_dir=tmp_path / "out", audit_only=False), train


def extractor(im):
    r, g, b = im.getpixel((0, 0))
    return np.array([r/255, (255-r)/255, g/255, .1+b/255])


def test_date_split_and_protected_targets(tmp_path):
    args, _ = inputs(tmp_path)
    rows, dev, removed, audit = probe.prepare(args.sup_zip, args.unsup_zip, args.dev_qa, args.gold_csv,
                                               minimum=2, strict=False)
    assert len(rows) == 24 and len(dev) == 4 and len(removed) == 4
    assert audit["exact_byte_overlap"] == audit["exact_rgb_overlap"] == 0
    dates = {s: {r["date"] for r in rows if r["split"] == s} for s in ["fit", "calibration", "internal_test"]}
    assert not (dates["fit"] & dates["calibration"] or dates["fit"] & dates["internal_test"] or dates["calibration"] & dates["internal_test"])
    assert {r["rgb_sha256"] for r in rows}.isdisjoint({r["rgb_sha256"] for r in dev})


def test_renamed_and_reencoded_eval_image_is_excluded(tmp_path):
    args, _ = inputs(tmp_path)
    with ZipFile(args.unsup_zip) as z:
        original = z.read("glass-insulator/test/good/official.png")
    prefix = "defect_supervised/glass-insulator/train/good/Fotos 01-12-2020_"
    with ZipFile(args.sup_zip, "a") as z:
        z.writestr(prefix + "renamed.png", original)
        z.writestr(prefix + "reencoded.png", png((50, 100, 0), "different bytes"))
    rows, _, removed, audit = probe.prepare(args.sup_zip, args.unsup_zip, args.dev_qa, args.gold_csv, minimum=2, strict=False)
    assert len(rows) == 24
    assert any(r["reason"] == "protected_size_crc" for r in removed)
    assert any(r["reason"] == "protected_rgb_sha256" for r in removed)


def test_conflicting_train_labels_fail_closed(tmp_path):
    args, train = inputs(tmp_path)
    with ZipFile(args.sup_zip, "a") as z:
        n, _, data = train[0]
        z.writestr(n.replace("/good/", "/missing-cap/") + ".png", data)
    with pytest.raises(ValueError, match="conflicting training labels"):
        probe.prepare(args.sup_zip, args.unsup_zip, args.dev_qa, args.gold_csv, minimum=2, strict=False)


def test_missing_protected_image_and_wrong_archive_rejected(tmp_path):
    args, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="expected frozen"):
        probe.prepare(args.sup_zip, args.unsup_zip, args.dev_qa, args.gold_csv)
    probe.write_csv(args.gold_csv, [{"image": "absent.png", "quality_status": "approve"}])
    with pytest.raises(ValueError, match="protected target missing"):
        probe.prepare(args.sup_zip, args.unsup_zip, args.dev_qa, args.gold_csv, minimum=2, strict=False)


def test_too_few_dates_or_class_support_rejected():
    with pytest.raises(ValueError, match="capture dates"):
        probe.assign_splits([{"date": "a", "label": 0}], 42, 1)
    with pytest.raises(ValueError, match="class support"):
        probe.assign_splits([{"date": str(i), "label": 0} for i in range(8)], 42, 1)


def test_cpu_pipeline_freezes_before_dev_and_preserves_inputs(tmp_path, monkeypatch):
    args, _ = inputs(tmp_path)
    hashes = {p: probe.sha_file(p) for p in [args.sup_zip, args.unsup_zip, args.dev_qa, args.gold_csv]}
    actual_features = probe.features
    def check_order(rows, *a, **kw):
        if rows[0]["split"] == "dev_report_only":
            assert (args.out_dir / "frozen_detector.json").is_file()
        return actual_features(rows, *a, **kw)
    monkeypatch.setattr(probe, "features", check_order)
    metrics = probe.execute(args, extractor=extractor, strict=False)
    assert len(metrics) == 4
    assert {p: probe.sha_file(p) for p in hashes} == hashes
    status = json.loads((args.out_dir / "run_status.json").read_text())
    assert status["complete"] and not status["gold_scored"]
    frozen = json.loads((args.out_dir / "frozen_detector.json").read_text())
    assert frozen["selection_uses"] == ["fit", "calibration"]
    assert next(m for m in metrics if m["split"] == "calibration")["normal_fpr"] <= .05
    assert next(m for m in metrics if m["split"] == "dev_report_only")["recall"] == 1
    # A second execution must reuse features; no extraction call is permitted.
    def forbidden_extractor(im):
        pytest.fail("cached features should be reused")
    assert probe.execute(args, extractor=forbidden_extractor, strict=False) == metrics


def test_test_and_dev_labels_cannot_change_head(tmp_path):
    args, _ = inputs(tmp_path)
    rows, dev, _, _ = probe.prepare(args.sup_zip, args.unsup_zip, args.dev_qa, args.gold_csv, minimum=2, strict=False)
    args.out_dir.mkdir()
    x = probe.features(rows, args.sup_zip, args.out_dir, {"test": 1}, extractor)
    a = probe.train_head(x, rows, args.out_dir)
    changed = [dict(r, label=1-r["label"]) if r["split"] == "internal_test" else r for r in rows]
    other = tmp_path / "other"
    other.mkdir()
    changed_x = x.copy()
    changed_x[[r["split"] == "internal_test" for r in rows]] += 100
    b = probe.train_head(changed_x, changed, other)
    assert a == b  # held-out labels/features cannot affect weights, scaler or threshold
    assert a["fit_images"] == sum(r["split"] == "fit" for r in rows)


def test_checksum_corruption_and_changed_inputs_stop_resume(tmp_path):
    args, _ = inputs(tmp_path)
    probe.execute(args, extractor=extractor, strict=False)
    p = next((args.out_dir / "features").glob("*.npy"))
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        probe.execute(args, extractor=extractor, strict=False)
    assert not json.loads((args.out_dir / "run_status.json").read_text())["complete"]
    assert not json.loads((args.out_dir / "run_metadata.json").read_text())["complete"]
    args.seed += 1
    with pytest.raises(ValueError, match="locked input/config changed"):
        probe.execute(args, extractor=extractor, strict=False)


def test_threshold_respects_fpr_and_rejects_invalid():
    y = np.array([0, 0, 1, 1])
    s = np.array([.1, .6, .5, .9])
    t = probe.threshold_from_calibration(y, s)
    assert ((s >= t)[y == 0]).sum() == 0
    assert ((s >= t)[y == 1]).sum() == 1
    with pytest.raises(ValueError, match="invalid calibration"):
        probe.threshold_from_calibration(y, np.array([.1, np.nan, .5, .9]))


def test_extraction_failure_never_freezes_or_completes(tmp_path):
    args, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="invalid extracted feature"):
        probe.execute(args, extractor=lambda im: np.array([np.nan, 1]), strict=False)
    assert not (args.out_dir / "frozen_detector.json").exists()
    assert not json.loads((args.out_dir / "run_status.json").read_text())["complete"]


def test_unrelated_existing_results_are_never_overwritten(tmp_path):
    args, _ = inputs(tmp_path)
    args.out_dir.mkdir()
    old = args.out_dir / "leaderboard.csv"
    old.write_text("prior research results", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing nonempty output"):
        probe.execute(args, extractor=extractor, strict=False)
    assert old.read_text(encoding="utf-8") == "prior research results"
    assert not (args.out_dir / "input_lock.json").exists()


def test_notebook_automatic_inputs_and_separate_outputs():
    nb = json.loads((SCRIPT.parents[1] / "notebooks/missing_component_probe_colab.ipynb").read_text(encoding="utf-8"))
    code = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            source = "".join(c["source"])
            compile(source, f"cell{i}", "exec")
            code.append(source)
            assert c["outputs"] == []
    joined = "\n".join(code)
    assert re.search(r'EXECUTION_COMMIT = "[0-9a-f]{40}"', joined)
    assert "PENDING_" not in joined
    for expected in ["uuid4()", "DRIVE_ROOT.rglob", "sha256(DEV_QA)", "GOLD_CSV",
                     "--out-dir", "verify_gold_revision(REPO_DIR, EXECUTION_COMMIT)",
                     "frozen_detector_sha256", "missing_component_probe_results.zip",
                     "transformers==5.14.0", "scikit-learn==1.7.2"]:
        assert expected in joined
    assert "BRANCH =" not in joined and '"origin", "main"' not in joined
    assert "files.upload" not in joined and "input(" not in joined
    assert "run_gold_v2_q23" not in joined
