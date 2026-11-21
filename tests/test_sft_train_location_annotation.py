import ast
import base64
import io
import json
from pathlib import Path
import re
import sys
import types
import zlib
from zipfile import ZipFile

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_sft_train_location_annotation_notebook as builder  # noqa: E402
import sft_train_location_annotation as a  # noqa: E402

CLASSES = [("glass-insulator", "missing-cap"), ("vari-grip", "rust"), ("vari-grip", "bird-nest")]


def photo(seed):
    buf = io.BytesIO()
    Image.new("RGB", (300, 180), (seed * 7 % 255, 140, 150)).save(buf, format="PNG")
    return buf.getvalue()


def setup(tmp, per_class=1):
    targets, archive = [], tmp / "sup.zip"
    with ZipFile(archive, "w") as z:
        for i, (asset, defect) in enumerate(c for c in CLASSES for _ in range(per_class)):
            name, payload = f"defect_supervised/{asset}/train/{defect}/photo{i}.png", photo(i)
            z.writestr(name, payload)
            targets.append(dict(core_id=f"sft-{i}", image=name, sha256=a.sha(payload), size=[300, 180], asset_category=asset,
                                defect_gt=defect, anomaly_gt="anomalous", sft_split="train",
                                content_fingerprint=f"{len(payload)}:{zlib.crc32(payload):08x}"))
    manifest = dict(protocol=a.PROTOCOL, source=dict(fixture=True), cohort="fixture", targets=targets)
    return manifest, archive, a.prepare(manifest, archive, tmp / (a.PREFIX + "_fixture"))


def test_draft_resume_confirm_and_screen_shows_only_photo_and_defect_type(tmp_path):
    m, z, s = setup(tmp_path)
    boxes = [[.05, .05, .1, .1], [.8, .8, .1, .1]]
    assert s.save(0, boxes, "draft", "memo", 0)["done"] == 0
    resumed = a.prepare(m, z, s.out)
    item = resumed.item(0)
    assert item["boxes"] == boxes and item["status"] == "draft" and item["revision"] == 1
    assert item["defect_label"].startswith("누락") and "원판" in item["guide"]
    assert resumed.item(1)["defect_label"].startswith("녹") and resumed.item(2)["defect_label"] == "새 둥지"
    assert not {"core_id", "image", "image_path", "defect_gt", "regions_gt", "proposal", "prediction", "bbox"} & item.keys()
    assert a.sha(base64.b64decode(item["image_url"].split(",")[1])) == m["targets"][0]["sha256"]
    assert resumed.save(0, boxes, "boxed", "memo", 1)["done"] == 1
    path, summary = resumed.export()
    assert summary["positive_spatial_support"] == 1 and not summary["complete"] and summary["multi_box_images"] == 1
    assert summary["region_counts"]["Upper-left"] == 1 and summary["region_counts"]["Lower-right"] == 1
    assert summary["boxed_by_defect"] == {"glass-insulator:missing-cap": 1}
    gt = a.read(s.out / "sft_train_location_gt.json")
    assert gt[0]["regions_gt"] == [1, 9] and gt[0]["normalized_bboxes_xywh"] == boxes and gt[1]["regions_gt"] is None
    assert gt[0]["label_basis"] == "human_train_box_centers" and gt[0]["sft_split"] == "train"
    assert len(list((s.out / "history/sft-0").glob("*.json"))) == 1
    with ZipFile(path) as archive:
        assert path.name == s.out.name + "_results.zip" and all("_images" not in n for n in archive.namelist())
        assert json.loads(archive.read("annotation/sft_train_location_gt.json")) == gt


def test_unknown_and_disputed_complete_the_photo_but_give_no_location(tmp_path):
    _, _, s = setup(tmp_path)
    s.save(0, [[.4, .4, .2, .2]], "boxed", "", 0)
    s.save(1, [], "unlocatable", "가려져 위치 판단 불가", 0)
    s.save(2, [], "disputed", "둥지인지 확신 불가", 0)
    _, summary = s.export()
    assert summary["complete"] and summary["positive_spatial_support"] == 1 and summary["unknown_spatial_support"] == 2
    assert summary["centre_only_images"] == 1
    rows = a.read(s.out / "sft_train_location_gt.json")
    assert all(r["anomaly_gt"] == "anomalous" for r in rows) and rows[1]["regions_gt"] is None and rows[2]["regions_gt"] is None
    assert a.read(s.out / "run_status.json")["complete"]
    for status in ("unlocatable", "disputed"):
        with pytest.raises(ValueError):
            s.save(0, [], status, "", 1)


def test_bad_boxes_and_stale_revisions_change_nothing(tmp_path):
    _, _, s = setup(tmp_path)
    for boxes in ([], [[float("nan"), 0, .1, .1]], [[-.1, 0, .1, .1]], [[.99, 0, .2, .1]], [[0, 0, .001, .1]], "not-list"):
        with pytest.raises(ValueError):
            s.save(0, boxes, "boxed", "", 0)
    assert s.progress()["done"] == 0 and not s.path.exists()
    s.save(0, [[.1, .1, .1, .1]], "draft", "", 0)
    with pytest.raises(ValueError, match="다른 화면"):
        s.save(0, [], "draft", "", 0)
    for index in (-1, 3, True, 1.0):
        with pytest.raises(ValueError):
            s.save(index, [], "draft", "", 0)
    assert s.item(0)["boxes"] == [[.1, .1, .1, .1]]


def test_editing_a_confirmed_photo_returns_it_to_draft_and_keeps_history(tmp_path):
    m, z, s = setup(tmp_path)
    s.save(0, [[.1, .1, .1, .1]], "boxed", "", 0)
    a.prepare(m, z, s.out).save(0, [[.6, .6, .1, .1]], "draft", "edited", 1)
    assert s.progress()["done"] == 0 and s.export()[1]["positive_spatial_support"] == 0
    assert a.read(next((s.out / "history/sft-0").glob("*.json")))["status"] == "boxed"


def test_only_anomalous_training_photos_without_model_fields_are_accepted(tmp_path):
    m, _, _ = setup(tmp_path)
    for key, value in [("core_id", "dev-0"), ("core_id", "../escape"), ("image", "../escape.png"),
                       ("image", "defect_supervised/glass-insulator/val/missing-cap/photo0.png"),
                       ("defect_gt", "corrosao"), ("anomaly_gt", "normal"), ("sft_split", "dev"), ("prediction", [5]),
                       ("bbox", [0, 0, .2, .2])]:
        x = json.loads(json.dumps(m))
        x["targets"][0][key] = value
        with pytest.raises(ValueError):
            a.validate_manifest(x)
    duplicate = json.loads(json.dumps(m))
    duplicate["targets"][1]["image"] = duplicate["targets"][0]["image"]
    with pytest.raises(ValueError):
        a.validate_manifest(duplicate)


def test_cache_is_reused_repaired_and_a_changed_target_list_is_refused(tmp_path):
    m, z, s = setup(tmp_path)
    s.save(0, [], "draft", "keep", 0)
    before = s.path.read_bytes()
    assert a.prepare(m, tmp_path / "no-such.zip", s.out).item(0)["note"] == "keep"     # verified cache: ZIP not needed
    s.paths["sft-0"].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="사진이 바뀌었습니다"):
        s.item(0)
    seen = []
    assert a.prepare(m, z, s.out, progress=lambda i, n: seen.append((i, n))).item(0)["index"] == 0   # restored from the ZIP
    assert seen == [(1, 1)]
    changed = json.loads(json.dumps(m))
    changed["cohort"] = "changed"
    with pytest.raises(ValueError, match="대상"):
        a.prepare(changed, z, s.out)
    assert s.path.read_bytes() == before
    with ZipFile(tmp_path / "tampered.zip", "w") as bad:
        for t in m["targets"]:
            bad.writestr(t["image"], b"other bytes")
    with pytest.raises(ValueError, match="SHA"):
        a.prepare(m, tmp_path / "tampered.zip", tmp_path / (a.PREFIX + "_new"))
    assert not (tmp_path / (a.PREFIX + "_new")).exists()


def test_colab_callbacks_and_screen_text(tmp_path, monkeypatch):
    _, _, s = setup(tmp_path)
    callbacks, shown = {}, []
    google, colab = types.ModuleType("google"), types.ModuleType("google.colab")
    colab.output = types.SimpleNamespace(register_callback=lambda n, f: callbacks.update({n: f}))
    ip, display = types.ModuleType("IPython"), types.ModuleType("IPython.display")
    display.HTML = display.JSON = lambda x: x
    display.display = shown.append
    for name, module in [("google", google), ("google.colab", colab), ("IPython", ip), ("IPython.display", display)]:
        monkeypatch.setitem(sys.modules, name, module)
    a.register_colab(s)
    assert set(callbacks) == {"sftloc.item", "sftloc.progress", "sftloc.save"}
    assert callbacks["sftloc.item"](0)["index"] == 0 and callbacks["sftloc.save"](0, [], "draft", "note", 0)["revision"] == 1
    html = shown[0]
    assert "devmissing" not in html and "item.defect_label" in html and "item.guide" in html and "초안 저장" in html
    assert set(re.findall(r"call\('(sftloc\.\w+)'", html)) == set(callbacks)


def data_folder(tmp, per_class):
    """A fixture of the SFT data folder: train QA, samples and the automatic-box table (only its split column is read)."""
    m, z, _ = setup(tmp, per_class)
    qa, boxes = [], ["core_id,split,bbox"]
    for t in m["targets"]:
        row = dict(core_id=t["core_id"], image=t["image"], asset_category=t["asset_category"], defect_gt=t["defect_gt"],
                   anomaly_gt="anomalous", content_fingerprint=t["content_fingerprint"])
        qa += [dict(row, question_type="anomaly_detection"), dict(row, question_type="defect_classification")]
        boxes.append(f"{t['core_id']},train,\"[0.0, 0.0, 0.2, 0.2]\"")
    qa.append(dict(core_id="sft-normal", image="x.png", asset_category="vari-grip", defect_gt="normal", anomaly_gt="normal",
                   content_fingerprint="1:0", question_type="anomaly_detection"))
    (tmp / "qa.jsonl").write_text("\n".join(json.dumps(r) for r in qa) + "\n", encoding="utf-8")
    (tmp / "samples.jsonl").write_text("fixture samples\n", encoding="utf-8")
    (tmp / "boxes.csv").write_text("\n".join(boxes) + "\n", encoding="utf-8")
    return z, a.sha((tmp / "samples.jsonl").read_bytes())


def test_target_list_is_tied_to_the_trained_data_and_ordered_in_balanced_rounds(tmp_path, monkeypatch):
    z, samples_sha = data_folder(tmp_path, per_class=4)
    monkeypatch.setattr(builder, "BLOCK", 2)
    args = (tmp_path / "qa.jsonl", tmp_path / "samples.jsonl", tmp_path / "boxes.csv", z)
    m = builder.targets(*args, expected_sha=samples_sha, per_class=4)
    classes = [(t["asset_category"], t["defect_gt"]) for t in m["targets"]]
    assert len(classes) == 12 and classes[:6] == [c for c in sorted(CLASSES) for _ in range(2)] and classes[:6] == classes[6:]
    assert all(set(t) == a.TARGET_KEYS for t in m["targets"]) and "bbox" not in json.dumps(m)
    assert builder.targets(*args, expected_sha=samples_sha, per_class=4) == m
    with pytest.raises(ValueError, match="trained on"):
        builder.targets(*args, per_class=4)
    with pytest.raises(ValueError, match="per class"):
        builder.targets(*args, expected_sha=samples_sha, per_class=3)
    qa = (tmp_path / "qa.jsonl").read_text(encoding="utf-8")
    fingerprint = m["targets"][0]["content_fingerprint"]
    (tmp_path / "qa.jsonl").write_text(qa.replace(fingerprint, "1:" + fingerprint.split(":")[1]), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the original ZIP"):
        builder.targets(*args, expected_sha=samples_sha, per_class=4)


def embedded_bundle(notebook):
    code = "".join(next(c for c in notebook["cells"] if "SOURCE_BYTES=" in "".join(c["source"]))["source"])
    payload = base64.b64decode(re.search(r"SOURCE_BYTES=base64\.b64decode\('([A-Za-z0-9+/=]+)'\)", code).group(1))
    assert a.sha(payload) == re.search(r"SOURCE_SHA256='([0-9a-f]{64})'", code).group(1)
    with ZipFile(io.BytesIO(payload)) as z:
        return {n: z.read(n) for n in z.namelist()}


def test_notebook_build_embeds_the_tool_and_runs_without_gpu(tmp_path, monkeypatch):
    z, samples_sha = data_folder(tmp_path, per_class=2)
    monkeypatch.setattr(builder, "DATA_SHA", samples_sha)
    monkeypatch.setattr(builder, "BLOCK", 1)
    fixture = builder.targets
    monkeypatch.setattr(builder, "targets", lambda *x: fixture(*x, expected_sha=samples_sha, per_class=2))
    manifest = builder.targets(tmp_path / "qa.jsonl", tmp_path / "samples.jsonl", tmp_path / "boxes.csv", z)
    notebook = builder.build(manifest, "sft_v1", tmp_path / "nb.ipynb", tmp_path / "evidence")
    polymer = builder.build(manifest, "polymer", tmp_path / "polymer.ipynb")
    folders = [re.search(r'RUN_FOLDER = "lvlm_insplad_results/([^"]+)"', "".join(n["cells"][1]["source"])).group(1) for n in (notebook, polymer)]
    assert folders == [a.PREFIX + "_sft_v1", a.PREFIX + "_sft_v2_polymer"]             # separate Drive folders and result ZIPs
    assert not "".join(notebook["cells"][0]["source"]).startswith(" ") and "@@" not in json.dumps(notebook)
    assert all(not line.startswith("    ") for line in notebook["cells"][0]["source"])     # markdown, not a code block
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == 4 and all(ast.parse(c) for c in code)
    assert "cuda" not in "".join(code).lower() and "register_colab(session)" in code[2] and "session.export()" in code[3]
    files = embedded_bundle(notebook)
    assert set(files) == {*builder.BUNDLE, "inputs/manifest.json"}
    assert all(files[n] == builder.source_bytes(n) and bytes([13]) not in files[n] for n in builder.BUNDLE)
    assert json.loads(files["inputs/manifest.json"]) == a.read(tmp_path / "evidence/manifest.json")
    assert a.read(tmp_path / "evidence/notebook_build.json")["images"] == 6


def test_committed_notebook_matches_the_current_tool_and_lists_300_balanced_photos():
    path = ROOT / "notebooks/lvlm_sft_train_location_annotation_colab.ipynb"
    if not path.exists():
        pytest.skip("notebook not built yet")
    files = embedded_bundle(json.loads(path.read_text(encoding="utf-8")))
    assert all(files[n] == builder.source_bytes(n) for n in builder.BUNDLE), "rebuild the notebook after changing the tool"
    manifest = json.loads(files["inputs/manifest.json"])
    a.validate_manifest(manifest)
    classes = [t["asset_category"] + ":" + t["defect_gt"] for t in manifest["targets"]]
    assert len(classes) == 300 and manifest["source"]["sft_samples_sha256"] == builder.DATA_SHA
    for start in range(0, 300, 60):
        assert sorted(set(classes[start:start + 60])) == sorted(set(classes)) and len(set(classes)) == 5
        assert all(classes[start:start + 60].count(c) == 12 for c in set(classes))


def test_polymer_corrosion_photos_live_in_the_portuguese_train_folder(tmp_path):
    import unicodedata
    payload, archive = photo(9), tmp_path / "sup.zip"
    name = "defect_supervised/polymer-insulator-upper-shackle/train/corrosão/p.png"
    with ZipFile(archive, "w") as z:
        z.writestr(name, payload)
    target = dict(core_id="sft-p", image=name, sha256=a.sha(payload), size=[300, 180], asset_category="polymer-insulator-upper-shackle",
                  defect_gt="corrosao", anomaly_gt="anomalous", sft_split="pending",
                  content_fingerprint=f"{len(payload)}:{zlib.crc32(payload):08x}")
    manifest = dict(protocol=a.PROTOCOL, source=dict(fixture=True), cohort="fixture", targets=[target])
    s = a.prepare(manifest, archive, tmp_path / (a.PREFIX + "_polymer"))
    assert s.item(0)["defect_label"] == "녹 / 부식"
    s.save(0, [[.4, .05, .2, .2]], "boxed", "", 0)
    path, summary = s.export()
    assert path.name == a.PREFIX + "_polymer_results.zip" and summary["region_counts"]["Upper-center"] == 1
    assert summary["boxed_by_defect"] == {"polymer-insulator-upper-shackle:corrosao": 1}
    decomposed = dict(manifest, targets=[dict(target, image=unicodedata.normalize("NFD", name))])
    a.validate_manifest(decomposed)                                    # same folder, other Unicode form
    for bad in (name.replace("corrosão", "rust"), name.replace("/train/", "/val/"), name.replace("polymer-insulator-upper-shackle", "vari-grip")):
        with pytest.raises(ValueError):
            a.validate_manifest(dict(manifest, targets=[dict(target, image=bad)]))


def test_polymer_target_list_comes_from_the_sft_selection_and_keeps_the_trained_photos(tmp_path):
    import test_agent_sft as sft_tests
    sup, unsup, dev_qa, dev_images = sft_tests.fixture(tmp_path)
    sft = sft_tests.sft
    unsup_rows = sft.parse_unsupervised_manifest(sft.devset.zip_manifest(unsup, tmp_path / "m/unsup.csv"))
    sup_rows, _ = sft.parse_supervised_manifest(sft.devset.zip_manifest(sup, tmp_path / "m/sup.csv"))
    base, _, _ = sft.select_train_images(sup_rows, unsup_rows, dev_images, 2, 42)
    trained = tmp_path / "trained.jsonl"
    trained.write_text("".join(json.dumps(dict(image=r["relative_path"])) + "\n" for r in base), encoding="utf-8")
    m = builder.polymer_targets(dev_qa, trained, sup, unsup, tmp_path / "work", per_class=2)
    assert len(m["targets"]) == 2 and all(t["defect_gt"] == "corrosao" and t["sft_split"] == "pending" for t in m["targets"])
    assert not {t["image"] for t in m["targets"]} & (dev_images | {r["relative_path"] for r in base})
    assert builder.polymer_targets(dev_qa, trained, sup, unsup, tmp_path / "work", per_class=2) == m
    trained.write_text(json.dumps(dict(image=base[0]["relative_path"])) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="moved the sft_v1 selection"):
        builder.polymer_targets(dev_qa, trained, sup, unsup, tmp_path / "work", per_class=2)


def test_committed_polymer_notebook_adds_60_new_photos_outside_dev_and_the_first_300():
    path = ROOT / builder.NOTEBOOKS["polymer"]
    if not path.exists():
        pytest.skip("polymer notebook not built yet")
    files = embedded_bundle(json.loads(path.read_text(encoding="utf-8")))
    assert all(files[n] == builder.source_bytes(n) for n in builder.BUNDLE), "rebuild the notebook after changing the tool"
    manifest = json.loads(files["inputs/manifest.json"])
    a.validate_manifest(manifest)
    first = json.loads(embedded_bundle(json.loads((ROOT / builder.NOTEBOOKS["sft_v1"]).read_text(encoding="utf-8")))["inputs/manifest.json"])
    dev = {json.loads(x)["image"] for x in (ROOT / "results/latest/gold_v2_dev/dev_master_qa_480.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    images = {t["image"] for t in manifest["targets"]}
    assert len(images) == 60 and not images & dev and not images & {t["image"] for t in first["targets"]}
    assert {(t["asset_category"], t["defect_gt"], t["sft_split"]) for t in manifest["targets"]} == {("polymer-insulator-upper-shackle", "corrosao", "pending")}
    assert not {t["core_id"] for t in manifest["targets"]} & {t["core_id"] for t in first["targets"]}
