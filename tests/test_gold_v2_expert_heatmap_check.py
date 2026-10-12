from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("gold_v2_expert_heatmap_check_t",
                                              ROOT / "scripts" / "build_gold_v2_expert_heatmap_check.py")
HC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HC)

BOX = [0.375, 0.375, 0.25, 0.25]          # centre box: 16x16 cells 6..9
# c1, c3 rust with a seg box; c5 bird nest without one; c0, c2, c4 normal
CORES = {"c0": ("lightning-rod-suspension", "normal"), "c1": ("lightning-rod-suspension", "rust"),
         "c2": ("lightning-rod-suspension", "normal"), "c3": ("lightning-rod-suspension", "rust"),
         "c4": ("vari-grip", "normal"), "c5": ("vari-grip", "bird_nest")}


def _rows():
    rows = []
    for core, (asset, defect) in CORES.items():
        common = {"core_id": core, "asset_category": asset, "image": f"sup/{core}.png",
                  "anomaly_gt": "normal" if defect == "normal" else "anomalous", "defect_gt": defect}
        rows.append({**common, "id": f"{core}-q1", "question_type": "anomaly_detection"})
        if defect == "rust":
            rows.append({**common, "id": f"{core}-q3", "question_type": "defect_localization",
                         "localization_gt": {"normalized_bboxes_xywh": [BOX], "empty_mask": False}})
        elif core == "c0":
            rows.append({**common, "id": f"{core}-q3", "question_type": "defect_localization",
                         "localization_gt": {"normalized_bboxes_xywh": [], "empty_mask": True}})
    return rows


def _setup(tmp: Path):
    sup = tmp / "sup.zip"
    with zipfile.ZipFile(sup, "w") as z:
        for core in CORES:
            buf = io.BytesIO()
            Image.new("RGB", (80, 60), (90, 90, 90)).save(buf, format="PNG")
            z.writestr(f"sup/{core}.png", buf.getvalue())
    art = tmp / "merged"
    rows = []
    rng = np.random.default_rng(0)
    for core, (asset, defect) in CORES.items():
        anomalous = defect != "normal"
        for expert in ("patchcore", "anomalyclip"):
            amap = rng.random((16, 16)).astype(np.float32) * 0.1
            if expert == "patchcore":
                amap[7:9, 7:9] += 5.0 if anomalous else 1.0    # peak in the centre box, louder on defects
            else:
                amap[0:2, 0:2] += 3.0                            # always the corner, same height
            d = art / expert
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{core}.npy", amap)
            Image.new("RGB", (80, 60), (200, 120, 60)).save(d / f"{core}__pure.png")
            roi = (30, 22, 50, 38) if expert == "patchcore" else (0, 0, 20, 15)
            rows.append({"query_id": core, "expert": expert, "asset_category": asset,
                         "model_pred_score": ("0.9" if anomalous else "0.1") if expert == "patchcore" else "",
                         "anomaly_score_top1pct": "1.0",
                         "roi_left": roi[0], "roi_top": roi[1], "roi_right": roi[2], "roi_bottom": roi[3],
                         "anomaly_map_path": f"{expert}/{core}.npy", "overlay_path": f"{expert}/{core}__pure.png"})
    manifest = art / "roi_manifest_all.csv"
    with manifest.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return sup, art, manifest


def _csv(path: Path):
    with path.open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def test_auroc_ties_and_missing_class():
    assert HC.auroc([0.9, 0.1], [True, False]) == 1.0
    assert HC.auroc([0.5, 0.5], [True, False]) == 0.5
    assert HC.auroc([0.5], [True]) is None


def test_heatmap_check_end_to_end(tmp_path):
    sup, art, manifest = _setup(tmp_path)
    out = tmp_path / "out"
    argv = ["--qa-dir", str(tmp_path), "--manifest", str(manifest), "--artifact-root", str(art),
            "--sup-zip", str(sup), "--out-dir", str(out), "--experts", "patchcore", "anomalyclip",
            "--rows-per-grid", "1", "--n-normal", "2", "--n-other", "1"]
    assert HC.main(argv, load_qa=lambda _: _rows()) == 0

    det = {(r["expert"], r["score_field"]): r for r in _csv(out / "detection.csv")}
    assert float(det[("patchcore", "model_pred_score")]["auroc_global"]) == 1.0
    assert float(det[("patchcore", "raw_map_max")]["auroc_global"]) == 1.0
    assert ("anomalyclip", "model_pred_score") not in det                    # no own score -> no row
    assert float(det[("anomalyclip", "anomaly_score_top1pct")]["auroc_global"]) == 0.5
    assert det[("patchcore", "model_pred_score")]["assets_with_both_classes"] == "2"

    loc = {r["pointer"]: r for r in _csv(out / "localization.csv")}
    assert loc["patchcore"]["gt_images"] == "2"
    assert float(loc["patchcore"]["peak_in_gt_rate"]) == 1.0
    assert float(loc["patchcore"]["roi_centre_in_gt_rate"]) == 1.0
    assert float(loc["patchcore"]["mean_heat_in_box_ratio"]) > 1.0
    assert float(loc["anomalyclip"]["peak_in_gt_rate"]) == 0.0
    assert float(loc["image_centre"]["peak_in_gt_rate"]) == 1.0

    heat = {(r["expert"], r["label"]) for r in _csv(out / "heat_on_normals.csv")}
    assert ("anomalyclip", "normal") in heat and ("patchcore", "anomalous") in heat

    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete" and status["gt_box_images"] == 2
    assert status["grids"] == ["grid_defect_1.png", "grid_defect_2.png", "grid_normal.png", "grid_other_defects.png"]
    with Image.open(out / "grid_defect_1.png") as g:
        assert g.width == HC.LABEL_W + HC.CELL_W * 3


def test_heatmap_check_notebook():
    nb = json.loads((ROOT / "notebooks" / "lvlm_gold_v2_expert_heatmap_check_colab.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"heatcheck_nb:{i}", "exec")
    code = "\n".join(cells)
    assert "build_gold_v2_expert_heatmap_check.py" in code and "roi_manifest_all.csv" in code
    assert '"--prefix", "gold_v2_expert_heatmap_check"' in code
    assert "torch" not in code                                                 # CPU only, no LVLM
