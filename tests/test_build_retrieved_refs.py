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
SPEC = importlib.util.spec_from_file_location("build_retrieved_refs_t", ROOT / "scripts" / "build_retrieved_refs.py")
REFS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REFS)


def _png(v: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (6, 6), (v % 255, (v * 3) % 255, (v * 7) % 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_flight_date_and_auroc():
    assert REFS.flight_date("x/Fotos 03-12-2020_DJI_0324_cadeia_876.jpg") == "03-12-2020"
    assert REFS.flight_date("x/no_date.jpg") is None
    assert REFS.auroc([0.9, 0.8, 0.2], [True, True, False]) == 1.0
    assert REFS.auroc([0.5, 0.5], [True, False]) == 0.5
    assert REFS.auroc([0.5], [True]) is None


def test_any_vs_diffdate_rules(tmp_path):
    sup, unsup = tmp_path / "sup.zip", tmp_path / "unsup.zip"
    same_day = "Fotos 03-12-2020_DJI_0324_x_1.png"
    with zipfile.ZipFile(unsup, "w") as z:
        z.writestr(f"vari-grip/train/good/{same_day}", _png(1))                       # same flight as the target
        z.writestr("vari-grip/train/good/Fotos 07-12-2020_DJI_0052_x_2.png", _png(2))
        z.writestr("vari-grip/test/good/Fotos 09-12-2020_DJI_0001_x_3.png", _png(3))
    with zipfile.ZipFile(sup, "w") as z:
        z.writestr("defect_supervised/vari-grip/train/good/Fotos 03-12-2020_DJI_0325_x_9.png", _png(9))
        z.writestr("defect_supervised/vari-grip/train/rust/Fotos 05-12-2020_DJI_0400_x_8.png", _png(8))
    qa = tmp_path / "qa.jsonl"
    qa.write_text("\n".join(json.dumps(r) for r in [
        {"core_id": "a", "image": "defect_supervised/vari-grip/train/good/Fotos 03-12-2020_DJI_0325_x_9.png",
         "asset_category": "vari-grip", "anomaly_gt": "normal"},
        {"core_id": "b", "image": "defect_supervised/vari-grip/train/rust/Fotos 05-12-2020_DJI_0400_x_8.png",
         "asset_category": "vari-grip", "anomaly_gt": "anomalous"}]) + "\n", encoding="utf-8")

    def factory():
        def embed(rows, which):
            out = {}
            for r in rows:
                name = Path(str(r["relative_path"])).name
                if "03-12-2020" in name:
                    v = np.array([1.0, 0.0])
                elif "05-12-2020" in name:                 # the defective target: far from every normal
                    v = np.array([0.0, 1.0])
                else:
                    v = np.array([0.6, 0.8])
                out[str(r["relative_path"])] = v.astype(np.float32)
            return out
        return embed

    out = tmp_path / "out"
    assert REFS.main(["--qa", str(qa), "--sup-zip", str(sup), "--unsup-zip", str(unsup), "--out-dir", str(out)],
                     embed_factory=factory) == 0
    with (out / "retrieved_refs.csv").open(encoding="utf-8") as h:
        rows = {r["core_id"]: r for r in csv.DictReader(h)}
    a = rows["a"]
    assert a["retrieved_any"].endswith(same_day) and a["any_same_date"] == "True"     # nearest is the same flight
    assert "07-12-2020" in a["retrieved_diffdate"] and a["diffdate_same_date"] == "False"
    assert all("/test/" not in r["retrieved_any"] for r in rows.values())             # leakage-safe pool only
    summary = json.loads((out / "retrieved_refs_summary.json").read_text(encoding="utf-8"))
    assert summary["diagnostics"]["any"]["by_label"]["normal"]["same_date_rate"] == 1.0
    assert summary["diagnostics"]["diffdate"]["by_label"]["normal"]["same_date_rate"] == 0.0
    assert summary["diagnostics"]["any"]["cosine_auroc_for_normal"] == 1.0
