from __future__ import annotations

import hashlib
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
SPEC = importlib.util.spec_from_file_location("build_gold_v2_dev_set_t", ROOT / "scripts" / "build_gold_v2_dev_set.py")
DEV = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DEV)


def _png(seed: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (seed % 255, (seed * 7) % 255, (seed * 13) % 255)).save(buf, format="PNG")
    return buf.getvalue()


def _zips(tmp: Path):
    sup, unsup = tmp / "sup.zip", tmp / "unsup.zip"
    shared_normal = _png(999)                 # same bytes in sup train and unsup train -> must be excluded
    with zipfile.ZipFile(unsup, "w") as z:
        n = 0
        for asset in DEV.ASSETS:
            for i in range(4):
                z.writestr(f"unsup/{asset}/train/good/{i}.png", _png(n)); n += 1
            z.writestr(f"unsup/{asset}/test/good/t.png", _png(5000 + n)); n += 1
        z.writestr("unsup/vari-grip/train/good/shared.png", shared_normal)
    with zipfile.ZipFile(sup, "w") as z:
        n = 100
        classes = {"glass-insulator": ["good", "missing-cap"], "lightning-rod-suspension": ["good", "rust"],
                   "polymer-insulator-upper-shackle": ["normal", "corrosão"], "vari-grip": ["good", "bird-nest", "rust"],
                   "yoke-suspension": ["good", "rust"]}
        for asset, cls in classes.items():
            for c in cls:
                for i in range(4):
                    z.writestr(f"defect_supervised/{asset}/train/{c}/{i}.png", _png(n)); n += 1
            z.writestr(f"defect_supervised/{asset}/val/good/v.png", _png(n)); n += 1
        z.writestr("defect_supervised/vari-grip/train/good/shared.png", shared_normal)
    gold = tmp / "gold"
    gold.mkdir()
    (gold / "master_qa_2.jsonl").write_text(json.dumps({"image": "defect_supervised/vari-grip/val/good/v.png"}) + "\n",
                                            encoding="utf-8")
    return sup, unsup, gold


def _fake_embed():
    def embed(rows, which):
        out = {}
        for r in rows:
            h = hashlib.sha256(str(r["relative_path"]).encode()).digest()
            v = np.frombuffer(h[:16], dtype=np.uint8).astype(np.float32) + 1
            out[str(r["relative_path"])] = v / np.linalg.norm(v)
        return out
    return embed


def test_dev_set_selection_rules_and_rows(tmp_path):
    sup, unsup, gold = _zips(tmp_path)
    out = tmp_path / "dev"
    argv = ["--sup-zip", str(sup), "--unsup-zip", str(unsup), "--gold-qa-dir", str(gold), "--out-dir", str(out),
            "--defect-per-class", "3", "--normal-per-asset", "3", "--vari-grip-per-defect", "2"]
    assert DEV.main(argv, embed_factory=_fake_embed) == 0
    summary = json.loads((out / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["quality_status"] == "FROZEN"
    assert summary["by_asset_defect"] == {
        "glass-insulator:missing-cap": 3, "glass-insulator:normal": 3, "lightning-rod-suspension:normal": 3,
        "lightning-rod-suspension:rust": 3, "polymer-insulator-upper-shackle:normal": 3, "vari-grip:bird-nest": 2,
        "vari-grip:normal": 3, "vari-grip:rust": 2, "yoke-suspension:normal": 3, "yoke-suspension:rust": 3}
    assert summary["selection"]["excluded"]["in_reference_pool"] == 1          # shared normal
    assert "class_not_used:polymer-insulator-upper-shackle:corrosao" in summary["selection"]["excluded"]
    rows = [json.loads(x) for x in (out / summary["file"]).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 * summary["images"]
    assert not any("shared.png" in r["image"] or "/val/" in r["image"] for r in rows)
    for r in rows:
        assert r["options"][r["answer"]] == r["answer_text"]
        assert r["normal_reference_image"] == summary["fixed_reference_by_asset"][r["asset_category"]]
        assert r["retrieved_reference_image"].startswith(f"unsup/{r['asset_category']}/train/good/")
    q2 = [r for r in rows if r["question_type"] == "defect_classification"]
    assert {frozenset(r["options"].values()) for r in q2} == {frozenset(DEV.DEFECT_TEXT.values())}
    # deterministic
    out2 = tmp_path / "dev2"
    assert DEV.main(argv[:-8] + ["--out-dir", str(out2)] + argv[-6:], embed_factory=_fake_embed) == 0
    assert (out2 / summary["file"]).read_text(encoding="utf-8") == (out / summary["file"]).read_text(encoding="utf-8")


def test_real_embed_path_reads_zips_and_caches(tmp_path, monkeypatch):
    import insplad_benchmark.reference_retrieval as rr

    calls = []

    class FakeEncoder:
        def __init__(self, *a, **k):
            self._p = rr.EmbeddingProvenance(model_id="fake", model_revision="r")

        @property
        def provenance(self):
            return self._p

        def encode(self, images):
            calls.append(len(images))
            return np.stack([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) + 1 for im in images])

    monkeypatch.setattr(rr, "DinoV2Encoder", FakeEncoder)
    sup, unsup, gold = _zips(tmp_path)
    out = tmp_path / "dev"
    argv = ["--sup-zip", str(sup), "--unsup-zip", str(unsup), "--gold-qa-dir", str(gold), "--out-dir", str(out),
            "--defect-per-class", "2", "--normal-per-asset", "2", "--vari-grip-per-defect", "2"]
    assert DEV.main(argv) == 0
    assert (out / "_dino_cache" / "dinov2_query.npz").is_file() and (out / "_dino_cache" / "dinov2_candidate.npz").is_file()
    first = sum(calls)
    assert DEV.main(argv) == 0
    assert sum(calls) == first                                   # second run served from the cache
    summary = json.loads((out / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["retrieval"]["model_id"] == "fake"


def test_shortfall_is_reported(tmp_path):
    sup, unsup, gold = _zips(tmp_path)
    out = tmp_path / "dev"
    assert DEV.main(["--sup-zip", str(sup), "--unsup-zip", str(unsup), "--gold-qa-dir", str(gold), "--out-dir", str(out),
                     "--defect-per-class", "9"], embed_factory=_fake_embed) == 0
    summary = json.loads((out / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["quality_status"] == "SHORTFALL" and summary["selection"]["shortfall"]


def test_dev_notebook_builds_checks_and_publishes_frozen_only():
    nb = json.loads((ROOT / "notebooks" / "lvlm_gold_v2_dev_set_colab.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"dev_nb:{i}", "exec")
    code = "\n".join(cells)
    assert "build_gold_v2_dev_set.py" in code and "고정 reference 검증" in code
    assert 'summary.get("quality_status") == "FROZEN"' in code and '"--prefix", "gold_v2_dev"' in code
