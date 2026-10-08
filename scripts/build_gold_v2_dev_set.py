#!/usr/bin/env python3
"""Build the gold-v2 development set for method selection (agent design), separate from gold v2.

gold v2 took every anomalous Aligned Core image, so development images come from the InsPLAD supervised
``train`` split. Selection (deterministic, seed 42):

- per asset: defect images per class (default 25; vari-grip 20 + 20) and 25 normal images
- polymer-insulator-upper-shackle train defects are ``corrosão`` (kept separate from ``rust`` by the data
  contract), so polymer contributes normal images only
- excluded: any image whose content fingerprint (size + CRC32) appears in the unsupervised test split, in the
  leakage-safe normal-reference pool (unsupervised train normals) or among the gold v2 images; duplicates

Questions use the gold v2 wording and option sets (Q1 normal/anomalous, Q2 defect type); there is no Q3.
Each image carries the gold-style fixed normal reference (one per asset, ``select_reference_by_asset`` seed 42)
and a retrieved one (DINOv2 cosine top-1 among the same-asset leakage-safe train normals).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_explainable_qa_gold as qa  # noqa: E402
from insplad_benchmark.dataset import (  # noqa: E402
    leakage_safe_train_normal_candidates,
    parse_supervised_manifest,
    parse_unsupervised_manifest,
    select_reference_by_asset,
    stable_id,
)

ASSETS = ("glass-insulator", "lightning-rod-suspension", "polymer-insulator-upper-shackle", "vari-grip",
          "yoke-suspension")
DEFECT_TEXT = qa.DEFECT_TEXT                         # same Q2 option texts as gold v2
Q1_TEXT = "Is the target power-line asset normal or anomalous?"
Q2_TEXT = "Which condition best describes the target power-line asset?"
PROTOCOL = "gold_v2_dev"


def zip_manifest(zip_path: Path, out_csv: Path) -> Path:
    """Same columns as scripts/inspect_zip_manifest.py (path, is_dir, size, extension, crc32)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zf, out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "is_dir", "size", "compressed_size", "extension", "crc32"])
        writer.writeheader()
        for info in zf.infolist():
            writer.writerow({"path": info.filename, "is_dir": info.is_dir(), "size": info.file_size,
                             "compressed_size": info.compress_size, "extension": Path(info.filename).suffix.lower(),
                             "crc32": f"{info.CRC:08x}"})
    return out_csv


def quotas(defect_per_class: int, normal_per_asset: int, vari_grip_per_defect: int) -> dict[tuple[str, str], int]:
    q = {}
    for asset in ASSETS:
        q[(asset, "normal")] = normal_per_asset
    q[("glass-insulator", "missing-cap")] = defect_per_class
    q[("lightning-rod-suspension", "rust")] = defect_per_class
    q[("vari-grip", "bird-nest")] = vari_grip_per_defect
    q[("vari-grip", "rust")] = vari_grip_per_defect
    q[("yoke-suspension", "rust")] = defect_per_class
    return q


def select_dev_images(sup_rows: Sequence[Mapping], unsup_rows: Sequence[Mapping], gold_images: set[str],
                      quota: Mapping[tuple[str, str], int], seed: int = 42) -> tuple[list[dict], dict]:
    test_fp = {str(r["content_fingerprint"]) for r in unsup_rows if r["split"] == "test"}
    pool_fp = {str(r["content_fingerprint"]) for rows in
               leakage_safe_train_normal_candidates(unsup_rows, ASSETS).values() for r in rows}
    sup_by_path = {str(r["relative_path"]): r for r in sup_rows}
    gold_fp = {str(sup_by_path[p]["content_fingerprint"]) for p in gold_images if p in sup_by_path}
    reasons: Counter = Counter()
    seen: set[str] = set()
    eligible: dict[tuple[str, str], list[Mapping]] = {}
    for r in sorted(sup_rows, key=lambda x: str(x["relative_path"])):
        if r["split"] != "train" or r["asset_category"] not in ASSETS:
            continue
        key = (str(r["asset_category"]), str(r["defect_class"]))
        if key not in quota:
            reasons[f"class_not_used:{key[0]}:{key[1]}"] += 1
            continue
        fp = str(r["content_fingerprint"])
        if fp in gold_fp:
            reasons["in_gold_v2"] += 1
        elif fp in test_fp:
            reasons["in_unsupervised_test"] += 1
        elif fp in pool_fp:
            reasons["in_reference_pool"] += 1
        elif fp in seen:
            reasons["duplicate_content"] += 1
        else:
            seen.add(fp)
            eligible.setdefault(key, []).append(r)
            continue
    selected = []
    shortfall = {}
    for key, n in sorted(quota.items()):
        pool = eligible.get(key, [])
        if len(pool) < n:
            shortfall[f"{key[0]}:{key[1]}"] = {"wanted": n, "available": len(pool)}
        pick = random.Random(f"{seed}:dev:{key[0]}:{key[1]}").sample(pool, min(n, len(pool)))
        selected.extend(sorted(pick, key=lambda x: str(x["relative_path"])))
    stats = {"excluded": dict(reasons), "eligible": {f"{a}:{d}": len(v) for (a, d), v in sorted(eligible.items())},
             "shortfall": shortfall}
    return selected, stats


def retrieve_similar_normals(dev_rows: Sequence[Mapping], unsup_rows: Sequence[Mapping],
                             embed: Callable[[Sequence[Mapping], str], Mapping[str, np.ndarray]]) -> dict[str, dict]:
    """``embed(rows, which)`` returns {relative_path: L2-normalized vector}; which is 'query' or 'candidate'."""
    candidates = leakage_safe_train_normal_candidates(unsup_rows, sorted({str(r["asset_category"]) for r in dev_rows}))
    q_emb = embed(dev_rows, "query")
    c_emb = embed([r for rows in candidates.values() for r in rows], "candidate")
    out = {}
    for asset, rows in candidates.items():
        cand = np.stack([c_emb[str(r["relative_path"])] for r in rows])
        for q in (r for r in dev_rows if r["asset_category"] == asset):
            scores = cand @ q_emb[str(q["relative_path"])]
            best = int(np.argmax(scores))
            if rows[best]["content_fingerprint"] == q["content_fingerprint"]:
                raise RuntimeError(f"{q['relative_path']}: retrieved reference is the query itself")
            out[str(q["relative_path"])] = {"retrieved_reference_image": str(rows[best]["relative_path"]),
                                            "retrieval_cosine": round(float(scores[best]), 6)}
    return out


def build_rows(dev_rows: Sequence[Mapping], fixed_refs: Mapping[str, Mapping], retrieved: Mapping[str, Mapping],
               seed: int = 42) -> tuple[list[dict], list[dict]]:
    images, questions = [], []
    for r in dev_rows:
        core = stable_id("dev", str(r["relative_path"]))
        defect = str(r["defect_class"])
        base = {
            "core_id": core, "asset_category": str(r["asset_category"]), "image": str(r["relative_path"]),
            "split": "supervised_train", "anomaly_gt": "normal" if defect == "normal" else "anomalous",
            "defect_gt": defect, "content_fingerprint": str(r["content_fingerprint"]),
            "normal_reference_image": str(fixed_refs[str(r["asset_category"])]["relative_path"]),
            **retrieved.get(str(r["relative_path"]), {}), "protocol": PROTOCOL,
        }
        images.append(base)
        a_text = "Normal." if defect == "normal" else "Anomalous."
        opts, ans = qa.make_options(["Normal.", "Anomalous."], a_text, seed_text=f"{seed}:{core}:anomaly")
        questions.append({**base, "id": f"{core}-q1", "task": "explainable_mcqa", "question_type": "anomaly_detection",
                          "question": Q1_TEXT, "options": opts, "answer": ans, "answer_text": a_text})
        d_text = DEFECT_TEXT[defect]
        opts, ans = qa.make_options(list(DEFECT_TEXT.values()), d_text, seed_text=f"{seed}:{core}:defect")
        questions.append({**base, "id": f"{core}-q2", "task": "explainable_mcqa",
                          "question_type": "defect_classification", "question": Q2_TEXT, "options": opts,
                          "answer": ans, "answer_text": d_text})
    return images, questions


def dino_embedder(sup_zip: Path, unsup_zip: Path, cache_dir: Path, device: str | None = None):
    import build_one_shot_plus as osp
    from insplad_benchmark.reference_retrieval import DinoV2Encoder, embed_rows_with_cache

    encoder = DinoV2Encoder(device=device)

    def embed(rows, which):
        zip_path = sup_zip if which == "query" else unsup_zip
        with osp.ZipImageLoader(zip_path) as loader:
            return embed_rows_with_cache(rows, encoder=encoder, image_loader=loader,
                                         cache_path=cache_dir / f"dinov2_{which}.npz", batch_size=32)
    return embed, encoder.provenance


def main(argv: Sequence[str] | None = None, *, embed_factory=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sup-zip", type=Path, required=True)
    p.add_argument("--unsup-zip", type=Path, required=True)
    p.add_argument("--gold-qa-dir", type=Path, required=True, help="gold_v2_final (images are excluded)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--defect-per-class", type=int, default=25)
    p.add_argument("--normal-per-asset", type=int, default=25)
    p.add_argument("--vari-grip-per-defect", type=int, default=20)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    man = args.out_dir / "_manifests"
    unsup_rows = parse_unsupervised_manifest(zip_manifest(args.unsup_zip, man / "unsupervised.csv"))
    sup_rows, _ = parse_supervised_manifest(zip_manifest(args.sup_zip, man / "supervised.csv"))
    gold_images = set()
    for f in sorted(Path(args.gold_qa_dir).glob("master_qa_*.jsonl")):
        gold_images |= {json.loads(x)["image"] for x in f.read_text(encoding="utf-8").splitlines() if x.strip()}
    if not gold_images:
        raise RuntimeError(f"no gold images found under {args.gold_qa_dir}")

    quota = quotas(args.defect_per_class, args.normal_per_asset, args.vari_grip_per_defect)
    dev_rows, stats = select_dev_images(sup_rows, unsup_rows, gold_images, quota, args.seed)
    fixed = select_reference_by_asset(unsup_rows, ASSETS, seed=42)
    if embed_factory is None:
        embed, provenance = dino_embedder(args.sup_zip, args.unsup_zip, args.out_dir / "_dino_cache", args.device)
        provenance = {"model_id": provenance.model_id, "model_revision": provenance.model_revision,
                      "embedding": provenance.embedding_kind}
    else:
        embed, provenance = embed_factory(), {"model_id": "test-fake"}
    retrieved = retrieve_similar_normals(dev_rows, unsup_rows, embed)
    images, questions = build_rows(dev_rows, fixed, retrieved, args.seed)

    qa_name = f"dev_master_qa_{len(questions)}.jsonl"
    with (args.out_dir / qa_name).open("w", encoding="utf-8") as h:
        for row in questions:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "dev_images.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(images[0]))
        w.writeheader()
        w.writerows(images)
    cos = [float(r["retrieval_cosine"]) for r in images]
    summary = {
        "protocol": PROTOCOL, "file": qa_name, "images": len(images), "questions": len(questions),
        "by_asset_defect": dict(Counter(f"{r['asset_category']}:{r['defect_gt']}" for r in images)),
        "quota": {f"{a}:{d}": n for (a, d), n in sorted(quota.items())}, "selection": stats,
        "fixed_reference_by_asset": {a: str(r["relative_path"]) for a, r in sorted(fixed.items())},
        "retrieval": {**provenance, "method": "cosine top-1 among same-asset leakage-safe unsupervised train normals",
                      "cosine_min": round(min(cos), 4), "cosine_mean": round(sum(cos) / len(cos), 4)},
        "seed": args.seed, "source_split": "supervised train",
        "excluded_from_dev": "unsupervised test, reference pool (unsupervised train normals), gold v2 images, duplicates",
        "note": "Method selection only. Paper numbers come from gold v2. No Q3 (no location labels).",
        "quality_status": "FROZEN" if not stats["shortfall"] else "SHORTFALL",
    }
    (args.out_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("images", "questions", "by_asset_defect", "quality_status")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
