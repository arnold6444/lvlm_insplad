#!/usr/bin/env python3
"""Retrieve similar normal references for any QA file, with and without same-flight candidates.

For every target image in ``--qa`` (supervised zip paths) this finds the DINOv2 cosine top-1 among the same-asset
leakage-safe unsupervised train normals under two rules:

- ``any``     : every candidate except the target's own content (size + CRC32)
- ``diffdate``: additionally drops candidates photographed on the same date as the target (the date in the
                InsPLAD file name, e.g. ``03-12-2020``), so a target is never matched to a near-identical frame
                of the same inspection flight

The development set showed that, under ``any``, normal targets get a same-date reference far more often than
defective ones and the cosine alone separates them (AUROC 0.886). The summary reports, per rule, the same-date
rate by label and the cosine-only AUROC, so the leakage can be checked on any set.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_gold_v2_dev_set as devset  # noqa: E402
from insplad_benchmark.dataset import (  # noqa: E402
    leakage_safe_train_normal_candidates,
    parse_supervised_manifest,
    parse_unsupervised_manifest,
)

DATE = re.compile(r"(\d{2}-\d{2}-\d{4})")
RULES = ("any", "diffdate")


def flight_date(path: str) -> str | None:
    m = DATE.search(Path(str(path)).name)
    return m.group(1) if m else None


def auroc(scores: Sequence[float], positive: Sequence[bool]) -> float | None:
    """Mann-Whitney AUROC (ties count half); None if one class is missing."""
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def retrieve(queries: Sequence[Mapping], candidates_by_asset: Mapping[str, Sequence[Mapping]],
             q_emb: Mapping[str, np.ndarray], c_emb: Mapping[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for asset, rows in candidates_by_asset.items():
        if not rows:
            continue
        cand = np.stack([c_emb[str(r["relative_path"])] for r in rows])
        fps = np.array([str(r["content_fingerprint"]) for r in rows])
        dates = np.array([flight_date(str(r["relative_path"])) or "" for r in rows])
        for q in (r for r in queries if r["asset_category"] == asset):
            base = cand @ q_emb[str(q["relative_path"])]
            own = fps == str(q["content_fingerprint"])
            q_date = flight_date(str(q["relative_path"]))
            item = {"query_date": q_date or ""}
            for rule in RULES:
                mask = own | ((dates == q_date) if (rule == "diffdate" and q_date) else np.zeros_like(own))
                scores = np.where(mask, -np.inf, base)
                best = int(np.argmax(scores))
                if not np.isfinite(scores[best]):
                    raise RuntimeError(f"{q['relative_path']}: no {rule} reference candidate")
                ref = str(rows[best]["relative_path"])
                item[f"retrieved_{rule}"] = ref
                item[f"cosine_{rule}"] = round(float(scores[best]), 6)
                item[f"{rule}_same_date"] = bool(q_date) and flight_date(ref) == q_date
            out[str(q["relative_path"])] = item
    return out


def main(argv: Sequence[str] | None = None, *, embed_factory: Callable | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa", type=Path, required=True, help="question jsonl (rows with core_id, image, asset_category)")
    p.add_argument("--sup-zip", type=Path, required=True)
    p.add_argument("--unsup-zip", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, help="DINOv2 cache (default: <out-dir>/_dino_cache)")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    rows = [json.loads(x) for x in args.qa.read_text(encoding="utf-8").splitlines() if x.strip()]
    images: dict[str, Mapping] = {}
    for r in rows:
        images.setdefault(str(r["image"]), r)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    man = args.out_dir / "_manifests"
    unsup_rows = parse_unsupervised_manifest(devset.zip_manifest(args.unsup_zip, man / "unsupervised.csv"))
    sup_rows, _ = parse_supervised_manifest(devset.zip_manifest(args.sup_zip, man / "supervised.csv"))
    sup_by_path = {str(r["relative_path"]): r for r in sup_rows}
    missing = [p for p in images if p not in sup_by_path]
    if missing:
        raise RuntimeError(f"{len(missing)} QA images not in the supervised zip, e.g. {missing[:2]}")
    queries = [sup_by_path[p] for p in sorted(images)]
    candidates = leakage_safe_train_normal_candidates(unsup_rows, sorted({str(q["asset_category"]) for q in queries}))

    if embed_factory is None:
        embed, prov = devset.dino_embedder(args.sup_zip, args.unsup_zip, args.cache_dir or args.out_dir / "_dino_cache",
                                           args.device)
        provenance = {"model_id": prov.model_id, "model_revision": prov.model_revision, "embedding": prov.embedding_kind}
    else:
        embed, provenance = embed_factory(), {"model_id": "test-fake"}
    q_emb = embed(queries, "query")
    c_emb = embed([r for rs in candidates.values() for r in rs], "candidate")
    found = retrieve(queries, candidates, q_emb, c_emb)

    out_rows = []
    for path in sorted(images):
        r = images[path]
        out_rows.append({"core_id": r.get("core_id", ""), "image": path, "asset_category": r["asset_category"],
                         "anomaly_gt": r.get("anomaly_gt", ""), **found[path]})
    with (args.out_dir / "retrieved_refs.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    labelled = [r for r in out_rows if r["anomaly_gt"] in ("normal", "anomalous")]
    diag = {}
    for rule in RULES:
        by_label = {}
        for lab in ("normal", "anomalous"):
            sub = [r for r in labelled if r["anomaly_gt"] == lab]
            if sub:
                by_label[lab] = {"n": len(sub), "same_date_rate": round(sum(r[f"{rule}_same_date"] for r in sub) / len(sub), 4),
                                 "cosine_mean": round(sum(r[f"cosine_{rule}"] for r in sub) / len(sub), 4)}
        a = auroc([r[f"cosine_{rule}"] for r in labelled], [r["anomaly_gt"] == "normal" for r in labelled])
        diag[rule] = {"by_label": by_label, "cosine_auroc_for_normal": None if a is None else round(a, 4)}
    summary = {"qa": args.qa.name, "images": len(out_rows), "retrieval": {**provenance, "rules": list(RULES)},
               "undated_queries": sum(not r["query_date"] for r in out_rows), "diagnostics": diag}
    (args.out_dir / "retrieved_refs_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
