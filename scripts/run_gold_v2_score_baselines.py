#!/usr/bin/env python3
"""Score-only Q1 baselines for gold v2 (no LVLM): what a single number already gets.

- ``patchcore_pct50``: anomalous when the within-asset PatchCore percentile over the 246-image gold cohort is >= 50
  (the transductive cue the PatchCore-score condition shows to the LVLMs)
- ``patchcore_auroc``: threshold-free separability of the raw PatchCore score
- ``cosine_<rule>``: anomalous when the retrieved normal reference's cosine is below a threshold chosen on the
  development set (max Q1 F1 there), then applied unchanged to gold; plus the gold AUROC

If an LVLM condition does not beat these, its gain could come from the score or the retrieval similarity alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_retrieved_refs as refs_mod  # noqa: E402
import run_gold_baselines as base  # noqa: E402
import run_gold_v2_baselines as v2  # noqa: E402
import run_gold_v2_loc_mmad_eval as loc  # noqa: E402
import run_gold_v2_vision_expert_full as vx  # noqa: E402

RULES = refs_mod.RULES


def q1_metrics(gold: Sequence[str], pred: Sequence[str]) -> dict:
    p, r, f1 = base.binary_f1(list(gold), list(pred), "anomalous")
    return {"q1_anomaly_f1": round(f1, 6), "q1_precision": round(p, 6), "q1_recall": round(r, 6)}


def best_threshold(cosines: Sequence[float], labels: Sequence[str]) -> float:
    """Threshold t maximizing Q1 F1 for 'anomalous if cosine < t' (ties -> smallest t)."""
    best_t, best_f = 0.0, -1.0
    for t in sorted(set(cosines)) + [max(cosines) + 1e-9]:
        f = base.binary_f1(list(labels), ["anomalous" if c < t else "normal" for c in cosines], "anomalous")[2]
        if f > best_f + 1e-12:
            best_t, best_f = t, f
    return best_t


def read_refs(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))


def main(argv: Sequence[str] | None = None, *, load_qa=v2.load_gold_qa_v2) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--roi-manifest", type=Path, required=True)
    p.add_argument("--gold-refs-csv", type=Path, required=True)
    p.add_argument("--dev-refs-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--expert", default="patchcore")
    args = p.parse_args(argv)

    rows = load_qa(args.qa_dir)
    q1 = {str(r["core_id"]): r for r in rows if r["question_type"] == "anomaly_detection"}
    cores = sorted(q1)
    gold = [str(q1[c]["anomaly_gt"]) for c in cores]
    out = []

    manifest = vx.read_csv(args.roi_manifest)
    pct = loc.expert_percentiles(manifest, set(cores), args.expert)
    raw = {str(r["query_id"]): vx.resolve_score(r) for r in manifest
           if r.get("expert") == args.expert and str(r.get("query_id")) in q1}
    out.append({"method": f"{args.expert}_pct50", **q1_metrics(gold, ["anomalous" if pct[c] >= 50 else "normal" for c in cores]),
                "auroc": None, "threshold": 50.0, "threshold_source": "fixed (within-asset percentile, gold cohort)"})
    a = refs_mod.auroc([raw[c] for c in cores], [g == "anomalous" for g in gold])
    out.append({"method": f"{args.expert}_raw_score", "auroc": round(a, 6) if a is not None else None,
                "threshold_source": "threshold-free"})

    gold_refs = {r["core_id"]: r for r in read_refs(args.gold_refs_csv)}
    dev_refs = [r for r in read_refs(args.dev_refs_csv) if r.get("anomaly_gt") in ("normal", "anomalous")]
    missing = [c for c in cores if c not in gold_refs]
    if missing:
        raise RuntimeError(f"gold refs lack {len(missing)} images, e.g. {missing[:2]}")
    for rule in RULES:
        t = best_threshold([float(r[f"cosine_{rule}"]) for r in dev_refs], [r["anomaly_gt"] for r in dev_refs])
        cos = [float(gold_refs[c][f"cosine_{rule}"]) for c in cores]
        a = refs_mod.auroc([-x for x in cos], [g == "anomalous" for g in gold])
        out.append({"method": f"cosine_{rule}", **q1_metrics(gold, ["anomalous" if x < t else "normal" for x in cos]),
                    "auroc": round(a, 6) if a is not None else None, "threshold": round(t, 6),
                    "threshold_source": "chosen on dev (max Q1 F1), applied unchanged to gold"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loc_fields = sorted({k for r in out for k in r}, key=lambda k: ["method", "q1_anomaly_f1", "q1_precision",
                                                                    "q1_recall", "auroc", "threshold",
                                                                    "threshold_source"].index(k))
    with (args.out_dir / "score_baselines.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=loc_fields)
        w.writeheader()
        w.writerows(out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
