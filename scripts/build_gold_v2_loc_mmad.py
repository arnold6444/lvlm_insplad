#!/usr/bin/env python3
"""Build the gold-v2 MMAD-style defect-location set (separate from the frozen gold v2 files).

gold v2 Q3 asks "where is the defect?" on 42 lightning-rod-suspension images, 22 of them normal with a
"no defect region" option, so it mixes defect presence with location. Following MMAD, this set asks location only
on anomalous images, states that a defect is present, and offers four 3x3-grid regions (no "none" option).
Location labels: InsPLAD-seg boxes where available, human-drawn boxes (``scripts/loc_box_annotation.py``) for the
rest. Images marked unlocatable are excluded and listed. Question text, options and seeds are deterministic.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_explainable_qa_gold as qa  # noqa: E402
import loc_box_annotation as lba  # noqa: E402
import run_gold_v2_baselines as v2  # noqa: E402

QUESTION = "There is a defect in the target power-line asset. Where is the defect located in the target image?"
REGIONS = [k for k in qa.REGION_TEXT if k != "none"]
PROTOCOL = "gold_v2_loc_mmad"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def build_rows(rows: Sequence[Mapping], loc_rows: Sequence[Mapping], annotations: Mapping[str, Mapping], *,
               seed: int = 42, allow_partial: bool = False) -> tuple[list[dict], dict]:
    anomalous = lba.anomalous_images(rows)
    seg = lba.seg_boxes_by_core(loc_rows)
    base_by_core = {str(r["core_id"]): r for r in rows if r.get("question_type") == lba.Q2}
    out: list[dict] = []
    excluded, missing = [], []
    for rec in anomalous:
        core = rec["core_id"]
        if core in seg:
            boxes, basis, annotator = seg[core], "insplad_seg_instance_annotation", ""
        else:
            ann = annotations.get(core)
            if ann is None:
                missing.append(core)
                continue
            if ann.get("status") == "unlocatable":
                excluded.append({"core_id": core, "asset_category": rec["asset_category"],
                                 "defect_gt": rec["defect_gt"], "note": ann.get("note", "")})
                continue
            boxes = lba.validate_boxes(ann.get("boxes_xywh") or [])
            if not boxes:
                raise RuntimeError(f"{core}: status boxed but no valid box")
            basis, annotator = "manual_box_annotation", str(ann.get("annotator", ""))
        region = qa.coarse_region_from_bboxes(boxes)
        correct = qa.REGION_TEXT[region]
        pool = [k for k in REGIONS if k != region]
        qa.stable_random(f"{seed}:{core}:loc-mmad-distractors").shuffle(pool)
        options, answer = qa.make_options([correct] + [qa.REGION_TEXT[k] for k in pool[:3]], correct,
                                          seed_text=f"{seed}:{core}:loc-mmad-options")
        base = base_by_core[core]
        keep = ("core_id", "asset_category", "image", "unsupervised_image", "normal_reference_image",
                "anomaly_gt", "defect_gt")
        out.append({
            **{k: base[k] for k in keep if k in base},
            "id": f"{core}-q3m", "task": "explainable_mcqa", "question_type": "defect_localization",
            "question": QUESTION, "options": options, "answer": answer, "answer_text": correct,
            "localization_gt": {"normalized_bboxes_xywh": boxes, "coarse_region": region,
                                "basis": basis, "annotator": annotator},
            "protocol": PROTOCOL, "review_status": "gold",
        })
    if missing and not allow_partial:
        raise RuntimeError(f"{len(missing)} anomalous images still need a box, e.g. {missing[:3]}")
    summary = {
        "protocol": PROTOCOL,
        "question": QUESTION,
        "anomalous_images": len(anomalous),
        "questions": len(out),
        "excluded_unlocatable": excluded,
        "missing_annotation": missing,
        "by_basis": dict(Counter(r["localization_gt"]["basis"] for r in out)),
        "by_asset": dict(Counter(r["asset_category"] for r in out)),
        "by_defect": dict(Counter(r["defect_gt"] for r in out)),
        "by_region": dict(Counter(r["localization_gt"]["coarse_region"] for r in out)),
        "by_answer_letter": dict(Counter(r["answer"] for r in out)),
        "seed": seed,
        "note": "MMAD-style: anomalous images only, defect presence stated, 4 grid-region options, no 'none'. "
                "gold v2 files are unchanged.",
    }
    return out, summary


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-dir", type=Path, required=True, help="gold_v2_final QA folder (master + localization)")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-partial", action="store_true", help="build from finished boxes only (preview)")
    args = p.parse_args(argv)

    rows = v2.load_gold_qa_v2(args.qa_dir)
    loc_files = sorted(Path(args.qa_dir).glob("localization_qa_*.jsonl"))
    loc_rows = read_jsonl(loc_files[-1]) if loc_files else []
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8")) if Path(args.annotations).is_file() else {}
    out, summary = build_rows(rows, loc_rows, annotations, seed=args.seed, allow_partial=args.allow_partial)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"localization_mmad_qa_{len(out)}{'_partial' if summary['missing_annotation'] else ''}.jsonl"
    with (args.out_dir / name).open("w", encoding="utf-8") as h:
        for row in out:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["file"] = name
    summary["quality_status"] = "PREVIEW" if summary["missing_annotation"] else "FROZEN"
    (args.out_dir / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("excluded_unlocatable",)}, ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
