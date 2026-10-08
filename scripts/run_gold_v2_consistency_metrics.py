#!/usr/bin/env python3
"""Cross-question consistency metrics for every gold-v2 condition (no new inference).

Each question is asked independently (MMAD protocol), so a model can call an image normal in Q1 and still name a
defect type in Q2. For every model x condition with stored per-question predictions this reports, with the same
conventions as ``scripts/evaluate.py``:

- ``consistency_error``: share of images (both Q1 and Q2 valid) whose Q1 and Q2 answers contradict
- ``joint_accuracy``: share of images with Q1 and Q2 both correct (invalid counts as wrong)
- ``false_defect_rate``: share of normal images whose valid Q2 answer is a defect type
- on the images that also have a Q3 (location) question: ``q3_accuracy``, ``q1q3_contradiction`` (Q1 normal but
  Q3 names a region, or Q1 anomalous but Q3 says no defect region) and ``q1q3_joint_accuracy``
plus Q1 anomaly F1 and Q2 defect macro-F1 so the rows can be checked against the published leaderboards.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_gold_baselines as base  # noqa: E402
import run_gold_v2_baselines as v2  # noqa: E402

Q1, Q2, Q3 = "anomaly_detection", "defect_classification", "defect_localization"
NONE_TEXT = "No localized defect region is present."
CENTER_TEXT = "Center region."
STEM = re.compile(r"^[a-z0-9_]+__[a-z0-9_]+$")   # skips Drive duplicates such as "x__score (1)"
MODELS = ("qwen3_5_4b", "internvl3_5_8b", "minicpm_v_4_5", "llava_onevision_7b")
REFERENCE_CONDITIONS = ("normal_reference", "positive_reference", "normal_positive_reference")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _choice(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text and text.lower() != "nan" else None


def discover(models: Sequence[str], image_only_csv: Path | None, reference_dir: Path | None,
             vx_pred_dir: Path | None, combined_dir: Path | None) -> dict[tuple[str, str], dict[str, str | None]]:
    """(model, condition) -> {question id: predicted letter} for every prediction file that exists."""
    found: dict[tuple[str, str], dict[str, str | None]] = {}
    if image_only_csv and Path(image_only_csv).is_file():
        with Path(image_only_csv).open(newline="", encoding="utf-8-sig") as handle:
            for r in csv.DictReader(handle):
                if r.get("model_key") in models:
                    found.setdefault((r["model_key"], "image_only"), {})[str(r["id"])] = _choice(r.get("predicted_choice"))
    for model in models:
        if reference_dir:
            for cond in REFERENCE_CONDITIONS:
                p = Path(reference_dir) / cond / "predictions" / f"{model}.jsonl"
                if p.is_file():
                    found[(model, cond)] = {str(r["id"]): _choice(r.get("predicted_choice")) for r in read_jsonl(p)}
        for root, prefix in ((vx_pred_dir, "vx"), (combined_dir, "ref+")):
            if not root or not (Path(root) / model).is_dir():
                continue
            for p in sorted((Path(root) / model).glob("*__*.jsonl")):
                if not STEM.match(p.stem):
                    continue
                expert, _, cond = p.stem.partition("__")
                name = f"vx:{expert}:{cond}" if prefix == "vx" else f"ref+{expert}:{cond}"
                found[(model, name)] = {str(r["id"]): _choice(r.get("predicted_choice")) for r in read_jsonl(p)}
    return found


def image_pairs(rows: Sequence[Mapping]) -> list[tuple[Mapping, Mapping]]:
    by_core: dict[str, dict[str, Mapping]] = {}
    for r in rows:
        if r.get("question_type") in (Q1, Q2):
            by_core.setdefault(str(r["core_id"]), {})[r["question_type"]] = r
    return [(q[Q1], q[Q2]) for _, q in sorted(by_core.items()) if Q1 in q and Q2 in q]


def q3_pairs(rows: Sequence[Mapping]) -> list[tuple[Mapping, Mapping]]:
    by_core: dict[str, dict[str, Mapping]] = {}
    for r in rows:
        if r.get("question_type") in (Q1, Q3):
            by_core.setdefault(str(r["core_id"]), {})[r["question_type"]] = r
    return [(q[Q1], q[Q3]) for _, q in sorted(by_core.items()) if Q1 in q and Q3 in q]


def q3_kind(row: Mapping, choice: str | None) -> str | None:
    """'none' / 'region' for a valid Q3 answer, None if invalid."""
    if not choice or choice not in dict(row["options"]):
        return None
    return "none" if dict(row["options"])[choice] == NONE_TEXT else "region"


def q3_row(pairs: Sequence[tuple[Mapping, Mapping]], choice: Mapping[str, str | None]) -> dict:
    if not pairs or any(str(q3["id"]) not in choice for _, q3 in pairs):
        return {}
    sem = base.predicted_semantic
    correct = joint = contradictions = valid = invalid = 0
    for q1, q3 in pairs:
        a = sem(q1, choice.get(str(q1["id"])))
        c3 = choice.get(str(q3["id"]))
        kind = q3_kind(q3, c3)
        ok3 = c3 == q3["answer"]
        correct += ok3
        joint += ok3 and a == q1["anomaly_gt"]
        invalid += kind is None
        if a is not None and kind is not None:
            valid += 1
            contradictions += (a == "normal") != (kind == "none")
    n = len(pairs)
    return {"q3_questions": n, "q3_accuracy": round(correct / n, 6),
            "q1q3_contradiction": round(contradictions / valid, 6) if valid else 0.0,
            "q1q3_contradictions": contradictions, "q1q3_valid_pairs": valid,
            "q1q3_joint_accuracy": round(joint / n, 6), "q3_invalid_rate": round(invalid / n, 6)}


def q3_baselines(pairs: Sequence[tuple[Mapping, Mapping]]) -> dict:
    n = len(pairs)
    if not n:
        return {}
    answer_text = [dict(q3["options"])[q3["answer"]] for _, q3 in pairs]
    return {"questions": n,
            "always_no_defect_region": round(sum(t == NONE_TEXT for t in answer_text) / n, 6),
            "always_center": round(sum(t == CENTER_TEXT for t in answer_text) / n, 6),
            "answer_regions": {t: answer_text.count(t) for t in sorted(set(answer_text))}}


def consistency_row(pairs: Sequence[tuple[Mapping, Mapping]], choice: Mapping[str, str | None]) -> dict:
    sem = base.predicted_semantic
    a_true, a_pred, d_true, d_pred = [], [], [], []
    joint = contradictions = valid_pairs = 0
    normal_n = false_defects = 0
    for q1, q2 in pairs:
        a = sem(q1, choice.get(str(q1["id"])))
        d = sem(q2, choice.get(str(q2["id"])))
        a_true.append(str(q1["anomaly_gt"]))
        a_pred.append(a)
        d_true.append(str(q2["defect_gt"]))
        d_pred.append(d)
        joint += (a == q1["anomaly_gt"]) and (d == q2["defect_gt"])
        if a is not None and d is not None:
            valid_pairs += 1
            contradictions += (a == "normal") != (d == "normal")
        if q1["anomaly_gt"] == "normal":
            normal_n += 1
            false_defects += d is not None and d != "normal"
    _, _, q1_f1 = base.binary_f1(a_true, a_pred, "anomalous")
    n = len(pairs)
    return {
        "images": n,
        "q1_anomaly_f1": round(q1_f1, 6),
        "q2_defect_macro_f1": round(base.macro_f1(d_true, d_pred), 6),
        "consistency_error": round(contradictions / valid_pairs, 6) if valid_pairs else 0.0,
        "contradictions": contradictions,
        "valid_pairs": valid_pairs,
        "joint_accuracy": round(joint / n, 6) if n else 0.0,
        "false_defect_rate": round(false_defects / normal_n, 6) if normal_n else 0.0,
        "q1_invalid_rate": round(sum(x is None for x in a_pred) / n, 6) if n else 0.0,
        "q2_invalid_rate": round(sum(x is None for x in d_pred) / n, 6) if n else 0.0,
    }


def main(argv: Sequence[str] | None = None, *, load_qa=v2.load_gold_qa_v2) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--image-only-csv", type=Path)
    p.add_argument("--reference-dir", type=Path, help="gold_reference_v2 folder (condition/predictions/model.jsonl)")
    p.add_argument("--vx-pred-dir", type=Path, help="vision expert full eval/predictions")
    p.add_argument("--combined-dir", type=Path, help="reference+expert predictions")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    args = p.parse_args(argv)

    rows = load_qa(args.qa_dir)
    pairs = image_pairs(rows)
    loc_pairs = q3_pairs(rows)
    found = discover(args.models, args.image_only_csv, args.reference_dir, args.vx_pred_dir, args.combined_dir)
    out, skipped = [], []
    for (model, cond), choice in sorted(found.items()):
        missing = sum(str(q["id"]) not in choice for pair in pairs for q in pair)
        if missing:
            skipped.append({"model_key": model, "condition": cond, "missing_q1_q2": missing})
            continue
        out.append({"model_key": model, "condition": cond, **consistency_row(pairs, choice), **q3_row(loc_pairs, choice)})
    write_csv(args.out_dir / "consistency_leaderboard.csv", out)
    meta = {
        "protocol": "gold_v2_consistency_metrics",
        "new_lvlm_inference": False,
        "images": len(pairs), "models": list(args.models), "rows": len(out), "skipped_incomplete": skipped,
        "q3_constant_baselines": q3_baselines(loc_pairs),
        "definitions": {
            "consistency_error": "Q1 normal but Q2 defect, or Q1 anomalous but Q2 normal; over images with both answers valid",
            "joint_accuracy": "Q1 and Q2 both correct; invalid counts as wrong; over all images",
            "false_defect_rate": "normal images whose valid Q2 answer is a defect type; over normal images",
            "q1q3_contradiction": "Q1 normal but Q3 names a region, or Q1 anomalous but Q3 says no defect region; "
                                  "over Q3 images with both answers valid",
            "q1q3_joint_accuracy": "Q1 and Q3 both correct; over Q3 images",
        },
        "source": "same conventions as scripts/evaluate.py; each question was asked independently",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"{len(out)} model x condition rows, {len(skipped)} skipped (incomplete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
