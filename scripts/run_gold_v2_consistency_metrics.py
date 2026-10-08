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
``significance_main.csv`` adds image-level paired bootstrap 95% CIs for the main condition comparisons.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

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


MAIN_PAIRS = (
    ("image_only", "normal_reference"),
    ("image_only", "vx:patchcore:score"),
    ("image_only", "ref+patchcore:score"),
    ("normal_reference", "ref+patchcore:score"),
)
BOOT_METRICS = ("q1_anomaly_f1", "q2_defect_macro_f1", "joint_accuracy", "consistency_error",
                "false_defect_rate", "q3_accuracy", "q1q3_contradiction")


def image_components(pairs: Sequence[tuple[Mapping, Mapping]], loc_pairs: Sequence[tuple[Mapping, Mapping]],
                     choice: Mapping[str, str | None]) -> dict:
    """Per-image 0/1 vectors from which every metric can be recomputed for a resampled image set."""
    sem = base.predicted_semantic
    n = len(pairs)
    index = {str(q1["core_id"]): i for i, (q1, _) in enumerate(pairs)}
    z = lambda: np.zeros(n)  # noqa: E731
    c = {k: z() for k in ("q1_tp", "q1_fp", "q1_fn", "joint", "contra", "valid", "normal", "false_def",
                          "q3_ok", "q3_n", "q13_contra", "q13_valid")}
    labels = sorted({str(q2["defect_gt"]) for _, q2 in pairs})
    c["q2"] = {lab: {"tp": z(), "fp": z(), "fn": z()} for lab in labels}
    for i, (q1, q2) in enumerate(pairs):
        a = sem(q1, choice.get(str(q1["id"])))
        d = sem(q2, choice.get(str(q2["id"])))
        pa, ga = a == "anomalous", q1["anomaly_gt"] == "anomalous"
        c["q1_tp"][i], c["q1_fp"][i], c["q1_fn"][i] = pa and ga, pa and not ga, ga and not pa
        for lab, v in c["q2"].items():
            v["tp"][i] = d == lab and q2["defect_gt"] == lab
            v["fp"][i] = d == lab and q2["defect_gt"] != lab
            v["fn"][i] = q2["defect_gt"] == lab and d != lab
        c["joint"][i] = a == q1["anomaly_gt"] and d == q2["defect_gt"]
        if a is not None and d is not None:
            c["valid"][i] = 1
            c["contra"][i] = (a == "normal") != (d == "normal")
        if q1["anomaly_gt"] == "normal":
            c["normal"][i] = 1
            c["false_def"][i] = d is not None and d != "normal"
    for q1, q3 in loc_pairs:
        i = index.get(str(q1["core_id"]))
        if i is None or str(q3["id"]) not in choice:
            continue
        a = sem(q1, choice.get(str(q1["id"])))
        kind = q3_kind(q3, choice.get(str(q3["id"])))
        c["q3_n"][i] = 1
        c["q3_ok"][i] = choice.get(str(q3["id"])) == q3["answer"]
        if a is not None and kind is not None:
            c["q13_valid"][i] = 1
            c["q13_contra"][i] = (a == "normal") != (kind == "none")
    return c


def _ratio(num, den):
    return np.divide(num, den, out=np.zeros_like(den, dtype=float), where=den > 0)


def metrics_from_weights(c: Mapping, w: np.ndarray) -> dict[str, np.ndarray]:
    """``w``: (B, n_images) resample counts. Equals consistency_row / q3_row when w is all ones."""
    w = np.atleast_2d(w)
    tp, fp, fn = w @ c["q1_tp"], w @ c["q1_fp"], w @ c["q1_fn"]
    f1s, present = [], []
    for v in c["q2"].values():
        t2, f2, n2 = w @ v["tp"], w @ v["fp"], w @ v["fn"]
        f1s.append(_ratio(2 * t2, 2 * t2 + f2 + n2))
        present.append((t2 + n2) > 0)
    f1s, present = np.array(f1s), np.array(present)
    return {
        "q1_anomaly_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "q2_defect_macro_f1": (f1s * present).sum(axis=0) / np.maximum(present.sum(axis=0), 1),
        "joint_accuracy": _ratio(w @ c["joint"], w.sum(axis=1)),
        "consistency_error": _ratio(w @ c["contra"], w @ c["valid"]),
        "false_defect_rate": _ratio(w @ c["false_def"], w @ c["normal"]),
        "q3_accuracy": _ratio(w @ c["q3_ok"], w @ c["q3_n"]),
        "q1q3_contradiction": _ratio(w @ c["q13_contra"], w @ c["q13_valid"]),
    }


def paired_bootstrap(pairs, loc_pairs, base_choice, new_choice, n_boot: int = 2000, seed: int = 0) -> dict:
    a = image_components(pairs, loc_pairs, base_choice)
    b = image_components(pairs, loc_pairs, new_choice)
    n = len(pairs)
    w = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot).astype(float)
    ones = np.ones((1, n))
    ma, mb = metrics_from_weights(a, w), metrics_from_weights(b, w)
    pa, pb = metrics_from_weights(a, ones), metrics_from_weights(b, ones)
    out = {}
    for k in BOOT_METRICS:
        delta = mb[k] - ma[k]
        lo, hi = float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))
        out[k] = {"base": float(pa[k][0]), "new": float(pb[k][0]), "delta": float(pb[k][0] - pa[k][0]),
                  "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)}
    return out


def significance_rows(pairs, loc_pairs, found, models, comparisons=MAIN_PAIRS, n_boot: int = 2000) -> list[dict]:
    rows = []
    for model in models:
        for base_cond, new_cond in comparisons:
            if (model, base_cond) not in found or (model, new_cond) not in found:
                continue
            stats = paired_bootstrap(pairs, loc_pairs, found[(model, base_cond)], found[(model, new_cond)], n_boot)
            for metric, st in stats.items():
                rows.append({"model_key": model, "baseline": base_cond, "condition": new_cond, "metric": metric,
                             **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in st.items()},
                             "n_boot": n_boot})
    return rows


def main(argv: Sequence[str] | None = None, *, load_qa=v2.load_gold_qa_v2) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--image-only-csv", type=Path)
    p.add_argument("--reference-dir", type=Path, help="gold_reference_v2 folder (condition/predictions/model.jsonl)")
    p.add_argument("--vx-pred-dir", type=Path, help="vision expert full eval/predictions")
    p.add_argument("--combined-dir", type=Path, help="reference+expert predictions")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--n-boot", type=int, default=2000)
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
    complete = {k: v for k, v in found.items() if not any(s["model_key"] == k[0] and s["condition"] == k[1] for s in skipped)}
    write_csv(args.out_dir / "significance_main.csv",
              significance_rows(pairs, loc_pairs, complete, args.models, n_boot=args.n_boot))
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
        "significance": {"pairs": [list(x) for x in MAIN_PAIRS], "method": f"image-level paired bootstrap, "
                         f"{args.n_boot} resamples, seed 0, two-sided 95% CI; significant = CI excludes 0",
                         "note": "consistency_error / false_defect_rate / q1q3_contradiction: lower is better"},
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"{len(out)} model x condition rows, {len(skipped)} skipped (incomplete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
