#!/usr/bin/env python3
"""Gold-v2 offline agent / ensemble analysis (no new LVLM inference).

Every policy is composed from predictions that already exist for the same 246-image
gold-v2 set: image-only, normal-reference-only, PatchCore-score expert-only and
normal-reference + PatchCore-score combined. A routed policy (e.g. "ask again with the
expert score only when the first answer conflicts with PatchCore") gives the same answers
as running it live, because the second-stage input equals an already-run condition.
Q1 / Q2 / Q3 are scored separately so a policy only counts as better when all three move.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_gold_baselines as base  # noqa: E402
import run_gold_v2_baselines as v2  # noqa: E402

Q1, Q2, Q3 = "anomaly_detection", "defect_classification", "defect_localization"
QTYPES = (Q1, Q2, Q3)
CONDITIONS = ("image_only", "normal_ref", "expert_only", "combined")
MODELS = ("qwen3_5_4b", "internvl3_5_8b", "minicpm_v_4_5", "llava_onevision_7b")
METRIC_KEYS = ("Q1_anomaly_f1", "Q2_defect_macro_f1", "Q3_loc_acc")
TIE_MODEL = "internvl3_5_8b"

Preds = dict[tuple[str, str], dict[str, str | None]]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _choice(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text and text.lower() != "nan" else None


def load_predictions(
    models: Sequence[str],
    image_only_csv: Path,
    normal_ref_dir: Path,
    expert_dir: Path,
    combined_dir: Path,
    expert_pair: str = "patchcore__score",
) -> tuple[Preds, dict[str, float]]:
    preds: Preds = {}
    with Path(image_only_csv).open(newline="", encoding="utf-8-sig") as handle:
        image_rows = list(csv.DictReader(handle))
    percentiles: dict[str, float] = {}
    for model in models:
        preds[(model, "image_only")] = {
            str(r["id"]): _choice(r.get("predicted_choice")) for r in image_rows if r.get("model_key") == model
        }
        preds[(model, "normal_ref")] = {
            str(r["id"]): _choice(r.get("predicted_choice")) for r in read_jsonl(normal_ref_dir / f"{model}.jsonl")
        }
        preds[(model, "expert_only")] = {
            str(r["id"]): _choice(r.get("predicted_choice"))
            for r in read_jsonl(expert_dir / model / f"{expert_pair}.jsonl")
        }
        combined = read_jsonl(combined_dir / model / f"{expert_pair}.jsonl")
        preds[(model, "combined")] = {str(r["id"]): _choice(r.get("predicted_choice")) for r in combined}
        for r in combined:
            percentiles[str(r["core_id"])] = float(r["score_percentile_full"])
    return preds, percentiles


def check_coverage(rows: Sequence[Mapping], preds: Preds, percentiles: Mapping[str, float]) -> None:
    for key, by_id in preds.items():
        missing = [str(r["id"]) for r in rows if str(r["id"]) not in by_id]
        if missing:
            raise RuntimeError(f"{key}: {len(missing)} predictions missing, e.g. {missing[:2]}")
    cores = {str(r["core_id"]) for r in rows}
    if not cores <= set(percentiles):
        raise RuntimeError(f"PatchCore percentile missing for {len(cores - set(percentiles))} images")


def group_by_core(rows: Sequence[Mapping]) -> dict[str, dict[str, Mapping]]:
    out: dict[str, dict[str, Mapping]] = collections.defaultdict(dict)
    for row in rows:
        out[str(row["core_id"])][str(row["question_type"])] = row
    return out


def score(rows: Sequence[Mapping], choice: Mapping[str, str | None]) -> dict[str, float]:
    q1 = [r for r in rows if r["question_type"] == Q1]
    q2 = [r for r in rows if r["question_type"] == Q2]
    q3 = [r for r in rows if r["question_type"] == Q3]
    p, rc, f1 = base.binary_f1(
        [r["anomaly_gt"] for r in q1], [base.predicted_semantic(r, choice[str(r["id"])]) for r in q1], "anomalous"
    )
    return {
        "Q1_anomaly_f1": f1,
        "Q1_precision": p,
        "Q1_recall": rc,
        "Q2_defect_macro_f1": base.macro_f1(
            [r["defect_gt"] for r in q2], [base.predicted_semantic(r, choice[str(r["id"])]) for r in q2]
        ),
        "Q3_loc_acc": (sum(choice[str(r["id"])] == r["answer"] for r in q3) / len(q3)) if q3 else 0.0,
    }


def _components(rows: Sequence[Mapping], choice: Mapping[str, str | None], cores: Sequence[str]) -> dict:
    """Per-image count vectors so any metric can be recomputed for a resampled set of images."""
    index = {c: i for i, c in enumerate(cores)}
    n = len(cores)
    zeros = lambda: np.zeros(n)  # noqa: E731
    comp = {"q1_tp": zeros(), "q1_fp": zeros(), "q1_fn": zeros(), "q3_ok": zeros(), "q3_n": zeros(), "q2": {}}
    labels = sorted({str(r["defect_gt"]) for r in rows if r["question_type"] == Q2})
    for label in labels:
        comp["q2"][label] = {"tp": zeros(), "fp": zeros(), "fn": zeros()}
    for row in rows:
        i = index[str(row["core_id"])]
        pred = choice[str(row["id"])]
        if row["question_type"] == Q1:
            p, g = base.predicted_semantic(row, pred) == "anomalous", row["anomaly_gt"] == "anomalous"
            comp["q1_tp"][i] += p and g
            comp["q1_fp"][i] += p and not g
            comp["q1_fn"][i] += g and not p
        elif row["question_type"] == Q2:
            p, g = base.predicted_semantic(row, pred), str(row["defect_gt"])
            for label, c in comp["q2"].items():
                c["tp"][i] += p == label and g == label
                c["fp"][i] += p == label and g != label
                c["fn"][i] += g == label and p != label
        elif row["question_type"] == Q3:
            comp["q3_ok"][i] += pred == row["answer"]
            comp["q3_n"][i] += 1
    return comp


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def metrics_from_weights(comp: Mapping, weights: np.ndarray) -> dict[str, np.ndarray]:
    """``weights``: (B, n_images) resampling counts. Matches ``score`` when weights are all ones."""
    w = np.atleast_2d(weights)
    q1 = _f1(w @ comp["q1_tp"], w @ comp["q1_fp"], w @ comp["q1_fn"])
    f1s, present = [], []
    for c in comp["q2"].values():
        f1s.append(_f1(w @ c["tp"], w @ c["fp"], w @ c["fn"]))
        present.append((w @ (c["tp"] + c["fn"])) > 0)
    f1s, present = np.array(f1s), np.array(present)
    q2 = (f1s * present).sum(axis=0) / np.maximum(present.sum(axis=0), 1)
    n3 = w @ comp["q3_n"]
    q3 = np.divide(w @ comp["q3_ok"], n3, out=np.zeros_like(n3, dtype=float), where=n3 > 0)
    return {"Q1_anomaly_f1": q1, "Q2_defect_macro_f1": q2, "Q3_loc_acc": q3}


def paired_bootstrap(rows: Sequence[Mapping], base_choice: Mapping[str, str | None],
                     new_choice: Mapping[str, str | None], n_boot: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Image-level paired bootstrap of (new - base) for Q1/Q2/Q3. p = share of resamples with delta <= 0."""
    cores = sorted({str(r["core_id"]) for r in rows})
    a, b = _components(rows, base_choice, cores), _components(rows, new_choice, cores)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(cores), np.full(len(cores), 1 / len(cores)), size=n_boot).astype(float)
    full = np.ones((1, len(cores)))
    ma, mb = metrics_from_weights(a, weights), metrics_from_weights(b, weights)
    pa, pb = metrics_from_weights(a, full), metrics_from_weights(b, full)
    out = {}
    for key in METRIC_KEYS:
        delta = mb[key] - ma[key]
        out[key] = {"delta": float(pb[key][0] - pa[key][0]),
                    "ci_low": float(np.percentile(delta, 2.5)), "ci_high": float(np.percentile(delta, 97.5)),
                    "p_le_0": float((delta <= 0).mean())}
    return out


def score_conflict(q1_row: Mapping, choice: str | None, percentile: float, low: float, high: float) -> bool:
    label = base.predicted_semantic(q1_row, choice)
    return label is None or (label == "normal" and percentile >= high) or (label == "anomalous" and percentile <= low)


def q12_inconsistent(q1_row: Mapping, q1_choice: str | None, q2_row: Mapping, q2_choice: str | None) -> bool:
    a = base.predicted_semantic(q1_row, q1_choice)
    d = base.predicted_semantic(q2_row, q2_choice)
    return a is None or d is None or ((a == "normal") != (d == "normal"))


def compose(
    rows: Sequence[Mapping],
    preds: Preds,
    model: str,
    source: Mapping[str, str],
    routed: Iterable[str] = (),
    routed_source: Mapping[str, str] | None = None,
) -> tuple[dict[str, str | None], float]:
    """Answer each question from ``source[qtype]``; for routed images, re-ask the
    question types in ``routed_source`` with that condition. Returns (answers, LVLM calls per question)."""
    routed = set(routed)
    routed_source = dict(routed_source or {})
    choice: dict[str, str | None] = {}
    extra = 0
    for row in rows:
        qtype = str(row["question_type"])
        cond = source[qtype]
        if qtype in routed_source and str(row["core_id"]) in routed:
            cond = routed_source[qtype]
            extra += 1
        choice[str(row["id"])] = preds[(model, cond)][str(row["id"])]
    return choice, 1 + extra / len(rows)


def vote(
    rows: Sequence[Mapping], voters: Sequence[Mapping[str, str | None]], tie_break: Mapping[str, str | None]
) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for row in rows:
        key = str(row["id"])
        ranked = collections.Counter(v[key] for v in voters).most_common()
        top, count = ranked[0]
        tied = len(ranked) > 1 and ranked[1][1] == count
        out[key] = tie_break[key] if tied else top
    return out


def score_only(rows: Sequence[Mapping], percentiles: Mapping[str, float], threshold: float,
               fallback: Mapping[str, str | None]) -> dict[str, str | None]:
    out = dict(fallback)
    for row in rows:
        if row["question_type"] != Q1:
            continue
        want = "Anomalous." if percentiles[str(row["core_id"])] >= threshold else "Normal."
        out[str(row["id"])] = next(k for k, v in dict(row["options"]).items() if v == want)
    return out


def routing_sets(rows: Sequence[Mapping], first: Mapping[str, str | None], percentiles: Mapping[str, float],
                 low: float, high: float) -> tuple[set[str], set[str]]:
    """Images whose first-pass (normal reference) answers conflict with PatchCore / contradict Q1 vs Q2."""
    cores = group_by_core(rows)
    conflict = {c for c, qs in cores.items()
                if score_conflict(qs[Q1], first[str(qs[Q1]["id"])], percentiles[c], low, high)}
    inconsistent = {c for c, qs in cores.items()
                    if q12_inconsistent(qs[Q1], first[str(qs[Q1]["id"])], qs[Q2], first[str(qs[Q2]["id"])])}
    return conflict, inconsistent


def run_policies(
    rows: Sequence[Mapping], preds: Preds, percentiles: Mapping[str, float], models: Sequence[str],
    low: float = 10.0, high: float = 90.0, score_only_threshold: float = 50.0,
) -> tuple[list[dict], list[dict], dict[tuple[str, str], dict]]:
    cores = group_by_core(rows)
    results: list[dict] = []
    route_stats: list[dict] = []
    choices: dict[tuple[str, str], dict] = {}
    all_q = lambda cond: {q: cond for q in QTYPES}  # noqa: E731
    split = {Q1: "normal_ref", Q2: "normal_ref", Q3: "image_only"}
    q12_combined = {Q1: "combined", Q2: "combined"}

    def add(model: str, policy: str, composed: tuple[Mapping, float], note: str = "") -> None:
        choice, calls = composed
        choices[(model, policy)] = dict(choice)
        results.append({"model_key": model, "policy": policy, **score(rows, choice),
                        "avg_lvlm_calls_per_q": round(calls, 4), "note": note})

    for model in models:
        for cond in CONDITIONS:
            add(model, f"fixed_{cond}", compose(rows, preds, model, all_q(cond)))
        nr = preds[(model, "normal_ref")]
        conflict, inconsistent = routing_sets(rows, nr, percentiles, low, high)
        either = conflict | inconsistent
        route_stats.append({"model_key": model, "images": len(cores),
                            "score_conflict_rate": len(conflict) / len(cores),
                            "q1q2_inconsistent_rate": len(inconsistent) / len(cores),
                            "either_rate": len(either) / len(cores)})

        add(model, "A1_all", compose(rows, preds, model, all_q("normal_ref"), conflict, all_q("combined")),
            "score conflict -> Q1-Q3 re-asked with combined")
        add(model, "A1_q1", compose(rows, preds, model, all_q("normal_ref"), conflict, {Q1: "combined"}),
            "score conflict -> Q1 re-asked with combined")
        add(model, "A1_q12", compose(rows, preds, model, all_q("normal_ref"), conflict, q12_combined),
            "score conflict -> Q1,Q2 re-asked with combined")
        add(model, "S_q3_image_only", compose(rows, preds, model, split),
            "Q3 from image-only (post-hoc rule)")
        add(model, "A1_q12+S", compose(rows, preds, model, split, conflict, q12_combined),
            "A1_q12 + Q3 from image-only")
        add(model, "A5_conflict_or_inconsistent+S", compose(rows, preds, model, split, either, q12_combined),
            "score conflict or Q1/Q2 inconsistency -> Q1,Q2 combined; Q3 image-only")
        votes = vote(rows, [preds[(model, c)] for c in ("normal_ref", "expert_only", "combined")], nr)
        add(model, "A4_condition_vote", (votes, 3.0), "majority of normal_ref / expert_only / combined")

    tie_model = TIE_MODEL if TIE_MODEL in models else models[0]
    add("patchcore_only", f"A2_score_only_t{int(score_only_threshold)}",
        (score_only(rows, percentiles, score_only_threshold, preds[(tie_model, "normal_ref")]), 0.0),
        f"Q1 only; Q2,Q3 from {tie_model} normal_ref")
    for cond in ("normal_ref", "combined"):
        add("vote_4models", f"A3_model_vote_{cond}",
            (vote(rows, [preds[(m, cond)] for m in models], preds[(tie_model, cond)]), float(len(models))),
            f"tie -> {tie_model}")
    add_flags(results, models)
    return results, route_stats, choices


def add_flags(results: list[dict], models: Sequence[str]) -> None:
    """d_* vs normal reference, and whether all three questions improve.
    ``all3_at_least_best_fixed``: every question >= the model's best fixed condition and at least one strictly
    better (policies that copy Q3 from image-only can tie but never exceed it, so strict '>' is not usable)."""
    tie_model = TIE_MODEL if TIE_MODEL in models else models[0]
    reference = {r["model_key"]: r for r in results if r["policy"] == "fixed_normal_ref"}
    best = {m: {k: max(r[k] for r in results if r["model_key"] == m and r["policy"].startswith("fixed_"))
                for k in METRIC_KEYS} for m in models}
    for r in results:
        key = r["model_key"] if r["model_key"] in reference else tie_model
        ref, top = reference[key], best[key]
        for k in METRIC_KEYS:
            r[f"d_{k.split('_')[0]}_vs_normal_ref"] = round(r[k] - ref[k], 6)
        r["all3_up_vs_normal_ref"] = all(r[k] > ref[k] for k in METRIC_KEYS)
        r["all3_beat_best_fixed"] = all(r[k] > top[k] for k in METRIC_KEYS)
        r["all3_at_least_best_fixed"] = (all(r[k] >= top[k] - 1e-12 for k in METRIC_KEYS)
                                         and any(r[k] > top[k] + 1e-12 for k in METRIC_KEYS))


SIGNIFICANCE_PAIRS = (
    # (baseline policy, new policy); the S_q3_image_only baselines isolate the agent from the Q3 image-only rule
    ("fixed_normal_ref", "A1_q12"),
    ("fixed_normal_ref", "S_q3_image_only"),
    ("fixed_normal_ref", "A1_q12+S"),
    ("fixed_normal_ref", "A5_conflict_or_inconsistent+S"),
    ("fixed_normal_ref", "A4_condition_vote"),
    ("S_q3_image_only", "A1_q12+S"),
    ("S_q3_image_only", "A5_conflict_or_inconsistent+S"),
)


def significance(rows: Sequence[Mapping], choices: Mapping[tuple[str, str], Mapping], models: Sequence[str],
                 pairs: Sequence[tuple[str, str]] = SIGNIFICANCE_PAIRS, n_boot: int = 2000,
                 seed: int = 0) -> list[dict]:
    out = []
    for model in models:
        for base_policy, new_policy in pairs:
            if (model, base_policy) not in choices or (model, new_policy) not in choices:
                continue
            stats = paired_bootstrap(rows, choices[(model, base_policy)], choices[(model, new_policy)], n_boot, seed)
            row = {"model_key": model, "baseline": base_policy, "policy": new_policy, "n_boot": n_boot}
            for key, st in stats.items():
                q = key.split("_")[0]
                row.update({f"{q}_delta": round(st["delta"], 6), f"{q}_ci_low": round(st["ci_low"], 6),
                            f"{q}_ci_high": round(st["ci_high"], 6), f"{q}_p_le_0": round(st["p_le_0"], 4)})
            out.append(row)
    return out


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None, load_qa: Callable[[Path], list[dict]] = v2.load_gold_qa_v2) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--image-only-csv", type=Path, required=True)
    p.add_argument("--normal-ref-dir", type=Path, required=True)
    p.add_argument("--expert-dir", type=Path, required=True)
    p.add_argument("--combined-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--expert-pair", default="patchcore__score")
    p.add_argument("--conflict-low", type=float, default=10.0)
    p.add_argument("--conflict-high", type=float, default=90.0)
    p.add_argument("--score-only-threshold", type=float, default=50.0)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)

    rows = load_qa(args.qa_dir)
    preds, percentiles = load_predictions(args.models, args.image_only_csv, args.normal_ref_dir,
                                          args.expert_dir, args.combined_dir, args.expert_pair)
    check_coverage(rows, preds, percentiles)
    results, route_stats, choices = run_policies(rows, preds, percentiles, args.models, args.conflict_low,
                                                 args.conflict_high, args.score_only_threshold)
    write_csv(args.out_dir / "leaderboard.csv", results)
    write_csv(args.out_dir / "route_stats.csv", route_stats)
    write_csv(args.out_dir / "significance.csv", significance(rows, choices, args.models, n_boot=args.n_boot))
    metadata = {
        "protocol": "gold_v2_offline_agent",
        "new_lvlm_inference": False,
        "gold_images": len(group_by_core(rows)),
        "qa_questions": len(rows),
        "models": list(args.models),
        "expert_pair": args.expert_pair,
        "conflict_thresholds": {"low": args.conflict_low, "high": args.conflict_high,
                                "source": "AgentIAD-lite defaults, fixed before looking at gold-v2 results"},
        "score_percentile": "within-expert, within-asset percentile over the 246-image cohort (transductive)",
        "post_hoc_rules": ["S_q3_image_only: Q3 answered from image-only because normal reference lowered "
                           "localization in earlier gold-v2 runs"],
        "calls_note": "avg_lvlm_calls_per_q counts one call per question plus one per re-asked question",
        "significance": f"image-level paired bootstrap, {args.n_boot} resamples, seed 0; "
                        "p_le_0 = share of resamples where the policy is not better than its baseline",
        "criteria": {"all3_beat_best_fixed": "every question > best fixed condition (strict)",
                     "all3_at_least_best_fixed": "every question >= best fixed and at least one >"},
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    for row in route_stats:
        print(json.dumps(row, ensure_ascii=False))
    print(f"wrote {len(results)} policy rows -> {args.out_dir / 'leaderboard.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
