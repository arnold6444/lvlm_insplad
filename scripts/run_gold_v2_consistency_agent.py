#!/usr/bin/env python3
"""Gold-v2 consistency re-ask agent (live LVLM calls only on routed images).

Stage 1 is the existing normal-reference run (``[normal reference, target]``), reused without inference.
An image is routed when its stage-1 answers contradict each other (Q1 normal but Q2 a defect type, or Q1
anomalous but Q2 normal) and, in the ``self_score`` variant, also when the Q1 answer strongly conflicts with
the PatchCore percentile. Routed images get Q1 and Q2 asked again with the same two images plus a review note
that quotes the stage-1 answers (and, for ``self_score``, the PatchCore score sentence). Q3 is left to the
fixed policies; ``+S`` rows take Q3 from image-only as in the offline analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gold_baselines as base  # noqa: E402
import run_gold_v2_baselines as v2  # noqa: E402
import run_gold_v2_offline_agent as oa  # noqa: E402

Q1, Q2, Q3 = oa.Q1, oa.Q2, oa.Q3
VARIANTS = ("self", "self_score")
PROMPT_VERSION = 1


def option_text(row: Mapping, choice: str | None) -> str:
    return str(dict(row["options"]).get(choice, "")) if choice else ""


def build_prompt(row: Mapping, first_q1: str, first_q2: str, *, inconsistent: bool, score_conflict: bool,
                 variant: str, percentile: float, expert: str = "patchcore") -> str:
    options = "\n".join(f"{k}. {v}" for k, v in dict(row["options"]).items())
    q1_text = first_q1 or "(no valid answer)"
    q2_text = first_q2 or "(no valid answer)"
    review = (
        "This is a second look at the same images. In a first pass you answered the anomaly question with "
        f"\"{q1_text}\" and the defect-type question with \"{q2_text}\". "
    )
    if inconsistent:
        review += ("These two answers contradict each other: a normal part has no defect type, "
                   "and an anomalous part must have one. ")
    if variant == "self_score":
        review += (
            f"The {expert} vision-expert anomaly score of the TARGET is at the {percentile:.1f}th percentile "
            "among same-asset images in this fixed unlabeled benchmark cohort. Higher is more suspicious. "
            "This relative score uses no ground-truth label and may be wrong. "
        )
        if score_conflict:
            review += f"The {expert} score disagrees with your first-pass anomaly answer. "
        review += "Do not assume the vision expert is correct. "
    review += "Re-examine the TARGET carefully; your first-pass answers may be wrong."
    return (
        "You are an industrial visual inspector. "
        "Image 1 is a verified NORMAL reference of the same asset category. "
        "Image 2 is the TARGET image. Use Image 1 only as the normal visual standard and answer about Image 2. "
        + review
        + "\n\n"
        f"Question: {row['question']}\n{options}\n\n"
        "Return exactly one option letter and nothing else."
    )


def routed_images(rows: Sequence[Mapping], first: Mapping[str, str | None], percentiles: Mapping[str, float],
                  variant: str, low: float, high: float) -> tuple[set[str], set[str], set[str]]:
    conflict, inconsistent = oa.routing_sets(rows, first, percentiles, low, high)
    routed = inconsistent if variant == "self" else inconsistent | conflict
    return routed, conflict, inconsistent


def normal_reference_by_core(rows: Sequence[Mapping]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for row in rows:
        value = str(row.get("normal_reference_image") or "")
        if value:
            if refs.get(str(row["core_id"]), value) != value:
                raise RuntimeError(f"{row['core_id']}: inconsistent normal_reference_image")
            refs[str(row["core_id"])] = value
    missing = sorted({str(r["core_id"]) for r in rows} - set(refs))
    if missing:
        raise RuntimeError(f"normal_reference_image missing for {len(missing)} images, e.g. {missing[:3]}")
    return refs


def reusable(old: Mapping | None, model_key: str, variant: str) -> bool:
    return bool(old and not old.get("error") and str(old.get("output", "")).strip()
                and old.get("model_key") == model_key and old.get("variant") == variant
                and old.get("prompt_version") == PROMPT_VERSION)


def run_variant(model_key: str, variant: str, rows: Sequence[Mapping], first: Mapping[str, str | None],
                percentiles: Mapping[str, float], images_for: Callable[[Mapping], list[Path]],
                out_dir: Path, holder, infer: Callable, low: float, high: float,
                checkpoint_every: int) -> tuple[Path, dict]:
    routed, conflict, inconsistent = routed_images(rows, first, percentiles, variant, low, high)
    cores = oa.group_by_core(rows)
    pred_path = out_dir / "predictions" / model_key / f"{variant}.jsonl"
    existing = {str(r["id"]): r for r in (oa.read_jsonl(pred_path) if pred_path.is_file() else [])}
    preds: list[dict] = []
    called = reused = errors = 0
    for core in sorted(routed):
        q1_row, q2_row = cores[core][Q1], cores[core][Q2]
        first_q1 = option_text(q1_row, first[str(q1_row["id"])])
        first_q2 = option_text(q2_row, first[str(q2_row["id"])])
        for row in (q1_row, q2_row):
            old = existing.get(str(row["id"]))
            if reusable(old, model_key, variant):
                preds.append(old)
                reused += 1
                continue
            prompt = build_prompt(row, first_q1, first_q2, inconsistent=core in inconsistent,
                                  score_conflict=core in conflict, variant=variant, percentile=percentiles[core])
            adapter = holder.get()
            started = time.time()
            output, error = "", ""
            try:
                output = str(infer(adapter, images_for(row), prompt))
            except Exception as exc:  # keep going; the row is retried on the next run
                error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                errors += 1
            preds.append({
                "id": row["id"], "core_id": core, "question_type": row["question_type"],
                "model_key": model_key, "variant": variant, "prompt_version": PROMPT_VERSION,
                "routed_by": ";".join(x for x, s in (("inconsistent", inconsistent), ("score_conflict", conflict))
                                      if core in s),
                "first_choice": first[str(row["id"])], "output": output,
                "predicted_choice": base.parse_choice(output, {str(k).upper() for k in row["options"]}),
                "latency_s": round(time.time() - started, 3), "error": error,
            })
            called += 1
            if checkpoint_every > 0 and called % checkpoint_every == 0:
                base.write_jsonl(pred_path, preds)
                print(f"[{model_key}/{variant}] {len(preds)}/{2 * len(routed)} checkpoint", flush=True)
    base.write_jsonl(pred_path, preds)
    status = {"model_key": model_key, "variant": variant, "routed_images": len(routed),
              "reasked_questions": 2 * len(routed), "called_this_run": called, "reused_predictions": reused,
              "row_errors": errors, "status": "complete" if errors == 0 else "has_errors"}
    return pred_path, status


def compose_live(rows: Sequence[Mapping], first: Mapping[str, str | None], reasked: Mapping[str, str | None],
                 q3_source: Mapping[str, str | None] | None = None) -> tuple[dict, float]:
    choice = dict(first)
    for key, value in reasked.items():
        choice[key] = value
    if q3_source is not None:
        for row in rows:
            if row["question_type"] == Q3:
                choice[str(row["id"])] = q3_source[str(row["id"])]
    return choice, 1 + len(reasked) / len(rows)


def main(argv: Sequence[str] | None = None, *, load_qa: Callable[[Path], list[dict]] = v2.load_gold_qa_v2,
         loader: Callable | None = None, infer: Callable | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--sup-zip", type=Path, required=True)
    p.add_argument("--unsup-zip", type=Path, required=True)
    p.add_argument("--image-only-csv", type=Path, required=True)
    p.add_argument("--normal-ref-dir", type=Path, required=True)
    p.add_argument("--expert-dir", type=Path, required=True)
    p.add_argument("--combined-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=list(oa.MODELS))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--expert-pair", default="patchcore__score")
    p.add_argument("--conflict-low", type=float, default=10.0)
    p.add_argument("--conflict-high", type=float, default=90.0)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)

    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer  # noqa: N813
    rows = load_qa(args.qa_dir)
    preds, percentiles = oa.load_predictions(args.models, args.image_only_csv, args.normal_ref_dir,
                                             args.expert_dir, args.combined_dir, args.expert_pair)
    oa.check_coverage(rows, preds, percentiles)

    normal_by_core = normal_reference_by_core(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_paths = base.prepare_images(rows, args.sup_zip, args.out_dir / "_images" / "target")
    normal_paths = {m: base.safe_extract(args.unsup_zip, m, args.out_dir / "_images" / "normal")
                    for m in sorted(set(normal_by_core.values()))}

    def images_for(row: Mapping) -> list[Path]:
        return [normal_paths[normal_by_core[str(row["core_id"])]], target_paths[str(row["image"])]]

    results, _, choices = oa.run_policies(rows, preds, percentiles, args.models,
                                          args.conflict_low, args.conflict_high)
    keep = {"fixed_image_only", "fixed_normal_ref", "fixed_expert_only", "fixed_combined",
            "S_q3_image_only", "A5_conflict_or_inconsistent+S"}
    results = [r for r in results if r["model_key"] in args.models and r["policy"] in keep]
    for r in results:
        r["source"] = "offline"
    statuses: list[dict] = []
    for model_key in args.models:
        spec = base.MODEL_REGISTRY[model_key]
        holder = base.LazyHFAdapter(spec, loader=loader)
        first = preds[(model_key, "normal_ref")]
        try:
            for variant in args.variants:
                print(f"\n=== {model_key} / consistency re-ask ({variant}) ===", flush=True)
                pred_path, status = run_variant(model_key, variant, rows, first, percentiles, images_for,
                                                args.out_dir, holder, infer, args.conflict_low,
                                                args.conflict_high, args.checkpoint_every)
                statuses.append(status)
                reasked = {str(r["id"]): r.get("predicted_choice") for r in oa.read_jsonl(pred_path)}
                for name, q3 in ((f"R_{variant}", None), (f"R_{variant}+S", preds[(model_key, "image_only")])):
                    choice, calls = compose_live(rows, first, reasked, q3)
                    choices[(model_key, name)] = choice
                    results.append({"model_key": model_key, "policy": name, **oa.score(rows, choice),
                                    "avg_lvlm_calls_per_q": round(calls, 4), "source": "live",
                                    "note": f"re-ask Q1,Q2 on {status['routed_images']} routed images"})
        finally:
            holder.release()

    oa.add_flags(results, args.models)
    pairs = []
    for variant in args.variants:
        pairs += [("fixed_normal_ref", f"R_{variant}"), ("fixed_normal_ref", f"R_{variant}+S"),
                  ("S_q3_image_only", f"R_{variant}+S"), ("A5_conflict_or_inconsistent+S", f"R_{variant}+S")]
    oa.write_csv(args.out_dir / "leaderboard.csv", results)
    oa.write_csv(args.out_dir / "significance.csv",
                 oa.significance(rows, choices, args.models, pairs, n_boot=args.n_boot))
    (args.out_dir / "run_status.json").write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    metadata = {
        "protocol": "gold_v2_consistency_agent",
        "prompt_version": PROMPT_VERSION,
        "models": list(args.models),
        "variants": {"self": "route Q1/Q2-inconsistent images; review note quotes stage-1 answers",
                     "self_score": "route inconsistent or PatchCore-conflict images; note adds the score sentence"},
        "stage1": "existing normal_reference predictions (no new inference)",
        "images": ["normal_reference", "target"],
        "conflict_thresholds": {"low": args.conflict_low, "high": args.conflict_high,
                                "source": "AgentIAD-lite defaults, fixed before looking at gold-v2 results"},
        "q3": "R_* keep stage-1 Q3; R_*+S take Q3 from image-only (post-hoc rule)",
        "significance": f"image-level paired bootstrap, {args.n_boot} resamples, seed 0",
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    errors = sum(s["row_errors"] for s in statuses)
    print(json.dumps(statuses, ensure_ascii=False, indent=2))
    if errors:
        print(f"WARNING: {errors} rows failed; re-run the same cell to retry only those rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
