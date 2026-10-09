#!/usr/bin/env python3
"""Training-free verification agent for the gold-v2 QA (development set first, gold once at the end).

Stage 1 reuses existing per-question answers (default: the same-flight-excluded retrieved normal reference,
``normal_ref_retrieved_diffdate``). An image is sent to verification when its stage-1 answers contradict each other
(Q1 normal but Q2 a defect, or Q1 anomalous but Q2 normal) or when the image-only Q1 answer disagrees with the
stage-1 Q1 answer. Verification asks the defect-type question once more with new evidence and rewrites the image's
answers from that single answer, so verified images are consistent by construction:

- ``reask``  : same two images, a neutral "second look" sentence (no new evidence; controls for asking again)
- ``zoom``   : enlarged central crops of the reference and the target (InsPLAD images are object-centred)
- ``define`` : the same two images plus short visual definitions of every option (domain knowledge)
- ``zoom_define``: both
- ``zoom_define_all``: zoom_define on every image (is the trigger worth it?)
- ``rule_q1_first``: no extra call; Q2/Q3 follow the stage-1 Q1 answer (the structural baseline)

Q1 becomes Normal/Anomalous from the verified defect type; Q3 (gold only) becomes "no defect region" when the
verified type is normal. Reported per variant: Q1 F1, Q2 macro-F1, contradiction, joint accuracy, FDR (and Q3 on
gold), LVLM calls per question, answer flips (changed, changed to correct, changed to wrong) and paired bootstrap CIs
against stage 1.
"""
from __future__ import annotations

import argparse
import csv
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
import run_gold_v2_consistency_metrics as cons  # noqa: E402

Q1, Q2, Q3 = cons.Q1, cons.Q2, cons.Q3
LLM_VARIANTS = ("reask", "zoom", "define", "zoom_define", "zoom_define_all")
VARIANTS = LLM_VARIANTS + ("rule_q1_first",)
PROMPT_VERSION = 1
ZOOM_BOX = (0.2, 0.2, 0.8, 0.8)            # central 60% x 60%
ZOOM_SCALE = 2.0
ZOOM_MAX_SIDE = 1344
# Visual definitions written from general inspection knowledge (not from gold images).
DEFINITIONS = {
    "Normal / no defect.": "all parts are present and intact; surfaces keep their usual uniform colour without "
                           "orange-brown stains, gaps or foreign material.",
    "Rust / corrosion.": "orange, reddish-brown or dark-brown discoloration, scaling or pitting on metal hardware "
                         "such as clamps, shackles, bolts or caps.",
    "Missing cap / missing insulator component.": "a disc, cap or other component that belongs to the regular "
                                                  "repeated structure is absent, leaving a visible gap or an exposed pin.",
    "Bird nest / foreign nesting material.": "a mass of twigs, straw, grass or other debris lodged on or around the "
                                             "hardware.",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def build_prompt(q2_row: Mapping, variant: str) -> str:
    zoom = variant.startswith("zoom")
    define = "define" in variant
    intro = ("You are an industrial visual inspector. Image 1 is a verified NORMAL reference of the same asset "
             "category. Image 2 is the TARGET image. ")
    if zoom:
        intro += ("Both images are enlarged central crops; the inspected asset is centred in both. ")
    intro += "Use Image 1 only as the normal visual standard and answer about Image 2. "
    intro += "This is a second look: re-examine the TARGET carefully before answering. "
    options = dict(q2_row["options"])
    body = ""
    if define:
        body = "Visual definitions of the options:\n" + "\n".join(
            f"- {text} {DEFINITIONS[text]}" for text in options.values()) + "\n\n"
    lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (intro + "\n\n" + body + f"Question: {q2_row['question']}\n{lines}\n\n"
            "Return exactly one option letter and nothing else.")


def zoom_image(src: Path, out: Path) -> Path:
    if out.is_file() and out.stat().st_size > 0:
        return out
    from PIL import Image

    with Image.open(src) as h:
        img = h.convert("RGB")
    w, h_ = img.size
    crop = img.crop((round(ZOOM_BOX[0] * w), round(ZOOM_BOX[1] * h_), round(ZOOM_BOX[2] * w), round(ZOOM_BOX[3] * h_)))
    scale = min(ZOOM_SCALE, ZOOM_MAX_SIDE / max(crop.size))
    if scale > 1:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.BICUBIC)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.png")
    crop.save(tmp, format="PNG")
    tmp.replace(out)
    return out


def letter_of(row: Mapping, text: str) -> str | None:
    return next((k for k, v in dict(row["options"]).items() if v == text), None)


def stage1_label(rows_by_core: Mapping[str, Mapping[str, Mapping]], first: Mapping[str, str | None], core: str) -> tuple:
    q = rows_by_core[core]
    a = base.predicted_semantic(q[Q1], first.get(str(q[Q1]["id"])))
    d = base.predicted_semantic(q[Q2], first.get(str(q[Q2]["id"])))
    return a, d


def apply_label(rows_by_core, choice: dict, core: str, label: str) -> None:
    """Rewrite Q1/Q2(/Q3) of one image from a verified defect label (normal or a defect type)."""
    q = rows_by_core[core]
    q2_text = next(t for t, lab in base.DEFECT_TEXT_TO_LABEL.items() if lab == label)
    choice[str(q[Q2]["id"])] = letter_of(q[Q2], q2_text)
    choice[str(q[Q1]["id"])] = letter_of(q[Q1], "Normal." if label == "normal" else "Anomalous.")
    if Q3 in q and label == "normal":
        none = letter_of(q[Q3], cons.NONE_TEXT)
        if none:
            choice[str(q[Q3]["id"])] = none


def triggers(rows_by_core, first, image_only, use: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for core, q in rows_by_core.items():
        reasons = []
        a, d = stage1_label(rows_by_core, first, core)
        if "contradiction" in use and (a is None or d is None or (a == "normal") != (d == "normal")):
            reasons.append("contradiction")
        if "disagreement" in use and image_only is not None:
            ai = base.predicted_semantic(q[Q1], image_only.get(str(q[Q1]["id"])))
            if ai != a:
                reasons.append("disagreement")
        if reasons:
            out[core] = reasons
    return out


def reusable(old: Mapping | None, model_key: str, variant: str) -> bool:
    return bool(old and not old.get("error") and str(old.get("output", "")).strip()
                and old.get("model_key") == model_key and old.get("variant") == variant
                and old.get("prompt_version") == PROMPT_VERSION)


def load_rows(qa_file: Path | None, qa_dir: Path | None, load_qa: Callable) -> list[dict]:
    if qa_file:
        return read_jsonl(qa_file)
    if qa_dir:
        return load_qa(qa_dir)
    raise RuntimeError("--qa-file or --qa-dir is required")


def main(argv: Sequence[str] | None = None, *, load_qa: Callable[[Path], list[dict]] = v2.load_gold_qa_v2,
         loader: Callable | None = None, infer: Callable | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-file", type=Path, help="dev_master_qa_*.jsonl")
    p.add_argument("--qa-dir", type=Path, help="gold_v2_final (Q1/Q2/Q3)")
    p.add_argument("--stage1-dir", type=Path, required=True, help="predictions/<model>/<condition>.jsonl root")
    p.add_argument("--stage1-condition", default="normal_ref_retrieved_diffdate")
    p.add_argument("--image-only-dir", type=Path, help="predictions/<model>/image_only.jsonl root (default: stage1-dir)")
    p.add_argument("--image-only-csv", type=Path, help="alternative: gold prediction_review.csv")
    p.add_argument("--refs-csv", type=Path, required=True, help="retrieved_refs.csv (reference for verification)")
    p.add_argument("--rule", default="diffdate", choices=["any", "diffdate"])
    p.add_argument("--sup-zip", type=Path, required=True)
    p.add_argument("--unsup-zip", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["internvl3_5_8b", "qwen3_5_4b"])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--triggers", nargs="+", default=["contradiction", "disagreement"],
                   choices=["contradiction", "disagreement"])
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)

    if infer is None and any(v in LLM_VARIANTS for v in args.variants):
        from insplad_benchmark.hf_adapters import infer_hf as infer  # noqa: N813
    rows = load_rows(args.qa_file, args.qa_dir, load_qa)
    rows_by_core: dict[str, dict[str, Mapping]] = {}
    for r in rows:
        rows_by_core.setdefault(str(r["core_id"]), {})[r["question_type"]] = r
    with args.refs_csv.open(newline="", encoding="utf-8-sig") as h:
        refs = {r["image"]: r[f"retrieved_{args.rule}"] for r in csv.DictReader(h)}
    missing = sorted({str(q[Q2]["image"]) for q in rows_by_core.values()} - set(refs))
    if missing:
        raise RuntimeError(f"--refs-csv lacks {len(missing)} images, e.g. {missing[:2]}")

    image_only_csv: dict[tuple[str, str], str | None] = {}
    if args.image_only_csv:
        with args.image_only_csv.open(newline="", encoding="utf-8-sig") as h:
            image_only_csv = {(r["model_key"], str(r["id"])): (r.get("predicted_choice") or None) for r in csv.DictReader(h)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zoom_dir = args.out_dir / "_images" / "zoom"
    needs_images = any(v in LLM_VARIANTS for v in args.variants)
    targets, normals = {}, {}
    if needs_images:
        targets = base.prepare_images([q[Q2] for q in rows_by_core.values()], args.sup_zip, args.out_dir / "_images" / "target")
        normals = {m: base.safe_extract(args.unsup_zip, m, args.out_dir / "_images" / "normal") for m in sorted(set(refs.values()))}

    def images_for(core: str, variant: str) -> list[Path]:
        q2 = rows_by_core[core][Q2]
        target, ref = targets[str(q2["image"])], normals[refs[str(q2["image"])]]
        if variant.startswith("zoom"):
            return [zoom_image(ref, zoom_dir / "ref" / f"{core}.png"), zoom_image(target, zoom_dir / "target" / f"{core}.png")]
        return [ref, target]

    results, flips, statuses, route_stats = [], [], [], []
    found: dict[tuple[str, str], dict] = {}
    for model_key in args.models:
        stage1_path = args.stage1_dir / model_key / f"{args.stage1_condition}.jsonl"
        if not stage1_path.is_file():
            raise FileNotFoundError(f"stage-1 predictions missing: {stage1_path}")
        first = {str(r["id"]): r.get("predicted_choice") for r in read_jsonl(stage1_path)}
        if any(str(r["id"]) not in first for r in rows):
            raise RuntimeError(f"{stage1_path}: stage-1 predictions do not cover every question")
        image_only = None
        io_path = (args.image_only_dir or args.stage1_dir) / model_key / "image_only.jsonl"
        if io_path.is_file():
            image_only = {str(r["id"]): r.get("predicted_choice") for r in read_jsonl(io_path)}
        elif image_only_csv:
            image_only = {i: c for (m, i), c in image_only_csv.items() if m == model_key}
        trig = triggers(rows_by_core, first, image_only, args.triggers)
        route_stats.append({"model_key": model_key, "images": len(rows_by_core), "routed": len(trig),
                            "routed_rate": round(len(trig) / len(rows_by_core), 4),
                            "by_reason": {r: sum(r in v for v in trig.values()) for r in args.triggers},
                            "image_only_available": image_only is not None})
        found[(model_key, "stage1")] = dict(first)
        holder = base.LazyHFAdapter(base.MODEL_REGISTRY[model_key], loader=loader)
        try:
            for variant in args.variants:
                choice = dict(first)
                extra_calls = 0
                if variant == "rule_q1_first":
                    for core in rows_by_core:
                        a, _ = stage1_label(rows_by_core, first, core)
                        if a == "normal":
                            apply_label(rows_by_core, choice, core, "normal")
                else:
                    print(f"\n=== {model_key} / {variant} ===", flush=True)
                    pred_path = args.out_dir / "predictions" / model_key / f"verify_{variant}.jsonl"
                    existing = {str(r["core_id"]): r for r in (read_jsonl(pred_path) if pred_path.is_file() else [])}
                    targets_cores = sorted(rows_by_core) if variant == "zoom_define_all" else sorted(trig)
                    out_rows, called, reused, errors = [], 0, 0, 0
                    for core in targets_cores:
                        q2 = rows_by_core[core][Q2]
                        old = existing.get(core)
                        if reusable(old, model_key, variant):
                            rec = old
                            reused += 1
                        else:
                            adapter = holder.get()
                            started = time.time()
                            output, error = "", ""
                            try:
                                output = str(infer(adapter, images_for(core, variant), build_prompt(q2, variant)))
                            except Exception as exc:  # retried on the next run
                                error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                                errors += 1
                            rec = {"core_id": core, "q2_id": q2["id"], "model_key": model_key, "variant": variant,
                                   "prompt_version": PROMPT_VERSION, "rule": args.rule,
                                   "routed_by": ";".join(trig.get(core, ["all"])), "output": output,
                                   "predicted_choice": base.parse_choice(output, {str(k).upper() for k in q2["options"]}),
                                   "latency_s": round(time.time() - started, 3), "error": error}
                            called += 1
                        out_rows.append(rec)
                        if args.checkpoint_every > 0 and called and called % args.checkpoint_every == 0:
                            base.write_jsonl(pred_path, out_rows)
                        label = base.predicted_semantic(q2, rec.get("predicted_choice"))
                        if label is not None:
                            apply_label(rows_by_core, choice, core, label)
                        extra_calls += 1
                    base.write_jsonl(pred_path, out_rows)
                    statuses.append({"model_key": model_key, "variant": variant, "called_this_run": called,
                                     "reused_predictions": reused, "row_errors": errors,
                                     "status": "complete" if errors == 0 else "has_errors"})
                found[(model_key, variant)] = choice
                n_q = len(rows)
                q1_rows = [q[Q1] for q in rows_by_core.values()]
                changed = [r for r in q1_rows if choice[str(r["id"])] != first[str(r["id"])]]
                q2_rows = [q[Q2] for q in rows_by_core.values()]
                changed2 = [r for r in q2_rows if choice[str(r["id"])] != first[str(r["id"])]]
                flips.append({
                    "model_key": model_key, "variant": variant,
                    "q1_changed": len(changed), "q1_changed_to_correct": sum(choice[str(r["id"])] == r["answer"] for r in changed),
                    "q1_changed_to_wrong": sum(first[str(r["id"])] == r["answer"] for r in changed),
                    "q2_changed": len(changed2), "q2_changed_to_correct": sum(choice[str(r["id"])] == r["answer"] for r in changed2),
                    "q2_changed_to_wrong": sum(first[str(r["id"])] == r["answer"] for r in changed2),
                })
                results.append({"model_key": model_key, "variant": variant,
                                **cons.consistency_row(cons.image_pairs(rows), choice),
                                **cons.q3_row(cons.q3_pairs(rows), choice),
                                "avg_lvlm_calls_per_q": round(1 + extra_calls / n_q, 4)})
        finally:
            adapter = None   # drop the loop's reference too, or the next model loads beside this one
            holder.release()
        results.append({"model_key": model_key, "variant": "stage1",
                        **cons.consistency_row(cons.image_pairs(rows), first),
                        **cons.q3_row(cons.q3_pairs(rows), first), "avg_lvlm_calls_per_q": 1.0})

    comparisons = [("stage1", v) for v in args.variants]
    sig = cons.significance_rows(cons.image_pairs(rows), cons.q3_pairs(rows), found, args.models, comparisons,
                                 n_boot=args.n_boot)
    if not cons.q3_pairs(rows):
        sig = [r for r in sig if r["metric"] not in ("q3_accuracy", "q1q3_contradiction")]
    cons.write_csv(args.out_dir / "leaderboard.csv", results)
    cons.write_csv(args.out_dir / "flips.csv", flips)
    cons.write_csv(args.out_dir / "significance.csv", sig)
    (args.out_dir / "route_stats.json").write_text(json.dumps(route_stats, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    (args.out_dir / "run_status.json").write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    meta = {"protocol": "gold_v2_verify_agent", "prompt_version": PROMPT_VERSION,
            "qa": (args.qa_file or args.qa_dir).name, "stage1_condition": args.stage1_condition, "rule": args.rule,
            "triggers": list(args.triggers), "variants": list(args.variants), "models": list(args.models),
            "zoom": {"box": ZOOM_BOX, "scale": ZOOM_SCALE, "max_side": ZOOM_MAX_SIDE}, "definitions": DEFINITIONS,
            "decision": "verified defect type rewrites Q1/Q2 (and Q3 -> no defect region when normal); invalid "
                        "verification output keeps the stage-1 answers",
            "significance": f"image-level paired bootstrap vs stage1, {args.n_boot} resamples, two-sided 95% CI"}
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    errors = sum(s["row_errors"] for s in statuses)
    print(json.dumps(route_stats, ensure_ascii=False, indent=2))
    if errors:
        print(f"WARNING: {errors} rows failed; re-run the same cell to retry only those rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
