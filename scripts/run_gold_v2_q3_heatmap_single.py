#!/usr/bin/env python3
"""Gold-v2 Q3 localization with expert evidence drawn on a single TARGET image.

The vision-expert full run gave heatmaps / ROI crops as a separate Image 2, and no expert condition beat
image-only on Q3. The oracle study showed that location marks drawn on the target itself (one image) help Q3
while a second same-scene image barely does. This runner asks only the 42 Q3 questions with:

- ``heatmap_single``: the expert's pure heatmap overlay of the TARGET (already a full target image), alone
- ``roi_box_single``: the TARGET with the expert's most suspicious ROI drawn as a red rectangle

and compares each against image-only and the matching two-image vision-expert condition with exact McNemar
tests. It also scores an expert-only (no LVLM) answer that maps the expert ROI centre to a region option, to
tell a bad localizer apart from a model that cannot use the mark.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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

Q3 = "defect_localization"
MODELS = ("qwen3_5_4b", "internvl3_5_8b", "minicpm_v_4_5", "llava_onevision_7b")
EXPERTS = ("patchcore", "dinomaly", "anomalyclip", "reconpatch_lite")
CONDITIONS = ("heatmap_single", "roi_box_single")
TWO_IMAGE_COUNTERPART = {"heatmap_single": "heatmap", "roi_box_single": "roi"}
NONE_TEXT = "No localized defect region is present."
# Same wording as scripts/build_explainable_qa_gold.py REGION_TEXT (Q3 options).
REGION_TEXT = {
    "upper-left": "Upper-left region.", "upper": "Upper-center region.", "upper-right": "Upper-right region.",
    "left": "Middle-left region.", "center": "Center region.", "right": "Middle-right region.",
    "lower-left": "Lower-left region.", "lower": "Lower-center region.", "lower-right": "Lower-right region.",
}
PROMPT_VERSION = 1


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def region_of_point(cx: float, cy: float) -> str:
    """Normalized point -> coarse region, same 3x3 rule as the QA builder."""
    x_name = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    y_name = "upper" if cy < 1 / 3 else ("lower" if cy > 2 / 3 else "middle")
    if x_name == "center" and y_name == "middle":
        return "center"
    if y_name == "middle":
        return x_name
    if x_name == "center":
        return y_name
    return f"{y_name}-{x_name}"


def build_prompt(row: Mapping, condition: str, expert: str) -> str:
    options = "\n".join(f"{k}. {v}" for k, v in dict(row["options"]).items())
    if condition == "heatmap_single":
        desc = (f"The image is the complete TARGET image with the {expert} vision-expert anomaly heatmap overlaid: "
                "red/orange tint marks more suspicious areas and darker areas are less suspicious. "
                "The heatmap may be noisy or wrong. ")
    elif condition == "roi_box_single":
        desc = (f"The image is the complete TARGET image. The red rectangle marks the area the {expert} "
                "vision expert finds most suspicious. It may be wrong. ")
    else:
        raise ValueError(condition)
    return (
        "You are an industrial visual inspector. " + desc
        + "Answer the multiple-choice question about the TARGET. Do not assume the vision expert is correct.\n\n"
        f"Question: {row['question']}\n{options}\n\n"
        "Return exactly one option letter and nothing else."
    )


def resolve(manifest_path: Path, value: str, artifact_root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (artifact_root if artifact_root else manifest_path.parent) / p


def draw_roi_box(target: Path, overlay: Path, expert_row: Mapping[str, str], out: Path) -> Path:
    """Draw the expert ROI (pixel coords of the expert source image) on the TARGET, scaling if sizes differ."""
    if out.is_file() and out.stat().st_size > 0:
        return out
    from PIL import Image, ImageDraw

    with Image.open(overlay) as h:
        src_w, src_h = h.size
    with Image.open(target) as h:
        img = h.convert("RGB")
    sx, sy = img.width / src_w, img.height / src_h
    box = (float(expert_row["roi_left"]) * sx, float(expert_row["roi_top"]) * sy,
           float(expert_row["roi_right"]) * sx - 1, float(expert_row["roi_bottom"]) * sy - 1)
    ImageDraw.Draw(img).rectangle(box, outline=(255, 0, 0), width=max(3, round(min(img.size) / 180)))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.png")
    img.save(tmp, format="PNG")
    tmp.replace(out)
    return out


def expert_only_choice(row: Mapping, expert_row: Mapping[str, str], width: int, height: int) -> str | None:
    cx = (float(expert_row["roi_left"]) + float(expert_row["roi_right"])) / 2 / width
    cy = (float(expert_row["roi_top"]) + float(expert_row["roi_bottom"])) / 2 / height
    want = REGION_TEXT[region_of_point(cx, cy)]
    return next((k for k, v in dict(row["options"]).items() if v == want), None)


def mcnemar_exact(base_ok: Sequence[bool], new_ok: Sequence[bool]) -> tuple[int, int, float]:
    """b = new right & base wrong, c = base right & new wrong; two-sided exact binomial p."""
    b = sum(n and not o for o, n in zip(base_ok, new_ok))
    c = sum(o and not n for o, n in zip(base_ok, new_ok))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def reusable(old: Mapping | None, model_key: str, expert: str, condition: str) -> bool:
    return bool(old and not old.get("error") and str(old.get("output", "")).strip()
                and old.get("model_key") == model_key and old.get("expert") == expert
                and old.get("condition") == condition and old.get("prompt_version") == PROMPT_VERSION)


def main(argv: Sequence[str] | None = None, *, load_qa: Callable[[Path], list[dict]] = v2.load_gold_qa_v2,
         loader: Callable | None = None, infer: Callable | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--sup-zip", type=Path, required=True)
    p.add_argument("--roi-manifest", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--image-only-csv", type=Path, required=True)
    p.add_argument("--vx-pred-dir", type=Path, required=True, help="vision expert full eval/predictions")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--experts", nargs="+", default=list(EXPERTS))
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--checkpoint-every", type=int, default=20)
    args = p.parse_args(argv)

    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer  # noqa: N813
    rows = [r for r in load_qa(args.qa_dir) if r.get("question_type") == Q3]
    if not rows:
        raise RuntimeError("no Q3 (defect_localization) rows")
    cores = {str(r["core_id"]) for r in rows}
    manifest = [r for r in read_csv(args.roi_manifest)
                if r.get("expert") in args.experts and str(r.get("query_id")) in cores]
    expert_index = {(str(r["query_id"]), str(r["expert"])): r for r in manifest}
    missing = [(c, e) for c in cores for e in args.experts if (c, e) not in expert_index]
    if missing:
        raise RuntimeError(f"ROI manifest missing {len(missing)} (core, expert) pairs, e.g. {missing[:2]}")
    overlay_paths = {k: resolve(args.roi_manifest, str(v["overlay_path"]), args.artifact_root)
                     for k, v in expert_index.items()}
    absent = [str(p) for p in overlay_paths.values() if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} heatmap overlays missing, e.g. {absent[:2]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_paths = base.prepare_images(rows, args.sup_zip, args.out_dir / "_images" / "target")

    def image_for(row: Mapping, expert: str, condition: str) -> Path:
        key = (str(row["core_id"]), expert)
        if condition == "heatmap_single":
            return overlay_paths[key]
        return draw_roi_box(target_paths[str(row["image"])], overlay_paths[key], expert_index[key],
                            args.out_dir / "_images" / "roi_box" / expert / f"{row['core_id']}.png")

    # expert-only (no LVLM): ROI centre -> region option
    from PIL import Image
    expert_only: dict[str, dict[str, str | None]] = {}
    for expert in args.experts:
        choice = {}
        for row in rows:
            key = (str(row["core_id"]), expert)
            with Image.open(overlay_paths[key]) as h:
                w, h_ = h.size
            choice[str(row["id"])] = expert_only_choice(row, expert_index[key], w, h_)
        expert_only[expert] = choice

    image_only = {(r["model_key"], str(r["id"])): (r.get("predicted_choice") or None)
                  for r in read_csv(args.image_only_csv)}
    statuses: list[dict] = []
    preds: dict[tuple[str, str, str], dict[str, str | None]] = {}
    for model_key in args.models:
        holder = base.LazyHFAdapter(base.MODEL_REGISTRY[model_key], loader=loader)
        try:
            for expert in args.experts:
                for condition in args.conditions:
                    print(f"\n=== {model_key} / {expert} / {condition} ===", flush=True)
                    pred_path = args.out_dir / "predictions" / model_key / f"{expert}__{condition}.jsonl"
                    existing = {str(r["id"]): r for r in (read_jsonl(pred_path) if pred_path.is_file() else [])}
                    out_rows: list[dict] = []
                    called = reused = errors = 0
                    for row in rows:
                        old = existing.get(str(row["id"]))
                        if reusable(old, model_key, expert, condition):
                            out_rows.append(old)
                            reused += 1
                            continue
                        image = image_for(row, expert, condition)
                        adapter = holder.get()
                        started = time.time()
                        output, error = "", ""
                        try:
                            output = str(infer(adapter, [image], build_prompt(row, condition, expert)))
                        except Exception as exc:  # retried on the next run
                            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                            errors += 1
                        out_rows.append({
                            "id": row["id"], "core_id": row["core_id"], "model_key": model_key, "expert": expert,
                            "condition": condition, "prompt_version": PROMPT_VERSION, "output": output,
                            "predicted_choice": base.parse_choice(output, {str(k).upper() for k in row["options"]}),
                            "latency_s": round(time.time() - started, 3), "error": error,
                        })
                        called += 1
                        if args.checkpoint_every > 0 and called % args.checkpoint_every == 0:
                            base.write_jsonl(pred_path, out_rows)
                    base.write_jsonl(pred_path, out_rows)
                    preds[(model_key, expert, condition)] = {str(r["id"]): r.get("predicted_choice") for r in out_rows}
                    statuses.append({"model_key": model_key, "expert": expert, "condition": condition,
                                     "called_this_run": called, "reused_predictions": reused, "row_errors": errors,
                                     "status": "complete" if errors == 0 else "has_errors"})
        finally:
            holder.release()

    region_rows = [r for r in rows if dict(r["options"]).get(r["answer"]) != NONE_TEXT]

    def acc(choice: Mapping[str, str | None], subset: Sequence[Mapping]) -> float:
        return sum(choice.get(str(r["id"])) == r["answer"] for r in subset) / len(subset) if subset else 0.0

    def correct(choice: Mapping[str, str | None]) -> list[bool]:
        return [choice.get(str(r["id"])) == r["answer"] for r in rows]

    leaderboard: list[dict] = []
    for model_key in args.models:
        img = {str(r["id"]): image_only.get((model_key, str(r["id"]))) for r in rows}
        leaderboard.append({"model_key": model_key, "expert": "", "condition": "image_only", "questions": len(rows),
                            "q3_acc": acc(img, rows), "q3_acc_region_only": acc(img, region_rows)})
        for expert in args.experts:
            for condition in args.conditions:
                new = preds[(model_key, expert, condition)]
                b, c, pval = mcnemar_exact(correct(img), correct(new))
                item = {"model_key": model_key, "expert": expert, "condition": condition, "questions": len(rows),
                        "q3_acc": acc(new, rows), "q3_acc_region_only": acc(new, region_rows),
                        "delta_vs_image_only": acc(new, rows) - acc(img, rows),
                        "gain_vs_image_only": b, "loss_vs_image_only": c, "mcnemar_p_vs_image_only": pval,
                        "valid_output_rate": sum(new.get(str(r["id"])) is not None for r in rows) / len(rows)}
                two_path = args.vx_pred_dir / model_key / f"{expert}__{TWO_IMAGE_COUNTERPART[condition]}.jsonl"
                if two_path.is_file():
                    two = {str(r["id"]): r.get("predicted_choice") for r in read_jsonl(two_path)}
                    b2, c2, p2 = mcnemar_exact(correct(two), correct(new))
                    item.update({"two_image_condition": TWO_IMAGE_COUNTERPART[condition],
                                 "two_image_q3_acc": acc(two, rows), "delta_vs_two_image": acc(new, rows) - acc(two, rows),
                                 "mcnemar_p_vs_two_image": p2})
                leaderboard.append(item)
    for item in leaderboard:
        for k, v in list(item.items()):
            if isinstance(v, float):
                item[k] = round(v, 6)
    expert_rows = [{"expert": e, "questions": len(rows), "q3_acc": round(acc(expert_only[e], rows), 6),
                    "q3_acc_region_only": round(acc(expert_only[e], region_rows), 6),
                    "note": "ROI centre mapped to 3x3 region; never answers 'none'"} for e in args.experts]

    write_csv(args.out_dir / "leaderboard.csv", leaderboard)
    write_csv(args.out_dir / "expert_only.csv", expert_rows)
    (args.out_dir / "run_status.json").write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    metadata = {
        "protocol": "gold_v2_q3_heatmap_single",
        "prompt_version": PROMPT_VERSION,
        "questions": len(rows), "region_questions": len(region_rows), "none_questions": len(rows) - len(region_rows),
        "models": list(args.models), "experts": list(args.experts), "conditions": list(args.conditions),
        "images": {"heatmap_single": "merged manifest overlay_path (pure heatmap on TARGET, alpha 0.42, no box)",
                   "roi_box_single": "TARGET with expert ROI as red rectangle (line max(3, min(W,H)/180))"},
        "baselines": {"image_only": "gold_v2_baseline prediction_review.csv",
                      "two_image": "vision expert full {expert}__heatmap / {expert}__roi predictions"},
        "test": "exact two-sided McNemar on the same Q3 questions",
        "selection_note": "Q3-only follow-up motivated by the gold-v2 oracle study; results are exploratory "
                          "on the same 42 gold questions",
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
