#!/usr/bin/env python3
"""Greedy re-run of MiniCPM-V-4.5 on the gold-v2 reference conditions used in the paper's Table 3.

The published MiniCPM gold results (image-only baseline and the three reference conditions) were sampled at
temperature 0.7 before the adapter fix in PR #47. This runner repeats exactly those four conditions with the current
greedy adapter. Prompts, image order and references are the original ones:

- ``image_only``                [target]                                   ``run_gold_baselines.prompt_for``
- ``normal_reference``          [fixed normal, target]                     ``run_gold_v2_reference_ablation``
- ``positive_reference``        [retrieved defect example, target]         ``run_gold_v2_reference_ablation``
- ``normal_positive_reference`` [fixed normal, defect example, target]     ``run_gold_v2_reference_ablation``

The defect examples come from the committed ``results/latest/gold_v2_reference/reference_manifest.csv``.
Every Q1, Q2 and Q3 question (534) is asked once per condition. Invalid outputs count as wrong, as before.
Nothing is published: results stay in the output folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_gold_v2_consistency_metrics as cons
import run_gold_v2_reference_ablation as ref

base, v2 = ref.base, ref.v2
MODEL = "minicpm_v_4_5"
PREFIX = "gold_v2_minicpm_greedy"
CONDITIONS = ("image_only", "normal_reference", "positive_reference", "normal_positive_reference")
COMPARISONS = [("image_only", c) for c in CONDITIONS[1:]]
REFERENCE_MANIFEST = ROOT / "results/latest/gold_v2_reference/reference_manifest.csv"
SAMPLED = ROOT / "results/latest/gold_v2_consistency_metrics/consistency_leaderboard.csv"
TABLE_KEYS = ("q1_anomaly_f1", "q2_defect_macro_f1", "joint_accuracy", "consistency_error", "false_defect_rate",
              "q3_accuracy", "q1_invalid_rate", "q2_invalid_rate", "q3_invalid_rate")


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def load_manifest(path, rows):
    """The published reference manifest, re-checked against the frozen QA (same target, same fixed normal)."""
    records = read_csv(path)
    by_core = {r["core_id"]: r for r in records}
    if len(by_core) != len(records):
        raise ValueError("duplicate core in reference manifest")
    for row in rows:
        m = by_core.get(str(row["core_id"]))
        if m is None:
            raise ValueError(f"{row['core_id']}: no reference in manifest")
        if m["target_image"] != row["image"] or m["normal_reference_image"] != row.get("normal_reference_image"):
            raise ValueError(f"{row['core_id']}: manifest differs from frozen QA")
        if m["target_gt_used_for_positive_selection"].lower() not in ("false", "0"):
            raise ValueError(f"{row['core_id']}: defect reference selection used target labels")
    return {core: by_core[core] for core in sorted({str(r["core_id"]) for r in rows})}


def prompt_for(row, condition):
    return base.prompt_for(dict(row)) if condition == "image_only" else ref.prompt_for_condition(dict(row), condition)


def images_for(row, condition, manifest, targets, normals, positives):
    if condition == "image_only":
        return [targets[str(row["image"])]]
    return ref.image_list_for(row, condition, manifest, targets, normals, positives)


def metrics(rows, choice):
    return {**cons.consistency_row(cons.image_pairs(rows), choice), **cons.q3_row(cons.q3_pairs(rows), choice)}


def sampled_rows(path):
    if not Path(path).is_file():
        return {}
    return {r["condition"]: r for r in read_csv(path) if r["model_key"] == MODEL and r["condition"] in CONDITIONS}


def main(argv=None, *, loader=None, infer=None, load_qa=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("qa-dir", "sup-zip", "unsup-zip", "out-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--reference-manifest", type=Path, default=REFERENCE_MANIFEST)
    p.add_argument("--sampled-leaderboard", type=Path, default=SAMPLED)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    if not args.out_dir.name.startswith(PREFIX) or args.checkpoint_every <= 0 or args.n_boot <= 0:
        raise ValueError("use a separate gold_v2_minicpm_greedy* output folder and positive check counts")
    summary = json.loads((args.qa_dir/"build_summary.json").read_text(encoding="utf-8"))
    if summary.get("quality_status") != "FROZEN":
        raise ValueError("gold QA must be FROZEN")
    rows = (load_qa or v2.load_gold_qa_v2)(args.qa_dir)
    if load_qa is None and (len(rows) != 534 or summary.get("final_master_images") != 246):
        raise ValueError("expected original gold246 QA (534 questions)")
    manifest = load_manifest(args.reference_manifest, rows)
    code_paths = [Path(__file__), Path(ref.__file__), Path(cons.__file__), ROOT/"scripts/run_gold_baselines.py",
                  ROOT/"scripts/run_gold_v2_baselines.py", ROOT/"src/insplad_benchmark/hf_adapters.py"]
    code = {str(f.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest() for f in code_paths}
    spec = base.MODEL_REGISTRY[MODEL]
    signature = hashlib.sha256(json.dumps({"rows": rows, "manifest": manifest, "code": code, "model": spec,
        "conditions": CONDITIONS, "n_boot": args.n_boot}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    meta_path = args.out_dir/"run_metadata.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if previous and previous.get("input_signature") != signature:
        raise ValueError("inputs/code changed: use a new folder; previous results preserved")
    img_root = args.out_dir/"_images"
    targets, normals, positives = {}, {}, {}
    for row in rows:
        m = manifest[str(row["core_id"])]
        targets.setdefault(row["image"], base.safe_extract(args.sup_zip, row["image"], img_root/"target"))
        normals.setdefault(m["normal_reference_image"], base.safe_extract(args.unsup_zip, m["normal_reference_image"], img_root/"normal"))
        positives.setdefault(m["positive_reference_image"], base.safe_extract(args.sup_zip, m["positive_reference_image"], img_root/"positive"))
    image_manifest = [{"role": role, "member": member, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for role, paths in (("target", targets), ("normal", normals), ("positive", positives))
                      for member, path in sorted(paths.items())]
    identity = hashlib.sha256(json.dumps([signature, image_manifest], sort_keys=True).encode()).hexdigest()
    if previous and previous.get("prediction_identity") != identity:
        raise ValueError("image bytes changed: use a new folder")
    meta = {**previous, "protocol": PREFIX, "model_key": MODEL, "model_spec": spec, "decoding": "greedy (sampling=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=80)",
        "conditions": list(CONDITIONS), "questions_per_condition": len(rows), "input_signature": signature,
        "prediction_identity": identity, "benchmark_commit": git_commit(),
        "code_sha256": code, "reference_manifest": str(args.reference_manifest),
        "reference_manifest_sha256": hashlib.sha256(Path(args.reference_manifest).read_bytes()).hexdigest(),
        "replaces": "sampled (temperature 0.7) MiniCPM results in gold_v2_baseline / gold_v2_reference",
        "published": False, "software": {}, "complete": False}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    base.write_csv(args.out_dir/"image_manifest.csv", image_manifest)
    base.write_jsonl(args.out_dir/"gold_qa_snapshot.jsonl", rows)
    path = args.out_dir/"predictions"/MODEL/"steps.jsonl"
    records = base.read_jsonl(path) if path.exists() else []
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate saved step IDs")
    existing = {r["id"]: r for r in records}
    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer
    holder = base.LazyHFAdapter(spec, loader=loader)
    used, counts = {}, Counter()
    choices = {c: {} for c in CONDITIONS}
    load_failure = ""

    def save():
        base.write_jsonl(path, [*used.values(), *(r for i, r in existing.items() if i not in used)])

    try:
        for condition in CONDITIONS:
            for row in rows:
                key = f"{condition}/{row['id']}"
                prompt = prompt_for(row, condition)
                images = images_for(row, condition, manifest, targets, normals, positives)
                letters = {str(k).upper() for k in row["options"]}
                rec = existing.get(key)
                if rec and rec.get("identity") != identity:
                    raise ValueError("saved prediction identity mismatch")
                new_call = not (rec and not rec.get("error") and rec.get("prompt") == prompt
                                and str(rec.get("output", "")).strip())
                if not new_call:
                    counts["reused_calls"] += 1
                else:
                    output, error, start = "", "", time.time()
                    if load_failure:
                        error = "blocked by adapter load failure: " + load_failure
                    else:
                        try:
                            adapter = holder.get()
                        except Exception as exc:
                            load_failure = error = f"{type(exc).__name__}: {exc}"
                        if not error:
                            try:
                                output = str(infer(adapter, images, prompt))
                            except Exception as exc:
                                error = f"{type(exc).__name__}: {exc}"
                        counts["called_this_run"] += 1
                    rec = {"id": key, "question_id": row["id"], "core_id": row["core_id"], "condition": condition,
                           "question_type": row["question_type"], "asset_category": row["asset_category"],
                           "identity": identity, "model_key": MODEL, "prompt": prompt,
                           "images": [str(Path(x).relative_to(img_root)).replace("\\", "/") for x in images],
                           "output": output, "error": error, "latency_s": round(time.time()-start, 3)}
                rec["predicted_choice"] = base.parse_choice(rec["output"], letters)
                used[key] = rec
                choices[condition][str(row["id"])] = rec["predicted_choice"]
                if new_call and counts["called_this_run"] % args.checkpoint_every == 0:
                    save()
                    print(f"[{condition}] {len(used)} steps saved", flush=True)
    finally:
        save()
        holder.release()
    sampled = sampled_rows(args.sampled_leaderboard)
    leaderboard, comparison, statuses, by_asset = [], [], [], []
    for condition in CONDITIONS:
        recs = [used[f"{condition}/{r['id']}"] for r in rows]
        errors = sum(bool(r["error"]) for r in recs)
        complete = errors == 0 and len(recs) == len(rows)
        statuses.append({"condition": condition, "status": "complete" if complete else "incomplete",
                         "row_errors": errors, "invalid_outputs": sum(r["predicted_choice"] is None for r in recs),
                         "questions": len(recs)})
        if not complete:
            continue
        m = metrics(rows, choices[condition])
        leaderboard.append({"model_key": MODEL, "condition": condition, "decoding": "greedy",
                            "latency_s": round(sum(r["latency_s"] for r in recs), 3), **m})
        for asset in sorted({r["asset_category"] for r in rows}):
            sub = [r for r in rows if r["asset_category"] == asset]
            by_asset.append({"model_key": MODEL, "condition": condition, "asset_category": asset, **metrics(sub, choices[condition])})
        old = sampled.get(condition)
        for key in TABLE_KEYS:
            if key in m:
                comparison.append({"condition": condition, "metric": key, "greedy": m[key],
                                   "sampled_published": float(old[key]) if old and old.get(key) not in (None, "") else None,
                                   "difference": round(m[key] - float(old[key]), 6) if old and old.get(key) not in (None, "") else None})
    found = {(MODEL, c): choices[c] for c in CONDITIONS if any(s["condition"] == c and s["status"] == "complete" for s in statuses)}
    sig = cons.significance_rows(cons.image_pairs(rows), cons.q3_pairs(rows), found, [MODEL], COMPARISONS, args.n_boot)
    for name, data in (("leaderboard.csv", leaderboard), ("metrics_by_asset.csv", by_asset),
                       ("comparison_vs_sampled.csv", comparison), ("significance.csv", sig)):
        base.write_csv(args.out_dir/name, data)
    complete = all(s["status"] == "complete" for s in statuses)
    meta.update({"complete": complete, "called_this_run": counts["called_this_run"], "reused_calls": counts["reused_calls"],
                 "unique_steps": len(used), "row_errors": sum(s["row_errors"] for s in statuses),
                 "invalid_outputs": sum(s["invalid_outputs"] for s in statuses)})
    for package in ("torch", "transformers", "numpy", "Pillow"):
        try:
            meta["software"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta["software"][package] = None
    try:
        import torch
        if torch.cuda.is_available():
            meta["gpu"] = {"name": torch.cuda.get_device_name(0),
                           "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)}
    except ImportError:
        pass
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir/"run_status.json").write_text(json.dumps({"status": "complete" if complete else "incomplete",
        "complete": complete, "conditions": statuses, "published": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**counts, "complete": complete}, indent=2), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
