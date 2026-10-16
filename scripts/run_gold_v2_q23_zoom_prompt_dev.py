#!/usr/bin/env python3
"""Exploratory dev-only 2x2 native-detail/prompt ablation; Q1 is read-only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from collections import Counter
from pathlib import Path

import run_gold_v2_q23_candidate as common
from insplad_benchmark import hf_adapters as hf

ROOT = Path(__file__).resolve().parents[1]
old, refined, agent = common.old, common.refined, common.agent
MODEL = "internvl3_5_8b"
PROTOCOL = "gold_v2_q23_zoom_prompt_dev_v1"
PREFIX = "gold_v2_q23_zoom_prompt_dev"
# Reference first, target second. Thumbnail is additional when the grid has >1 cell.
BUDGETS = {"standard": [3, 3], "zoom": [3, 6]}
FACTORS = {"control": ("standard", False), "zoom_only": ("zoom", False),
           "prompt_only": ("standard", True), "zoom_prompt": ("zoom", True)}
POLICIES = ("cached_baseline", *FACTORS)
COMPARISONS = [("cached_baseline", "control"), ("control", "zoom_only"),
               ("control", "prompt_only"), ("control", "zoom_prompt"),
               ("zoom_only", "zoom_prompt"), ("prompt_only", "zoom_prompt")]
SHORT_DEFINITIONS = {
    "Normal / no defect.": "visible components are complete, without corrosion or attached nesting material; lighting and background differences alone are not defects.",
    "Rust / corrosion.": "irregular orange, brown or dark rough/flaky corrosion on metal, rather than smooth paint or shadow.",
    "Missing cap / missing insulator component.": "glass disc/skirt or another required repeated component is absent or broken. An exposed metal cap/pin without its glass also counts. Check local repetition within the TARGET; do not infer loss merely from different total counts or viewpoints in the reference.",
    "Bird nest / foreign nesting material.": "twigs, straw or grass attached to the hardware, including sparse strands; background vegetation and normal cable strands do not count.",
}


def prompts(q2, q3, short):
    q2prompt = refined.definition_prompt(q2, True)
    task, presence = old.presence_task(q3)
    position = old.position_prompt(q3)
    if short:
        for text, definition in refined.DEFINITIONS.items():
            q2prompt = q2prompt.replace(definition, SHORT_DEFINITIONS[text])
        task = {**task, "question": "Does the TARGET show visible defect evidence, including a missing/broken component or gap, corrosion, or attached nesting material? A gap or exposed structure also counts as a defect region. Lighting, viewpoint and background alone do not count."}
        presence = old.direct_prompt(task)
        position_task = {**q3, "question": "Where is the defect evidence in the TARGET? For missing material, locate the gap or broken structure. For corrosion or attached twigs, locate the affected hardware. Choose no-region if no defect evidence is visible."}
        position = old.position_prompt(position_task)
    # Identical option letters/order and routing for both wording levels.
    return q2prompt, task, presence, position


def baseline(path, rows):
    """Validate the actual dev-baseline schema; never generate or repair Q1 here."""
    records = agent.read_jsonl(path)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate baseline prediction IDs")
    by_id = {r["id"]: r for r in records}
    first = {}
    for row in rows:
        p = by_id.get(row["id"], {})
        if (p.get("model_key"), p.get("condition"), p.get("prompt_version"), p.get("core_id")) != (
                MODEL, "normal_ref_retrieved_diffdate", 1, row["core_id"]):
            raise ValueError("missing baseline or model/condition/question identity mismatch")
        choice = agent.base.parse_choice(str(p.get("output", "")), set(row["options"]))
        if p.get("error") or choice is None or choice != p.get("predicted_choice"):
            raise ValueError("baseline failed/invalid: repair the original dev baseline first")
        first[row["id"]] = choice
    return first, hashlib.sha256(path.read_bytes()).hexdigest()


def tile_manifest(targets, normals):
    from PIL import Image
    result = []
    for role, paths in (("target", targets), ("reference", normals)):
        for member, path in sorted(paths.items()):
            with Image.open(path) as source:
                im = source.convert("RGB")
                result.append({"role": role, "member": member, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "width": im.width, "height": im.height,
                    "standard_tiles": len(hf._dynamic_preprocess(im, max_num=3)),
                    "zoom_tiles": len(hf._dynamic_preprocess(im, max_num=6 if role == "target" else 3))})
    return result


def main(argv=None, *, loader=None, infer=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dev-qa", "stage1-dir", "refs-csv", "sup-zip", "unsup-zip", "out-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--n-images", type=int, default=240)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    if not args.out_dir.name.startswith(PREFIX) or min(args.n_boot, args.checkpoint_every) < 1:
        raise ValueError("use a separate gold_v2_q23_zoom_prompt_dev* folder and positive check counts")
    summary = json.loads((args.dev_qa.parent/"build_summary.json").read_text(encoding="utf-8"))
    if summary.get("quality_status") != "FROZEN":
        raise ValueError("expected FROZEN dev QA")
    rows, _ = refined.select_rows(agent.read_jsonl(args.dev_qa), args.n_images, args.seed)
    q3 = [{**r, "protocol": PROTOCOL} for r in old.q3_rows(rows)]
    refs = common.read_references(args.refs_csv, rows)
    source = args.stage1_dir/MODEL/"normal_ref_retrieved_diffdate.jsonl"
    first, source_sha = baseline(source, rows)
    spec = agent.base.MODEL_REGISTRY[MODEL]
    if spec.get("adapter") != "internvl_chat" or spec.get("max_total_tiles", 6) != 6:
        raise ValueError("expected the unchanged InternVL default tile budget6")
    code_paths = [Path(__file__), Path(common.__file__), Path(old.__file__), Path(refined.__file__),
        Path(agent.__file__), Path(agent.cons.__file__), Path(agent.base.__file__), Path(old.diagnostic.__file__),
        Path(old.diagnostic.dev.__file__), Path(hf.__file__), ROOT/"scripts/run_gold_v2_reference_ablation.py",
        ROOT/"src/insplad_benchmark/metrics.py"]
    code = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in code_paths}
    audit = []
    q2rows = [r for r in rows if r["question_type"] == agent.Q2]
    for r2, r3 in zip(q2rows, q3):
        for policy, (view, short) in FACTORS.items():
            text2, _, presence, position = prompts(r2, r3, short)
            for row, step, text in ((r2, "q2", text2), (r3, "presence", presence), (r3, "position", position)):
                audit.append({"policy": policy, "question_id": row["id"], "step": step,
                              "image_tile_budgets": BUDGETS[view], "prompt": text})
    signature = old.diagnostic.digest({"protocol": PROTOCOL, "rows": rows, "q3": q3, "refs": refs,
        "source_sha256": source_sha, "spec": spec, "code_sha256": code, "prompts": audit, "n_boot": args.n_boot})
    out = args.out_dir
    meta_path = out/"run_metadata.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if previous and previous.get("input_signature") != signature:
        raise ValueError("inputs/code/protocol changed: use a new folder; previous results preserved")
    if not previous and out.exists() and any(out.iterdir()):
        raise ValueError("unrecognized nonempty output folder; use a new folder")
    for parent in (args.dev_qa.parent, args.stage1_dir, args.refs_csv.parent):
        if out.resolve().is_relative_to(parent.resolve()):
            raise ValueError("output must be separate from existing dev/reference/baseline folders")
    # Bind partial extraction to this experiment too, so interruption can resume safely.
    if not previous:
        old.atomic_json(meta_path, {"input_signature": signature, "complete": False, "preflight_only": True})
        old.atomic_json(out/"run_status.json", {"status": "preparing", "complete": False, "policies": []})
    targets = old.verified_images(args.sup_zip, [r["image"] for r in rows], out/"_images/target")
    normals = old.verified_images(args.unsup_zip, list(refs.values()), out/"_images/normal")
    manifest = tile_manifest(targets, normals)
    identity = old.diagnostic.digest([signature, manifest])
    if previous.get("prediction_identity") and previous["prediction_identity"] != identity:
        raise ValueError("input image bytes changed; use a new folder")
    path = out/"predictions"/MODEL/"steps.jsonl"
    records = agent.read_jsonl(path) if path.exists() else []
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate saved step IDs")
    existing = {r["id"]: r for r in records}
    meta = {"protocol": PROTOCOL, "benchmark_commit": agent.git_commit(), "input_signature": signature,
        "prediction_identity": identity, "code_sha256": code, "model_spec": spec, "images": len(q3),
        "policies": POLICIES, "factors": FACTORS, "image_tile_budgets": BUDGETS,
        "baseline_source": str(source), "baseline_source_sha256": source_sha,
        "baseline_provenance_limit": "original dev records do not store reference bytes; original condition metadata and saved choices are validated",
        "q1_inference_calls": 0, "q1": "exact cached dev baseline for every policy",
        "q3_location_accuracy": None, "n_boot": args.n_boot, "complete": False,
        "independence": "each question uses only its own images/text/answers; GT only for scoring",
        "claim_limit": "already studied dev240; exploratory only, no gold selection or spatial localization claim"}
    old.atomic_json(meta_path, meta)
    old.atomic_json(out/"run_status.json", {"status": "running", "complete": False, "policies": []})
    common.save_records(out/"subset_qa_q12.jsonl", rows)
    common.save_records(out/"dev_q3_partial_qa.jsonl", q3)
    common.save_records(out/"prompts.jsonl", audit)
    (out/"cached_dev_baseline.jsonl").write_bytes(source.read_bytes())
    (out/"retrieved_refs.csv").write_bytes(args.refs_csv.read_bytes())
    agent.cons.write_csv(out/"image_manifest.csv", manifest)
    holder = agent.base.LazyHFAdapter(spec, loader=loader)
    infer = hf.infer_hf if infer is None else infer
    used, counts = {}, Counter()
    load_failure = ""
    view_failures = {}

    def call(row, prompt, step, view):
        nonlocal load_failure
        key = row["id"]+"/"+step
        rec = existing.get(key)
        images = [normals[refs[row["image"]]], targets[row["image"]]]
        budgets = BUDGETS[view]
        if rec and rec.get("identity") != identity:
            raise ValueError("saved prediction identity mismatch")
        valid = (rec and not rec.get("error") and rec.get("prompt") == prompt
            and rec.get("image_tile_budgets") == budgets and rec.get("images") == list(map(str, images))
            and rec.get("predicted_choice") in row["options"]
            and agent.base.parse_choice(str(rec.get("output", "")), set(row["options"])) == rec["predicted_choice"])
        if valid:
            counts["reused_calls"] += 1
        else:
            output, error, start = "", "", time.time()
            if load_failure or view in view_failures:
                error = "blocked by adapter/view failure: "+(load_failure or view_failures[view])
                counts["adapter_load_blocked_steps"] += 1
            else:
                try:
                    adapter = holder.get()
                except Exception as exc:
                    load_failure = error = f"{type(exc).__name__}: {exc}"
                if not error:
                    try:
                        # Shallow copy shares weights; default adapter and its reference budget never mutate.
                        output = str(infer({**adapter, "image_tile_budgets": list(budgets)}, images, prompt))
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        if "out of memory" in error.lower():
                            view_failures[view] = error
                counts["called_this_run"] += 1
            rec = {"id": key, "question_id": row["id"], "question_type": row["question_type"], "step": step,
                "model_key": MODEL, "identity": identity, "images": list(map(str, images)), "image_tile_budgets": budgets,
                "prompt": prompt, "output": output, "predicted_choice": agent.base.parse_choice(output, set(row["options"])),
                "error": error, "latency_s": round(time.time()-start, 3)}
        used[key] = rec
        if len(used) % args.checkpoint_every == 0:
            common.save_records(path, [*used.values(), *(r for i, r in existing.items() if i not in used)])
            print(f"[{MODEL}] {len(used)} Q2/Q3 steps saved", flush=True)
        return rec

    finals = {policy: dict(first) for policy in POLICIES}
    locs = {policy: {} for policy in POLICIES}
    deps = {policy: [] for policy in POLICIES}
    try:
        for r2, r3 in zip(q2rows, q3):
            direct = call(r3, old.direct_prompt(r3), "cached_baseline/direct", "standard")
            locs["cached_baseline"][r3["id"]] = direct["predicted_choice"]
            deps["cached_baseline"].append(direct)
            for policy, (view, short) in FACTORS.items():
                text2, task, presence_text, position_text = prompts(r2, r3, short)
                rec2 = call(r2, text2, policy+"/q2", view)
                presence = call(task, presence_text, policy+"/presence", view)
                positive = task["options"].get(presence["predicted_choice"]) == "A visible localized defect region."
                position = call(r3, position_text, policy+"/position", view) if positive else None
                finals[policy][r2["id"]] = rec2["predicted_choice"]
                # Invalid is retained as invalid; it never becomes a successful no-region answer.
                locs[policy][r3["id"]] = (position["predicted_choice"] if position else
                    r3["none_choice"] if presence["predicted_choice"] in task["options"] else None)
                deps[policy].extend([rec2, presence]+([position] if position else []))
    finally:
        common.save_records(path, [*used.values(), *(r for i, r in existing.items() if i not in used)])
        holder.release()
    statuses, eligible = [], {}
    for policy in POLICIES:
        errors = sum(bool(r["error"]) for r in deps[policy])
        invalids = sum(r["predicted_choice"] is None for r in deps[policy])
        fixed = all(finals[policy][r["id"]] == first[r["id"]] for r in rows if r["question_type"] == agent.Q1)
        coverage = len(locs[policy]) == len(q3) and all(finals[policy][r["id"]] in r["options"] for r in rows)
        eligible[policy] = fixed and coverage and not errors and not invalids
        statuses.append({"policy": policy, "eligible_for_comparison": eligible[policy], "q1_fixed": fixed,
            "row_errors": errors, "invalid_tool_answers": invalids, "q1_inference_calls": 0, "tool_calls": len(deps[policy])})
    common.evaluate(rows, q3, first, finals, locs, eligible, deps, out, MODEL, args.n_boot,
        policies=POLICIES, comparisons=COMPARISONS, scope=f"exploratory dev{len(q3)}")
    q3bydefect = []
    gt = {r["core_id"]: r["defect_gt"] for r in q2rows}
    for policy in POLICIES:
        if not eligible[policy]:
            continue
        for defect in sorted(set(gt.values())):
            subset = [r for r in q3 if gt[r["core_id"]] == defect]
            predicted = [locs[policy][r["id"]] != r["none_choice"] for r in subset]
            q3bydefect.append({"policy": policy, "defect": defect, "images": len(subset),
                "region_present": sum(predicted), "region_none": len(subset)-sum(predicted),
                "missed_defects": sum(r["anomaly_gt"] == "anomalous" and not v for r,v in zip(subset,predicted)),
                "normal_false_regions": sum(r["anomaly_gt"] == "normal" and v for r,v in zip(subset,predicted))})
    agent.cons.write_csv(out/"q3_by_defect.csv", q3bydefect)
    complete = all(eligible.values())
    counts.update({"row_errors": sum(bool(r["error"]) for r in used.values()),
        "invalid_answers": sum(r["predicted_choice"] is None for r in used.values()), "unique_tool_calls": len(used)})
    meta.update({**{k: counts[k] for k in ("row_errors", "invalid_answers", "called_this_run", "reused_calls", "unique_tool_calls", "adapter_load_blocked_steps")},
        "complete": complete, "gpu": agent.gpu_info() or previous.get("gpu"),
        "software": {}, "q2_definition_reference": "control=fresh revised definitions; short=concise evidence definitions"})
    for package in ("torch", "transformers", "numpy", "Pillow"):
        try:
            meta["software"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta["software"][package] = None
    old.atomic_json(meta_path, meta)
    old.atomic_json(out/"run_status.json", {"status": "complete" if complete else "incomplete", "complete": complete,
        "policies": statuses, "selection_performed": False, "result_published": False})
    print(json.dumps({**counts, "complete": complete, "q1_inference_calls": 0}, indent=2), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
