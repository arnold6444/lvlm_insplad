#!/usr/bin/env python3
"""Dev-only Q2/Q3 tools with frozen Q1 answers and partial Q3 presence evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_dev_prompt_diagnostic as diagnostic
import run_gold_v2_verify_agent as agent
from dev_part_crop_annotation import atomic_json
from insplad_benchmark.metrics import anomaly_classification_summary

VERSION = 1
PROTOCOL = "gold_v2_q2_q3_independent_dev"
NONE = agent.cons.NONE_TEXT
REGIONS = tuple(agent._CELLS)
POLICIES = ("baseline", "q2_define", "q2_define_q3_independent", "matched_reask")
COMPARISONS = [("baseline", p) for p in POLICIES[1:]] + [("matched_reask", "q2_define_q3_independent")]


def verified_images(zip_path, members, root):
    """Bind cached images to current ZIP bytes without replacing an earlier input."""
    paths = {}
    with ZipFile(zip_path) as archive:
        names = {n.replace("\\", "/").lstrip("/"): n for n in archive.namelist()}
        for member in sorted(set(members)):
            path = root / member
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("unsafe image member")
            payload = archive.read(names[member.replace("\\", "/").lstrip("/")])
            if path.exists() and path.read_bytes() != payload:
                raise ValueError("input image bytes changed: use a new output folder")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            paths[member] = path
    return paths


def options_for(core, texts, purpose):
    texts = list(texts)
    random.Random(diagnostic.digest([core, purpose, VERSION])).shuffle(texts)
    return {chr(65+i): text for i, text in enumerate(texts)}


def q3_rows(rows):
    """GT never determines options/routing. Positive spatial answers remain unknown."""
    out = []
    for row in rows:
        if row["question_type"] != agent.Q1:
            continue
        if row["anomaly_gt"] not in ("normal", "anomalous"):
            raise ValueError("unknown image-level GT")
        options = options_for(row["core_id"], [*REGIONS, NONE], "q3_grid")
        none = next(k for k, value in options.items() if value == NONE)
        out.append({"id": row["core_id"]+"-q3-dev-presence", "core_id": row["core_id"],
                    "image": row["image"], "asset_category": row["asset_category"], "protocol": PROTOCOL,
                    "question_type": agent.Q3, "anomaly_gt": row["anomaly_gt"],
                    "question": "Where is a localized defect region in the TARGET image? If no visible defect region is present, choose the no-region option.",
                    "options": options, "none_choice": none,
                    "answer": none if row["anomaly_gt"] == "normal" else None,
                    "location_gt_known": False})
    return out


def direct_prompt(row):
    return diagnostic.dev.build_prompt(row, "normal_ref_retrieved_diffdate")


def presence_task(row):
    options = options_for(row["core_id"], ["No visible localized defect region.", "A visible localized defect region."], "q3_presence")
    task = {**row, "options": options,
            "question": "For the localization task, does the TARGET image contain a visible localized defect region? Assess the TARGET itself, not the normal reference."}
    return task, direct_prompt(task)


def position_prompt(row):
    text = direct_prompt(row)
    instructions = ("Use the full TARGET frame as a 3 by 3 grid. Top/middle/bottom and left/center/right refer to the "
                    "original frame, not an imagined crop. Choose a grid region only when a localized defect is visible; "
                    "a difference in lighting or viewpoint from the reference alone does not establish a defect.")
    return text.replace("\n\nQuestion:", "\n\n"+instructions+"\n\nQuestion:", 1)


def q3_semantic(row, choice):
    if choice not in row["options"]:
        return "invalid"
    return "normal" if row["options"][choice] == NONE else "anomalous"


def presence_metrics(rows, choices, q12_correct=None):
    true = [r["anomaly_gt"] for r in rows]
    pred = [q3_semantic(r, choices.get(r["id"])) for r in rows]
    s = anomaly_classification_summary(true, pred)
    return {"q3_presence_balanced_accuracy": s["balanced_accuracy"],
            "q3_presence_accuracy": sum(a == b for a, b in zip(true, pred))/len(rows),
            "q3_normal_none_accuracy": s["specificity"],
            "q3_normal_false_region_rate": s["fp"]/s["normal_n"] if s["normal_n"] else None,
            "q3_anomaly_false_none_rate": sum(a == "anomalous" and b == "normal" for a, b in zip(true, pred))/s["anomalous_n"] if s["anomalous_n"] else None,
            "q3_anomaly_region_recall": s["recall"], "q3_invalid_rate": s["invalid_output_rate"],
            "q3_normal_images": s["normal_n"], "q3_anomalous_images": s["anomalous_n"],
            "q1_q2_q3_presence_joint": sum(bool(q12_correct.get(r["core_id"])) and a == b
                                         for r, a, b in zip(rows, true, pred))/len(rows) if q12_correct is not None else None,
            "q3_location_accuracy": None, "q3_anomalous_location_labels": 0}


def q3_bootstrap(rows, found, joint, models, n_boot=2000, comparisons=None,
                 scope="dev presence only, no localization claim"):
    out = []
    metrics = ("q3_presence_balanced_accuracy", "q3_normal_false_region_rate", "q3_anomaly_false_none_rate", "q1_q2_q3_presence_joint")
    for model in models:
        for base_policy, policy in COMPARISONS if comparisons is None else comparisons:
            baseline = (model, base_policy)
            key = (model, policy)
            if key not in found or baseline not in found:
                continue
            base = presence_metrics(rows, found[baseline], joint[baseline])
            new = presence_metrics(rows, found[key], joint[key])
            samples = {k: [] for k in metrics}
            rng = np.random.default_rng(42)
            for _ in range(n_boot):
                subset = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
                a = presence_metrics(subset, found[baseline], joint[baseline])
                b = presence_metrics(subset, found[key], joint[key])
                for metric in metrics:
                    if a[metric] is not None and b[metric] is not None:
                        samples[metric].append(b[metric]-a[metric])
            for metric in metrics:
                if samples[metric] and base[metric] is not None and new[metric] is not None:
                    low, high = np.percentile(samples[metric], [2.5, 97.5])
                    out.append({"model_key": model, "baseline": base_policy, "condition": policy, "metric": metric,
                                "base": base[metric], "new": new[metric], "delta": new[metric]-base[metric],
                                "ci_low": float(low), "ci_high": float(high), "significant": bool(low > 0 or high < 0),
                                "n_boot": n_boot, "scope": scope})
    return out


def main(argv=None, *, loader=None, infer=None, refinement=None):
    protocol = refinement.PROTOCOL if refinement else PROTOCOL
    version = refinement.VERSION if refinement else VERSION
    policies = refinement.POLICIES if refinement else POLICIES
    comparisons = refinement.COMPARISONS if refinement else COMPARISONS
    prefix = refinement.FOLDER_PREFIX if refinement else "gold_v2_q2_q3_independent_dev"
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dev-qa", "stage1-dir", "refs-csv", "sup-zip", "unsup-zip", "out-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=list(diagnostic.dev.MODELS), default=["internvl3_5_8b"])
    p.add_argument("--n-images", type=int, default=240 if refinement else 40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    if not args.out_dir.name.startswith(prefix):
        raise ValueError(f"use a separate {prefix}* output directory")
    if len(set(args.models)) != len(args.models) or args.n_boot <= 0:
        raise ValueError("duplicate models or invalid bootstrap count")
    if refinement:
        rows, cohorts = refinement.select_rows(agent.read_jsonl(args.dev_qa), args.n_images, args.seed)
    else:
        rows = diagnostic.select_subset(agent.read_jsonl(args.dev_qa), args.n_images, args.seed)
        cohorts = {}
    for i in range(0, len(rows), 2):
        if any(rows[i].get(k) != rows[i+1].get(k) for k in ("asset_category", "anomaly_gt", "defect_gt")):
            raise ValueError("Q1/Q2 image labels disagree")
    q3 = q3_rows(rows)
    with args.refs_csv.open(encoding="utf-8-sig", newline="") as h:
        refs = {r["image"]: r["retrieved_diffdate"] for r in csv.DictReader(h)}
    if any(not refs.get(r["image"]) for r in rows):
        raise ValueError("missing diffdate reference")
    first, first_latency = {}, {}
    for model in args.models:
        records = agent.read_jsonl(args.stage1_dir / model / "normal_ref_retrieved_diffdate.jsonl")
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("duplicate baseline prediction IDs")
        by_id = {r["id"]: r for r in records}
        if any(r["id"] not in by_id or by_id[r["id"]].get("error") or by_id[r["id"]].get("predicted_choice") not in r["options"] for r in rows):
            raise ValueError("baseline is incomplete/failed/invalid; retry the dev baseline first")
        first[model] = {r["id"]: by_id[r["id"]]["predicted_choice"] for r in rows}
        first_latency[model] = sum(float(by_id[r["id"]].get("latency_s") or 0) for r in rows)
    prompt_audit = []
    for m in args.models:
        for r in rows:
            if r["question_type"] == agent.Q2:
                for tool in ("reask", "define"):
                    prompt_audit.append({"model_key": m, "id": r["id"], "step": "q2_"+tool,
                                         "prompt": agent.build_prompt(r, first[m][r["id"]], tool)})
        for r in q3:
            for step, prompt in (("q3_direct", direct_prompt(r)), ("q3_presence", presence_task(r)[1]), ("q3_position", position_prompt(r))):
                prompt_audit.append({"model_key": m, "id": r["id"], "step": step, "prompt": prompt})
    if refinement:
        prompt_audit = refinement.audit_prompts(rows, q3, first, args.models)
    signature_data = {"protocol": protocol, "version": version, "rows": rows, "q3": q3,
                                    "first": first, "seed": args.seed, "refs": {r["image"]: refs[r["image"]] for r in rows},
                                    "prompts": prompt_audit, "specs": {m: agent.base.MODEL_REGISTRY[m] for m in args.models}}
    if refinement:
        signature_data["code_sha256"] = {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                          for p in (__file__, refinement.__file__)}
        signature_data["cohorts"] = {k: sorted(v) for k, v in cohorts.items()}
    signature = diagnostic.digest(signature_data)
    meta_path = args.out_dir / "run_metadata.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    if previous and previous["input_signature"] != signature:
        raise ValueError("different inputs/protocol: use a new output folder, existing results are preserved")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = verified_images(args.sup_zip, [r["image"] for r in rows], args.out_dir / "_images" / "target")
    normals = verified_images(args.unsup_zip, [refs[r["image"]] for r in rows], args.out_dir / "_images" / "normal")
    manifest = [{"role": role, "member": member, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for role, paths in (("target", targets), ("reference", normals)) for member, path in paths.items()]
    identity = diagnostic.digest([signature, [{k: r[k] for k in ("role", "member", "sha256")} for r in manifest]])
    if previous and previous["prediction_identity"] != identity:
        raise ValueError("input image bytes changed: use a new output folder")
    meta = {"protocol": protocol, "version": version, "input_signature": signature, "prediction_identity": identity,
            "benchmark_commit": agent.git_commit(), "models": args.models, "model_specs": {m: agent.base.MODEL_REGISTRY[m] for m in args.models},
            "images": args.n_images, "questions_q12": len(rows), "questions_q3": len(q3), "seed": args.seed,
            "policies": policies, "q1": "exact cached stage1 answers; zero Q1 inference calls",
            "q2": "own-question reask/define, valid output replaces own Q2; invalid keeps cached own Q2",
            "q3": "independent presence task -> optional position task; invalid keeps own direct Q3, no Q1/Q2 answers used",
            "q3_schema": "nine original-frame grid regions plus no-region, order shuffled without GT",
            "q3_evaluation": "none-vs-region from image-level GT only; anomalous spatial GT unknown; not gold-compatible localization accuracy",
            "q3_control": "repeat direct Q3 prompt once or twice, matching agent additional call count per image",
            "independence": "each prompt/route/final uses only that question's inputs; contradictions are evaluation only",
            "cost": "incremental Q2/Q3 tool calls, cached Q1/Q2 baseline excluded; shared calls reported separately",
            "selection": "no automatic selection or frozen config; no gold execution", "gpu": None,
            "software": {}, "claim_limit": "exploratory dev validation; Q3 location accuracy remains unknown"}
    if refinement:
        meta.update({"q2": "fresh old-definition ablation and balanced revised definitions; no previous answer in fresh tools",
                     "q3": "refined presence AND position on every image; disagree -> own-Q3 review; invalid -> valid own position/direct",
                     "q3_control": "two/three direct Q3 repeats matching the refined presence/position/review budget",
                     "cohorts": {k: sorted(v) for k, v in cohorts.items()}, "code_sha256": signature_data["code_sha256"],
                     "primary_validation": "additional_dev (excluded from design40 diagnosis, still dev, not formal holdout)",
                     "claim_limit": "exploratory development validation; all/design/additional separately; Q3 spatial accuracy unknown"})
    atomic_json(meta_path, meta)
    agent.base.write_jsonl(args.out_dir / "subset_qa_q12.jsonl", rows)
    agent.base.write_jsonl(args.out_dir / "dev_q3_partial_qa.jsonl", q3)
    agent.base.write_jsonl(args.out_dir / "prompts.jsonl", prompt_audit)
    agent.cons.write_csv(args.out_dir / "image_manifest.csv", manifest)
    if refinement:
        agent.cons.write_csv(args.out_dir / "cohort_manifest.csv", [
            {"core_id": r["core_id"], "asset_category": r["asset_category"], "anomaly_gt": r["anomaly_gt"],
             "cohort": "design_dev40" if r["core_id"] in cohorts["design_dev40"] else "additional_dev"} for r in q3])
    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer
    q3_by_core = {r["core_id"]: r for r in q3}
    rows_by_core = {r["core_id"]: {q["question_type"]: q for q in rows if q["core_id"] == r["core_id"]} for r in rows[::2]}
    all_found, q3_found, joint, leaderboard, statuses, answers, by_asset, by_defect = {}, {}, {}, [], [], [], [], []
    total_new = total_reused = 0
    run_errors = 0
    for model in args.models:
        holder = agent.base.LazyHFAdapter(agent.base.MODEL_REGISTRY[model], loader=loader)
        path = args.out_dir / "predictions" / model / "steps.jsonl"
        existing = {r["id"]: r for r in (agent.read_jsonl(path) if path.exists() else [])}
        used = {}

        def call(row, prompt, step):
            nonlocal total_new, total_reused
            key = row["id"]+"/"+step
            old = existing.get(key)
            if old and old.get("identity") == identity and not old.get("error") and old.get("output", "").strip():
                rec = old
                total_reused += 1
            else:
                images = [normals[refs[row["image"]]], targets[row["image"]]]
                output, error, start = "", "", time.time()
                try:
                    output = str(infer(holder.get(), images, prompt))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                rec = {"id": key, "question_id": row["id"], "question_type": row["question_type"], "step": step,
                       "model_key": model, "identity": identity, "images": list(map(str, images)), "prompt": prompt,
                       "output": output, "predicted_choice": agent.base.parse_choice(output, set(row["options"])),
                       "latency_s": round(time.time()-start, 3), "error": error}
                total_new += 1
            used[key] = rec
            if args.checkpoint_every > 0 and len(used) % args.checkpoint_every == 0:
                agent.base.write_jsonl(path, list(used.values())+[r for i, r in existing.items() if i not in used])
                print(f"[{model}] {len(used)} Q2/Q3 calls saved", flush=True)
            return rec

        finals = {policy: dict(first[model]) for policy in policies}
        q3_final = {policy: {} for policy in policies}
        deps = {policy: [] for policy in policies}
        try:
            for core, qs in rows_by_core.items():
                q2, r3 = qs[agent.Q2], q3_by_core[core]
                if refinement:
                    results = refinement.run_image(q2, r3, first[model][q2["id"]], call)
                    for policy, (choice2, choice3, dependencies) in results.items():
                        finals[policy][q2["id"]] = choice2
                        q3_final[policy][r3["id"]] = choice3
                        deps[policy].extend(dependencies)
                    continue
                reask = call(q2, agent.build_prompt(q2, first[model][q2["id"]], "reask"), "q2_reask")
                define = call(q2, agent.build_prompt(q2, first[model][q2["id"]], "define"), "q2_define")
                direct = call(r3, direct_prompt(r3), "q3_direct")
                task, prompt = presence_task(r3)
                presence = call(task, prompt, "q3_presence")
                decision = task["options"].get(presence["predicted_choice"])
                needs_position = decision == "A visible localized defect region."
                position = call(r3, position_prompt(r3), "q3_position") if needs_position else None
                control = [call(r3, direct_prompt(r3), f"q3_reask_{i+1}") for i in range(1+int(needs_position))]
                baseline_choice = direct["predicted_choice"]
                if decision == "No visible localized defect region.":
                    agent_choice = r3["none_choice"]
                elif position and position["predicted_choice"]:
                    agent_choice = position["predicted_choice"]
                else:
                    agent_choice = baseline_choice
                for policy in POLICIES:
                    q3_final[policy][r3["id"]] = baseline_choice
                    deps[policy].append(direct)
                for policy in ("q2_define", "q2_define_q3_independent"):
                    finals[policy][q2["id"]] = define["predicted_choice"] or first[model][q2["id"]]
                    deps[policy].append(define)
                q3_final["q2_define_q3_independent"][r3["id"]] = agent_choice
                deps["q2_define_q3_independent"].extend([presence]+([position] if position else []))
                finals["matched_reask"][q2["id"]] = reask["predicted_choice"] or first[model][q2["id"]]
                q3_final["matched_reask"][r3["id"]] = next((r["predicted_choice"] for r in reversed(control) if r["predicted_choice"]), baseline_choice)
                deps["matched_reask"].extend([reask, *control])
            agent.base.write_jsonl(path, list(used.values()))
        finally:
            holder.release()
        run_errors += sum(bool(r["error"]) for r in used.values())
        for policy in policies:
            final, loc = finals[policy], q3_final[policy]
            if any(final[r["id"]] != first[model][r["id"]] for r in rows if r["question_type"] == agent.Q1):
                raise AssertionError("Q1 changed")
            errors = sum(bool(r["error"]) for r in deps[policy])
            statuses.append({"model_key": model, "policy": policy, "row_errors": errors,
                             "invalid_tool_answers": sum(r["predicted_choice"] is None for r in deps[policy]),
                             "q1_inference_calls": 0, "q2_q3_calls": len(deps[policy]),
                             "status": "complete" if errors == 0 else "has_errors", "eligible_for_comparison": errors == 0})
            for r in [*rows, *q3]:
                choice = loc.get(r["id"]) if r["question_type"] == agent.Q3 else final[r["id"]]
                answers.append({"model_key": model, "policy": policy, "id": r["id"], "core_id": r["core_id"],
                                "question_type": r["question_type"], "asset_category": r["asset_category"],
                                "predicted_choice": choice, "predicted_text": r["options"].get(choice, "INVALID"),
                                "cached_baseline_choice": first[model].get(r["id"]),
                                "answer": r.get("answer"), "spatial_gt_known": False if r["question_type"] == agent.Q3 else None,
                                "q3_presence_correct": q3_semantic(r, choice) == r["anomaly_gt"] if r["question_type"] == agent.Q3 else None,
                                "q12_correct": choice == r["answer"] if r["question_type"] != agent.Q3 else None})
            if errors:
                continue
            all_found[(model, policy)], q3_found[(model, policy)] = final, loc
            correct_pairs = {core: all(final[q["id"]] == q["answer"] for q in qs.values()) for core, qs in rows_by_core.items()}
            joint[(model, policy)] = correct_pairs
            metrics = {"model_key": model, "policy": policy, **agent.summary_row(rows, first[model], final, set(), {}),
                       **presence_metrics(q3, loc, correct_pairs), "new_q2_q3_calls": len(deps[policy]),
                       "tool_latency_s": round(sum(r["latency_s"] for r in deps[policy]), 3),
                       "cached_stage1_latency_s": round(first_latency[model], 3)}
            # These are tool counts, not the legacy recheck cost convention.
            for key in ("rechecks", "q1_rechecks", "q2_rechecks", "avg_lvlm_calls_per_q", "recheck_latency_s"):
                metrics.pop(key, None)
            leaderboard.append(metrics)
            for asset in sorted({r["asset_category"] for r in rows}):
                subset = [r for r in rows if r["asset_category"] == asset]
                subset3 = [r for r in q3 if r["asset_category"] == asset]
                by_asset.append({"model_key": model, "policy": policy, "asset_category": asset,
                                 **agent.summary_row(subset, first[model], final, set(), {}), **presence_metrics(subset3, loc, correct_pairs)})
            for defect in sorted({r["defect_gt"] for r in rows if r["question_type"] == agent.Q2}):
                subset = [r for r in rows if r["question_type"] == agent.Q2 and r["defect_gt"] == defect]
                by_defect.append({"model_key": model, "policy": policy, "defect": defect, "images": len(subset),
                                  "q2_correct": sum(final[r["id"]] == r["answer"] for r in subset),
                                  "q2_accuracy": sum(final[r["id"]] == r["answer"] for r in subset)/len(subset)})
    sig = agent.cons.significance_rows(agent.cons.image_pairs(rows), [], all_found, args.models,
                                       comparisons, n_boot=args.n_boot)
    sig = [r for r in sig if r["metric"] not in ("q3_accuracy", "q1q3_contradiction")]
    sig += q3_bootstrap(q3, q3_found, joint, args.models, args.n_boot, comparisons)
    if refinement:
        cohort_metrics, cohort_sig, cohort_assets, cohort_defects = [], [], [], []
        for cohort, cores in cohorts.items():
            subset = [r for r in rows if r["core_id"] in cores]
            subset3 = [r for r in q3 if r["core_id"] in cores]
            if not subset:
                continue
            for (model, policy), final in all_found.items():
                metrics = {**agent.summary_row(subset, first[model], final, set(), {}),
                           **presence_metrics(subset3, q3_found[(model, policy)], joint[(model, policy)])}
                cohort_metrics.append({"cohort": cohort, "model_key": model, "policy": policy, **metrics})
                for asset in sorted({r["asset_category"] for r in subset}):
                    asset12 = [r for r in subset if r["asset_category"] == asset]
                    asset3 = [r for r in subset3 if r["asset_category"] == asset]
                    cohort_assets.append({"cohort": cohort, "model_key": model, "policy": policy, "asset_category": asset,
                        **agent.summary_row(asset12, first[model], final, set(), {}),
                        **presence_metrics(asset3, q3_found[(model, policy)], joint[(model, policy)])})
                for defect in sorted({r["defect_gt"] for r in subset if r["question_type"] == agent.Q2}):
                    defect_rows = [r for r in subset if r["question_type"] == agent.Q2 and r["defect_gt"] == defect]
                    correct = sum(final[r["id"]] == r["answer"] for r in defect_rows)
                    cohort_defects.append({"cohort": cohort, "model_key": model, "policy": policy, "defect": defect,
                                          "images": len(defect_rows), "q2_correct": correct, "q2_accuracy": correct/len(defect_rows)})
            if cohort == "all_dev":
                stats = sig  # This is the same cohort as the main paired comparisons.
            else:
                stats = agent.cons.significance_rows(agent.cons.image_pairs(subset), [], all_found, args.models, comparisons, n_boot=args.n_boot)
                stats = [s for s in stats if s["metric"] not in ("q3_accuracy", "q1q3_contradiction")]
                stats += q3_bootstrap(subset3, q3_found, joint, args.models, args.n_boot, comparisons)
            cohort_sig.extend({"cohort": cohort, **s} for s in stats)
        for filename, records in (("cohort_metrics.csv", cohort_metrics), ("cohort_significance.csv", cohort_sig),
                                  ("cohort_metrics_by_asset.csv", cohort_assets), ("cohort_q2_by_defect.csv", cohort_defects)):
            agent.cons.write_csv(args.out_dir / filename, records)
    for filename, records in (("leaderboard.csv", leaderboard), ("answers.csv", answers), ("metrics_by_asset.csv", by_asset),
                              ("q2_by_defect.csv", by_defect), ("significance.csv", sig)):
        agent.cons.write_csv(args.out_dir / filename, records)
    controls = []
    for name, label in (("always_none", "normal"), ("always_region", "anomalous")):
        choices = {r["id"]: r["none_choice"] if label == "normal" else next(k for k in r["options"] if k != r["none_choice"]) for r in q3}
        controls.append({"control": name, **presence_metrics(q3, choices)})
    agent.cons.write_csv(args.out_dir / "q3_constant_baselines.csv", controls)
    meta.update({"gpu": agent.gpu_info(), "called_this_run": total_new, "reused_calls": total_reused,
                 "unique_tool_calls": total_new+total_reused, "row_errors": run_errors,
                 "images_by_gt": dict(Counter(r["anomaly_gt"] for r in q3)),
                 "q3_location_accuracy": None})
    import importlib.metadata
    for package in ("torch", "transformers", "numpy", "Pillow"):
        try:
            meta["software"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta["software"][package] = None
    atomic_json(meta_path, meta)
    atomic_json(args.out_dir / "run_status.json", statuses)
    print(json.dumps({"row_errors": run_errors, "called_this_run": total_new, "reused_calls": total_reused,
                      "q1_calls": 0, "q3_location_accuracy": None}, indent=2), flush=True)
    return 1 if run_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
