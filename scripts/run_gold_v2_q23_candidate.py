#!/usr/bin/env python3
"""Locked dev-selected Q2/legacy-Q3 candidate on gold, retaining baseline Q1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_gold_v2_q2_q3_refined_dev as refined
import build_retrieved_refs as retrieval
from insplad_benchmark.metrics import anomaly_classification_summary, balanced_accuracy

old = refined.old
agent = old.agent
PROTOCOL = "gold_v2_q1_fixed_q2_cues_q3_legacy"
PREFIX = "gold_v2_q23_candidate"
CONFIG = ROOT / "configs/q1_fixed_q2_cues_q3_legacy_gold_v1.json"
POLICIES = ("baseline", "previous_agent", "candidate")
COMPARISONS = [("baseline", "candidate"), ("previous_agent", "candidate")]


def q12_metrics(rows, choices):
    pairs = agent.cons.image_pairs(rows)
    q1_true = [a["anomaly_gt"] for a, _ in pairs]
    q1_pred = [agent.base.predicted_semantic(a, choices.get(a["id"])) or "invalid" for a, _ in pairs]
    q2_true = [b["defect_gt"] for _, b in pairs]
    q2_pred = [agent.base.predicted_semantic(b, choices.get(b["id"])) or "invalid" for _, b in pairs]
    summary = anomaly_classification_summary(q1_true, q1_pred)
    return {**agent.cons.consistency_row(pairs, choices),
        **{f"q1_{key}": summary[key] for key in ("balanced_accuracy", "precision", "recall", "tp", "fp", "tn", "fn", "has_both_classes")},
        "q1_normal_images": summary["normal_n"], "q1_anomalous_images": summary["anomalous_n"],
        "q1_accuracy": sum(a == b for a, b in zip(q1_true, q1_pred))/len(pairs),
        "q2_accuracy": sum(a == b for a, b in zip(q2_true, q2_pred))/len(pairs),
        "q2_defect_balanced_accuracy": balanced_accuracy(q2_true, q2_pred)}


def save_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    agent.base.write_jsonl(tmp, list(records))
    tmp.replace(path)


def gold_rows(rows):
    """Require paired gold Q1/Q2; never reinterpret dev or original Q3 answers."""
    if not rows or any(r.get("protocol") == "gold_v2_dev" for r in rows):
        raise ValueError("expected gold QA, not dev")
    rows = [r for r in rows if r["question_type"] in (agent.Q1, agent.Q2)]
    pairs = {}
    ids = set()
    for r in rows:
        if r["id"] in ids or r["question_type"] in pairs.setdefault(r["core_id"], {}):
            raise ValueError("duplicate QA ID or question in a core")
        ids.add(r["id"])
        pairs[r["core_id"]][r["question_type"]] = r
        if r.get("answer") not in r["options"] or r.get("anomaly_gt") not in ("normal", "anomalous"):
            raise ValueError("missing gold Q1/Q2 labels")
        if r["question_type"] == agent.Q2 and set(r["options"].values()) != set(refined.DEFINITIONS):
            raise ValueError("unexpected Q2 schema")
    for qs in pairs.values():
        if set(qs) != {agent.Q1, agent.Q2}:
            raise ValueError("each gold image needs Q1 and Q2")
        a, b = qs[agent.Q1], qs[agent.Q2]
        if any(a.get(k) != b.get(k) for k in ("image", "asset_category", "anomaly_gt", "defect_gt")):
            raise ValueError("Q1/Q2 inputs or labels disagree")
    return [pairs[c][qt] for c in sorted(pairs) for qt in (agent.Q1, agent.Q2)]


def read_references(path, rows):
    with path.open(encoding="utf-8-sig", newline="") as h:
        records = list(csv.DictReader(h))
    if len({r["image"] for r in records}) != len(records):
        raise ValueError("duplicate reference mapping")
    refs = {r["image"]: r["retrieved_diffdate"] for r in records}
    for row in rows:
        ref = refs.get(row["image"])
        if not ref or "/train/good/" not in "/" + ref.replace("\\", "/"):
            raise ValueError("missing train-normal diffdate reference")
        if ref.replace("\\", "/").split("/")[0] != row["asset_category"]:
            raise ValueError("reference asset differs from target")
        date = retrieval.flight_date(row["image"])
        if date and date == retrieval.flight_date(ref):
            raise ValueError("same-flight reference in diffdate condition")
    return {r["image"]: refs[r["image"]] for r in rows}


def cached_baseline(path, rows, refs, model):
    """Only import valid predictions from the existing gold main-condition runner."""
    if path is None or not path.is_file():
        return {}, None
    records = agent.read_jsonl(path)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate baseline prediction IDs")
    by_id = {r["id"]: r for r in records}
    found = {}
    for row in rows:
        p = by_id.get(row["id"])
        if not p or p.get("error") or not str(p.get("output", "")).strip():
            continue  # missing/error baseline calls are generated only in the new folder
        if (p.get("model_key"), p.get("condition"), p.get("prompt_version"),
            p.get("core_id"), p.get("question_type")) != (
                model, "normal_ref_retrieved_diffdate", 1, row["core_id"], row["question_type"]):
            raise ValueError("baseline protocol/model/question identity mismatch")
        ref_path = str(p.get("reference_image", "")).replace("\\", "/")
        if not ref_path.endswith("/" + refs[row["image"]].replace("\\", "/")):
            raise ValueError("baseline reference differs from current diffdate mapping")
        choice = agent.base.parse_choice(p["output"], set(row["options"]))
        if choice is not None and choice == p.get("predicted_choice"):
            found[row["id"]] = p
    return found, hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(rows, q3, first, finals, locations, eligible, deps, out, model, n_boot):
    found, loc_found, joint = {}, {}, {}
    leaderboard, answers, assets, defects = [], [], [], []
    for policy in POLICIES:
        final, loc = finals[policy], locations[policy]
        for r in [*rows, *q3]:
            choice = loc.get(r["id"]) if r["question_type"] == agent.Q3 else final.get(r["id"])
            answers.append({"model_key": model, "policy": policy, "id": r["id"], "core_id": r["core_id"],
                "question_type": r["question_type"], "asset_category": r["asset_category"],
                "predicted_choice": choice, "predicted_text": r["options"].get(choice, "INVALID"),
                "cached_baseline_choice": first.get(r["id"]), "answer": r.get("answer"),
                "spatial_gt_known": False if r["question_type"] == agent.Q3 else None,
                "eligible_for_comparison": eligible[policy]})
        if not eligible[policy]:
            continue
        found[model, policy], loc_found[model, policy] = final, loc
        pairs = agent.cons.image_pairs(rows)
        joint[model, policy] = {a["core_id"]: final[a["id"]] == a["answer"] and final[b["id"]] == b["answer"] for a, b in pairs}
        def metrics(sub12, sub3):
            return {**q12_metrics(sub12, final),
                **old.presence_metrics(sub3, loc, joint[model, policy])}
        leaderboard.append({"model_key": model, "policy": policy, **metrics(rows, q3),
            "incremental_tool_calls": len(deps[policy]), "tool_latency_s": sum(p["latency_s"] for p in deps[policy])})
        for asset in sorted({r["asset_category"] for r in rows}):
            sub12 = [r for r in rows if r["asset_category"] == asset]
            sub3 = [r for r in q3 if r["asset_category"] == asset]
            assets.append({"model_key": model, "policy": policy, "asset_category": asset, **metrics(sub12, sub3)})
        for defect in sorted({r["defect_gt"] for r in rows}):
            sub = [r for r in rows if r["question_type"] == agent.Q2 and r["defect_gt"] == defect]
            correct = sum(final[r["id"]] == r["answer"] for r in sub)
            defects.append({"model_key": model, "policy": policy, "defect": defect, "images": len(sub),
                "q2_correct": correct, "q2_accuracy": correct/len(sub)})
    sig = agent.cons.significance_rows(agent.cons.image_pairs(rows), [], found, [model], COMPARISONS, n_boot)
    sig = [r for r in sig if r["metric"] not in ("q3_accuracy", "q1q3_contradiction")]
    sig += old.q3_bootstrap(q3, loc_found, joint, [model], n_boot, COMPARISONS,
                            scope=f"gold{len(q3)} presence only, no localization claim")
    sig = [{**r, "scope": r.get("scope") or f"gold{len(q3)} Q1/Q2 image-level paired comparison"} for r in sig]
    for name, records in (("leaderboard.csv", leaderboard), ("answers.csv", answers),
                          ("metrics_by_asset.csv", assets), ("q2_by_defect.csv", defects), ("significance.csv", sig)):
        agent.cons.write_csv(out/name, records)
    controls = []
    for control, none in (("always_none", True), ("always_region", False)):
        choices = {r["id"]: r["none_choice"] if none else next(k for k in r["options"] if k != r["none_choice"]) for r in q3}
        controls.append({"control": control, **old.presence_metrics(q3, choices)})
    agent.cons.write_csv(out/"q3_constant_baselines.csv", controls)
    macro = []
    for policy in POLICIES:
        sub = [r for r in assets if r["policy"] == policy]
        if sub:
            keys = ("q1_balanced_accuracy", "q1_anomaly_f1", "q1_precision", "q1_recall", "q1_invalid_rate",
                    "q2_defect_balanced_accuracy", "q2_defect_macro_f1", "q2_invalid_rate", "joint_accuracy")
            macro.append({"model_key": model, "policy": policy, "assets": len(sub),
                          **{k: sum(r[k] for r in sub)/len(sub) for k in keys}})
    agent.cons.write_csv(out/"metrics_asset_macro.csv", macro)


def main(argv=None, *, loader=None, infer=None, load_qa=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("qa-dir", "refs-csv", "sup-zip", "unsup-zip", "out-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--baseline-predictions", type=Path)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    if not args.out_dir.name.startswith(PREFIX) or args.n_boot <= 0 or args.checkpoint_every <= 0:
        raise ValueError("use a separate gold_v2_q23_candidate* output folder and positive check counts")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config != json.loads(CONFIG.read_text(encoding="utf-8")):
        raise ValueError("configuration differs from the committed pre-gold lock")
    model = config["models"][0]
    summary = json.loads((args.qa_dir/"build_summary.json").read_text(encoding="utf-8"))
    if summary.get("quality_status") != "FROZEN":
        raise ValueError("gold QA must be FROZEN")
    rows = gold_rows((load_qa or agent.v2.load_gold_qa_v2)(args.qa_dir))
    if load_qa is None and (len(rows) != 492 or summary.get("final_master_images") != 246):
        raise ValueError("expected original gold246 Q1/Q2 QA")
    q3 = [{**r, "id": r["core_id"]+"-q3-gold-presence", "protocol": PROTOCOL}
          for r in old.q3_rows(rows)]
    refs = read_references(args.refs_csv, rows)
    external, baseline_sha = cached_baseline(args.baseline_predictions, rows, refs, model)
    code_paths = [Path(__file__), Path(refined.__file__), Path(old.__file__), Path(agent.__file__),
                  Path(agent.cons.__file__), Path(old.diagnostic.dev.__file__),
                  ROOT/"scripts/run_gold_v2_reference_ablation.py", ROOT/"src/insplad_benchmark/hf_adapters.py"]
    code = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in code_paths}
    signature = old.diagnostic.digest({"rows": rows, "q3": q3, "refs": refs, "config": config, "code": code,
        "baseline_sha256": baseline_sha, "model_spec": agent.base.MODEL_REGISTRY[model], "n_boot": args.n_boot})
    meta_path = args.out_dir/"run_metadata.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if previous and previous.get("input_signature") != signature:
        raise ValueError("inputs/config/code changed: use a new folder; previous results preserved")
    targets = old.verified_images(args.sup_zip, [r["image"] for r in rows], args.out_dir/"_images/target")
    normals = old.verified_images(args.unsup_zip, list(refs.values()), args.out_dir/"_images/normal")
    manifest = [{"role": role, "member": member, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for role, paths in (("target", targets), ("reference", normals)) for member, path in sorted(paths.items())]
    identity = old.diagnostic.digest([signature, manifest])
    if previous and previous.get("prediction_identity") != identity:
        raise ValueError("image bytes changed: use a new folder")
    path = args.out_dir/"predictions"/model/"steps.jsonl"
    records = agent.read_jsonl(path) if path.exists() else []
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate saved step IDs")
    existing = {r["id"]: r for r in records}
    meta = {**previous, "protocol": PROTOCOL, "input_signature": signature, "prediction_identity": identity,
        "benchmark_commit": agent.git_commit(), "config": config, "code_sha256": code,
        "images": len(q3), "questions_q12": len(rows), "questions_q3_auxiliary": len(q3),
        "original_gold_q3": "untouched original four-option questions; not rescored here",
        "q1": "direct baseline fixed; zero candidate Q1 inference", "q3_location_accuracy": None,
        "independence": "Q2 own inputs only; Q3 own inputs/answers only; contradictions evaluation only",
        "baseline_source": str(args.baseline_predictions) if args.baseline_predictions else None,
        "baseline_source_sha256": baseline_sha, "n_boot": args.n_boot,
        "gpu": previous.get("gpu"), "model_spec": agent.base.MODEL_REGISTRY[model], "software": {}, "complete": False}
    old.atomic_json(meta_path, meta)
    old.atomic_json(args.out_dir/"protocol_lock.json", {"config": config, "input_signature": signature,
        "prediction_identity": identity, "note": "pre-inference protocol, not a successful-result or selection marker"})
    old.atomic_json(args.out_dir/"run_status.json", {"status": "running", "complete": False, "policies": []})
    save_records(args.out_dir/"gold_qa_q12_snapshot.jsonl", rows)
    save_records(args.out_dir/"q3_auxiliary_partial_qa.jsonl", q3)
    agent.cons.write_csv(args.out_dir/"image_manifest.csv", manifest)
    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer
    holder = agent.base.LazyHFAdapter(agent.base.MODEL_REGISTRY[model], loader=loader)
    used, first, baseline_deps = {}, {}, []
    finals = {policy: {} for policy in POLICIES}
    locations = {policy: {} for policy in POLICIES}
    deps = {policy: [] for policy in POLICIES}
    counts = Counter()
    load_failure = ""

    def call(row, prompt, step, cached=None):
        nonlocal load_failure
        key = row["id"]+"/"+step
        rec = existing.get(key)
        if rec and rec.get("identity") != identity:
            raise ValueError("saved prediction identity mismatch")
        valid = (rec and not rec.get("error") and rec.get("prompt") == prompt
            and rec.get("predicted_choice") in row["options"]
            and agent.base.parse_choice(str(rec.get("output", "")), set(row["options"])) == rec["predicted_choice"])
        if valid:
            counts["reused_calls"] += 1
        else:
            images = [normals[refs[row["image"]]], targets[row["image"]]]
            output, error, latency = "", "", 0.0
            if cached:
                output, latency = cached["output"], float(cached.get("latency_s") or 0)
                counts["imported_baseline_calls"] += 1
            else:
                start = time.time()
                if load_failure:
                    error = "blocked by adapter load failure: " + load_failure
                    counts["adapter_load_blocked_steps"] += 1
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
                    counts["baseline_calls_this_run" if step == "baseline" else "tool_calls_this_run"] += 1
                latency = round(time.time()-start, 3)
            rec = {"id": key, "question_id": row["id"], "question_type": row["question_type"], "step": step,
                "identity": identity, "model_key": model, "prompt": prompt, "images": list(map(str, images)),
                "output": output, "predicted_choice": agent.base.parse_choice(output, set(row["options"])),
                "error": error, "latency_s": latency, "source": "imported_baseline" if cached else "inference"}
        used[key] = rec
        if len(used) % args.checkpoint_every == 0:
            save_records(path, [*used.values(), *(r for i, r in existing.items() if i not in used)])
            print(f"[{model}] {len(used)} baseline/tool steps saved", flush=True)
        return rec

    try:
        for row in rows:
            rec = call(row, old.direct_prompt(row), "baseline", external.get(row["id"]))
            first[row["id"]] = rec["predicted_choice"]
            baseline_deps.append(rec)
        # Invalid/failed direct baselines cannot be treated as a fixed valid Q1 reference.
        if any(r["error"] or r["predicted_choice"] is None for r in baseline_deps):
            counts["baseline_blocked"] = 1
        else:
            finals = {policy: dict(first) for policy in POLICIES}
            for q2, r3 in zip((r for r in rows if r["question_type"] == agent.Q2), q3):
                updated = call(q2, refined.definition_prompt(q2, True), "q2_cues")
                defined = call(q2, agent.build_prompt(q2, first[q2["id"]], "define"), "q2_define")
                direct = call(r3, old.direct_prompt(r3), "q3_direct")
                task, prompt = old.presence_task(r3)
                presence = call(task, prompt, "q3_presence")
                decision = task["options"].get(presence["predicted_choice"])
                position = call(r3, old.position_prompt(r3), "q3_position") if decision == "A visible localized defect region." else None
                choice = (r3["none_choice"] if decision == "No visible localized defect region." else
                    position["predicted_choice"] if position and position["predicted_choice"] else direct["predicted_choice"])
                locations["baseline"][r3["id"]] = direct["predicted_choice"]
                deps["baseline"].append(direct)
                for policy, q2rec in (("previous_agent", defined), ("candidate", updated)):
                    finals[policy][q2["id"]] = q2rec["predicted_choice"] or first[q2["id"]]
                    locations[policy][r3["id"]] = choice
                    deps[policy].extend([q2rec, direct, presence]+([position] if position else []))
    finally:
        # Even unexpected interruptions preserve the unprocessed cache tail.
        save_records(path, [*used.values(), *(r for i, r in existing.items() if i not in used)])
        holder.release()
    statuses, eligible = [], {}
    for policy in POLICIES:
        dependencies = baseline_deps+deps[policy]
        errors = sum(bool(r["error"]) for r in dependencies)
        invalids = sum(r["predicted_choice"] is None for r in dependencies)
        coverage = len(finals[policy]) == len(rows) and len(locations[policy]) == len(q3)
        q1_fixed = all(finals[policy].get(r["id"]) == first[r["id"]] for r in rows if r["question_type"] == agent.Q1)
        eligible[policy] = coverage and q1_fixed and not errors and not invalids
        statuses.append({"model_key": model, "policy": policy, "status": "complete" if eligible[policy] else "incomplete",
            "eligible_for_comparison": eligible[policy], "row_errors": errors, "invalid_tool_answers": invalids,
            "q12_answers": len(finals[policy]), "q3_answers": len(locations[policy]),
            "q1_fixed": q1_fixed, "q1_candidate_calls": 0, "tool_calls": len(deps[policy])})
    evaluate(rows, q3, first, finals, locations, eligible, deps, args.out_dir, model, args.n_boot)
    complete = all(eligible.values())
    counts.update({"unique_calls": len(used), "row_errors": sum(bool(r["error"]) for r in used.values()),
                   "invalid_answers": sum(r["predicted_choice"] is None for r in used.values())})
    meta.update({**{key: counts[key] for key in ("called_this_run", "reused_calls", "imported_baseline_calls", "baseline_calls_this_run",
        "tool_calls_this_run", "adapter_load_blocked_steps", "unique_calls", "row_errors", "invalid_answers")}, "complete": complete,
        "images_by_gt": dict(Counter(r["anomaly_gt"] for r in q3)), "q1_candidate_calls": 0,
        "gpu": agent.gpu_info() or previous.get("gpu")})
    for package in ("torch", "transformers", "numpy", "Pillow"):
        try:
            meta["software"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta["software"][package] = None
    old.atomic_json(meta_path, meta)
    old.atomic_json(args.out_dir/"run_status.json", {"status": "complete" if complete else "incomplete", "complete": complete,
        "policies": statuses, "selection_performed": False, "result_published": False})
    print(json.dumps({**counts, "complete": complete, "q1_candidate_calls": 0}, indent=2), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
