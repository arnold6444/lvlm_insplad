#!/usr/bin/env python3
"""Matched-call control for the locked gold agent: is the gain structure or just more calls?

Adds ``matched_reask`` next to the three PR56/57 gold policies. It spends the same LVLM calls as
``previous_agent`` on each image but adds no new evidence: Q2 gets one own-answer re-check without
definitions, Q3 gets as many sequential own-answer re-checks as the staged presence/position Q3 used
for that image. Q1 stays fixed. Saved candidate-run steps are reused when prompt and image bytes match,
so only the control calls are new and the published gold policies stay identical.
"""
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
import run_gold_v2_q23_candidate as candidate

refined, old, agent = candidate.refined, candidate.old, candidate.agent
PROTOCOL = "gold_v2_matched_call_control"
PREFIX = "gold_v2_matched_call"
CONFIG = ROOT / "configs/gold_v2_matched_call_v1.json"
POLICIES = ("baseline", "previous_agent", "candidate", "matched_reask")
# (a, b) reports b - a. The first row is the primary question.
COMPARISONS = [("matched_reask", "previous_agent"), ("baseline", "matched_reask"),
               ("matched_reask", "candidate"), ("baseline", "candidate"), ("previous_agent", "candidate")]


def image_member(path):
    """Role/member part of a saved image path, e.g. ``target/<zip member>``."""
    text = str(path).replace("\\", "/")
    if "/_images/" not in text:
        raise ValueError("unexpected saved image path")
    return text.split("/_images/", 1)[1]


def read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as h:
        return {(r["role"], r["member"]): r["sha256"] for r in csv.DictReader(h)}


def load_reuse(steps_path, manifest_path, manifest):
    """Valid saved steps of an earlier gold run whose image bytes equal this run's images."""
    if steps_path is None:
        return {}, None
    if manifest_path is None or not steps_path.is_file() or not manifest_path.is_file():
        raise ValueError("reuse needs both the saved steps and that run's image manifest")
    saved = read_manifest(manifest_path)
    current = {(r["role"], r["member"]): r["sha256"] for r in manifest}
    if any(saved.get(key, sha) != sha for key, sha in current.items()):
        raise ValueError("reused run image bytes differ from this run")
    roles = {"target": "target", "normal": "reference"}
    records = agent.read_jsonl(steps_path)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate reused step IDs")
    reuse = {}
    for rec in records:
        members = [image_member(p).split("/", 1) for p in rec.get("images", [])]
        if rec.get("error") or not members or any(
                (roles.get(role), member) not in current or (roles.get(role), member) not in saved
                for role, member in members):
            continue
        reuse[rec["id"]] = rec
    digest = hashlib.sha256(steps_path.read_bytes() + manifest_path.read_bytes()).hexdigest()
    return reuse, digest


def call_budget(rows, q3, baseline_deps, deps):
    """Calls each policy's final Q2/Q3 answers depend on (the shared Q1 call is excluded)."""
    q2_ids = {r["id"] for r in rows if r["question_type"] == agent.Q2}
    q2_base = sum(r["question_id"] in q2_ids for r in baseline_deps)
    out = []
    for policy, records in deps.items():
        q2 = sum(r["question_type"] == agent.Q2 for r in records) + (0 if policy == "candidate" else q2_base)
        q3n = sum(r["question_type"] == agent.Q3 for r in records)
        out.append({"policy": policy, "images": len(q3), "q2_calls": q2, "q3_calls": q3n,
                    "q2_calls_per_image": q2/len(q3), "q3_calls_per_image": q3n/len(q3),
                    "q2_q3_calls_per_image": (q2+q3n)/len(q3)})
    return out


def main(argv=None, *, loader=None, infer=None, load_qa=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("qa-dir", "refs-csv", "sup-zip", "unsup-zip", "out-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--baseline-predictions", type=Path)
    p.add_argument("--reuse-steps", type=Path, help="steps.jsonl of the completed gold_v2_q23_candidate run")
    p.add_argument("--reuse-manifest", type=Path, help="image_manifest.csv of that same run")
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    if not args.out_dir.name.startswith(PREFIX) or args.n_boot <= 0 or args.checkpoint_every <= 0:
        raise ValueError("use a separate gold_v2_matched_call* output folder and positive check counts")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config != json.loads(CONFIG.read_text(encoding="utf-8")):
        raise ValueError("configuration differs from the committed lock")
    model = config["models"][0]
    summary = json.loads((args.qa_dir/"build_summary.json").read_text(encoding="utf-8"))
    if summary.get("quality_status") != "FROZEN":
        raise ValueError("gold QA must be FROZEN")
    rows = candidate.gold_rows((load_qa or agent.v2.load_gold_qa_v2)(args.qa_dir))
    if load_qa is None and (len(rows) != 492 or summary.get("final_master_images") != 246):
        raise ValueError("expected original gold246 Q1/Q2 QA")
    # Same Q3 IDs/options as the candidate run, so its saved Q3 steps stay reusable.
    q3 = [{**r, "id": r["core_id"]+"-q3-gold-presence", "protocol": candidate.PROTOCOL}
          for r in old.q3_rows(rows)]
    refs = candidate.read_references(args.refs_csv, rows)
    external, baseline_sha = candidate.cached_baseline(args.baseline_predictions, rows, refs, model)
    code_paths = [Path(__file__), Path(candidate.__file__), Path(refined.__file__), Path(old.__file__),
                  Path(agent.__file__), Path(agent.cons.__file__), Path(old.diagnostic.dev.__file__),
                  ROOT/"scripts/run_gold_v2_reference_ablation.py", ROOT/"src/insplad_benchmark/hf_adapters.py"]
    code = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in code_paths}
    targets = old.verified_images(args.sup_zip, [r["image"] for r in rows], args.out_dir/"_images/target")
    normals = old.verified_images(args.unsup_zip, list(refs.values()), args.out_dir/"_images/normal")
    manifest = [{"role": role, "member": member, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for role, paths in (("target", targets), ("reference", normals)) for member, path in sorted(paths.items())]
    reuse, reuse_sha = load_reuse(args.reuse_steps, args.reuse_manifest, manifest)
    signature = old.diagnostic.digest({"rows": rows, "q3": q3, "refs": refs, "config": config, "code": code,
        "baseline_sha256": baseline_sha, "reuse_sha256": reuse_sha,
        "model_spec": agent.base.MODEL_REGISTRY[model], "n_boot": args.n_boot})
    meta_path = args.out_dir/"run_metadata.json"
    previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if previous and previous.get("input_signature") != signature:
        raise ValueError("inputs/config/code changed: use a new folder; previous results preserved")
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
        "q1": "direct baseline fixed; zero Q1 inference", "q3_location_accuracy": None,
        "independence": "Q2 own inputs only; Q3 own inputs/answers only; contradictions evaluation only",
        "baseline_source": str(args.baseline_predictions) if args.baseline_predictions else None,
        "baseline_source_sha256": baseline_sha, "reuse_source": str(args.reuse_steps) if args.reuse_steps else None,
        "reuse_source_sha256": reuse_sha, "n_boot": args.n_boot,
        "gpu": previous.get("gpu"), "model_spec": agent.base.MODEL_REGISTRY[model], "software": {}, "complete": False}
    old.atomic_json(meta_path, meta)
    old.atomic_json(args.out_dir/"protocol_lock.json", {"config": config, "input_signature": signature,
        "prediction_identity": identity, "note": "pre-inference protocol, not a successful-result or selection marker"})
    old.atomic_json(args.out_dir/"run_status.json", {"status": "running", "complete": False, "policies": []})
    candidate.save_records(args.out_dir/"gold_qa_q12_snapshot.jsonl", rows)
    candidate.save_records(args.out_dir/"q3_auxiliary_partial_qa.jsonl", q3)
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

    def valid(rec, row, prompt, images=None):
        return bool(rec and not rec.get("error") and rec.get("prompt") == prompt
            and rec.get("predicted_choice") in row["options"]
            and agent.base.parse_choice(str(rec.get("output", "")), set(row["options"])) == rec["predicted_choice"]
            and (images is None or list(map(image_member, rec.get("images", []))) == list(map(image_member, images))))

    def call(row, prompt, step, cached=None):
        nonlocal load_failure
        key = row["id"]+"/"+step
        rec = existing.get(key)
        if rec and rec.get("identity") != identity:
            raise ValueError("saved prediction identity mismatch")
        images = [normals[refs[row["image"]]], targets[row["image"]]]
        if valid(rec, row, prompt):
            counts["reused_calls"] += 1
        elif valid(reuse.get(key), row, prompt, images):
            saved = reuse[key]
            rec = {**saved, "identity": identity, "images": list(map(str, images)), "source": "reused_candidate_run",
                   "reused_identity": saved.get("identity")}
            counts["reused_candidate_steps"] += 1
        else:
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
            candidate.save_records(path, [*used.values(), *(r for i, r in existing.items() if i not in used)])
            print(f"[{model}] {len(used)} steps saved", flush=True)
        return rec

    try:
        for row in rows:
            rec = call(row, old.direct_prompt(row), "baseline", external.get(row["id"]))
            first[row["id"]] = rec["predicted_choice"]
            baseline_deps.append(rec)
        if any(r["error"] or r["predicted_choice"] is None for r in baseline_deps):
            counts["baseline_blocked"] = 1
        else:
            finals = {policy: dict(first) for policy in POLICIES}
            for q2, r3 in zip((r for r in rows if r["question_type"] == agent.Q2), q3):
                # The three published gold policies, unchanged (same prompts and step names).
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
                # Matched control: same per-image call count as previous_agent, no definitions or staging.
                reask = call(q2, agent.build_prompt(q2, first[q2["id"]], "reask"), "q2_reask")
                finals["matched_reask"][q2["id"]] = reask["predicted_choice"] or first[q2["id"]]
                own, control = direct["predicted_choice"], []
                for i in range(1 + (position is not None)):
                    rec = call(r3, agent.build_prompt(r3, own, "reask"), f"q3_own_reask_{i+1}")
                    control.append(rec)
                    own = rec["predicted_choice"] or own
                locations["matched_reask"][r3["id"]] = own
                deps["matched_reask"].extend([reask, direct, *control])
    finally:
        # Even unexpected interruptions preserve the unprocessed cache tail.
        candidate.save_records(path, [*used.values(), *(r for i, r in existing.items() if i not in used)])
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
            "q1_fixed": q1_fixed, "q1_calls": 0, "tool_calls": len(deps[policy])})
    candidate.evaluate(rows, q3, first, finals, locations, eligible, deps, args.out_dir, model, args.n_boot,
                       policies=POLICIES, comparisons=COMPARISONS)
    agent.cons.write_csv(args.out_dir/"call_budget.csv", call_budget(rows, q3, baseline_deps, deps))
    complete = all(eligible.values())
    counts.update({"unique_calls": len(used), "row_errors": sum(bool(r["error"]) for r in used.values()),
                   "invalid_answers": sum(r["predicted_choice"] is None for r in used.values())})
    meta.update({**{key: counts[key] for key in ("called_this_run", "reused_calls", "reused_candidate_steps",
        "imported_baseline_calls", "baseline_calls_this_run", "tool_calls_this_run", "adapter_load_blocked_steps",
        "unique_calls", "row_errors", "invalid_answers")}, "complete": complete,
        "images_by_gt": dict(Counter(r["anomaly_gt"] for r in q3)), "q1_calls": 0,
        "gpu": agent.gpu_info() or previous.get("gpu")})
    for package in ("torch", "transformers", "numpy", "Pillow"):
        try:
            meta["software"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta["software"][package] = None
    old.atomic_json(meta_path, meta)
    old.atomic_json(args.out_dir/"run_status.json", {"status": "complete" if complete else "incomplete", "complete": complete,
        "policies": statuses, "selection_performed": False, "result_published": False})
    print(json.dumps({**counts, "complete": complete, "q1_calls": 0}, indent=2), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
