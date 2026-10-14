#!/usr/bin/env python3
"""CPU-only reporting repair for saved gold candidate steps, without inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gold_v2_q23_candidate as runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    source = args.run_dir
    if args.n_boot <= 0:
        raise ValueError("positive bootstrap count required")
    meta = json.loads((source/"run_metadata.json").read_text(encoding="utf-8"))
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    if meta.get("protocol") != runner.PROTOCOL or meta.get("config") != config:
        raise ValueError("saved gold protocol/config mismatch")
    model = config["models"][0]
    rows = runner.gold_rows(runner.agent.read_jsonl(source/"gold_qa_q12_snapshot.jsonl"))
    q3 = runner.agent.read_jsonl(source/"q3_auxiliary_partial_qa.jsonl")
    expected = [{**r, "id": r["core_id"]+"-q3-gold-presence", "protocol": runner.PROTOCOL}
                for r in runner.old.q3_rows(rows)]
    if q3 != expected or len(q3) != meta["images"]:
        raise ValueError("saved QA coverage/schema mismatch")
    with (source/"image_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if runner.old.diagnostic.digest([meta["input_signature"], manifest]) != meta["prediction_identity"]:
        raise ValueError("saved input manifest identity mismatch")
    path = source/"predictions"/model/"steps.jsonl"
    records = runner.agent.read_jsonl(path)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate saved prediction IDs")
    if any(r.get("identity") != meta["prediction_identity"] or r.get("model_key") != model for r in records):
        raise ValueError("saved prediction identity/model mismatch")
    by_id = {r["id"]: r for r in records}
    first, baseline = {}, []
    finals = {p: {} for p in runner.POLICIES}
    locations = {p: {} for p in runner.POLICIES}
    deps = {p: [] for p in runner.POLICIES}

    def step(row, prompt, name):
        key = row["id"]+"/"+name
        rec = by_id.get(key)
        if rec is None:
            return {"id": key, "predicted_choice": None, "error": "missing saved step", "latency_s": 0}
        if rec.get("question_id") != row["id"] or rec.get("question_type") != row["question_type"] or rec.get("prompt") != prompt:
            raise ValueError("saved prompt/question mismatch")
        parsed = runner.agent.base.parse_choice(str(rec.get("output", "")), set(row["options"]))
        if parsed != rec.get("predicted_choice"):
            raise ValueError("saved choice differs from raw response")
        return rec

    for row in rows:
        rec = step(row, runner.old.direct_prompt(row), "baseline")
        baseline.append(rec)
        first[row["id"]] = rec["predicted_choice"]
    if not any(r["error"] or r["predicted_choice"] is None for r in baseline):
        finals = {p: dict(first) for p in runner.POLICIES}
        for q2, r3 in zip((r for r in rows if r["question_type"] == runner.agent.Q2), q3):
            revised = step(q2, runner.refined.definition_prompt(q2, True), "q2_cues")
            defined = step(q2, runner.agent.build_prompt(q2, first[q2["id"]], "define"), "q2_define")
            direct = step(r3, runner.old.direct_prompt(r3), "q3_direct")
            task, prompt = runner.old.presence_task(r3)
            presence = step(task, prompt, "q3_presence")
            decision = task["options"].get(presence["predicted_choice"])
            position = step(r3, runner.old.position_prompt(r3), "q3_position") if decision == "A visible localized defect region." else None
            choice = (r3["none_choice"] if decision == "No visible localized defect region." else
                      position["predicted_choice"] if position and position["predicted_choice"] else direct["predicted_choice"])
            locations["baseline"][r3["id"]] = direct["predicted_choice"]
            deps["baseline"].append(direct)
            for policy, q2rec in (("previous_agent", defined), ("candidate", revised)):
                finals[policy][q2["id"]] = q2rec["predicted_choice"] or first[q2["id"]]
                locations[policy][r3["id"]] = choice
                deps[policy].extend([q2rec, direct, presence]+([position] if position else []))
    eligible, statuses = {}, []
    for policy in runner.POLICIES:
        required = baseline+deps[policy]
        errors = sum(bool(r["error"]) for r in required)
        invalids = sum(r["predicted_choice"] is None for r in required)
        coverage = len(finals[policy]) == len(rows) and len(locations[policy]) == len(q3)
        fixed = all(finals[policy].get(r["id"]) == first[r["id"]] for r in rows if r["question_type"] == runner.agent.Q1)
        eligible[policy] = bool(coverage and fixed and not errors and not invalids)
        statuses.append({"model_key": model, "policy": policy, "eligible_for_comparison": eligible[policy],
                         "row_errors": errors, "invalid_answers": invalids, "full_coverage": coverage, "q1_fixed": fixed})
    # Never overwrite source reports, metadata, responses or inference identities.
    out = source/"aggregation_v2"
    out.mkdir(exist_ok=True)
    runner.old.atomic_json(out/"run_status.json", {"status": "running", "complete": False, "policies": []})
    runner.evaluate(rows, q3, first, finals, locations, eligible, deps, out, model, args.n_boot)
    complete = all(eligible.values())
    runner.old.atomic_json(out/"run_status.json", {"status": "complete" if complete else "incomplete", "complete": complete,
        "policies": statuses, "inference_calls": 0, "selection_performed": False})
    runner.old.atomic_json(out/"aggregation_metadata.json", {"source_run_dir": str(source),
        "source_benchmark_commit": meta["benchmark_commit"], "source_prediction_identity": meta["prediction_identity"],
        "source_predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "aggregation_commit": runner.agent.git_commit(), "aggregation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_boot": args.n_boot, "images": len(q3), "inference_calls": 0, "complete": complete,
        "q3_location_accuracy": None, "scope": f"gold{len(q3)} presence only, no localization claim"})
    print(json.dumps({"report_dir": str(out), "complete": complete, "inference_calls": 0}, indent=2), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
