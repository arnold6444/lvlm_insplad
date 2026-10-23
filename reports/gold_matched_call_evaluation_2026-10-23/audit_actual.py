"""CPU audit of the actual matched-call gold run (no model loading, no inference).

Usage: python -I audit_actual.py <code_dir at main@36458d5> <this run's raw dir> <earlier candidate run raw dir> <out json>
Checks: reused steps equal the earlier audited gold run; every new prompt equals the locked design; finals are rebuilt
from raw outputs; leaderboard, per-defect and paired-bootstrap rows are recomputed with the repository functions.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

code, raw, prev, out = map(Path, sys.argv[1:5])
sys.path[:0] = [str(code/"scripts"), str(code/"src")]
import run_gold_v2_matched_call as runner

c, old, agent, refined = runner.candidate, runner.old, runner.agent, runner.refined
report = {}

def read_csv(path):
    with path.open(encoding="utf-8") as h:
        return list(csv.DictReader(h))

steps = {r["id"]: r for r in agent.read_jsonl(raw/"predictions/internvl3_5_8b/steps.jsonl")}
prev_steps = {r["id"]: r for r in agent.read_jsonl(prev/"predictions/internvl3_5_8b/steps.jsonl")}
meta = json.loads((raw/"run_metadata.json").read_text(encoding="utf-8"))

# 1. Reuse source is exactly the earlier audited run (same digest the runner recorded).
digest = hashlib.sha256((prev/"predictions/internvl3_5_8b/steps.jsonl").read_bytes() + (prev/"image_manifest.csv").read_bytes()).hexdigest()
report["reuse_source_digest_matches_earlier_zip"] = digest == meta["reuse_source_sha256"]
report["image_manifest_identical"] = read_csv(raw/"image_manifest.csv") == read_csv(prev/"image_manifest.csv")
reused = [k for k, r in steps.items() if r.get("source") == "reused_candidate_run"]
new = [k for k, r in steps.items() if r.get("source") == "inference"]
report["reused_steps"], report["new_steps"] = len(reused), len(new)
report["reused_equal_to_earlier"] = all(
    all(steps[k][f] == prev_steps[k][f] for f in ("prompt", "output", "predicted_choice", "question_id", "step"))
    for k in reused) and set(reused) == set(prev_steps)
report["new_step_counts"] = dict(Counter(steps[k]["step"].rsplit("_", 1)[0] if steps[k]["step"].startswith("q3_own")
                                         else steps[k]["step"] for k in new))
report["errors"] = sum(bool(r["error"]) for r in steps.values())
report["invalid"] = sum(r["predicted_choice"] is None for r in steps.values())

# 2. Rebuild every prompt and every final answer from raw outputs with the locked routing.
rows = c.gold_rows(agent.read_jsonl(raw/"gold_qa_q12_snapshot.jsonl"))
q3 = [{**r, "id": r["core_id"]+"-q3-gold-presence", "protocol": c.PROTOCOL} for r in old.q3_rows(rows)]
assert [r["options"] for r in q3] == [r["options"] for r in agent.read_jsonl(raw/"q3_auxiliary_partial_qa.jsonl")]
mismatch = []

def get(row, prompt, step):
    rec = steps[row["id"]+"/"+step]
    if rec["prompt"] != prompt or agent.base.parse_choice(rec["output"], set(row["options"])) != rec["predicted_choice"]:
        mismatch.append(rec["id"])
    return rec

policies = runner.POLICIES
first = {r["id"]: get(r, old.direct_prompt(r), "baseline")["predicted_choice"] for r in rows}
finals = {p: dict(first) for p in policies}
locs = {p: {} for p in policies}
calls = {p: Counter() for p in policies}
for q2, r3 in zip((r for r in rows if r["question_type"] == agent.Q2), q3):
    cues = get(q2, refined.definition_prompt(q2, True), "q2_cues")
    define = get(q2, agent.build_prompt(q2, first[q2["id"]], "define"), "q2_define")
    direct = get(r3, old.direct_prompt(r3), "q3_direct")
    task, prompt = old.presence_task(r3)
    decision = task["options"].get(get(task, prompt, "q3_presence")["predicted_choice"])
    position = get(r3, old.position_prompt(r3), "q3_position") if decision == "A visible localized defect region." else None
    staged = (r3["none_choice"] if decision == "No visible localized defect region." else
              position["predicted_choice"] if position and position["predicted_choice"] else direct["predicted_choice"])
    reask = get(q2, agent.build_prompt(q2, first[q2["id"]], "reask"), "q2_reask")
    own, k = direct["predicted_choice"], 1 + (position is not None)
    for i in range(k):
        own = get(r3, agent.build_prompt(r3, own, "reask"), f"q3_own_reask_{i+1}")["predicted_choice"] or own
    finals["previous_agent"][q2["id"]], finals["candidate"][q2["id"]] = define["predicted_choice"], cues["predicted_choice"]
    finals["matched_reask"][q2["id"]] = reask["predicted_choice"]
    locs["baseline"][r3["id"]] = direct["predicted_choice"]
    for p in ("previous_agent", "candidate"):
        locs[p][r3["id"]] = staged
    locs["matched_reask"][r3["id"]] = own
    calls["previous_agent"]["q3"] += 1 + 1 + (position is not None)
    calls["matched_reask"]["q3"] += 1 + k
report["prompt_or_parse_mismatch"] = len(mismatch)
report["q3_calls_previous_agent_vs_matched"] = [calls["previous_agent"]["q3"], calls["matched_reask"]["q3"]]
answers = read_csv(raw/"answers.csv")
saved = {(a["policy"], a["id"]): a["predicted_choice"] or None for a in answers}
rebuilt = {(p, i): v for p in policies for i, v in {**finals[p], **locs[p]}.items()}
report["answers_csv_equals_rebuilt"] = saved == rebuilt
report["q1_identical_all_policies"] = all(finals[p][r["id"]] == first[r["id"]] for p in policies
                                          for r in rows if r["question_type"] == agent.Q1)
# Q2 answers that changed between the direct answer and the matched re-ask, by ground truth.
changes = Counter()
for r in rows:
    if r["question_type"] == agent.Q2 and finals["matched_reask"][r["id"]] != first[r["id"]]:
        changes[(r["defect_gt"], "fixed" if finals["matched_reask"][r["id"]] == r["answer"] else
                 "broken" if first[r["id"]] == r["answer"] else "wrong_to_wrong")] += 1
report["matched_q2_changes_by_class"] = {f"{a}/{b}": n for (a, b), n in sorted(changes.items())}

# 3. Recompute metrics and paired bootstrap independently of the saved reports.
pairs = agent.cons.image_pairs(rows)
found = {("internvl3_5_8b", p): finals[p] for p in policies}
loc_found = {("internvl3_5_8b", p): locs[p] for p in policies}
joint = {k: {a["core_id"]: v[a["id"]] == a["answer"] and v[b["id"]] == b["answer"] for a, b in pairs} for k, v in found.items()}
recomputed = {p: {**c.q12_metrics(rows, finals[p]), **old.presence_metrics(q3, locs[p], joint[("internvl3_5_8b", p)])}
              for p in policies}
lb = {r["policy"]: r for r in read_csv(raw/"leaderboard.csv")}
keys = ("q2_defect_macro_f1", "joint_accuracy", "q3_presence_balanced_accuracy", "q3_normal_false_region_rate",
        "q3_anomaly_false_none_rate", "consistency_error", "false_defect_rate", "q1_anomaly_f1")
report["leaderboard_max_abs_diff"] = max(abs(float(lb[p][k]) - float(recomputed[p][k])) for p in policies for k in keys)
report["leaderboard"] = {p: {k: round(float(recomputed[p][k]), 6) for k in keys} for p in policies}
prev_lb = {r["policy"]: r for r in read_csv(prev/"aggregation_v2/leaderboard.csv")}
report["published_policies_unchanged"] = all(abs(float(prev_lb[p][k]) - float(recomputed[p][k])) < 1e-9
                                            for p in ("baseline", "previous_agent", "candidate") for k in keys)
sig = agent.cons.significance_rows(pairs, [], found, ["internvl3_5_8b"], runner.COMPARISONS, 2000)
sig += old.q3_bootstrap(q3, loc_found, joint, ["internvl3_5_8b"], 2000, runner.COMPARISONS, scope="presence")
saved_sig = {(r["baseline"], r["condition"], r["metric"]): r for r in read_csv(raw/"significance.csv")}
diffs = [max(abs(float(s[f]) - float(saved_sig[(s["baseline"], s["condition"], s["metric"])][f])) for f in ("delta", "ci_low", "ci_high"))
         for s in sig if (s["baseline"], s["condition"], s["metric"]) in saved_sig]
report["significance_rows_compared"], report["significance_max_abs_diff"] = len(diffs), max(diffs)
report["primary"] = {f'{s["condition"]}-{s["baseline"]}:{s["metric"]}': [round(float(s[f]), 4) for f in ("delta", "ci_low", "ci_high")]
                     for s in sig if s["metric"] in keys[:5] and (s["baseline"], s["condition"]) in
                     (("matched_reask", "previous_agent"), ("baseline", "matched_reask"))}
defects = Counter()
for r in rows:
    if r["question_type"] == agent.Q2:
        for p in policies:
            defects[(p, r["defect_gt"])] += finals[p][r["id"]] == r["answer"]
report["q2_correct_by_class"] = {f"{p}/{d}": n for (p, d), n in sorted(defects.items())}
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
