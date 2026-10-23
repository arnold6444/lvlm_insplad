"""CPU-only: paired bootstrap CIs from saved gold answers, using the repository's own functions."""
import csv
import json
import sys
from pathlib import Path

code, raw = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(code/"scripts"), str(code/"src")]
import run_gold_v2_q23_candidate as c

agent, old = c.agent, c.old
rows = c.gold_rows(agent.read_jsonl(raw/"gold_qa_q12_snapshot.jsonl"))
q3 = agent.read_jsonl(raw/"q3_auxiliary_partial_qa.jsonl")
answers = list(csv.DictReader((raw/"aggregation_v2/answers.csv").open(encoding="utf-8")))
model = "internvl3_5_8b"
finals, locs = {}, {}
for a in answers:
    target = locs if a["question_type"] == agent.Q3 else finals
    target.setdefault(a["policy"], {})[a["id"]] = a["predicted_choice"] or None
pairs = agent.cons.image_pairs(rows)
found = {(model, p): finals[p] for p in finals}
loc_found = {(model, p): locs[p] for p in locs}
joint = {(model, p): {x["core_id"]: finals[p][x["id"]] == x["answer"] and finals[p][y["id"]] == y["answer"] for x, y in pairs}
         for p in finals}
comparisons = [("baseline", "candidate"), ("baseline", "previous_agent")]
sig = agent.cons.significance_rows(pairs, [], found, [model], comparisons, 2000)
sig += old.q3_bootstrap(q3, loc_found, joint, [model], 2000, comparisons, scope="presence")
keep = {"q2_defect_macro_f1", "joint_accuracy", "q3_presence_balanced_accuracy", "q3_normal_false_region_rate",
        "q3_anomaly_false_none_rate"}
for r in sig:
    if r["metric"] in keep:
        print(r["baseline"], r["condition"], r["metric"], round(float(r["delta"]), 3),
              round(float(r["ci_low"]), 3), round(float(r["ci_high"]), 3))
