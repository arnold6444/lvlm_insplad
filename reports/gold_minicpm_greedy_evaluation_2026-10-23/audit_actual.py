"""CPU audit of the actual MiniCPM greedy gold run (no model loading, no inference).

Usage: python -I audit_actual.py <code_dir at main@49059c6> <run raw dir> <matched-call raw dir> <out json>
Rebuilds every prompt and image list from the run's QA snapshot and the committed reference manifest, re-parses all
outputs, recomputes the leaderboard, comparison and paired bootstrap rows, derives the accuracy columns of the
paper's baseline table, and measures how often greedy answers differ from the published sampled image-only answers.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
code, raw, matched, out = map(Path, sys.argv[1:5])
sys.path[:0] = [str(code/"scripts"), str(code/"src")]
import run_gold_v2_minicpm_greedy as r

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))

report = {}
rows = r.base.read_jsonl(raw/"gold_qa_snapshot.jsonl")
meta = json.loads((raw/"run_metadata.json").read_text(encoding="utf-8"))
report["questions"] = dict(Counter(x["question_type"] for x in rows))
manifest = r.load_manifest(r.REFERENCE_MANIFEST, rows)
report["reference_manifest_sha256_matches_committed"] = (
    hashlib.sha256(r.REFERENCE_MANIFEST.read_bytes()).hexdigest() == meta["reference_manifest_sha256"])
# Q1/Q2 rows equal the matched-call run's frozen snapshot; target image bytes equal across runs.
q12 = [x for x in rows if x["question_type"] != "defect_localization"]
prev = {x["id"]: x for x in r.base.read_jsonl(matched/"gold_qa_q12_snapshot.jsonl")}
report["q12_rows_equal_matched_run"] = all(prev.get(x["id"]) == x for x in q12) and len(prev) == len(q12)
img = read_csv(raw/"image_manifest.csv")
report["images_by_role"] = dict(Counter(x["role"] for x in img))
prev_t = {x["member"]: x["sha256"] for x in read_csv(matched/"image_manifest.csv") if x["role"] == "target"}
report["target_sha_equal_matched_run"] = all(prev_t.get(x["member"]) == x["sha256"] for x in img if x["role"] == "target")

steps = {s["id"]: s for s in r.base.read_jsonl(raw/"predictions"/r.MODEL/"steps.jsonl")}
roles = {"target": lambda row, m: "target/" + row["image"], "normal": lambda row, m: "normal/" + m["normal_reference_image"],
         "positive": lambda row, m: "positive/" + m["positive_reference_image"]}
order = {"image_only": ["target"], "normal_reference": ["normal", "target"], "positive_reference": ["positive", "target"],
         "normal_positive_reference": ["normal", "positive", "target"]}
mismatch, choices = [], {c: {} for c in r.CONDITIONS}
for c in r.CONDITIONS:
    for row in rows:
        s = steps[f"{c}/{row['id']}"]
        m = manifest[str(row["core_id"])]
        want_images = [roles[k](row, m) for k in order[c]]
        letters = {str(k).upper() for k in row["options"]}
        if (s["prompt"] != r.prompt_for(row, c) or s["images"] != want_images or s["error"]
                or r.base.parse_choice(s["output"], letters) != s["predicted_choice"]):
            mismatch.append(s["id"])
        choices[c][row["id"]] = s["predicted_choice"]
report["steps"], report["prompt_image_parse_mismatch"] = len(steps), len(mismatch)

lb = {x["condition"]: x for x in read_csv(raw/"leaderboard.csv")}
recomputed = {c: r.metrics(rows, choices[c]) for c in r.CONDITIONS}
keys = ("q1_anomaly_f1", "q2_defect_macro_f1", "joint_accuracy", "consistency_error", "false_defect_rate", "q3_accuracy")
report["leaderboard_max_abs_diff"] = max(abs(float(lb[c][k]) - recomputed[c][k]) for c in r.CONDITIONS for k in keys)
sig = r.cons.significance_rows(r.cons.image_pairs(rows), r.cons.q3_pairs(rows),
                               {(r.MODEL, c): choices[c] for c in r.CONDITIONS}, [r.MODEL], r.COMPARISONS, 2000)
saved = {(x["baseline"], x["condition"], x["metric"]): x for x in read_csv(raw/"significance.csv")}
report["significance_rows"] = len(sig)
report["significance_max_abs_diff"] = max(abs(float(s[f]) - float(saved[(s["baseline"], s["condition"], s["metric"])][f]))
                                          for s in sig for f in ("delta", "ci_low", "ci_high"))
sampled = r.sampled_rows(r.SAMPLED)
report["comparison"] = {c: {k: [round(recomputed[c][k], 3), round(float(sampled[c][k]), 3)] for k in keys} for c in r.CONDITIONS}

# Accuracy columns of the baseline table (image-only, all 534 questions are valid letters).
by_type = {}
for qt in ("anomaly_detection", "defect_classification", "defect_localization"):
    sub = [x for x in rows if x["question_type"] == qt]
    by_type[qt] = round(sum(choices["image_only"][x["id"]] == x["answer"] for x in sub) / len(sub), 6)
report["image_only_accuracy"] = by_type
report["image_only_valid_rate"] = sum(v is not None for v in choices["image_only"].values()) / len(rows)

# How often greedy differs from the published sampled image-only answers, per question.
pub = {x["id"]: (x["predicted_choice"] or None) for x in read_csv(code/"results/latest/gold_v2_baseline/prediction_review.csv")
       if x["model_key"] == r.MODEL}
changed = Counter(x["question_type"] for x in rows if pub.get(x["id"]) != choices["image_only"][x["id"]])
report["image_only_answers_changed_vs_sampled"] = {"total": sum(changed.values()), "of": len(rows), **dict(changed)}
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
