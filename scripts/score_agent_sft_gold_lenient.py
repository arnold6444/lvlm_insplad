"""Post-hoc: re-score a saved gold run of the tool-choice agent after forgiving format-only errors. No model call.

The gold protocol counts every output that is not the strict JSON action as invalid. The untrained model often
answers "B. Anomalous." instead, so its strict score mixes format and perception. This script reads the saved raw
outputs of one condition and accepts an answer when it is unambiguous:

  json     the strict JSON finish action (as in the run)
  letter   the whole output is one option letter, or a letter followed by exactly that option's text
  text     the whole output is exactly one option's text
  embedded exactly one finish JSON object inside other text

A letter followed by another option's text is a conflict and stays invalid, and so does prose that picks no option;
nothing is inferred from the meaning of a sentence. The rules were fixed from the shapes of the outputs before any
re-scored number was computed. The strict rows remain the reported result; the lenient rows are a reference.

    python scripts/score_agent_sft_gold_lenient.py --run-dir <unpacked>/gold_base \
        --loc-qa <unpacked>/gold_inputs/localization_mmad_qa_133.jsonl --train-qa <unpacked>/data/sft_train_qa.jsonl \
        --out-dir results/latest/agent_sft_gold_eval/gold_base_lenient
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_agent_sft_gold_eval as gold  # noqa: E402

pilot, sft = gold.pilot, gold.sft
PROTOCOL = "agent_sft_gold_lenient_rescore_v1"
RULES = ("json: the strict JSON finish action; letter: the whole output is one option letter, or a letter followed by "
         "exactly that option's text; text: the whole output is exactly one option's text; embedded: exactly one finish "
         "JSON object inside other text. A letter with another option's text, and prose that picks no option, stay invalid.")


def _norm(text):
    return re.sub(r"[\s.]+$", "", str(text).strip()).lower()


def lenient_choice(output, options):
    """(choice or None, rule). Accepts only an unambiguous answer; never reads the meaning of a sentence."""
    text = str(output).strip()
    try:
        action = pilot.parse_action(text, options)
        return (action["choice"], "json") if action.get("action") == "finish" else (None, "tool request")
    except Exception:  # noqa: BLE001 - any parse failure falls through to the lenient rules
        pass
    by_text = {_norm(v): k for k, v in options.items()}
    match = re.fullmatch(r"([A-Z])\s*[.):]?\s*(.*)", text, flags=re.S)
    if match and match.group(1) in options:
        letter, rest = match.group(1), match.group(2).strip()
        if rest == "" or _norm(rest) == _norm(options[letter]):
            return letter, "letter"
        if _norm(rest) in by_text:
            return None, "conflict"
    if _norm(text) in by_text:  # checked for every output: "Center region." also starts with an option letter
        return by_text[_norm(text)], "text"
    found = re.findall(r'\{\s*"action"\s*:\s*"finish"\s*,\s*"choice"\s*:\s*"([A-Z])"\s*\}', text)
    if len(found) == 1 and found[0] in options:
        return found[0], "embedded"
    return None, "no parsable answer"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="unpacked run folder with episodes/")
    parser.add_argument("--loc-qa", type=Path, required=True)
    parser.add_argument("--train-qa", type=Path, nargs="*", default=[], help="training lists, for the capture-date overlap")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--policy", default="direct_reference")
    args = parser.parse_args(argv)

    qa = read_jsonl(args.run_dir / "qa_snapshot.jsonl")
    rows = [r for r in qa if r["question_type"] in (gold.Q1, gold.Q2)]
    q3 = [r for r in qa if r["question_type"] == gold.Q3]
    gt, boxes = gold.load_location(args.loc_qa, rows)
    by_id = {r["id"]: r for r in qa}
    records = [a for a in read_jsonl(args.run_dir / "answers.jsonl") if a["policy"] == args.policy]
    if sorted(r["id"] for r in records) != sorted(by_id):
        raise ValueError("the saved answers do not cover the saved questions")
    per_image = read_csv(args.run_dir / "per_image.csv")
    primary = {r["core_id"] for r in per_image if r["scored_in_primary"] == "True"}

    relaxed, rule_of, rules = [], {}, Counter()
    for rec in records:
        row = by_id[rec["id"]]
        calls = read_jsonl(args.run_dir / "episodes" / args.policy / (pilot.digest(rec["id"]) + ".jsonl"))
        choice, rule = lenient_choice(calls[-1]["output"], row["options"])
        if rule == "json" and (choice != rec["predicted_choice"] or rec["invalid"]):
            raise ValueError(f"the saved output of {rec['id']} does not match its saved answer")
        rule_of[rec["id"]] = rule
        rules[(row["question_type"], rule)] += 1
        relaxed.append({**rec, "predicted_choice": choice if choice else rec["predicted_choice"],
                        "invalid": "" if choice else (rec["invalid"] or "no parsable answer")})

    published = {r["subset"]: r for r in read_csv(args.run_dir / "leaderboard.csv") if r["policy"] == args.policy}
    board = []
    for subset in gold.SUBSETS:
        cores = primary if subset == gold.PRIMARY else {r["core_id"] for r in rows}
        keep = lambda items: gold.keep(items, cores)  # noqa: E731
        strict = gold.score(keep(rows), keep(q3), keep(records), args.policy, gt, boxes)
        for key in ("q1_anomaly_f1", "q2_defect_macro_f1", "joint_accuracy", "q3_position_hit", "invalid_episodes"):
            if abs(float(published[subset][key]) - float(strict[key])) > 1e-6:
                raise ValueError(f"strict re-scoring does not reproduce the run's {subset} {key}")
        loose = gold.score(keep(rows), keep(q3), keep(relaxed), args.policy, gt, boxes)
        for parsing, result in (("strict", strict), ("lenient", loose)):
            board.append({"subset": subset, "parsing": parsing, **result,
                          "scope": "post hoc; the strict row is the reported result"})

    choices = {r["id"]: r["predicted_choice"] for r in relaxed}
    cells = gold.reported_cells(q3, choices)
    first = {r["core_id"]: r for r in rows if r["question_type"] == gold.Q1}
    second = {r["core_id"]: r for r in rows if r["question_type"] == gold.Q2}
    third = {r["core_id"]: r for r in q3}
    images = [{"core_id": core, "asset_category": first[core]["asset_category"], "anomaly_gt": first[core]["anomaly_gt"],
               "defect_gt": first[core]["defect_gt"], "scored_in_primary": core in primary,
               "q1_correct": choices[first[core]["id"]] == first[core]["answer"], "q1_rule": rule_of[first[core]["id"]],
               "q2_correct": choices[second[core]["id"]] == second[core]["answer"], "q2_rule": rule_of[second[core]["id"]],
               "q3_cell": cells[core] or "", "q3_rule": rule_of[third[core]["id"]]} for core in sorted(first)]
    by_defect = []
    for subset in gold.SUBSETS:
        chosen = [r for r in images if subset == "all" or r["scored_in_primary"]]
        for defect in sorted({r["defect_gt"] for r in chosen}):
            group = [r for r in chosen if r["defect_gt"] == defect]
            by_defect.append({"subset": subset, "parsing": "lenient", "policy": args.policy, "defect_gt": defect,
                              "images": len(group), "q2_correct": sum(r["q2_correct"] for r in group)})

    overlap = None
    if args.train_qa:
        train = {r["image"] for path in args.train_qa for r in read_jsonl(path)}
        dates = {sft.photo_key(p)[0] for p in train}
        gold_dates = Counter(sft.photo_key(first[c]["image"])[0] for c in primary)
        overlap = {"training_photos": len(train), "training_capture_dates": len(dates),
                   "primary_gold_capture_dates": len(gold_dates),
                   "primary_gold_photos_on_a_training_date": sum(n for d, n in gold_dates.items() if d in dates),
                   "primary_gold_photos": len(primary)}

    source = json.loads((args.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "leaderboard.csv", board)
    write_csv(args.out_dir / "per_image.csv", images)
    write_csv(args.out_dir / "q2_by_defect.csv", by_defect)
    write_csv(args.out_dir / "rule_counts.csv", [{"question_type": q, "rule": r, "outputs": n} for (q, r), n in sorted(rules.items())])
    meta = {"protocol": PROTOCOL, "policy": args.policy, "post_hoc_rules": RULES,
            "source_run": {k: source.get(k) for k in ("protocol", "run_id", "model", "benchmark_commit", "lora_adapter", "gpu")},
            "strict_reproduces_the_run": True, "capture_date_overlap": overlap,
            "scope": "post hoc reference computed after the run from the saved raw outputs; no model call"}
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "run_status.json").write_text(json.dumps({"complete": True, "status": "rescored", "outputs": len(records)}, indent=2) + "\n",
                                                  encoding="utf-8")
    for row in board:
        print(f"{row['subset']:14s} {row['parsing']:7s} Q1 F1 {row['q1_anomaly_f1']:.3f} Q2 F1 {row['q2_defect_macro_f1']:.3f} "
              f"joint {row['joint_accuracy']:.3f} position {row['q3_position_hit']} invalid {row['invalid_episodes']}")
    print("rules:", dict(rules))
    print("capture-date overlap:", overlap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
