"""Differences between models in the fine-tuning section, from the saved per-image tables. No model call.

    python compute_sft_gold_differences.py [results/latest/agent_sft_gold_eval]

The gold protocol (docs/agent_sft_gold_eval.md) fixed before the run that differences between models are computed
afterwards from per_image.csv. The interval is the image-level paired bootstrap of
scripts/score_gold_v2_q3_position.py (2,000 resamples, seed 42). Writes model_differences.csv next to the inputs.
"""
import csv
import random
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "results/latest/agent_sft_gold_eval"
RUNS = {"lora2": "gold_lora_sft_v2_human", "lora1": "gold_lora_sft_v1", "base": "gold_base"}
SINGLE, TOOLS = "direct_reference", "adaptive_tools"
TRUE = lambda value: str(value) == "True"


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cells(text):
    return {int(x) for x in re.findall(r"\d+", text or "")}


def bootstrap(a, b, n=2000, seed=42):
    diff = [int(x) - int(y) for x, y in zip(a, b)]
    rng, stats = random.Random(seed), []
    for _ in range(n):
        stats.append(sum(diff[rng.randrange(len(diff))] for _ in diff) / len(diff))
    stats.sort()
    return sum(diff) / len(diff), stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def main():
    runs = {key: {r["core_id"]: r for r in table(ROOT / name / "per_image.csv")} for key, name in RUNS.items()}
    small = {r["core_id"]: r for r in table(ROOT / "classifier_dinov2_linear" / "per_image.csv") if r["eval_set"] == "gold"}
    every = sorted(runs["lora2"])
    assert all(sorted(run) == every for run in runs.values()) and sorted(small) == every, "the runs score different photos"

    def both(key, policy):
        return lambda i: TRUE(runs[key][i][f"{policy}_q1_correct"]) and TRUE(runs[key][i][f"{policy}_q2_correct"])

    def one(key, policy, question):
        return lambda i: TRUE(runs[key][i][f"{policy}_{question}_correct"])

    def hit(key, policy):
        def value(i):
            cell = runs[key][i][f"{policy}_q3_cell"]
            return bool(cell) and int(cell) in cells(runs[key][i]["gt_cells"])
        return value

    small_both = lambda i: TRUE(small[i]["q1_correct"]) and TRUE(small[i]["q2_correct"])
    # The untrained model after forgiving format-only errors (scripts/score_agent_sft_gold_lenient.py).
    loose = {r["core_id"]: r for r in table(ROOT / "gold_base_lenient" / "per_image.csv")}
    assert sorted(loose) == every, "the lenient re-score covers different photos"
    loose_both = lambda i: TRUE(loose[i]["q1_correct"]) and TRUE(loose[i]["q2_correct"])
    plan = [  # metric, new, baseline, new value, baseline value, located photos only
        ("q1_q2_both_correct", "lora2 single call", "base single call, lenient", both("lora2", SINGLE), loose_both, False),
        ("q1_q2_both_correct", "lora2 single call", "dinov2_linear", both("lora2", SINGLE), small_both, False),
        ("q1_correct", "lora2 single call", "dinov2_linear", one("lora2", SINGLE, "q1"), lambda i: TRUE(small[i]["q1_correct"]), False),
        ("q2_correct", "lora2 single call", "dinov2_linear", one("lora2", SINGLE, "q2"), lambda i: TRUE(small[i]["q2_correct"]), False),
        ("q1_q2_both_correct", "lora1 single call", "dinov2_linear", both("lora1", SINGLE), small_both, False),
        ("q1_q2_both_correct", "lora2 single call", "base single call", both("lora2", SINGLE), both("base", SINGLE), False),
        ("q1_q2_both_correct", "lora2 single call", "lora1 single call", both("lora2", SINGLE), both("lora1", SINGLE), False),
        ("q1_q2_both_correct", "lora2 tool choice", "lora1 tool choice", both("lora2", TOOLS), both("lora1", TOOLS), False),
        ("q3_position_hit", "lora2 single call", "lora1 single call", hit("lora2", SINGLE), hit("lora1", SINGLE), True),
        ("q3_position_hit", "lora2 tool choice", "lora1 tool choice", hit("lora2", TOOLS), hit("lora1", TOOLS), True),
        ("q3_position_hit", "lora2 tool choice", "lora1 single call", hit("lora2", TOOLS), hit("lora1", SINGLE), True),
    ]
    rows = []
    for subset in ("frame_disjoint", "all"):
        ids = [i for i in every if subset == "all" or TRUE(runs["lora2"][i]["scored_in_primary"])]
        located = [i for i in ids if cells(runs["lora2"][i]["gt_cells"])]
        for metric, new, base, f_new, f_base, on_located in plan:
            use = located if on_located else ids
            a, b = [f_new(i) for i in use], [f_base(i) for i in use]
            delta, low, high = bootstrap(a, b)
            rows.append(dict(subset=subset, metric=metric, condition=new, baseline=base, images=len(use), new_correct=sum(a),
                             base_correct=sum(b), delta=round(delta, 6), ci_low=round(low, 6), ci_high=round(high, 6),
                             significant=bool(low > 0 or high < 0), resamples=2000, seed=42,
                             scope="post hoc from per_image.csv; image-level paired bootstrap"))
    with (ROOT / "model_differences.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['subset']:14s} {r['metric']:19s} {r['condition']:18s} - {r['baseline']:18s} {r['new_correct']:3d} vs {r['base_correct']:3d} / "
              f"{r['images']} d={r['delta']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] {'*' if r['significant'] else ''}")


if __name__ == "__main__":
    main()
