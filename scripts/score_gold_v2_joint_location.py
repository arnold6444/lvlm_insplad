"""Post-hoc Q1/Q2/location joint scoring of saved predictions; no inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("baseline", "matched_reask", "previous_agent", "candidate")
Q1, Q2, Q3 = "anomaly_detection", "defect_classification", "defect_localization"
NONE = "No localized defect region is present."
CELL_TEXTS = (
    "Upper-left region.", "Upper-center region.", "Upper-right region.",
    "Middle-left region.", "Center region.", "Middle-right region.",
    "Lower-left region.", "Lower-center region.", "Lower-right region.",
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    paths = {
        "answers": ROOT / "results/latest/gold_v2_matched_call/answers.csv",
        "labels": ROOT / "results/latest/gold_v2_final/manual_review_final.csv",
        "positions": ROOT / "results/latest/gold_v2_q3_position/per_image.csv",
        "existing_metrics": ROOT / "results/latest/gold_v2_matched_call/leaderboard.csv",
        "existing_positions": ROOT / "results/latest/gold_v2_q3_position/leaderboard.csv",
    }
    labels = {r["core_id"]: r for r in read_csv(paths["labels"])}
    positions = {r["core_id"]: r for r in read_csv(paths["positions"])}
    require(len(labels) == 246 and len(positions) == 133, "Unexpected cohort size")
    require(set(positions) <= set(labels), "Unknown position IDs")
    normal = {i for i, r in labels.items() if r["anomaly_gt"] == "normal"}
    require(len(normal) == 107 and not normal.intersection(positions), "Unexpected normal cohort")
    ids = sorted(normal | set(positions))
    excluded = sorted(set(labels) - set(ids))
    require(len(ids) == 240 and len(excluded) == 6, "Unexpected scoring scope")

    answers = {p: {} for p in POLICIES}
    for row in read_csv(paths["answers"]):
        p, i, q = row["policy"], row["core_id"], row["question_type"]
        require(p in answers and i in labels and q in (Q1, Q2, Q3), "Unknown answer key")
        require(row["model_key"] == "internvl3_5_8b", "Mixed models")
        require(row["eligible_for_comparison"] == "True", "Ineligible answer")
        require((i, q) not in answers[p], "Duplicate answer")
        require(bool(row["predicted_choice"]), "Missing prediction")
        answers[p][i, q] = row
    expected_keys = {(i, q) for i in labels for q in (Q1, Q2, Q3)}
    for p in POLICIES:
        require(set(answers[p]) == expected_keys, "Incomplete answer coverage: " + p)

    per_image, board, all_hits = [], [], {}
    existing = {r["policy"]: r for r in read_csv(paths["existing_metrics"])}
    existing_pos = {r["policy"]: r for r in read_csv(paths["existing_positions"])}
    for p in POLICIES:
        hits, joint12, joint_presence, location_hits = {}, 0, 0, 0
        for i in sorted(labels):
            r1, r2, r3 = (answers[p][i, q] for q in (Q1, Q2, Q3))
            for q, row in ((Q1, r1), (Q2, r2)):
                require(bool(row["answer"]), "Missing Q1/Q2 GT")
                require(row["answer"] == answers["baseline"][i, q]["answer"], "GT differs by policy")
            c1 = r1["predicted_choice"] == r1["answer"]
            c2 = r2["predicted_choice"] == r2["answer"]
            require(r3["predicted_text"] in (*CELL_TEXTS, NONE), "Unknown location output")
            is_none = r3["predicted_text"] == NONE
            require(bool(r3["answer"]) == (i in normal), "Q3 normal-label mismatch")
            joint12 += int(c1 and c2)
            joint_presence += int(c1 and c2 and (is_none == (i in normal)))
            if i not in ids:
                continue
            if i in normal:
                c3 = is_none
            else:
                require(positions[i]["defect_gt"] == labels[i]["defect_gt"], "Defect GT mismatch")
                cell = None if is_none else CELL_TEXTS.index(r3["predicted_text"]) + 1
                require(positions[i][p] == ("" if cell is None else str(cell)), "Position-file mismatch")
                c3 = cell in {int(c) for c in positions[i]["gt_cells"].split()}
                location_hits += int(c3)
            hits[i] = int(c1 and c2 and c3)
            per_image.append(dict(policy=p, core_id=i, defect_gt=labels[i]["defect_gt"],
                                  q1_correct=int(c1), q2_correct=int(c2),
                                  q3_location_or_none_correct=int(c3), all_correct=hits[i]))
        require(abs(joint12 / 246 - float(existing[p]["joint_accuracy"])) < 1e-6, "Q1/Q2 metric mismatch")
        require(abs(joint_presence / 246 - float(existing[p]["q1_q2_q3_presence_joint"])) < 1e-6,
                "Presence-joint metric mismatch")
        require(location_hits == int(existing_pos[p]["position_hit"]), "Position total mismatch")
        all_hits[p] = hits
        for cohort, members in (("normal_plus_locatable_defect", ids),
                                ("normal_only", sorted(normal)), ("locatable_defect_only", sorted(positions))):
            count = sum(hits[i] for i in members)
            board.append(dict(policy=p, cohort=cohort, images=len(members), all_correct=count,
                              joint_accuracy=count / len(members)))

    comparisons = []
    for a, b in (("matched_reask", "baseline"), ("previous_agent", "baseline"),
                 ("candidate", "baseline"), ("candidate", "previous_agent")):
        differences = [all_hits[a][i] - all_hits[b][i] for i in ids]
        rng = random.Random(42)
        samples = sorted(sum(rng.choices(differences, k=len(ids))) / len(ids) for _ in range(2000))
        comparisons.append(dict(comparison=a + " - " + b, images=len(ids),
                                additional_correct=sum(differences), delta=sum(differences) / len(ids),
                                ci_low=samples[49], ci_high=samples[1949], resamples=2000, seed=42))
    out = ROOT / "results/latest/gold_v2_joint_location"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "leaderboard.csv", board)
    write_csv(out / "per_image.csv", per_image)
    write_csv(out / "significance.csv", comparisons)
    by_defect = []
    for p in POLICIES:
        for defect in sorted({r["defect_gt"] for r in per_image}):
            rows = [r for r in per_image if r["policy"] == p and r["defect_gt"] == defect]
            correct = sum(r["all_correct"] for r in rows)
            by_defect.append(dict(policy=p, defect_gt=defect, images=len(rows),
                                  all_correct=correct, joint_accuracy=correct / len(rows)))
    write_csv(out / "metrics_by_defect.csv", by_defect)
    metadata = dict(post_hoc=True, new_lvlm_inference=False, model="OpenGVLab/InternVL3_5-8B",
                    baseline="Direct questions with retrieved different-date normal reference; not image-only",
                    rule="Q1 correct AND Q2 correct AND Q3 reported cell contains at least one GT box center; normal requires no region",
                    scope="107 normal + 133 locatable defective images; six defects without position GT excluded",
                    limitation="Single-cell any-box hit, not exact multiple-region or box overlap scoring; historical gold reuse",
                    excluded_ids=excluded, verification="Reproduced all four existing Q1/Q2 joint, presence joint and position-hit totals",
                    scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    inputs={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()})
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(leaderboard=board, comparisons=comparisons), indent=2))


if __name__ == "__main__":
    main()
