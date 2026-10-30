"""Score the saved gold Q3 nine-region answers for position, with no new inference.

The gold agent run scored Q3 for presence only. This script reads its saved answers and checks
the reported cell against the frozen 133-image location set (manual boxes + InsPLAD-seg), using
the same box-center cell rule as q3_multiregion.py. A defective image counts as a position hit
when its single reported cell is the center cell of one of its boxes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import q3_multiregion as q3m

PROTOCOL = "gold_v2_q3_position_offline_v1"
TEXT_TO_CELL = {
    "Upper-left region.": 1, "Upper-center region.": 2, "Upper-right region.": 3,
    "Middle-left region.": 4, "Center region.": 5, "Middle-right region.": 6,
    "Lower-left region.": 7, "Lower-center region.": 8, "Lower-right region.": 9,
    "No localized defect region is present.": None,
}
POLICIES = ("baseline", "matched_reask", "previous_agent", "candidate")


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def location_gt(loc_rows):
    gt = {}
    for r in loc_rows:
        if r.get("protocol") != "gold_v2_loc_mmad" or r["core_id"] in gt:
            raise ValueError("unexpected or duplicate location row")
        g = r["localization_gt"]
        gt[r["core_id"]] = dict(cells=q3m.boxes_to_regions(g["normalized_bboxes_xywh"]), basis=g["basis"],
                                defect_gt=r["defect_gt"], asset_category=r["asset_category"])
    return gt


def q3_predictions(answer_rows):
    preds = defaultdict(dict)
    for r in answer_rows:
        if r["question_type"] != "defect_localization":
            continue
        if r["predicted_text"] not in TEXT_TO_CELL:
            raise ValueError("unparsed Q3 answer: " + repr(r["predicted_text"]))
        if r["core_id"] in preds[r["policy"]]:
            raise ValueError("duplicate Q3 answer")
        preds[r["policy"]][r["core_id"]] = TEXT_TO_CELL[r["predicted_text"]]
    return preds


def count(ids, gt, pred):
    reported = sum(pred[i] is not None for i in ids)
    hit = sum(pred[i] is not None and pred[i] in gt[i]["cells"] for i in ids)
    exact = sum(pred[i] is not None and [pred[i]] == gt[i]["cells"] for i in ids)
    return dict(images=len(ids), region_reported=reported, position_hit=hit, exact_cell_set=exact,
                position_hit_rate=hit / len(ids) if ids else None,
                hit_given_reported=hit / reported if reported else None)


def bootstrap(ids, gt, a, b, n=2000, seed=42):
    hits = lambda p, i: int(p[i] is not None and p[i] in gt[i]["cells"])
    diff = [hits(a, i) - hits(b, i) for i in ids]
    rng, stats = random.Random(seed), []
    for _ in range(n):
        sample = [diff[rng.randrange(len(diff))] for _ in diff]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    return dict(delta=sum(diff) / len(diff), ci_low=stats[int(0.025 * n)], ci_high=stats[int(0.975 * n) - 1],
                resamples=n, seed=seed)


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--loc-qa", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)
    gt = location_gt(read_jsonl(args.loc_qa))
    with open(args.answers, encoding="utf-8") as f:
        preds = q3_predictions(csv.DictReader(f))
    ids = sorted(gt)
    for policy in POLICIES:
        if any(i not in preds[policy] for i in ids):
            raise ValueError("missing Q3 answer for " + policy)
    constants = {f"always_cell_{c}": {i: c for i in ids} for c in range(1, 10)}
    board = [dict(policy=k, **count(ids, gt, preds[k])) for k in POLICIES]
    board += [dict(policy=k, **count(ids, gt, v)) for k, v in constants.items()]
    by_defect, by_asset = [], []
    for k in POLICIES:
        for key, out in (("defect_gt", by_defect), ("asset_category", by_asset)):
            groups = defaultdict(list)
            for i in ids:
                groups[gt[i][key]].append(i)
            for g, members in sorted(groups.items()):
                out.append(dict(policy=k, **{key: g}, **count(members, gt, preds[k])))
    per_image = [dict(core_id=i, defect_gt=gt[i]["defect_gt"], asset_category=gt[i]["asset_category"],
                      label_basis=gt[i]["basis"], gt_cells=" ".join(map(str, gt[i]["cells"])),
                      **{k: "" if preds[k][i] is None else preds[k][i] for k in POLICIES}) for i in ids]
    significance = [dict(comparison="previous_agent - baseline", metric="position_hit_rate",
                         **bootstrap(ids, gt, preds["previous_agent"], preds["baseline"]))]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "leaderboard.csv", board)
    write_csv(args.out_dir / "metrics_by_defect.csv", by_defect)
    write_csv(args.out_dir / "metrics_by_asset.csv", by_asset)
    write_csv(args.out_dir / "per_image.csv", per_image)
    write_csv(args.out_dir / "significance.csv", significance)
    meta = dict(protocol=PROTOCOL, new_lvlm_inference=False, model_id="OpenGVLab/InternVL3_5-8B",
                source_run="gold_v2_matched_call (agent gold run 0c577ffe3bdd + matched re-ask 902a9f8a1752)",
                location_images=len(ids), label_basis=dict(Counter(gt[i]["basis"] for i in ids)),
                cell_rule="each box center mapped to a 3x3 cell (q3_multiregion.boxes_to_regions); "
                          "hit = the single reported cell is one of those cells",
                post_hoc=True, inputs={name: hashlib.sha256(path.read_bytes()).hexdigest()
                                       for name, path in (("answers", args.answers), ("loc_qa", args.loc_qa))})
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "run_status.json").write_text(json.dumps(dict(status="complete", complete=True)) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(dict(leaderboard=board[:len(POLICIES)], significance=significance), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
