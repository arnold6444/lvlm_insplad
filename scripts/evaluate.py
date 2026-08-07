#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.metrics import balanced_accuracy, binary_f1, consistency_error, false_defect_rate, invalid_output_rate, joint_accuracy, macro_f1, per_class_prf
from insplad_benchmark.parsing import parse_anomaly_output, parse_structured_output


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def eval_anomaly(gt: list[dict], predictions: dict[str, str], out_dir: Path, regime: str) -> None:
    def summarize(rows: list[dict]) -> dict:
        parsed = [parse_anomaly_output(predictions.get(r["id"], "")) for r in rows]
        valid_pairs = [(r["answer"], p.anomaly) for r, p in zip(rows, parsed) if p.valid and p.anomaly is not None]
        y_true = [a for a, _ in valid_pairs]; y_pred = [b for _, b in valid_pairs]
        return {"regime": regime, "n": len(rows), "valid_n": len(valid_pairs), "balanced_accuracy": balanced_accuracy(y_true, y_pred, ["normal", "anomalous"]) if y_true else 0.0, "f1": binary_f1(y_true, y_pred) if y_true else 0.0, "invalid_output_rate": invalid_output_rate([p.valid for p in parsed])}
    write_csv(out_dir / "anomaly_overall.csv", [summarize(gt)])
    by_asset = []; grouped: dict[str, list[dict]] = defaultdict(list)
    for row in gt: grouped[row["asset_category"]].append(row)
    for asset, rows in sorted(grouped.items()):
        data = summarize(rows); data["asset_category"] = asset; by_asset.append(data)
    write_csv(out_dir / "anomaly_by_asset.csv", by_asset)


def eval_explainability(gt: list[dict], predictions: dict[str, str], out_dir: Path) -> None:
    parsed = [parse_structured_output(predictions.get(r["id"], ""), set(r["valid_defect_classes"])) for r in gt]
    def summarize(indices: list[int]) -> dict:
        valid = [i for i in indices if parsed[i].valid]
        a_t = [gt[i]["answer"]["anomaly"] for i in valid]; a_p = [parsed[i].anomaly for i in valid]
        d_t = [gt[i]["answer"]["defect_class"] for i in valid]; d_p = [parsed[i].defect_class for i in valid]
        return {"n": len(indices), "valid_n": len(valid), "defect_balanced_accuracy": balanced_accuracy(d_t, d_p) if valid else 0.0, "defect_macro_f1": macro_f1(d_t, d_p) if valid else 0.0, "joint_accuracy": joint_accuracy(a_t, a_p, d_t, d_p) if valid else 0.0, "false_defect_rate": false_defect_rate(a_t, d_p) if valid else 0.0, "consistency_error": consistency_error(a_p, d_p) if valid else 0.0, "invalid_output_rate": invalid_output_rate([parsed[i].valid for i in indices])}
    all_idx = list(range(len(gt))); write_csv(out_dir / "explainability_overall.csv", [summarize(all_idx)])
    by_asset = []; grouped: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(gt): grouped[row["asset_category"]].append(i)
    for asset, idx in sorted(grouped.items()):
        data = summarize(idx); data["asset_category"] = asset; by_asset.append(data)
    write_csv(out_dir / "explainability_by_asset.csv", by_asset)
    valid_idx = [i for i in all_idx if parsed[i].valid]
    d_t = [gt[i]["answer"]["defect_class"] for i in valid_idx]; d_p = [parsed[i].defect_class for i in valid_idx]
    defect_rows = [{"defect_class": defect, **stat} for defect, stat in (per_class_prf(d_t, d_p).items() if valid_idx else [])]
    write_csv(out_dir / "explainability_by_defect.csv", defect_rows)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--task", choices=["anomaly", "explainability"], required=True); parser.add_argument("--qa", type=Path, required=True); parser.add_argument("--predictions", type=Path, required=True, help='JSONL rows with {"id": ..., "output": "..."}'); parser.add_argument("--out-dir", type=Path, default=Path("results")); parser.add_argument("--regime", choices=["zero_shot", "one_shot"], default="zero_shot"); args = parser.parse_args()
    gt = read_jsonl(args.qa); predictions = {r["id"]: str(r.get("output", "")) for r in read_jsonl(args.predictions)}
    if args.task == "anomaly": eval_anomaly(gt, predictions, args.out_dir, args.regime)
    else: eval_explainability(gt, predictions, args.out_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
