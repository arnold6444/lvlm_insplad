#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(
    eval_dir: Path,
    *,
    models: list[str],
    experts: list[str],
    conditions: list[str],
) -> tuple[pd.DataFrame, dict]:
    leaderboard_path = eval_dir / "leaderboard.csv"
    if not leaderboard_path.is_file():
        raise FileNotFoundError(leaderboard_path)

    leaderboard = pd.read_csv(leaderboard_path)
    screen = leaderboard[
        leaderboard["model_key"].isin(models)
        & leaderboard["expert"].isin(experts)
        & leaderboard["condition"].isin(conditions)
    ].copy()

    expected = len(models) * len(experts) * len(conditions)
    if len(screen) != expected:
        raise RuntimeError(f"leaderboard coverage mismatch: {len(screen)} != {expected}")

    required = {"f1", "balanced_accuracy", "invalid_output_rate", "tp", "tn", "n"}
    missing = sorted(required - set(screen.columns))
    if missing:
        raise RuntimeError(f"leaderboard missing required columns: {missing}")

    # run_hf_evidence_ablation intentionally does not emit an `accuracy` column.
    # Invalid outputs count as incorrect, so ordinary accuracy is (TP + TN) / N.
    if "accuracy" not in screen.columns:
        denominator = pd.to_numeric(screen["n"], errors="raise")
        if (denominator <= 0).any():
            raise RuntimeError("leaderboard contains non-positive n")
        screen["accuracy"] = (
            pd.to_numeric(screen["tp"], errors="raise")
            + pd.to_numeric(screen["tn"], errors="raise")
        ) / denominator

    summary = (
        screen.groupby(["expert", "condition"], as_index=False)
        .agg(
            models=("model_key", "nunique"),
            mean_f1=("f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_invalid_output_rate=("invalid_output_rate", "mean"),
        )
        .sort_values(
            ["mean_f1", "mean_balanced_accuracy", "mean_invalid_output_rate"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    summary.to_csv(eval_dir / "screening_summary.csv", index=False)

    row = summary.iloc[0]
    winner = {
        "winner_expert": str(row["expert"]),
        "winner_condition": str(row["condition"]),
        "mean_f1": float(row["mean_f1"]),
        "mean_balanced_accuracy": float(row["mean_balanced_accuracy"]),
        "mean_accuracy": float(row["mean_accuracy"]),
        "mean_invalid_output_rate": float(row["mean_invalid_output_rate"]),
        "selection_rule": (
            "highest mean anomaly F1 across selected LVLMs; tie-break by mean balanced accuracy; "
            "then lower mean invalid-output rate"
        ),
        "models": models,
        "experts": experts,
        "conditions": conditions,
    }
    (eval_dir / "winner.json").write_text(
        json.dumps(winner, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "protocol": "gold_v2_vision_expert_selection_dev50",
        "selection_scope": "development smoke only",
        "models": models,
        "experts": experts,
        "conditions": conditions,
        "expected_combinations": expected,
        "observed_combinations": len(screen),
        "accuracy_definition": "(tp + tn) / n; invalid outputs count as incorrect",
        "winner": winner,
    }
    (eval_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary, winner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--experts", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    args = parser.parse_args()

    summary, winner = summarize(
        args.eval_dir,
        models=args.models,
        experts=args.experts,
        conditions=args.conditions,
    )
    print(summary.to_string(index=False))
    print(json.dumps(winner, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
