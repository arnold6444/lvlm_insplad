from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from insplad_benchmark.sampling import (
    balanced_binary_sample,
    balanced_structured_sample,
    matched_one_shot_sample,
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_balanced_samples(root: Path, out_dir: Path, n: int, seed: int) -> dict:
    qa_dir = root / "data" / "qa"
    zero_all = read_jsonl(qa_dir / "anomaly_zero_shot.jsonl")
    one_all = read_jsonl(qa_dir / "anomaly_one_shot.jsonl")
    structured_all = read_jsonl(qa_dir / "defect_qa.jsonl")

    zero = balanced_binary_sample(zero_all, n, seed=seed)
    one = matched_one_shot_sample(zero, one_all)
    structured = balanced_structured_sample(structured_all, n, seed=seed)

    write_jsonl(out_dir / "anomaly_zero_shot.jsonl", zero)
    write_jsonl(out_dir / "anomaly_one_shot.jsonl", one)
    write_jsonl(out_dir / "defect.jsonl", structured)

    summary = {
        "seed": seed,
        "count_per_task": n,
        "zero_shot_labels": dict(Counter(row["answer"] for row in zero)),
        "zero_one_same_query_ids": [row["id"] for row in zero]
        == [row["id"] for row in one],
        "structured_assets": dict(Counter(row["asset_category"] for row in structured)),
        "structured_defects": dict(
            Counter(row["answer"]["defect_class"] for row in structured)
        ),
    }
    (out_dir / "sample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic balanced QA subsets from the full benchmark QA."
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("samples/balanced100")
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    summary = build_balanced_samples(root, args.out_dir, args.count, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
