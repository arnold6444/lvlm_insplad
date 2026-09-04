#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.evidence_ablation import build_evidence_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--base-roi-manifest", type=Path, required=True)
    parser.add_argument("--base-artifact-root", type=Path, required=True)
    parser.add_argument("--reconpatch-manifest", type=Path, required=True)
    parser.add_argument("--reconpatch-artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = build_evidence_manifest(
        qa_path=args.qa,
        workspace_root=args.workspace_root,
        expert_sources=[
            (args.base_roi_manifest, args.base_artifact_root),
            (args.reconpatch_manifest, args.reconpatch_artifact_root),
        ],
        out_dir=args.out_dir,
        expected_experts=("patchcore", "dinomaly", "anomalyclip", "reconpatch_lite"),
    )
    print(f"evidence rows={len(rows)} -> {args.out_dir / 'evidence_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
