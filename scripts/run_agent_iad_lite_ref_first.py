#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import run_agent_iad_lite_smoke as base


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare fixed DINO reference against AgentIAD-lite RefFirst only."
    )
    p.add_argument("--qa", type=Path, required=True)
    p.add_argument("--workspace-root", type=Path, required=True)
    p.add_argument("--unsup-zip", type=Path, required=True)
    p.add_argument("--evidence-manifest", type=Path, required=True)
    p.add_argument("--patchcore-roi-manifest", type=Path, required=True)
    p.add_argument("--patchcore-artifact-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--image-cache",
        type=Path,
        default=Path("/content/agent_iad_ref_first_reference_cache"),
    )
    p.add_argument("--min-confidence", type=float, default=0.70)
    p.add_argument("--conflict-low", type=float, default=10.0)
    p.add_argument("--conflict-high", type=float, default=90.0)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--retry-attempts", type=int, default=2)
    args = p.parse_args()

    if not 0 <= args.min_confidence <= 1:
        p.error("min-confidence must be in [0,1]")
    if args.retry_attempts <= 0 or args.checkpoint_every <= 0:
        p.error("retry-attempts and checkpoint-every must be positive")

    qa = base.read_jsonl(args.qa)
    if not qa:
        raise RuntimeError("QA is empty")
    queries = base.workspace_queries(args.workspace_root)
    evidence_rows = base.read_csv(args.evidence_manifest)
    patch_score = {
        row["query_id"]: float(row["score_percentile_smoke"])
        for row in evidence_rows
        if row.get("expert") == "patchcore"
    }
    roi_rows = base.read_csv(args.patchcore_roi_manifest)
    roi = {row["query_id"]: row for row in roi_rows if row.get("expert") == "patchcore"}
    expected = {str(row["id"]) for row in qa}
    if set(patch_score) != expected or set(roi) != expected or set(queries) != expected:
        raise RuntimeError("query coverage mismatch among QA/workspace/PatchCore evidence")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.image_cache.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / "agent_predictions.jsonl"
    old = {str(row["id"]): row for row in base.read_jsonl(pred_path)}
    adapter = base.load_hf_adapter(base.MODEL_SPEC)
    rows: list[dict] = []

    try:
        for idx, gt in enumerate(qa, 1):
            qid = str(gt["id"])
            query_path = queries[qid]
            ref_path = base.safe_extract(
                args.unsup_zip,
                str(gt["reference_image"]),
                args.image_cache,
            )
            roi_path = Path(str(roi[qid]["roi_path"]))
            if not roi_path.is_absolute():
                roi_path = args.patchcore_artifact_root / roi_path
            if not roi_path.is_file():
                raise FileNotFoundError(roi_path)

            pctl = patch_score[qid]
            existing = dict(old.get(qid, {}))
            row = {
                "id": qid,
                "asset_category": gt["asset_category"],
                "answer": gt["answer"],
                "query_image": gt["image"],
                "reference_image": gt["reference_image"],
                "patchcore_percentile_smoke": pctl,
                "prompt_schema_version": base.PROMPT_SCHEMA_VERSION,
            }

            if base.reuse_stage(existing, "reference", {"none", "zoom"}):
                ro = existing["reference_output"]
                re_ = existing.get("reference_error", "")
                rl = float(existing.get("reference_latency_seconds", 0.0))
                rd = base.parse_agent_decision(ro, {"none", "zoom"})
                r_attempts = int(existing.get("reference_attempts", 1))
            else:
                ro, re_, rl, rd, r_attempts = base.call_decision(
                    adapter,
                    [ref_path, query_path],
                    base.reference_prompt(),
                    {"none", "zoom"},
                    args.retry_attempts,
                )
            route_r, reasons_r = base.should_route(
                rd,
                expected_tool="zoom",
                patchcore_percentile=pctl,
                min_confidence=args.min_confidence,
                conflict_low=args.conflict_low,
                conflict_high=args.conflict_high,
            )
            row.update(
                {
                    "reference_output": ro,
                    "reference_error": re_,
                    "reference_latency_seconds": rl,
                    "reference_prediction": rd.anomaly or "<invalid>",
                    "reference_confidence": rd.confidence,
                    "reference_request_tool": rd.request_tool,
                    "reference_attempts": r_attempts,
                    "reference_route": route_r,
                    "reference_route_reasons": ";".join(reasons_r),
                }
            )

            zo = ze = ""
            zl = 0.0
            zd = None
            z_attempts = 0
            if route_r:
                if base.reuse_stage(existing, "zoom", {"none"}):
                    zo = existing["zoom_output"]
                    ze = existing.get("zoom_error", "")
                    zl = float(existing.get("zoom_latency_seconds", 0.0))
                    zd = base.parse_agent_decision(zo, {"none"})
                    z_attempts = int(existing.get("zoom_attempts", 1))
                else:
                    zo, ze, zl, zd, z_attempts = base.call_decision(
                        adapter,
                        [ref_path, query_path, roi_path],
                        base.zoom_prompt(),
                        {"none"},
                        args.retry_attempts,
                    )
            row.update(
                {
                    "zoom_output": zo,
                    "zoom_error": ze,
                    "zoom_latency_seconds": zl,
                    "zoom_prediction": zd.anomaly if zd and zd.anomaly else "",
                    "zoom_confidence": zd.confidence if zd else None,
                    "zoom_attempts": z_attempts,
                }
            )

            row["fixed_prompt_prediction"] = (
                rd.anomaly if rd.valid and rd.anomaly else "<invalid>"
            )
            if route_r:
                final = zd.anomaly if zd and zd.valid and zd.anomaly else rd.anomaly
                calls = 2
                zoom_calls = 1
            else:
                final = rd.anomaly
                calls = 1
                zoom_calls = 0
            row.update(
                {
                    "agent_ref_first_prediction": final or "<invalid>",
                    "agent_ref_first_lvlm_calls": calls,
                    "agent_ref_first_reference_calls": 1,
                    "agent_ref_first_zoom_calls": zoom_calls,
                    "agent_ref_first_latency_seconds": float(rl)
                    + (float(zl) if zoom_calls else 0.0),
                }
            )
            rows.append(row)
            if idx % args.checkpoint_every == 0 or idx == len(qa):
                base.atomic_jsonl(pred_path, rows)
                print(f"agent ref-first dev: {idx}/{len(qa)}")
    finally:
        adapter = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    base.atomic_jsonl(pred_path, rows)
    leaderboard: list[dict] = []
    for variant, field, calls, zooms, latency in (
        (
            "fixed_reference_agent_prompt",
            "fixed_prompt_prediction",
            None,
            None,
            "reference_latency_seconds",
        ),
        (
            "agent_ref_first",
            "agent_ref_first_prediction",
            "agent_ref_first_lvlm_calls",
            "agent_ref_first_zoom_calls",
            "agent_ref_first_latency_seconds",
        ),
    ):
        eval_rows = [{**row, "final_prediction": row[field]} for row in rows]
        summary = base.classification_summary(eval_rows)
        summary.update(
            {
                "variant": variant,
                "model_key": "internvl3_5_8b",
                "model_id": base.MODEL_SPEC["model_id"],
            }
        )
        if calls:
            summary["avg_lvlm_calls"] = sum(float(row[calls]) for row in rows) / len(rows)
            summary["reference_call_rate"] = 1.0
            summary["zoom_call_rate"] = sum(float(row[zooms]) for row in rows) / len(rows)
        else:
            summary["avg_lvlm_calls"] = 1.0
            summary["reference_call_rate"] = 1.0
            summary["zoom_call_rate"] = 0.0
        summary["avg_latency_seconds"] = sum(float(row[latency]) for row in rows) / len(rows)
        leaderboard.append(summary)

    base.write_csv(args.out_dir / "agent_leaderboard.csv", leaderboard)
    base.write_csv(args.out_dir / "agent_prediction_review.csv", rows)
    config = {
        "model": base.MODEL_SPEC,
        "router": {
            "min_confidence": args.min_confidence,
            "conflict_low": args.conflict_low,
            "conflict_high": args.conflict_high,
            "retry_attempts": args.retry_attempts,
            "patchcore_score_source": (
                "expanded-dev unlabeled percentile screening only; replace with "
                "train-normal calibration before final holdout"
            ),
        },
        "variants": {
            "fixed_reference_agent_prompt": "DINO normal reference -> InternVL",
            "agent_ref_first": "DINO normal reference -> InternVL -> conditional PatchCore zoom",
        },
        "evaluation_note": (
            "Expanded development-set recheck. Query-first is intentionally omitted "
            "because the 50-row smoke showed a structural failure and would add cost."
        ),
    }
    (args.out_dir / "agent_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
