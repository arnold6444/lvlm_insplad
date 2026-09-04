#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.agent_iad_lite import (
    classification_summary,
    parse_agent_decision,
    should_route,
)
from insplad_benchmark.hf_adapters import infer_hf, load_hf_adapter

PROMPT_SCHEMA_VERSION = 1

MODEL_SPEC = {
    "backend": "hf",
    "provider": "huggingface",
    "model_id": "OpenGVLab/InternVL3_5-8B",
    "trust_remote_code": True,
    "adapter": "internvl_chat",
    "max_total_tiles": 6,
    "role": "internvl_family",
}


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_extract(zip_path: Path, member: str, cache: Path) -> Path:
    key = member.replace("\\", "/").lstrip("/")
    out = cache / key
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        names = {n.replace("\\", "/").lstrip("/"): n for n in archive.namelist()}
        if key not in names:
            raise FileNotFoundError(member)
        out.write_bytes(archive.read(names[key]))
    return out


def workspace_queries(root: Path) -> dict[str, Path]:
    result = {}
    for row in read_csv(root / "workspace_manifest.csv"):
        if row.get("role") == "query":
            result[row["query_id"]] = root / row["local_path"]
    return result


def query_prompt() -> str:
    return (
        "Inspect Image 1, the TARGET industrial asset. Decide whether it is normal or anomalous. "
        "If the image alone is insufficient, request a normal reference. "
        "Return exactly one JSON object and no explanation: "
        '{"anomaly":"normal|anomalous","confidence":0.0,"request_tool":"none|reference"}. '
        "confidence must be between 0 and 1."
    )


def reference_prompt() -> str:
    return (
        "Image 1 is a retrieved NORMAL same-category reference. Image 2 is the TARGET. "
        "Compare fine-grained structure, material, missing parts, corrosion, breakage, and shape. "
        "If the comparison is still visually ambiguous, request a zoom. "
        "Return exactly one JSON object and no explanation: "
        '{"anomaly":"normal|anomalous","confidence":0.0,"request_tool":"none|zoom"}. '
        "confidence must be between 0 and 1."
    )


def zoom_prompt() -> str:
    return (
        "Image 1 is a retrieved NORMAL same-category reference. Image 2 is the full TARGET. "
        "Image 3 is a high-resolution zoom selected by PatchCore from the TARGET. "
        "The zoom is only a hint and can point to a normal region, so preserve full-image context. "
        "Make the final decision. Return exactly one JSON object and no explanation: "
        '{"anomaly":"normal|anomalous","confidence":0.0,"request_tool":"none"}. '
        "confidence must be between 0 and 1."
    )


def call(adapter, images, prompt):
    started = time.perf_counter()
    output = ""
    error = ""
    try:
        output = str(infer_hf(adapter, images, prompt))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
    return output, error, round(time.perf_counter() - started, 6)


def call_decision(adapter, images, prompt, allowed_tools, attempts: int):
    total_latency = 0.0
    last_output = ""
    last_error = ""
    for attempt in range(1, attempts + 1):
        prompt_text = prompt
        if attempt > 1:
            prompt_text += " Previous output was invalid. Output only the exact JSON object now."
        output, error, latency = call(adapter, images, prompt_text)
        total_latency += latency
        last_output, last_error = output, error
        decision = parse_agent_decision(output, allowed_tools)
        if not error and decision.valid:
            return output, "", round(total_latency, 6), decision, attempt
    return (
        last_output,
        last_error or "invalid_structured_output",
        round(total_latency, 6),
        parse_agent_decision(last_output, allowed_tools),
        attempts,
    )


def reuse_stage(existing: Mapping[str, object] | None, stage: str, allowed_tools) -> bool:
    if not existing or existing.get("prompt_schema_version") != PROMPT_SCHEMA_VERSION:
        return False
    if existing.get(f"{stage}_error"):
        return False
    output = str(existing.get(f"{stage}_output", "")).strip()
    return bool(output and parse_agent_decision(output, allowed_tools).valid)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qa", type=Path, required=True)
    p.add_argument("--workspace-root", type=Path, required=True)
    p.add_argument("--unsup-zip", type=Path, required=True)
    p.add_argument("--evidence-manifest", type=Path, required=True)
    p.add_argument("--patchcore-roi-manifest", type=Path, required=True)
    p.add_argument("--patchcore-artifact-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--image-cache", type=Path, default=Path("/content/agent_iad_reference_cache"))
    p.add_argument("--min-confidence", type=float, default=0.70)
    p.add_argument("--conflict-low", type=float, default=10.0)
    p.add_argument("--conflict-high", type=float, default=90.0)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--retry-attempts", type=int, default=2)
    args = p.parse_args()
    if not 0 <= args.min_confidence <= 1:
        p.error("min-confidence must be in [0,1]")
    if args.retry_attempts <= 0:
        p.error("retry-attempts must be positive")

    qa = read_jsonl(args.qa)
    queries = workspace_queries(args.workspace_root)
    evidence_rows = read_csv(args.evidence_manifest)
    patch_score = {
        row["query_id"]: float(row["score_percentile_smoke"])
        for row in evidence_rows
        if row.get("expert") == "patchcore"
    }
    roi_rows = read_csv(args.patchcore_roi_manifest)
    roi = {row["query_id"]: row for row in roi_rows if row.get("expert") == "patchcore"}
    expected = {str(x["id"]) for x in qa}
    if set(patch_score) != expected or set(roi) != expected or set(queries) != expected:
        raise RuntimeError("query coverage mismatch among QA/workspace/PatchCore evidence")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.image_cache.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / "agent_predictions.jsonl"
    old = {str(r["id"]): r for r in read_jsonl(pred_path)}
    adapter = load_hf_adapter(MODEL_SPEC)
    rows: list[dict] = []

    try:
        for idx, gt in enumerate(qa, 1):
            qid = str(gt["id"])
            query_path = queries[qid]
            ref_path = safe_extract(args.unsup_zip, str(gt["reference_image"]), args.image_cache)
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
                "prompt_schema_version": PROMPT_SCHEMA_VERSION,
            }

            if reuse_stage(existing, "query", {"none", "reference"}):
                qo = existing["query_output"]
                qe = existing.get("query_error", "")
                ql = existing.get("query_latency_seconds", 0.0)
                qd = parse_agent_decision(qo, {"none", "reference"})
                q_attempts = int(existing.get("query_attempts", 1))
            else:
                qo, qe, ql, qd, q_attempts = call_decision(
                    adapter,
                    [query_path],
                    query_prompt(),
                    {"none", "reference"},
                    args.retry_attempts,
                )
            route_q, reasons_q = should_route(
                qd,
                expected_tool="reference",
                patchcore_percentile=pctl,
                min_confidence=args.min_confidence,
                conflict_low=args.conflict_low,
                conflict_high=args.conflict_high,
            )
            row.update(
                {
                    "query_output": qo,
                    "query_error": qe,
                    "query_latency_seconds": ql,
                    "query_prediction": qd.anomaly or "<invalid>",
                    "query_confidence": qd.confidence,
                    "query_request_tool": qd.request_tool,
                    "query_attempts": q_attempts,
                    "query_route": route_q,
                    "query_route_reasons": ";".join(reasons_q),
                }
            )

            if reuse_stage(existing, "reference", {"none", "zoom"}):
                ro = existing["reference_output"]
                re_ = existing.get("reference_error", "")
                rl = existing.get("reference_latency_seconds", 0.0)
                rd = parse_agent_decision(ro, {"none", "zoom"})
                r_attempts = int(existing.get("reference_attempts", 1))
            else:
                ro, re_, rl, rd, r_attempts = call_decision(
                    adapter,
                    [ref_path, query_path],
                    reference_prompt(),
                    {"none", "zoom"},
                    args.retry_attempts,
                )
            route_r, reasons_r = should_route(
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

            need_zoom = route_r
            zo = ze = ""
            zl = 0.0
            zd = None
            z_attempts = 0
            if need_zoom:
                if reuse_stage(existing, "zoom", {"none"}):
                    zo = existing["zoom_output"]
                    ze = existing.get("zoom_error", "")
                    zl = existing.get("zoom_latency_seconds", 0.0)
                    zd = parse_agent_decision(zo, {"none"})
                    z_attempts = int(existing.get("zoom_attempts", 1))
                else:
                    zo, ze, zl, zd, z_attempts = call_decision(
                        adapter,
                        [ref_path, query_path, roi_path],
                        zoom_prompt(),
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
                final_a = zd.anomaly if zd and zd.valid and zd.anomaly else rd.anomaly
                calls_a = 2
                zoom_a = 1
            else:
                final_a = rd.anomaly
                calls_a = 1
                zoom_a = 0
            row.update(
                {
                    "agent_ref_first_prediction": final_a or "<invalid>",
                    "agent_ref_first_lvlm_calls": calls_a,
                    "agent_ref_first_reference_calls": 1,
                    "agent_ref_first_zoom_calls": zoom_a,
                    "agent_ref_first_latency_seconds": float(rl) + (float(zl) if zoom_a else 0.0),
                }
            )

            if not route_q:
                final_b = qd.anomaly
                calls_b = 1
                ref_b = 0
                zoom_b = 0
                lat_b = float(ql)
            elif not route_r:
                final_b = rd.anomaly
                calls_b = 2
                ref_b = 1
                zoom_b = 0
                lat_b = float(ql) + float(rl)
            else:
                final_b = zd.anomaly if zd and zd.valid and zd.anomaly else rd.anomaly
                calls_b = 3
                ref_b = 1
                zoom_b = 1
                lat_b = float(ql) + float(rl) + float(zl)
            row.update(
                {
                    "agent_query_first_prediction": final_b or "<invalid>",
                    "agent_query_first_lvlm_calls": calls_b,
                    "agent_query_first_reference_calls": ref_b,
                    "agent_query_first_zoom_calls": zoom_b,
                    "agent_query_first_latency_seconds": lat_b,
                }
            )
            rows.append(row)
            if idx % args.checkpoint_every == 0 or idx == len(qa):
                atomic_jsonl(pred_path, rows)
                print(f"agent smoke: {idx}/{len(qa)}")
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

    atomic_jsonl(pred_path, rows)
    leaderboard = []
    for variant, field, calls, refs, zooms, latency in (
        (
            "fixed_reference_agent_prompt",
            "fixed_prompt_prediction",
            None,
            None,
            None,
            "reference_latency_seconds",
        ),
        (
            "agent_ref_first",
            "agent_ref_first_prediction",
            "agent_ref_first_lvlm_calls",
            "agent_ref_first_reference_calls",
            "agent_ref_first_zoom_calls",
            "agent_ref_first_latency_seconds",
        ),
        (
            "agent_query_first",
            "agent_query_first_prediction",
            "agent_query_first_lvlm_calls",
            "agent_query_first_reference_calls",
            "agent_query_first_zoom_calls",
            "agent_query_first_latency_seconds",
        ),
    ):
        eval_rows = [{**r, "final_prediction": r[field]} for r in rows]
        summary = classification_summary(eval_rows)
        summary.update(
            {
                "variant": variant,
                "model_key": "internvl3_5_8b",
                "model_id": MODEL_SPEC["model_id"],
            }
        )
        if calls:
            summary["avg_lvlm_calls"] = sum(float(r[calls]) for r in rows) / len(rows)
            summary["reference_call_rate"] = sum(float(r[refs]) for r in rows) / len(rows)
            summary["zoom_call_rate"] = sum(float(r[zooms]) for r in rows) / len(rows)
        else:
            summary["avg_lvlm_calls"] = 1.0
            summary["reference_call_rate"] = 1.0
            summary["zoom_call_rate"] = 0.0
        summary["avg_latency_seconds"] = sum(float(r[latency]) for r in rows) / len(rows)
        leaderboard.append(summary)
    write_csv(args.out_dir / "agent_leaderboard.csv", leaderboard)
    write_csv(args.out_dir / "agent_prediction_review.csv", rows)
    config = {
        "model": MODEL_SPEC,
        "router": {
            "min_confidence": args.min_confidence,
            "conflict_low": args.conflict_low,
            "conflict_high": args.conflict_high,
            "retry_attempts": args.retry_attempts,
            "patchcore_score_source": (
                "fixed-smoke unlabeled percentile screening only; replace with "
                "train-normal calibration before holdout"
            ),
        },
        "variants": {
            "agent_ref_first": (
                "DINO normal reference -> InternVL -> conditional PatchCore zoom"
            ),
            "agent_query_first": (
                "query-only -> conditional DINO normal reference -> conditional PatchCore zoom"
            ),
        },
        "evaluation_note": (
            "Query/reference stages are precomputed for paired comparison; reported "
            "calls and latency are sums for the stages each policy would invoke."
        ),
    }
    (args.out_dir / "agent_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
