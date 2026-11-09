#!/usr/bin/env python3
"""Build SFT tool-use trajectories for the dev240 tool-choice agent. Training data only; dev240 stays held out.

Images come from the InsPLAD supervised ``train`` split (same five assets and Q1/Q2/Q3 wording as dev240).
Excluded before sampling:

- every dev240 image, its content fingerprint (size + CRC32), and every crop of the same drone photo
  (same flight date + ``DJI_nnnn`` frame number), so no dev240 scene is trained on. All 35 dev240 flight
  dates cover the training split, so a date-disjoint split is impossible; the frame rule is the next best unit
- unsupervised ``test`` fingerprints (this also excludes every gold v2 image), duplicates, fixed references
- polymer ``corrosão`` (not merged with rust by the data contract; dev240 has no corrosion option)

Each image gets the different-date DINOv2 Top-1 normal reference (``build_retrieved_refs.py``, the dev240 rule).
The crop box is automatic, never a human label: DINOv2 patch tokens of the target are compared with the
reference patches (1 - best cosine) and ``vision_expert.select_roi_box`` boxes the hottest component. Boxes can be
wrong; they are audited in ``boxes.csv``. Trajectories use the eval runner's own prompt and history format:

- anomalous: reference -> crop(auto box) -> finish
- normal:    reference -> crop -> finish when the patch difference is at least the asset's normal median,
             otherwise reference -> finish (the policy must learn when a crop is worth requesting)

Final answers are the GT option letters. Q3 anomalous answers use the 3 x 3 cell containing the auto box centre,
so Q3 location supervision is as noisy as the box. Outputs are JSON actions only (the eval parser rejects text).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import build_gold_v2_dev_set as devset  # noqa: E402
import build_retrieved_refs as refsmod  # noqa: E402
import run_agent_tool_pilot as pilot  # noqa: E402
from insplad_benchmark.dataset import (  # noqa: E402
    parse_supervised_manifest,
    parse_unsupervised_manifest,
    select_reference_by_asset,
)
from insplad_benchmark.vision_expert import normalize_anomaly_map, select_roi_box  # noqa: E402

PROTOCOL = "agent_sft_train_v1"
DINO_ID = "facebook/dinov2-base"
DINO_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
PATCH_INPUT = 448                       # 32 x 32 patches of 14 px
DJI = re.compile(r"DJI_(\d+)")
CELL_ORDER = ("Upper-left region.", "Upper-center region.", "Upper-right region.",
              "Middle-left region.", "Center region.", "Middle-right region.",
              "Lower-left region.", "Lower-center region.", "Lower-right region.")
REF_RESULT = "Normal reference attached."
CROP_RESULT = "Requested crop attached; use original-frame coordinates."


def photo_key(path: str):
    """Drone frame identity: crops of the same photo share flight date and DJI frame number."""
    date, frame = refsmod.flight_date(path), DJI.search(Path(str(path)).name)
    return (date, frame.group(1)) if date and frame else None


def select_train_images(sup_rows, unsup_rows, dev_images: set[str], per_class: int, seed: int):
    dev_rows = [r for r in sup_rows if r["relative_path"] in dev_images]
    if len(dev_rows) != len(dev_images):
        raise ValueError("dev240 images missing from the supervised manifest")
    dev_fp = {str(r["content_fingerprint"]) for r in dev_rows}
    dev_keys = {photo_key(p) for p in dev_images} - {None}
    reasons = Counter()
    kept = []
    for r in sup_rows:
        path = str(r["relative_path"])
        if r["split"] == "train" and r["asset_category"] in devset.ASSETS:
            if path in dev_images or str(r["content_fingerprint"]) in dev_fp:
                reasons["dev240_image"] += 1
                continue
            if photo_key(path) in dev_keys:
                reasons["dev240_same_drone_photo"] += 1
                continue
        kept.append(r)
    quota = devset.quotas(per_class, per_class, per_class)
    fixed = select_reference_by_asset(unsup_rows, devset.ASSETS, seed=42)
    chosen, stats = devset.select_dev_images(kept, unsup_rows, set(), quota, seed, fixed)
    stats["excluded_before_sampling"] = dict(reasons)
    stats["undated_or_unframed_selected"] = sum(photo_key(r["relative_path"]) is None for r in chosen)
    if {photo_key(r["relative_path"]) for r in chosen} & dev_keys or {r["relative_path"] for r in chosen} & dev_images:
        raise AssertionError("dev240 overlap survived selection")
    return chosen, stats, fixed


def train_rows(chosen, fixed, seed):
    _, questions = devset.build_rows(chosen, fixed, {}, seed)
    rows = []
    for q in questions:
        q = dict(q, protocol=PROTOCOL, split="supervised_train_sft")
        q["core_id"] = "sft-" + q["core_id"].removeprefix("dev-")
        q["id"] = "sft-" + q["id"].removeprefix("dev-")
        rows.append(q)
    q3 = [dict(r, protocol=PROTOCOL) for r in pilot.support.q3_rows(rows)]
    return rows, q3


def difference_map(target_tokens: np.ndarray, reference_tokens: np.ndarray, grid: int) -> np.ndarray:
    """1 - best cosine of each target patch to any reference patch (inputs L2-normalised, [n, d])."""
    if target_tokens.shape != reference_tokens.shape or target_tokens.shape[0] != grid * grid:
        raise ValueError("patch token shape mismatch")
    best = (target_tokens @ reference_tokens.T).max(axis=1)
    return np.clip(1.0 - best, 0.0, 2.0).reshape(grid, grid).astype(np.float32)


def map_score(diff: np.ndarray, top_fraction: float = 0.02) -> float:
    flat = np.sort(diff.reshape(-1))
    return float(flat[-max(1, int(math.ceil(flat.size * top_fraction))):].mean())


def box_is_fallback(diff: np.ndarray) -> bool:
    """select_roi_box silently returns a centred box when the robust-normalised map is flat."""
    return float(normalize_anomaly_map(diff).max()) <= 1e-8


def normalized_box(diff: np.ndarray, size) -> list[float]:
    box = select_roi_box(diff, size)
    w, h = size
    out = [round(box.left / w, 3), round(box.top / h, 3), round(box.right / w, 3), round(box.bottom / h, 3)]
    out = [min(1.0, max(0.0, v)) for v in out]
    if not (out[0] < out[2] and out[1] < out[3]):
        raise ValueError(f"degenerate auto box {out}")
    return out


def cell_text(bbox: Sequence[float]) -> str:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    col, row = min(2, int(cx * 3)), min(2, int(cy * 3))
    return CELL_ORDER[row * 3 + col]


class PatchEncoder:
    """DINOv2-base patch tokens on a full-frame 448 x 448 resize (no centre crop, so boxes map to the frame)."""

    def __init__(self, device):
        import torch
        from transformers import AutoModel
        self.torch, self.device = torch, torch.device(device)
        self.model = AutoModel.from_pretrained(DINO_ID, revision=DINO_REVISION).to(self.device).eval()

    def __call__(self, images) -> np.ndarray:
        arrays = []
        for image in images:
            x = np.asarray(image.convert("RGB").resize((PATCH_INPUT, PATCH_INPUT), resample=3), dtype=np.float32) / 255
            arrays.append((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
        batch = self.torch.tensor(np.stack(arrays).transpose(0, 3, 1, 2), dtype=self.torch.float32, device=self.device)
        with self.torch.inference_mode():
            tokens = self.model(pixel_values=batch).last_hidden_state[:, 1:, :].float()
            tokens = self.torch.nn.functional.normalize(tokens, dim=-1)
        return tokens.cpu().numpy()


def trajectory(row, target, reference, crop_box, answer, crop_dir, max_calls=3):
    """Supervised steps rebuilt with the eval runner's prompt_for/history so train and test prompts match."""
    history, use_ref, bbox, crop = [], False, None, None
    plan = [{"action": "reference"}] + ([{"action": "crop", "bbox": crop_box}] if crop_box else [])
    plan.append({"action": "finish", "choice": answer})
    if len(plan) > max_calls:
        raise ValueError("trajectory exceeds the eval call budget")
    steps = []
    for index, action in enumerate(plan):
        allowed = index < max_calls - 1
        if action["action"] != "finish" and not allowed:
            raise ValueError("tool step on a final-answer turn")
        prompt = pilot.prompt_for(row, history, use_ref, bbox, allowed)
        images = [target] + ([reference] if use_ref else []) + ([crop] if crop else [])
        output = json.dumps(action, separators=(",", ":"))
        pilot.parse_action(output, row["options"])          # every target must pass the strict eval parser
        steps.append(dict(step=index, images=[str(p) for p in images], prompt=prompt, target=output))
        if action["action"] == "reference":
            use_ref = True
            history.append(dict(output=action, result=REF_RESULT))
        elif action["action"] == "crop":
            bbox = action["bbox"]
            crop = pilot.make_crop(target, bbox, crop_dir / (pilot.digest([row["id"], bbox]) + ".png"))
            history.append(dict(output=action, result=CROP_RESULT))
    return steps


def main(argv: Sequence[str] | None = None, *, patch_encoder=None, embed_factory=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("dev-qa", "sup-zip", "unsup-zip", "out-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--per-class", type=int, default=60)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    if not 1 <= args.per_class <= 400 or not 0 <= args.val_fraction < 0.5:
        raise ValueError("per-class must be 1..400 and val-fraction 0..0.5")
    out = args.out_dir
    if (out / "build_summary.json").exists():
        old = json.loads((out / "build_summary.json").read_text(encoding="utf-8"))
        if old.get("settings") == dict(per_class=args.per_class, val_fraction=args.val_fraction, seed=args.seed):
            print("SFT data already built:", out)
            return 0
        raise ValueError("existing SFT data was built with other settings; use a new folder")
    dev_rows = pilot.agent.read_jsonl(args.dev_qa)
    if any(r.get("protocol") != "gold_v2_dev" for r in dev_rows):
        raise ValueError("--dev-qa must be the FROZEN gold_v2_dev QA")
    dev_images = {r["image"] for r in dev_rows}
    man = out / "_manifests"
    unsup_rows = parse_unsupervised_manifest(devset.zip_manifest(args.unsup_zip, man / "unsupervised.csv"))
    sup_rows, _ = parse_supervised_manifest(devset.zip_manifest(args.sup_zip, man / "supervised.csv"))
    chosen, stats, fixed = select_train_images(sup_rows, unsup_rows, dev_images, args.per_class, args.seed)
    rows, q3 = train_rows(chosen, fixed, args.seed)
    qa_path = out / "sft_train_qa.jsonl"
    pilot.base.write_jsonl(qa_path, rows)

    refs_dir = out / "refs"
    refsmod.main(["--qa", str(qa_path), "--sup-zip", str(args.sup_zip), "--unsup-zip", str(args.unsup_zip),
                  "--out-dir", str(refs_dir), "--device", args.device], embed_factory=embed_factory)
    with (refs_dir / "retrieved_refs.csv").open(encoding="utf-8", newline="") as h:
        refs = {r["image"]: r["retrieved_diffdate"] for r in csv.DictReader(h)}
    targets = pilot.support.verified_images(args.sup_zip, [r["image"] for r in rows], out / "_images/target")
    normals = pilot.support.verified_images(args.unsup_zip, sorted(set(refs.values())), out / "_images/normal")

    from PIL import Image
    encode = patch_encoder or PatchEncoder(args.device)
    grid = PATCH_INPUT // 14
    images = {}
    for r in rows:
        if r["question_type"] != pilot.Q1:
            continue
        target, ref = targets[r["image"]], normals[refs[r["image"]]]
        if pilot.file_sha(target) == pilot.file_sha(ref):
            raise ValueError("target and reference are the same bytes")
        with Image.open(target) as t, Image.open(ref) as n:
            tokens = encode([t, n])
            size = t.size
        diff = difference_map(tokens[0], tokens[1], grid)
        images[r["core_id"]] = dict(core_id=r["core_id"], image=r["image"], asset_category=r["asset_category"],
                                    anomaly_gt=r["anomaly_gt"], defect_gt=r["defect_gt"], reference=refs[r["image"]],
                                    score=round(map_score(diff), 6), bbox=normalized_box(diff, size),
                                    box_fallback=box_is_fallback(diff))
    medians = {a: float(np.median([v["score"] for v in images.values() if v["asset_category"] == a and v["anomaly_gt"] == "normal"]))
               for a in sorted({v["asset_category"] for v in images.values()})}
    order = sorted(images, key=lambda c: pilot.digest([args.seed, "sft-val", c]))
    val = set(order[:int(round(len(order) * args.val_fraction))])
    for v in images.values():
        v["crop"] = v["anomaly_gt"] == "anomalous" or v["score"] >= medians[v["asset_category"]]
        v["q3_cell"] = cell_text(v["bbox"]) if v["anomaly_gt"] == "anomalous" else pilot.support.NONE
        v["split"] = "val" if v["core_id"] in val else "train"

    samples = []
    for r in rows + q3:
        info = images[r["core_id"]]
        if r["question_type"] == pilot.Q3:
            answer = next(k for k, text in r["options"].items() if text == info["q3_cell"])
        else:
            answer = r["answer"]
        steps = trajectory(r, targets[r["image"]], normals[info["reference"]],
                           info["bbox"] if info["crop"] else None, answer, out / "_crops")
        for s in steps:
            samples.append(dict(sample_id=f"{r['id']}#s{s['step']}", id=r["id"], core_id=r["core_id"],
                                question_type=r["question_type"], split=info["split"],
                                images=[Path(x).relative_to(out).as_posix() for x in s["images"]],
                                prompt=s["prompt"], target=s["target"]))
    pilot.base.write_jsonl(out / "sft_samples.jsonl", samples)
    pilot.base.write_jsonl(out / "sft_train_q3.jsonl", q3)
    base_fields = ["core_id", "image", "asset_category", "anomaly_gt", "defect_gt", "reference", "score", "bbox",
                   "box_fallback", "crop", "q3_cell", "split"]
    with (out / "boxes.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=base_fields)
        w.writeheader()
        for v in images.values():
            w.writerow({k: json.dumps(v[k]) if k == "bbox" else v[k] for k in base_fields})
    summary = dict(
        protocol=PROTOCOL, settings=dict(per_class=args.per_class, val_fraction=args.val_fraction, seed=args.seed),
        images=len(images), questions=len(rows) + len(q3), samples=len(samples),
        samples_by_split=dict(Counter(s["split"] for s in samples)),
        by_asset_defect=dict(Counter(f"{v['asset_category']}:{v['defect_gt']}" for v in images.values())),
        trajectories=dict(Counter("crop" if v["crop"] else "no_crop" for v in images.values())),
        crop_by_label=dict(Counter(f"{v['anomaly_gt']}:{'crop' if v['crop'] else 'no_crop'}" for v in images.values())),
        normal_score_median_by_asset=medians, selection=stats,
        centred_fallback_boxes=sum(v["box_fallback"] for v in images.values()),
        box_source=f"automatic {DINO_ID}@{DINO_REVISION} patch difference vs the different-date reference; no human labels",
        rationale="JSON actions only; the eval parser rejects explanation text",
        held_out="dev240 images, fingerprints and same drone photos; unsupervised test fingerprints (includes gold v2)",
    )
    (out / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("images", "samples", "by_asset_defect", "trajectories")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
