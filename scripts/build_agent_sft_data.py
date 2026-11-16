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

``--human-boxes`` (protocol ``agent_sft_train_v2_human_boxes``) replaces the automatic box, because the sft_v1
adapter learned no location from it (341 of 365 dev240 crops were the same upper-left corner):

- anomalous: reference -> crop(largest human box, grown to at least 0.22 of each side) -> finish; the Q3 answer is
             the cell of that human box's centre, the rule the label tool itself uses
- normal:    reference -> finish (there is no human box on a normal photo, and no DINOv2 patch model is loaded)

``--add-polymer-corrosion`` adds the polymer shackle ``corrosão`` photos as a training class. Training only: their Q2
answer is the rust option text, because val and gold v2 name the same corroded shackle ``rust`` (user decision
2026-11-16). Evaluation labels and the slug contract do not change.
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
HUMAN_PROTOCOL = "agent_sft_train_v2_human_boxes"
MIN_CROP_SIDE = 0.22                    # select_roi_box's min_side_fraction, so sft_v1 crops were never smaller
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


# Supervised train names the polymer shackle defect ``corrosão``; val and gold v2 name the same corroded shackle
# ``rust``. dev240 has no polymer defect, so the first SFT set (sft_v1) had none and the adapter saw polymer only as
# normal, while gold v2 has 24 such defect photos. Adding the class is opt-in so sft_v1 stays reproducible.
POLYMER_CORROSION = ("polymer-insulator-upper-shackle", "corrosao")


def select_train_images(sup_rows, unsup_rows, dev_images: set[str], per_class: int, seed: int,
                        extra_classes: Sequence[tuple[str, str]] = ()):
    """``extra_classes`` adds (asset, defect) classes; each class has its own RNG, so the other picks do not move."""
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
    if set(extra_classes) & set(quota):
        raise ValueError("extra class is already in the base quota")
    quota.update({key: per_class for key in extra_classes})
    fixed = select_reference_by_asset(unsup_rows, devset.ASSETS, seed=42)
    chosen, stats = devset.select_dev_images(kept, unsup_rows, set(), quota, seed, fixed)
    stats["excluded_before_sampling"] = dict(reasons)
    stats["undated_or_unframed_selected"] = sum(photo_key(r["relative_path"]) is None for r in chosen)
    if {photo_key(r["relative_path"]) for r in chosen} & dev_keys or {r["relative_path"] for r in chosen} & dev_images:
        raise AssertionError("dev240 overlap survived selection")
    return chosen, stats, fixed


def train_rows(chosen, fixed, seed, protocol=PROTOCOL):
    # Training side only: a polymer ``corrosão`` photo is asked and answered as rust (see the module docstring).
    chosen = [dict(r, defect_class="rust") if (r["asset_category"], str(r["defect_class"])) == POLYMER_CORROSION else r
              for r in chosen]
    _, questions = devset.build_rows(chosen, fixed, {}, seed)
    rows = []
    for q in questions:
        q = dict(q, protocol=protocol, split="supervised_train_sft")
        q["core_id"] = "sft-" + q["core_id"].removeprefix("dev-")
        q["id"] = "sft-" + q["id"].removeprefix("dev-")
        rows.append(q)
    q3 = [dict(r, protocol=protocol) for r in pilot.support.q3_rows(rows)]
    return rows, q3


def load_human_boxes(folders: Sequence[Path]):
    """Finished box-label runs of sft_train_location_annotation.py (``manifest.json`` + ``annotations.json``).

    Returns {image path: label} and one receipt per folder. A run with an undecided photo is refused: that photo
    would otherwise train as "anomalous but no crop".
    """
    import sft_train_location_annotation as ann
    labels, receipts = {}, []
    for folder in map(Path, folders):
        manifest, raw = ann.read(folder / "manifest.json"), (folder / "annotations.json").read_bytes()
        state, identity = json.loads(raw.decode("utf-8")), ann.validate_manifest(manifest)
        if state.get("manifest_identity") != identity:
            raise ValueError(f"{folder.name}: annotations belong to another target list")
        counts = Counter()
        for t in manifest["targets"]:
            rec = state["records"].get(t["core_id"], {})
            status = rec.get("status") if rec.get("status") in ann.FINAL else "pending"
            counts[status] += 1
            if status == "pending":
                continue
            if rec["image_sha256"] != t["sha256"] or t["image"] in labels:
                raise ValueError(f"{folder.name}: label does not match its photo or the photo is labelled twice")
            boxes = ann.clean_boxes(rec["boxes_xywh"], rec["image_size"]) if status == "boxed" else []
            labels[t["image"]] = dict(core_id=t["core_id"], sha256=t["sha256"], status=status, boxes=boxes)
        if counts["pending"]:
            raise ValueError(f"{folder.name}: {counts['pending']} of {len(manifest['targets'])} photos are not "
                             "labelled yet; finish the box-label Colab first")
        receipts.append(dict(folder=folder.name, manifest_identity=identity, targets=len(manifest["targets"]),
                             annotations_sha256=pilot.file_sha(folder / "annotations.json"), status=dict(counts)))
    return labels, receipts


def human_crop_box(boxes_xywh, min_side: float = MIN_CROP_SIDE):
    """Crop request and Q3 cell from a photo's human boxes: the largest box and the cell of its centre.

    A side shorter than ``min_side`` is grown around the box centre and shifted to stay inside the frame, so a
    missing cap (median box area 1% of the photo) is still cropped with some context.
    """
    import sft_train_location_annotation as ann
    x, y, w, h = max(boxes_xywh, key=lambda b: (b[2] * b[3], -b[1], -b[0]))
    cell = CELL_ORDER[ann.box_regions([[x, y, w, h]])[0] - 1]
    spans = []
    for start, side in ((x, w), (y, h)):
        grown = min(1.0, max(side, min_side))
        low = min(max(start + side / 2 - grown / 2, 0.0), 1.0 - grown)
        spans.append((round(low, 3), min(1.0, round(low + grown, 3))))
    box = [spans[0][0], spans[1][0], spans[0][1], spans[1][1]]
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f"degenerate human crop box {box}")
    return box, cell, bool(w < min_side or h < min_side)


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
    p.add_argument("--human-boxes", type=Path, nargs="*", default=[],
                   help="finished box-label run folders; their boxes replace the automatic DINOv2 box")
    p.add_argument("--add-polymer-corrosion", action="store_true",
                   help="add the polymer shackle corrosão photos, answered as rust (training only)")
    args = p.parse_args(argv)
    if not 1 <= args.per_class <= 400 or not 0 <= args.val_fraction < 0.5:
        raise ValueError("per-class must be 1..400 and val-fraction 0..0.5")
    out = args.out_dir
    human = bool(args.human_boxes)
    labels, receipts = load_human_boxes(args.human_boxes) if human else ({}, [])
    extra = [POLYMER_CORROSION] if args.add_polymer_corrosion else []
    protocol = HUMAN_PROTOCOL if human else PROTOCOL
    settings = dict(per_class=args.per_class, val_fraction=args.val_fraction, seed=args.seed)
    if extra:
        settings["extra_classes"] = [":".join(c) for c in extra]
    if human:           # a relabelled photo changes the data, so the label bytes are part of the settings
        settings.update(human_boxes=[r["annotations_sha256"] for r in receipts], min_crop_side=MIN_CROP_SIDE)
    if (out / "build_summary.json").exists():
        old = json.loads((out / "build_summary.json").read_text(encoding="utf-8"))
        if old.get("settings") == settings:
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
    chosen, stats, fixed = select_train_images(sup_rows, unsup_rows, dev_images, args.per_class, args.seed, extra)
    unlocated = set()
    if human:
        anomalous = {str(r["relative_path"]) for r in chosen if str(r["defect_class"]) != "normal"}
        if anomalous != set(labels):
            raise ValueError(f"human boxes do not cover the selected anomalous photos: {len(anomalous - set(labels))} "
                             f"unlabelled, {len(set(labels) - anomalous)} labelled but not selected")
        # A photo the annotator could not locate has no crop target, so it leaves training rather than teach "no crop".
        unlocated = {p for p, v in labels.items() if v["status"] != "boxed"}
        chosen = [r for r in chosen if str(r["relative_path"]) not in unlocated]
    relabelled = sum((r["asset_category"], str(r["defect_class"])) == POLYMER_CORROSION for r in chosen)
    rows, q3 = train_rows(chosen, fixed, args.seed, protocol)
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
    encode = None if human else patch_encoder or PatchEncoder(args.device)
    grid = PATCH_INPUT // 14
    images = {}
    for r in rows:
        if r["question_type"] != pilot.Q1:
            continue
        target, ref = targets[r["image"]], normals[refs[r["image"]]]
        target_sha = pilot.file_sha(target)
        if target_sha == pilot.file_sha(ref):
            raise ValueError("target and reference are the same bytes")
        info = dict(core_id=r["core_id"], image=r["image"], asset_category=r["asset_category"],
                    anomaly_gt=r["anomaly_gt"], defect_gt=r["defect_gt"], reference=refs[r["image"]])
        if human:
            label = labels.get(r["image"])
            if (label is None) != (r["anomaly_gt"] == "normal") or (label and label["sha256"] != target_sha):
                raise ValueError("human box and photo disagree: " + r["core_id"])
            box, cell, grown = human_crop_box(label["boxes"]) if label else (None, pilot.support.NONE, False)
            info.update(score="", bbox=box, box_fallback=False, crop=label is not None, q3_cell=cell,
                        box_source="human" if label else "none", grown=grown,
                        human_boxes=label["boxes"] if label else [])
        else:
            with Image.open(target) as t, Image.open(ref) as n:
                tokens = encode([t, n])
                size = t.size
            diff = difference_map(tokens[0], tokens[1], grid)
            info.update(score=round(map_score(diff), 6), bbox=normalized_box(diff, size),
                        box_fallback=box_is_fallback(diff))
        images[r["core_id"]] = info
    medians = {} if human else {
        a: float(np.median([v["score"] for v in images.values() if v["asset_category"] == a and v["anomaly_gt"] == "normal"]))
        for a in sorted({v["asset_category"] for v in images.values()})}
    order = sorted(images, key=lambda c: pilot.digest([args.seed, "sft-val", c]))
    val = set(order[:int(round(len(order) * args.val_fraction))])
    for v in images.values():
        if not human:
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
                   "box_fallback", "crop", "q3_cell", "split"] + (["box_source", "grown", "human_boxes"] if human else [])
    with (out / "boxes.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=base_fields)
        w.writeheader()
        for v in images.values():
            w.writerow({k: json.dumps(v[k]) if k in ("bbox", "human_boxes") else v[k] for k in base_fields})
    cropped = [v for v in images.values() if v["crop"]]
    summary = dict(
        protocol=protocol, settings=settings,
        images=len(images), questions=len(rows) + len(q3), samples=len(samples),
        samples_by_split=dict(Counter(s["split"] for s in samples)),
        by_asset_defect=dict(Counter(f"{v['asset_category']}:{v['defect_gt']}" for v in images.values())),
        trajectories=dict(Counter("crop" if v["crop"] else "no_crop" for v in images.values())),
        crop_by_label=dict(Counter(f"{v['anomaly_gt']}:{'crop' if v['crop'] else 'no_crop'}" for v in images.values())),
        normal_score_median_by_asset=medians, selection=stats,
        centred_fallback_boxes=sum(v["box_fallback"] for v in images.values()),
        box_source=("human boxes on the anomalous photos (largest box, each side at least "
                    f"{MIN_CROP_SIDE}); normal photos request no crop") if human else
        f"automatic {DINO_ID}@{DINO_REVISION} patch difference vs the different-date reference; no human labels",
        # How concentrated the supervision is: a constant answer already scores 72/133 on the gold location set.
        q3_cells=dict(Counter(v["q3_cell"] for v in cropped)), distinct_crop_boxes=len({tuple(v["bbox"]) for v in cropped}),
        human_box_runs=receipts, grown_small_boxes=sum(v.get("grown", False) for v in images.values()),
        unlocated_photos_left_out=len(unlocated),
        polymer_corrosion_photos_answered_as_rust=relabelled,
        rationale="JSON actions only; the eval parser rejects explanation text",
        held_out="dev240 images, fingerprints and same drone photos; unsupervised test fingerprints (includes gold v2)",
    )
    (out / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("images", "samples", "by_asset_defect", "trajectories")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
