#!/usr/bin/env python3
"""Gold-v2 vision-expert heatmap check: does each expert detect and localize defects? (CPU, no LVLM)

AnomalyCLIP's heatmap gave the highest LVLM Q1 F1 among the 20 expert conditions, but with the highest recall and
false-defect rate. This script looks at the experts themselves, with no LVLM in the loop. It reads the merged
vision-expert manifest (one row per gold image x expert: raw anomaly map .npy, the pure heatmap overlay the LVLMs
saw, the ROI box and the scores) plus the gold-v2 QA, and writes:

- ``detection.csv``: image-level AUROC of each expert score against the binary GT (246 images), global and
  averaged over assets (raw scores are not comparable across assets)
- ``localization.csv`` / ``localization_by_image.csv``: on the gold images that have an InsPLAD-seg defect box
  (the Q3 rust set), whether the heatmap peak / ROI centre falls inside the GT box, ROI IoU, and how much of the
  heat lies inside the box relative to its area. An "image centre" pointer is the trivial baseline because every
  Q3 rust answer is 'center'
- ``heat_on_normals.csv``: how much of each per-image-normalised heatmap is hot on normal vs anomalous images
  (every map is stretched to [0, 1] per image, so normal images get a hot spot too)
- ``grid_*.png``: target + GT box next to each expert heatmap (GT box red, expert ROI yellow, heat peak cyan)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gold_baselines as base  # noqa: E402
import run_gold_v2_baselines as v2  # noqa: E402

Q1, Q3 = "anomaly_detection", "defect_localization"
EXPERTS = ("patchcore", "anomalyclip", "dinomaly", "reconpatch_lite")
SCORE_FIELDS = ("model_pred_score", "anomaly_score_top1pct", "raw_map_max", "raw_map_top1pct")
HOT = 0.6            # normalised heat counted as "hot"
CELL_W, CELL_H, HEAD_H, LABEL_W = 320, 240, 28, 230


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve(manifest_path: Path, value: str, artifact_root: Path | None) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (artifact_root or manifest_path.parent) / p


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Mann-Whitney AUROC (ties count half); None without both classes."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def number(value: object) -> float | None:
    try:
        out = float(str(value).strip())
    except ValueError:
        return None
    return out if out == out else None


def inside(x: float, y: float, boxes: Sequence[Sequence[float]]) -> bool:
    return any(bx <= x <= bx + bw and by <= y <= by + bh for bx, by, bw, bh in boxes)


def union_box(boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """xywh boxes -> (left, top, right, bottom) of their union."""
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def map_stats(raw, boxes: Sequence[Sequence[float]] | None) -> dict[str, float]:
    import numpy as np

    from insplad_benchmark.vision_expert import normalize_anomaly_map

    raw = np.asarray(raw, dtype=np.float32)
    raw = raw.reshape(raw.shape[-2:]) if raw.ndim > 2 else raw
    norm = normalize_anomaly_map(raw)
    h, w = norm.shape
    py, px = np.unravel_index(int(np.argmax(norm)), norm.shape)
    flat = raw[np.isfinite(raw)].reshape(-1)
    top = np.sort(flat)[-max(1, int(np.ceil(flat.size * 0.01))):]
    out = {"raw_map_max": float(flat.max()), "raw_map_top1pct": float(top.mean()),
           "peak_x": (px + 0.5) / w, "peak_y": (py + 0.5) / h,
           "hot_area_frac": float((norm >= HOT).mean()), "median_heat": float(np.median(norm))}
    if boxes:
        xs = (np.arange(w) + 0.5) / w
        ys = (np.arange(h) + 0.5) / h
        mask = np.zeros((h, w), dtype=bool)
        for bx, by, bw, bh in boxes:
            mask |= ((ys >= by) & (ys <= by + bh))[:, None] & ((xs >= bx) & (xs <= bx + bw))[None, :]
        total = float(norm.sum())
        area = float(mask.mean())
        share = float(norm[mask].sum()) / total if total > 0 else 0.0
        out.update({"gt_area_frac": area, "heat_in_box_share": share,
                    "heat_in_box_ratio": share / area if area > 0 else 0.0})
    return out


def pick(rng: random.Random, groups: Mapping[str, Sequence[str]], n: int) -> list[str]:
    """Round-robin over sorted groups, a seeded random member each turn."""
    pools = {k: rng.sample(sorted(v), len(v)) for k, v in sorted(groups.items())}
    out: list[str] = []
    while len(out) < n and any(pools.values()):
        for key in sorted(pools):
            if pools[key] and len(out) < n:
                out.append(pools[key].pop())
    return out


def font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fit(img, w: int, h: int):
    from PIL import Image

    img = img.copy()
    img.thumbnail((w, h), Image.Resampling.BILINEAR)
    cell = Image.new("RGB", (w, h), (24, 32, 51))
    cell.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return cell


def draw_marks(img, boxes, roi, peak):
    """GT boxes red, expert ROI yellow, heat peak cyan cross; coordinates normalised to the image."""
    from PIL import ImageDraw

    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    lw = max(3, round(min(w, h) / 120))
    for bx, by, bw, bh in boxes or []:
        d.rectangle((bx * w, by * h, (bx + bw) * w, (by + bh) * h), outline=(255, 40, 40), width=lw)
    if roi:
        d.rectangle((roi[0] * w, roi[1] * h, roi[2] * w, roi[3] * h), outline=(255, 230, 0), width=lw)
    if peak:
        x, y, r = peak[0] * w, peak[1] * h, max(10, round(min(w, h) / 25))
        d.line((x - r, y, x + r, y), fill=(0, 230, 255), width=lw)
        d.line((x, y - r, x, y + r), fill=(0, 230, 255), width=lw)
    return img


def make_grid(path: Path, title: str, cores: Sequence[str], info: Mapping[str, Mapping], targets: Mapping[str, Path],
              overlays: Mapping[tuple[str, str], Path], marks: Mapping[tuple[str, str], Mapping], experts: Sequence[str],
              boxes: Mapping[str, list]) -> None:
    from PIL import Image, ImageDraw

    cols = ["target + GT"] + list(experts)
    sheet = Image.new("RGB", (LABEL_W + CELL_W * len(cols), HEAD_H * 2 + (CELL_H + HEAD_H) * len(cores)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 4), title, fill=(23, 32, 51), font=font(18))
    for j, name in enumerate(cols):
        d.text((LABEL_W + j * CELL_W + 8, HEAD_H + 4), name, fill=(23, 32, 51), font=font(16))
    for i, core in enumerate(cores):
        y0 = HEAD_H * 2 + i * (CELL_H + HEAD_H)
        meta = info[core]
        for k, line in enumerate((core[-10:], str(meta["asset_category"])[:28], f"GT: {meta['defect_gt']}")):
            d.text((8, y0 + 8 + 22 * k), line, fill=(23, 32, 51), font=font(15))
        with Image.open(targets[core]) as h:
            sheet.paste(fit(draw_marks(h, boxes.get(core), None, None), CELL_W - 6, CELL_H), (LABEL_W, y0))
        for j, expert in enumerate(experts, 1):
            key = (core, expert)
            x0 = LABEL_W + j * CELL_W
            if key not in overlays or not overlays[key].is_file():
                d.text((x0 + 8, y0 + CELL_H // 2), "missing", fill=(180, 84, 26), font=font(15))
                continue
            m = marks.get(key, {})
            with Image.open(overlays[key]) as h:
                sheet.paste(fit(draw_marks(h, boxes.get(core), m.get("roi"), m.get("peak")), CELL_W - 6, CELL_H), (x0, y0))
            d.text((x0 + 4, y0 + CELL_H + 4), m.get("caption", ""), fill=(63, 71, 86), font=font(14))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main(argv: Sequence[str] | None = None, *, load_qa: Callable[[Path], list[dict]] = v2.load_gold_qa_v2) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True, help="merged_evidence/roi_manifest_all.csv")
    p.add_argument("--artifact-root", type=Path, default=None)
    p.add_argument("--sup-zip", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--experts", nargs="+", default=list(EXPERTS))
    p.add_argument("--rows-per-grid", type=int, default=5)
    p.add_argument("--n-normal", type=int, default=5)
    p.add_argument("--n-other", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    import numpy as np
    from PIL import Image

    rows = load_qa(args.qa_dir)
    q1 = {str(r["core_id"]): r for r in rows if r.get("question_type") == Q1}
    boxes: dict[str, list] = {}
    for r in rows:
        gt = r.get("localization_gt") or {}
        if r.get("question_type") == Q3 and not gt.get("empty_mask") and gt.get("normalized_bboxes_xywh"):
            boxes[str(r["core_id"])] = [[float(v) for v in b] for b in gt["normalized_bboxes_xywh"]]
    manifest = [r for r in read_csv(args.manifest) if r.get("expert") in args.experts and str(r.get("query_id")) in q1]
    index = {(str(r["query_id"]), str(r["expert"])): r for r in manifest}
    missing_rows = [(c, e) for c in q1 for e in args.experts if (c, e) not in index]
    if missing_rows:
        raise RuntimeError(f"manifest misses {len(missing_rows)} (core, expert) pairs, e.g. {missing_rows[:2]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlays = {k: resolve(args.manifest, v["overlay_path"], args.artifact_root) for k, v in index.items()}
    stats: dict[tuple[str, str], dict] = {}
    marks: dict[tuple[str, str], dict] = {}
    missing_maps: list[str] = []
    for n, (key, row) in enumerate(sorted(index.items()), 1):
        core, expert = key
        map_path = resolve(args.manifest, row.get("anomaly_map_path", ""), args.artifact_root)
        s: dict[str, float | None] = {f: number(row.get(f)) for f in ("model_pred_score", "anomaly_score_top1pct")}
        if row.get("anomaly_map_path") and map_path.is_file():
            s.update(map_stats(np.load(map_path, allow_pickle=False), boxes.get(core)))
        else:
            missing_maps.append(f"{expert}/{core}")
        roi = None
        if overlays[key].is_file() and all(number(row.get(k)) is not None for k in ("roi_left", "roi_top", "roi_right", "roi_bottom")):
            with Image.open(overlays[key]) as h:
                ow, oh = h.size
            roi = (float(row["roi_left"]) / ow, float(row["roi_top"]) / oh,
                   float(row["roi_right"]) / ow, float(row["roi_bottom"]) / oh)
            s.update({"roi_cx": (roi[0] + roi[2]) / 2, "roi_cy": (roi[1] + roi[3]) / 2})
        stats[key] = s
        peak = (s["peak_x"], s["peak_y"]) if "peak_x" in s else None
        caption = ""
        if core in boxes and peak:
            caption = f"peak in GT: {'yes' if inside(peak[0], peak[1], boxes[core]) else 'no'}"
        elif s.get("hot_area_frac") is not None:
            caption = f"hot area {s['hot_area_frac']:.0%}"
        marks[key] = {"roi": roi, "peak": peak, "caption": caption}
        if n % 200 == 0 or n == len(index):
            print(f"maps {n}/{len(index)} (missing {len(missing_maps)})", flush=True)

    # detection: global AUROC and asset-macro AUROC per score field
    cores = sorted(q1)
    labels = {c: str(q1[c].get("anomaly_gt")) == "anomalous" for c in cores}
    detection = []
    for expert in args.experts:
        for field in SCORE_FIELDS:
            have = [c for c in cores if stats[(c, expert)].get(field) is not None]
            if not have:
                continue
            g = auroc([stats[(c, expert)][field] for c in have], [labels[c] for c in have])
            per_asset = defaultdict(list)
            for c in have:
                per_asset[str(q1[c]["asset_category"])].append(c)
            macro = [a for a in (auroc([stats[(c, expert)][field] for c in cs], [labels[c] for c in cs])
                                 for cs in per_asset.values()) if a is not None]
            detection.append({"expert": expert, "score_field": field, "images": len(have),
                              "anomalous": sum(labels[c] for c in have), "normal": sum(not labels[c] for c in have),
                              "auroc_global": round(g, 4) if g is not None else "",
                              "auroc_asset_macro": round(sum(macro) / len(macro), 4) if macro else "",
                              "assets_with_both_classes": len(macro)})

    # localization on GT-box images
    loc_cores = sorted(c for c in boxes if c in q1)
    by_image, localization = [], []
    for expert in list(args.experts) + ["image_centre"]:
        hits_peak, hits_roi, ious, ratios = [], [], [], []
        for core in loc_cores:
            b = boxes[core]
            if expert == "image_centre":
                rec = {"peak_in_gt": inside(0.5, 0.5, b)}
            else:
                s = stats[(core, expert)]
                rec = {}
                if "peak_x" in s:
                    rec["peak_in_gt"] = inside(s["peak_x"], s["peak_y"], b)
                    rec["heat_in_box_ratio"] = round(s["heat_in_box_ratio"], 4)
                    rec["gt_area_frac"] = round(s["gt_area_frac"], 4)
                if "roi_cx" in s:
                    roi = marks[(core, expert)]["roi"]
                    rec["roi_centre_in_gt"] = inside(s["roi_cx"], s["roi_cy"], b)
                    rec["roi_iou"] = round(iou(roi, union_box(b)), 4)
            by_image.append({"core_id": core, "pointer": expert, **rec})
            if "peak_in_gt" in rec:
                hits_peak.append(rec["peak_in_gt"])
            if "roi_centre_in_gt" in rec:
                hits_roi.append(rec["roi_centre_in_gt"])
                ious.append(rec["roi_iou"])
            if "heat_in_box_ratio" in rec:
                ratios.append(rec["heat_in_box_ratio"])
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else ""  # noqa: E731
        localization.append({"pointer": expert, "gt_images": len(loc_cores), "peak_in_gt_rate": mean(hits_peak),
                             "roi_centre_in_gt_rate": mean(hits_roi), "mean_roi_iou": mean(ious),
                             "mean_heat_in_box_ratio": mean(ratios),
                             "note": "point at the image centre (trivial baseline)" if expert == "image_centre"
                             else "ratio > 1 = heat concentrated on the GT box; 1 = spread like uniform heat"})

    heat = []
    for expert in args.experts:
        for label in ("normal", "anomalous"):
            vals = [stats[(c, expert)] for c in cores if labels[c] == (label == "anomalous")
                    and "hot_area_frac" in stats[(c, expert)]]
            if vals:
                heat.append({"expert": expert, "label": label, "images": len(vals),
                             f"mean_hot_area_frac_ge_{HOT}": round(sum(v["hot_area_frac"] for v in vals) / len(vals), 4),
                             "mean_median_heat": round(sum(v["median_heat"] for v in vals) / len(vals), 4)})

    write_csv(args.out_dir / "detection.csv", detection)
    write_csv(args.out_dir / "localization.csv", localization)
    write_csv(args.out_dir / "localization_by_image.csv", by_image)
    write_csv(args.out_dir / "heat_on_normals.csv", heat)

    # grids
    rng = random.Random(args.seed)
    normal_groups, other_groups = defaultdict(list), defaultdict(list)
    for c in cores:
        if not labels[c]:
            normal_groups[str(q1[c]["asset_category"])].append(c)
        elif c not in boxes:
            other_groups[str(q1[c].get("defect_gt"))].append(c)
    normals = pick(rng, normal_groups, args.n_normal)
    others = pick(rng, other_groups, args.n_other)
    shown = loc_cores + normals + others
    targets = base.prepare_images([q1[c] for c in shown], args.sup_zip, args.out_dir / "_images")
    tpath = {c: targets[str(q1[c]["image"])] for c in shown}
    grids = []
    step = max(1, args.rows_per_grid)
    for k in range(0, len(loc_cores), step):
        name = f"grid_defect_{k // step + 1}.png"
        make_grid(args.out_dir / name, f"Gold defect images with an InsPLAD-seg box ({k + 1}-{min(k + step, len(loc_cores))} "
                  f"of {len(loc_cores)}): red = GT box, yellow = expert ROI, cyan = heat peak",
                  loc_cores[k:k + step], q1, tpath, overlays, marks, args.experts, boxes)
        grids.append(name)
    for name, sel, title in (("grid_normal.png", normals, "Gold NORMAL images: every map is stretched per image, so a hot spot always appears"),
                             ("grid_other_defects.png", others, "Gold anomalous images without a seg box (bird nest / missing cap)")):
        if sel:
            make_grid(args.out_dir / name, title, sel, q1, tpath, overlays, marks, args.experts, boxes)
            grids.append(name)

    status = {"status": "complete" if not missing_maps else "partial", "images": len(cores),
              "experts": list(args.experts), "missing_anomaly_maps": len(missing_maps),
              "missing_examples": missing_maps[:10], "gt_box_images": len(loc_cores), "grids": grids}
    (args.out_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "protocol": "gold_v2_expert_heatmap_check",
        "new_lvlm_inference": False,
        "inputs": {"manifest": "vision expert full merged_evidence/roi_manifest_all.csv (pure heatmap overlays)",
                   "qa": "gold_v2_final master + localization QA", "gt_boxes": "InsPLAD-seg defect instance boxes"},
        "score_fields": {"model_pred_score": "expert's own image score when present",
                         "anomaly_score_top1pct": "mean of the top 1% of the per-image-normalised map (prompt fallback)",
                         "raw_map_max": "max of the raw anomaly map", "raw_map_top1pct": "mean of the top 1% raw map values"},
        "prompt_score_rule": "LVLM prompts used model_pred_score, else anomaly_score_top1pct, as a within-asset percentile",
        "hot_threshold": HOT,
        "normalisation": "per image, 1st-99.5th percentile -> [0, 1] (insplad_benchmark.vision_expert.normalize_anomaly_map)",
        "localization_subset": "gold images with a non-empty InsPLAD-seg box (all lightning-rod-suspension rust, all 'center')",
        "grids": {"seed": args.seed, "n_normal": args.n_normal, "n_other": args.n_other},
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
