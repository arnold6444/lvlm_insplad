#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFECT_TEXT = {
    "normal": "Normal / no defect.",
    "rust": "Rust / corrosion.",
    "missing-cap": "Missing cap / missing insulator component.",
    "bird-nest": "Bird nest / foreign nesting material.",
}
REGION_TEXT = {
    "upper-left": "Upper-left region.",
    "upper": "Upper-center region.",
    "upper-right": "Upper-right region.",
    "left": "Middle-left region.",
    "center": "Center region.",
    "right": "Middle-right region.",
    "lower-left": "Lower-left region.",
    "lower": "Lower-center region.",
    "lower-right": "Lower-right region.",
    "none": "No localized defect region is present.",
}
EXPECTED_ASSETS = {
    "glass-insulator",
    "lightning-rod-suspension",
    "polymer-insulator-upper-shackle",
    "vari-grip",
    "yoke-suspension",
}


def stable_random(seed_text: str) -> random.Random:
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = [dict(r) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def balanced_gold_sample(
    rows: Sequence[Mapping[str, str]], n: int, *, seed: int = 42
) -> list[dict[str, str]]:
    rows = [dict(r) for r in rows]
    anomalous = [r for r in rows if r["anomaly_answer"] == "anomalous"]
    if len(anomalous) > n:
        raise ValueError(f"anomalous rows={len(anomalous)} exceeds target={n}")

    assets = sorted({r["asset_category"] for r in rows})
    normal_by_asset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["anomaly_answer"] == "normal":
            normal_by_asset[row["asset_category"]].append(row)

    selected = list(anomalous)
    selected_counts = Counter(r["asset_category"] for r in selected)
    for asset in assets:
        stable_random(f"{seed}:normal:{asset}").shuffle(normal_by_asset[asset])

    while len(selected) < n:
        available = [a for a in assets if normal_by_asset[a]]
        if not available:
            break
        asset = min(available, key=lambda a: (selected_counts[a], a))
        selected.append(normal_by_asset[asset].pop())
        selected_counts[asset] += 1

    if len(selected) != n:
        raise ValueError(f"requested {n} rows but only selected {len(selected)}")
    stable_random(f"{seed}:final").shuffle(selected)
    return selected


def make_options(
    texts: Sequence[str], correct_text: str, *, seed_text: str
) -> tuple[dict[str, str], str]:
    values = list(dict.fromkeys(texts))
    if correct_text not in values:
        raise ValueError("correct_text missing from choices")
    stable_random(seed_text).shuffle(values)
    options = {LETTERS[i]: value for i, value in enumerate(values)}
    answer = next(k for k, value in options.items() if value == correct_text)
    return options, answer


def build_master_rows(
    aligned_rows: Sequence[Mapping[str, str]], n: int, *, seed: int
) -> tuple[list[dict], list[dict]]:
    selected = balanced_gold_sample(aligned_rows, n, seed=seed)
    images: list[dict] = []
    questions: list[dict] = []

    for row in selected:
        base = {
            "core_id": row["core_id"],
            "asset_category": row["asset_category"],
            "image": row["supervised_image"],
            "unsupervised_image": row["unsupervised_image"],
            "normal_reference_image": row["reference_image"],
            "anomaly_gt": row["anomaly_answer"],
            "defect_gt": row["defect_class"],
            "anomaly_gt_basis": "verified_aligned_source_label",
            "defect_gt_basis": "verified_supervised_source_label",
            "exact_sha256_match": True,
            "binary_gt_agreement": True,
            "sample_policy": "all_anomalies_then_asset_balanced_normals",
        }
        images.append(base)

        anomaly_text = "Normal." if base["anomaly_gt"] == "normal" else "Anomalous."
        options, answer = make_options(
            ["Normal.", "Anomalous."],
            anomaly_text,
            seed_text=f"{seed}:{base['core_id']}:anomaly",
        )
        questions.append(
            {
                **base,
                "id": f"{base['core_id']}-q1",
                "task": "explainable_mcqa",
                "question_type": "anomaly_detection",
                "question": "Is the target power-line asset normal or anomalous?",
                "options": options,
                "answer": answer,
                "answer_text": anomaly_text,
                "gt_basis": "verified_aligned_source_label",
                "review_status": "gold",
            }
        )

        defect_text = DEFECT_TEXT[base["defect_gt"]]
        options, answer = make_options(
            list(DEFECT_TEXT.values()),
            defect_text,
            seed_text=f"{seed}:{base['core_id']}:defect",
        )
        questions.append(
            {
                **base,
                "id": f"{base['core_id']}-q2",
                "task": "explainable_mcqa",
                "question_type": "defect_classification",
                "question": "Which condition best describes the target power-line asset?",
                "options": options,
                "answer": answer,
                "answer_text": defect_text,
                "gt_basis": "verified_supervised_source_label",
                "review_status": "gold",
            }
        )
    return images, questions


def canonical_filename_key(name: str) -> str:
    name = PurePosixPath(str(name).replace("\\", "/")).name
    name = re.sub(r"\.rf\.[^.]+(?=\.[^.]+$)", "", name, flags=re.I)
    stem = Path(name).stem
    stem = re.sub(r"_(jpg|jpeg|png|bmp|webp|tif|tiff)$", "", stem, flags=re.I)
    stem = (
        unicodedata.normalize("NFKD", stem)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def asset_hint_from_category_names(names: Sequence[str]) -> str | None:
    normalized = [str(x).lower().replace("_", "-") for x in names]
    for asset in sorted(EXPECTED_ASSETS, key=len, reverse=True):
        if any(asset in name for name in normalized):
            return asset
    return None


def is_defect_category(category_name: str) -> bool:
    name = category_name.lower().replace("_", "-")
    return any(
        token in name
        for token in ["rust", "missing-cap", "missingcap", "bird-nest", "birdnest"]
    )


def expected_defect_match(source_defect: str, category_name: str) -> bool:
    source = source_defect.lower()
    name = category_name.lower().replace("_", "-")
    if source == "rust":
        return "rust" in name
    if source == "missing-cap":
        return "missing-cap" in name or "missingcap" in name
    if source == "bird-nest":
        return "bird-nest" in name or "birdnest" in name or name == "nest"
    return False


def norm_bbox_xywh(
    bbox: Sequence[float], width: float, height: float
) -> list[float]:
    x, y, w, h = map(float, bbox)
    return [
        max(0.0, min(1.0, x / width)),
        max(0.0, min(1.0, y / height)),
        max(0.0, min(1.0, w / width)),
        max(0.0, min(1.0, h / height)),
    ]


def normalize_polygon(
    segmentation: object, width: float, height: float
) -> list[list[list[float]]]:
    out: list[list[list[float]]] = []
    if not isinstance(segmentation, list):
        return out
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2:
            continue
        points = []
        for i in range(0, len(poly), 2):
            points.append(
                [
                    max(0.0, min(1.0, float(poly[i]) / width)),
                    max(0.0, min(1.0, float(poly[i + 1]) / height)),
                ]
            )
        out.append(points)
    return out


def coarse_region_from_bboxes(bboxes: Sequence[Sequence[float]]) -> str:
    if not bboxes:
        return "none"
    total_area = cx_sum = cy_sum = 0.0
    for x, y, w, h in bboxes:
        area = max(float(w) * float(h), 1e-9)
        cx_sum += (float(x) + float(w) / 2) * area
        cy_sum += (float(y) + float(h) / 2) * area
        total_area += area
    cx, cy = cx_sum / total_area, cy_sum / total_area
    x_name = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    y_name = "upper" if cy < 1 / 3 else ("lower" if cy > 2 / 3 else "middle")
    if x_name == "center" and y_name == "middle":
        return "center"
    if y_name == "middle":
        return x_name
    if x_name == "center":
        return y_name
    return f"{y_name}-{x_name}"


def load_coco(seg_root: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    images: dict[str, dict] = {}
    annotations: dict[str, list[dict]] = defaultdict(list)
    coco_files = sorted(seg_root.rglob("_annotations.coco.json"))
    if not coco_files:
        raise FileNotFoundError(f"no _annotations.coco.json under {seg_root}")

    for coco_path in coco_files:
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        categories = {int(c["id"]): c["name"] for c in data.get("categories", [])}
        for image in data.get("images", []):
            key = f"{coco_path.parent}:{image['id']}"
            images[key] = {
                **image,
                "_coco_path": str(coco_path),
                "_image_path": str(coco_path.parent / image["file_name"]),
            }
        for ann in data.get("annotations", []):
            key = f"{coco_path.parent}:{ann['image_id']}"
            item = dict(ann)
            item["_category_name"] = categories.get(
                int(ann["category_id"]), str(ann["category_id"])
            )
            annotations[key].append(item)
    return images, annotations


def build_localization_rows(
    aligned_rows: Sequence[Mapping[str, str]],
    master_images: Sequence[Mapping[str, object]],
    master_qa: Sequence[Mapping[str, object]],
    seg_root: Path,
    *,
    seed: int,
    seg_version: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    seg_images, seg_annotations = load_coco(seg_root)
    core_index: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in aligned_rows:
        key = canonical_filename_key(PurePosixPath(row["supervised_image"]).name)
        core_index[key].append(row)

    master_ids = {str(r["core_id"]) for r in master_images}
    master_by_id = {str(r["core_id"]): dict(r) for r in master_images}
    qa_by_core: dict[str, list[dict]] = defaultdict(list)
    for row in master_qa:
        qa_by_core[str(row["core_id"])].append(dict(row))

    audit: list[dict] = []
    localization_images: list[dict] = []

    for seg_key, image in seg_images.items():
        canon = canonical_filename_key(image["file_name"])
        candidates = list(core_index.get(canon, []))
        anns = seg_annotations.get(seg_key, [])
        category_names = [str(a["_category_name"]) for a in anns]
        asset_hint = asset_hint_from_category_names(category_names)

        if len(candidates) > 1 and asset_hint:
            candidates = [r for r in candidates if r["asset_category"] == asset_hint]

        if not candidates:
            audit.append(
                {
                    "seg_file": image["file_name"],
                    "canonical_key": canon,
                    "status": "no_aligned_core_match",
                    "asset_hint": asset_hint or "",
                    "categories": "|".join(sorted(set(category_names))),
                }
            )
            continue
        if len(candidates) != 1:
            audit.append(
                {
                    "seg_file": image["file_name"],
                    "canonical_key": canon,
                    "status": "ambiguous_aligned_core_match",
                    "candidate_count": len(candidates),
                    "asset_hint": asset_hint or "",
                    "categories": "|".join(sorted(set(category_names))),
                }
            )
            continue

        core = candidates[0]
        core_id = core["core_id"]
        expected = core["defect_class"]
        defect_anns = [a for a in anns if is_defect_category(a["_category_name"])]

        if expected == "normal":
            selected_defect_anns: list[dict] = []
            usable = not defect_anns
            reason = (
                "normal_empty_defect_mask"
                if usable
                else "normal_label_but_defect_mask_present"
            )
        else:
            selected_defect_anns = [
                a
                for a in defect_anns
                if expected_defect_match(expected, a["_category_name"])
            ]
            usable = bool(selected_defect_anns)
            reason = (
                "matched_expected_defect_mask"
                if usable
                else "missing_expected_defect_mask"
            )

        in_master = core_id in master_ids
        audit.append(
            {
                "seg_file": image["file_name"],
                "canonical_key": canon,
                "status": reason,
                "core_id": core_id,
                "asset_category": core["asset_category"],
                "anomaly_gt": core["anomaly_answer"],
                "defect_gt": expected,
                "in_master_sample": in_master,
                "asset_hint": asset_hint or "",
                "categories": "|".join(sorted(set(category_names))),
                "expected_mask_instances": len(selected_defect_anns),
            }
        )
        if not (usable and in_master):
            continue

        width, height = float(image["width"]), float(image["height"])
        bboxes = [
            norm_bbox_xywh(a["bbox"], width, height)
            for a in selected_defect_anns
            if a.get("bbox")
        ]
        polygons: list[list[list[float]]] = []
        for ann in selected_defect_anns:
            polygons.extend(normalize_polygon(ann.get("segmentation"), width, height))

        region = coarse_region_from_bboxes(bboxes)
        base = master_by_id[core_id]
        localization_images.append(
            {
                **base,
                "segmentation_source": {
                    "dataset": "andreluizbvs/insplad-seg",
                    "version": seg_version,
                    "format": "coco-segmentation",
                    "roboflow_file": image["file_name"],
                    "coco_file": image["_coco_path"],
                    "preprocessed_width": int(width),
                    "preprocessed_height": int(height),
                },
                "localization_gt": {
                    "defect_class": expected,
                    "normalized_bboxes_xywh": bboxes,
                    "normalized_polygons_xy": polygons,
                    "coarse_region": region,
                    "empty_mask": expected == "normal",
                },
                "localization_gt_basis": (
                    "insplad_seg_instance_annotation"
                    if expected != "normal"
                    else "verified_normal_source_label_plus_insplad_seg_image"
                ),
                "localization_review_status": "gold",
            }
        )

    localization_qa: list[dict] = []
    for loc in localization_images:
        core_id = str(loc["core_id"])
        for question in qa_by_core[core_id]:
            localization_qa.append(
                {
                    **question,
                    "localization_gt": loc["localization_gt"],
                    "localization_gt_basis": loc["localization_gt_basis"],
                }
            )

        region = loc["localization_gt"]["coarse_region"]
        correct_text = REGION_TEXT[region]
        if region == "none":
            distractor_keys = ["center", "upper-left", "lower-right"]
        else:
            pool = [k for k in REGION_TEXT if k not in {region, "none"}]
            stable_random(f"{seed}:{core_id}:localization-distractors").shuffle(pool)
            distractor_keys = pool[:2] + ["none"]
        options, answer = make_options(
            [correct_text] + [REGION_TEXT[k] for k in distractor_keys],
            correct_text,
            seed_text=f"{seed}:{core_id}:localization-options",
        )
        q_base = {
            k: v
            for k, v in loc.items()
            if k not in {"segmentation_source", "localization_gt"}
        }
        localization_qa.append(
            {
                **q_base,
                "id": f"{core_id}-q3",
                "task": "explainable_mcqa",
                "question_type": "defect_localization",
                "question": "Where is the defect region located in the target image?",
                "options": options,
                "answer": answer,
                "answer_text": correct_text,
                "localization_gt": loc["localization_gt"],
                "segmentation_source": loc["segmentation_source"],
                "gt_basis": loc["localization_gt_basis"],
                "review_status": "gold",
            }
        )
    return localization_images, localization_qa, audit


def count_rows(images: Sequence[Mapping[str, object]]) -> list[dict]:
    out = []
    for asset in sorted({str(r["asset_category"]) for r in images}):
        rows = [r for r in images if r["asset_category"] == asset]
        out.append(
            {
                "asset_category": asset,
                "total": len(rows),
                "normal": sum(r["anomaly_gt"] == "normal" for r in rows),
                "anomalous": sum(r["anomaly_gt"] == "anomalous" for r in rows),
                "defect_counts": json.dumps(
                    dict(sorted(Counter(str(r["defect_gt"]) for r in rows).items())),
                    ensure_ascii=False,
                ),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-core-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--master-count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seg-root", type=Path)
    parser.add_argument("--seg-version", type=int, default=7)
    args = parser.parse_args()

    aligned_rows = read_csv(args.aligned_core_csv)
    if len(aligned_rows) != 1378:
        raise ValueError(f"expected 1378 Aligned Core rows, got {len(aligned_rows)}")
    assets = {r["asset_category"] for r in aligned_rows}
    if assets != EXPECTED_ASSETS:
        raise ValueError(f"unexpected Aligned Core assets: {sorted(assets)}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    master_images, master_qa = build_master_rows(
        aligned_rows, args.master_count, seed=args.seed
    )
    write_jsonl(out / f"master_images_{len(master_images)}.jsonl", master_images)
    write_jsonl(out / f"master_qa_{len(master_qa)}.jsonl", master_qa)
    write_csv(out / "master_counts.csv", count_rows(master_images))

    localization_images: list[dict] = []
    localization_qa: list[dict] = []
    audit: list[dict] = []
    if args.seg_root:
        localization_images, localization_qa, audit = build_localization_rows(
            aligned_rows,
            master_images,
            master_qa,
            args.seg_root,
            seed=args.seed,
            seg_version=args.seg_version,
        )
        write_jsonl(out / "localization_images.jsonl", localization_images)
        write_jsonl(
            out / f"localization_qa_{len(localization_qa)}.jsonl", localization_qa
        )
        write_csv(out / "localization_match_audit.csv", audit)

    loc_by_core = {str(r["core_id"]): r for r in localization_images}
    write_csv(
        out / "manual_review.csv",
        [
            {
                "core_id": r["core_id"],
                "asset_category": r["asset_category"],
                "image": r["image"],
                "anomaly_gt": r["anomaly_gt"],
                "defect_gt": r["defect_gt"],
                "has_localization_gt": r["core_id"] in loc_by_core,
                "localization_region": (
                    loc_by_core[str(r["core_id"])]["localization_gt"]["coarse_region"]
                    if str(r["core_id"]) in loc_by_core
                    else ""
                ),
                "manual_review_status": "",
                "manual_notes": "",
            }
            for r in master_images
        ],
    )

    summary = {
        "aligned_core_images": len(aligned_rows),
        "master_images": len(master_images),
        "master_questions": len(master_qa),
        "master_question_types": ["anomaly_detection", "defect_classification"],
        "localization_source_available": bool(args.seg_root),
        "localization_images": len(localization_images),
        "localization_questions": len(localization_qa),
        "localization_question_types": (
            ["anomaly_detection", "defect_classification", "defect_localization"]
            if localization_images
            else []
        ),
        "sampling_policy": "all_anomalies_then_asset_balanced_normals",
        "seed": args.seed,
        "object_bbox_included": False,
        "description_analysis_gold_included": False,
    }
    (out / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    card = f"""# InsPLAD Explainable QA Gold v1

- Aligned Core source images: {len(aligned_rows)}
- Main images: {len(master_images)}
- Main QA: {len(master_qa)} (`anomaly_detection`, `defect_classification`)
- Localization images: {len(localization_images)}
- Localization QA: {len(localization_qa)}
- Q1/Q2 basis: verified InsPLAD source labels
- Q3 basis: official InsPLAD-seg instance annotation when matched
- Asset object bbox: excluded for cropped fault images
- Defect description/analysis: excluded from gold because original InsPLAD has no human-authored explanation GT
"""
    (out / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
