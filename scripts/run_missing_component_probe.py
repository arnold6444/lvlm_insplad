#!/usr/bin/env python3
"""Standalone supervised probe. Never writes QA, LVLM answers or old results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODEL = "facebook/dinov2-base"
REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
PROTOCOL = "missing_component_probe_v1"
FEATURE = "full RGB square padding (124,116,104), resize518, no center crop, L2-normalized CLS"
ASSET = "glass-insulator"
EXTENSIONS = {".jpg", ".jpeg", ".png"}
NORMAL = {"good", "normal"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty report: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def lock_json(path, value):
    if Path(path).exists():
        if json.loads(Path(path).read_text(encoding="utf-8")) != value:
            raise ValueError(f"locked input/config changed; use a NEW output directory: {path}")
    else:
        write_json(path, value)


def normalized(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError(f"unsafe archive path: {name}")
    return str(p)


def image_hashes(data):
    byte_sha = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        pixel_sha = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
    return byte_sha, pixel_sha


def entries(z):
    result = {}
    for info in z.infolist():
        name = normalized(info.filename)
        if not info.is_dir() and PurePosixPath(name).suffix.lower() in EXTENSIONS:
            if name in result:
                raise ValueError(f"duplicate ZIP member: {name}")
            result[name] = info
    return result


def read_targets(dev_qa, gold_csv):
    raw = [json.loads(x) for x in Path(dev_qa).read_text(encoding="utf-8").splitlines() if x.strip()]
    dev = {}
    for r in raw:
        name = normalized(r["image"])
        identity = (r["asset_category"], r["defect_gt"], r["anomaly_gt"])
        if name in dev and dev[name]["identity"] != identity:
            raise ValueError("conflicting dev labels")
        dev[name] = {"image": name, "identity": identity, "label": int(r["defect_gt"] == "missing-cap")}
    with Path(gold_csv).open(encoding="utf-8-sig", newline="") as f:
        gold = list(csv.DictReader(f))
    if not dev or not gold or any(r.get("quality_status") != "approve" for r in gold):
        raise ValueError("empty/unapproved dev or gold protection set")
    return dev, {normalized(r["image"]) for r in gold}


def prepare(sup_zip, unsup_zip, dev_qa, gold_csv, *, seed=42, minimum=10, strict=True):
    """No learned statistics here. All official evaluation images are exclusion-only."""
    dev, gold = read_targets(dev_qa, gold_csv)
    if strict and (len(dev) != 240 or len(gold) != 246):
        raise ValueError("expected frozen dev240 and gold246")
    if strict:
        for given, locked in [(dev_qa, ROOT / "results/latest/gold_v2_dev/dev_master_qa_480.jsonl"),
                              (gold_csv, ROOT / "results/latest/gold_v2_final/manual_review_final.csv")]:
            if sha_file(given) != sha_file(locked):
                raise ValueError("protection list differs from frozen committed dev/gold cohort")
    with ZipFile(sup_zip) as sup, ZipFile(unsup_zip) as unsup:
        sources = {"sup": sup, "unsup": unsup}
        index = {k: entries(v) for k, v in sources.items()}
        forbidden = [(k, n) for k, ix in index.items() for n in ix
                     if (k == "sup" and "/val/" in n) or (k == "unsup" and "/test/" in n)]
        counts = Counter(k for k, _ in forbidden)
        if strict and (counts["sup"] != 6417 or counts["unsup"] != 6481):
            raise ValueError(f"incomplete/wrong official archives: {dict(counts)}")
        for n in set(dev) | gold:
            if n not in index["sup"]:
                raise ValueError(f"protected target missing from supervised ZIP: {n}")
            forbidden.append(("sup", n))
        forbidden = sorted(set(forbidden))
        protected_names = {PurePosixPath(n).name for _, n in forbidden}
        protected_crc = {(index[k][n].file_size, index[k][n].CRC) for k, n in forbidden}
        # Decode every forbidden image, including other assets: renamed/re-encoded
        # identical pixels are blocked even when names, file size and CRC differ.
        byte_shas, pixel_shas = set(), set()
        for i, (k, n) in enumerate(forbidden):
            b, p = image_hashes(sources[k].read(n))
            byte_shas.add(b)
            pixel_shas.add(p)
            if (i + 1) % 500 == 0:
                print(f"protection hashes {i + 1}/{len(forbidden)}", flush=True)
        candidates, removed, seen = [], [], {}
        train_prefix = f"defect_supervised/{ASSET}/train/"
        for n, info in sorted(index["sup"].items()):
            if not n.startswith(train_prefix):
                continue
            cls = PurePosixPath(n).parent.name
            if cls not in NORMAL | {"missing-cap"}:
                raise ValueError(f"unexpected glass training label: {cls}")
            y = int(cls == "missing-cap")
            b, p = image_hashes(sup.read(n))
            reason = None
            if PurePosixPath(n).name in protected_names:
                reason = "protected_name"
            elif (info.file_size, info.CRC) in protected_crc:
                reason = "protected_size_crc"
            elif b in byte_shas:
                reason = "protected_byte_sha256"
            elif p in pixel_shas:
                reason = "protected_rgb_sha256"
            if reason:
                removed.append({"image": n, "label": y, "reason": reason})
                continue
            if p in seen:
                if seen[p] != y:
                    raise ValueError("conflicting training labels for identical pixels")
                removed.append({"image": n, "label": y, "reason": "duplicate_training_pixels"})
                continue
            seen[p] = y
            match = re.search(r"(?<!\d)(\d{2}-\d{2}-\d{4})(?!\d)", PurePosixPath(n).name)
            if not match:
                raise ValueError(f"missing capture date: {n}")
            candidates.append({"image": n, "label": y, "date": match[1], "byte_sha256": b,
                               "rgb_sha256": p, "size": info.file_size, "crc32": f"{info.CRC:08x}"})
        candidates = assign_splits(candidates, seed, minimum)
        dev_rows = []
        for n, r in sorted(dev.items()):
            if r["identity"][0] != ASSET:
                continue
            if r["identity"][1] not in {"normal", "missing-cap"}:
                raise ValueError("unexpected dev glass label")
            b, p = image_hashes(sup.read(n))
            if p in seen:
                raise ValueError("dev/train RGB overlap")
            dev_rows.append({"image": n, "label": r["label"], "split": "dev_report_only",
                             "byte_sha256": b, "rgb_sha256": p})
        if strict and Counter(x["label"] for x in dev_rows) != {0: 25, 1: 25}:
            raise ValueError("expected dev glass 25 normal + 25 missing-cap")
    audit = {"protocol": PROTOCOL, "protected_official_counts": dict(counts),
             "protected_targets": {"dev": len(dev), "gold": len(gold)},
             "protected_images_hashed": len(forbidden), "eligible_train_images": len(candidates),
             "excluded": dict(Counter(r["reason"] for r in removed)),
             "counts": dict(Counter(f'{r["split"]}:{r["label"]}' for r in candidates)),
             "exact_byte_overlap": 0, "exact_rgb_overlap": 0, "gold_scored": False,
             "internal_date_overlap": 0,
             "limitation": "Exact bytes/RGB and names excluded; near-duplicate similarity is not certified. Dev is exploratory and may share capture dates with train."}
    return candidates, dev_rows, removed, audit


def assign_splits(rows, seed, minimum):
    dates = sorted({r["date"] for r in rows})
    if len(dates) < 6:
        raise ValueError("need >=6 capture dates for independent fit/calibration/internal_test")
    n = max(1, round(len(dates) * .2))
    rng = np.random.default_rng(seed)
    # Feasibility uses TRAIN class counts only, never features/scores/dev/gold labels.
    for attempt in range(1000):
        perm = rng.permutation(dates).tolist()
        mapping = {d: "internal_test" if i < n else "calibration" if i < 2*n else "fit"
                   for i, d in enumerate(perm)}
        counts = Counter((mapping[r["date"]], r["label"]) for r in rows)
        if all(counts[s, y] >= minimum for s in ("fit", "calibration", "internal_test") for y in (0, 1)):
            return [dict(r, split=mapping[r["date"]]) for r in rows]
    raise ValueError("insufficient class support in date-disjoint splits")


class DinoFeatures:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
            raise RuntimeError("Use an L4/A100 GPU with >=20 GB VRAM")
        self.torch = torch
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION, backend="pil")
        self.model = AutoModel.from_pretrained(MODEL, revision=REVISION).to("cuda").eval()
        self.model.requires_grad_(False)

    def __call__(self, image):
        # Preserve the full asset. A center crop can remove a missing end component.
        side = max(image.size)
        padded = Image.new("RGB", (side, side), (124, 116, 104))
        padded.paste(image, ((side-image.width)//2, (side-image.height)//2))
        inputs = self.processor(images=padded, return_tensors="pt", do_center_crop=False,
                                size={"height": 518, "width": 518})
        with self.torch.inference_mode():
            out = self.model(**{k: v.to("cuda") for k, v in inputs.items()}).last_hidden_state[:, 0]
        return out[0].float().cpu().numpy()


def features(rows, sup_zip, out, identity, extractor=None):
    """One content-addressed feature per image; successful features survive restarts."""
    cache = out / "features"
    cache.mkdir(exist_ok=True)
    lock_json(cache / "identity.json", identity)
    vectors = []
    with ZipFile(sup_zip) as z:
        for i, r in enumerate(rows):
            p = cache / f'{r["byte_sha256"]}.npy'
            check = p.with_suffix(".sha256")
            if p.exists() and check.exists():
                if sha_file(p) != check.read_text(encoding="utf-8").strip():
                    raise ValueError("feature cache checksum mismatch")
                x = np.load(p, allow_pickle=False)
            else:
                if extractor is None:
                    extractor = DinoFeatures()
                data = z.read(r["image"])
                if hashlib.sha256(data).hexdigest() != r["byte_sha256"]:
                    raise ValueError("source changed since audit")
                with Image.open(io.BytesIO(data)) as im:
                    x = np.asarray(extractor(ImageOps.exif_transpose(im).convert("RGB")), dtype=np.float32)
                if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all() or np.linalg.norm(x) <= 0:
                    raise ValueError("invalid extracted feature")
                x = x / np.linalg.norm(x)
                with p.with_suffix(".tmp").open("wb") as f:
                    np.save(f, x, allow_pickle=False)
                os.replace(p.with_suffix(".tmp"), p)
                check.write_text(sha_file(p), encoding="utf-8")
            if x.ndim != 1 or not np.isfinite(x).all() or np.linalg.norm(x) <= 0:
                raise ValueError("invalid cached feature")
            vectors.append(x)
            if (i + 1) % 20 == 0 or i + 1 == len(rows):
                print(f"features {i + 1}/{len(rows)}", flush=True)
                write_json(out / "feature_progress.json", {"complete": i + 1, "total": len(rows)})
    return np.stack(vectors)


def threshold_from_calibration(y, scores, fpr_cap=.05):
    if set(np.asarray(y).tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("invalid calibration")
    unique = np.unique(scores)
    # Midpoints avoid putting the decision boundary exactly on a calibration
    # positive; the above-maximum candidate permits a valid all-normal outcome.
    choices = np.r_[unique[0], (unique[:-1] + unique[1:]) / 2,
                    np.nextafter(unique[-1], np.inf)]
    feasible = []
    for t in choices:
        pred = scores >= t
        fpr = float(pred[y == 0].mean())
        recall = float(pred[y == 1].mean())
        if fpr <= fpr_cap:
            feasible.append(((1 - fpr + recall) / 2, float(t)))
    return max(feasible)[1]  # deterministic, higher threshold breaks BAcc ties


def train_head(x, rows, out, fpr_cap=.05):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    x = np.asarray(x, dtype=np.float64)
    fit = np.array([r["split"] == "fit" for r in rows])
    cal = np.array([r["split"] == "calibration" for r in rows])
    y = np.array([r["label"] for r in rows])
    scaler = StandardScaler().fit(x[fit])
    # Fixed hyperparameters. No internal-test/dev/gold selection.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000,
                                   random_state=42).fit(scaler.transform(x[fit]), y[fit])
    threshold = threshold_from_calibration(y[cal], model.predict_proba(scaler.transform(x[cal]))[:, 1], fpr_cap)
    artifact = {"protocol": PROTOCOL, "model_id": MODEL, "model_revision": REVISION,
                "feature": FEATURE, "fit_calibration_manifest_digest": digest([r for r in rows if r["split"] in {"fit", "calibration"}]),
                "scope": "fit_and_calibration_only; evaluation completion required before use",
                "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                "threshold": threshold, "threshold_rule": "max calibration BAcc with normal FPR <= cap; higher threshold tie",
                "calibration_fpr_cap": fpr_cap, "fit_images": int(fit.sum()),
                "calibration_images": int(cal.sum()), "C": 1.0,
                "class_weight": "balanced", "selection_uses": ["fit", "calibration"],
                "evaluation_uses": ["internal_test", "dev_report_only"], "gold_scored": False}
    # Artifact is immutable, and written BEFORE any internal-test/dev prediction.
    lock_json(out / "frozen_detector.json", artifact)
    return artifact


def predict(artifact, x):
    z = ((x - np.array(artifact["mean"])) / np.array(artifact["scale"])) @ np.array(artifact["coef"]) + artifact["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def report(rows, scores, threshold, split):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.array([r["label"] for r in rows])
    p = scores >= threshold
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    tn, fn = int((~p & (y == 0)).sum()), int((~p & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn)
    return {"split": split, "asset": ASSET, "images": len(y), "normal": tn+fp, "missing_cap": tp+fn,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "balanced_accuracy": (recall + tn/(tn+fp))/2,
            "normal_fpr": fp/(tn+fp), "missing_cap_fnr": fn/(tp+fn),
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)), "threshold": threshold,
            "invalid_output_rate": 0., "always_normal_balanced_accuracy": .5,
            "always_missing_balanced_accuracy": .5}


def execute(args, *, extractor=None, strict=True):
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not (out / "input_lock.json").is_file():
        raise ValueError("refusing nonempty output without this experiment's input lock; use a NEW directory")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    input_identity = {"protocol": PROTOCOL, "script_sha256": sha_file(__file__),
                      "inputs": {k: sha_file(getattr(args, k)) for k in ("sup_zip", "unsup_zip", "dev_qa", "gold_csv")},
                      "model_id": MODEL, "model_revision": REVISION,
                      "feature": FEATURE,
                      "seed": args.seed, "minimum_per_class": args.minimum_per_class, "fpr_cap": .05}
    lock_json(out / "input_lock.json", input_identity)
    write_json(out / "run_status.json", {"complete": False, "phase": "leakage_audit"})
    write_json(out / "run_metadata.json", {**input_identity, "complete": False,
                                            "condition": "supervised_assisted_standalone", "gold_scored": False})
    try:
        rows, dev, removed, audit = prepare(args.sup_zip, args.unsup_zip, args.dev_qa, args.gold_csv,
                                            seed=args.seed, minimum=args.minimum_per_class, strict=strict)
        lock_json(out / "split_manifest.json", rows)
        lock_json(out / "dev_report_manifest.json", dev)
        write_json(out / "leakage_audit.json", audit)
        write_csv(out / "image_manifest.csv", rows)
        if removed:
            write_csv(out / "excluded_images.csv", removed)
        if args.audit_only:
            write_json(out / "run_status.json", {"complete": False, "audit_complete": True, "phase": "audit_only_no_training"})
            return audit
        x = features(rows, args.sup_zip, out, input_identity, extractor)
        write_json(out / "run_status.json", {"complete": False, "phase": "training"})
        artifact = train_head(x, rows, out)
        artifact_sha = sha_file(out / "frozen_detector.json")
        # Dev images/features are not even fed through the encoder until freezing.
        xd = features(dev, args.sup_zip, out, input_identity, extractor)
        all_rows = rows + dev
        scores = predict(artifact, np.concatenate([x, xd]))
        if sha_file(out / "frozen_detector.json") != artifact_sha:
            raise ValueError("frozen model changed during evaluation")
        metrics, predictions = [], []
        for split in ("fit", "calibration", "internal_test", "dev_report_only"):
            indices = [i for i, r in enumerate(all_rows) if r["split"] == split]
            subset = [all_rows[i] for i in indices]
            metrics.append(report(subset, scores[indices], artifact["threshold"], split))
            predictions.extend({"image": r["image"], "split": split, "label": r["label"],
                                "missing_probability": float(s), "predicted_missing": int(s >= artifact["threshold"])}
                               for r, s in zip(subset, scores[indices]))
        write_csv(out / "leaderboard.csv", metrics)
        write_csv(out / "metrics_by_asset.csv", metrics)
        write_csv(out / "predictions.csv", predictions)
        import sklearn
        metadata = {**input_identity, "complete": True, "condition": "supervised_assisted_standalone",
                    "frozen_detector_sha256": artifact_sha, "images": len(all_rows), "row_errors": 0,
                    "invalid_outputs": 0, "q1_inference_calls": 0, "gold_scored": False,
                    "wall_time_s": round(time.time()-start, 3), "python": sys.version,
                    "numpy": np.__version__, "sklearn": sklearn.__version__, "gpu": "CPU fixture"}
        if extractor is None:
            import torch
            import transformers
            metadata.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__)
        try:
            metadata["benchmark_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except subprocess.CalledProcessError:
            metadata["benchmark_commit"] = None
        write_json(out / "run_metadata.json", metadata)
        write_json(out / "github_publish_manifest.json", {"protocol": PROTOCOL, "auto_publish": False,
                                                          "gold_scored": False, "files": [p.name for p in out.iterdir() if p.is_file()]})
        write_json(out / "run_status.json", {"complete": True, "phase": "complete", "row_errors": 0,
                                             "gold_scored": False, "frozen_detector_sha256": artifact_sha})
        return metrics
    except Exception as e:
        write_json(out / "run_status.json", {"complete": False, "phase": "failed", "error": f"{type(e).__name__}: {e}"})
        write_json(out / "run_metadata.json", {**input_identity, "complete": False,
                                                "error": f"{type(e).__name__}: {e}", "gold_scored": False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("sup-zip", "unsup-zip", "dev-qa", "gold-csv", "out-dir"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--minimum-per-class", type=int, default=10)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
