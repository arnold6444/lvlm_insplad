"""A small supervised baseline on the fine-tuning labels: frozen DINOv2 features and one linear classifier.

It trains on the photos and labels of one sft_train_qa.jsonl (the LoRA run's own training set) and answers Q1 and Q2
of frozen gold v2, and of dev240 when given, scored with the functions the agent runs use. It sees the target photo
only (no reference photo), gives no location, and nothing is tuned: scikit-learn defaults and one fit, fixed before
any gold photo is embedded (docs/agent_sft_gold_eval.md). The features are saved, so any other small classifier can
be tried later without a GPU.
"""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import build_gold_v2_dev_set as devset  # noqa: E402
import run_agent_sft_gold_eval as gold  # noqa: E402
from insplad_benchmark.dataset import parse_supervised_manifest  # noqa: E402
from insplad_benchmark.metrics import anomaly_classification_summary  # noqa: E402

pilot = gold.pilot
agent, base, cons, support = pilot.agent, pilot.base, pilot.cons, pilot.support
Q1, Q2 = pilot.Q1, pilot.Q2
PROTOCOL = "sft_label_classifier_baseline_v1"
NAME = "dinov2_linear"
CLASSIFIER = "StandardScaler + LogisticRegression(max_iter=5000), scikit-learn defaults otherwise"


def training_labels(path):
    """photo -> its Q2 answer text (the class), and class -> Q1 answer text, both read from the training QA."""
    rows = agent.read_jsonl(path)
    if not rows or any(r.get("split") != "supervised_train_sft" for r in rows):
        raise ValueError(f"not an SFT training QA file: {path}")
    pairs = cons.image_pairs(rows)
    if len(rows) != 2 * len(pairs) or len({a["image"] for a, _ in pairs}) != len(pairs):
        raise ValueError("expected one Q1 and one Q2 per training photo")
    labels, q1_text = {}, {}
    for a, b in pairs:
        label = b["options"][b["answer"]]
        labels[a["image"]] = label
        if q1_text.setdefault(label, a["options"][a["answer"]]) != a["options"][a["answer"]]:
            raise ValueError("one defect class has two Q1 answers")
    if len(q1_text) < 2:
        raise ValueError("training labels have a single class")
    return labels, q1_text


def embed_photos(images, sup_zip, cache_dir, device, embed_factory):
    """Normalized DINOv2 CLS features of supervised-ZIP photos, with the cache the reference search uses."""
    sup_rows, _ = parse_supervised_manifest(devset.zip_manifest(sup_zip, cache_dir / "_manifests/supervised.csv"))
    by_path = {str(r["relative_path"]): r for r in sup_rows}
    missing = sorted(p for p in images if p not in by_path)
    if missing:
        raise FileNotFoundError(f"{len(missing)} photos are not in the supervised ZIP, e.g. {missing[0]}")
    if embed_factory is None:
        embed, source = devset.dino_embedder(sup_zip, sup_zip, cache_dir / "_dino_cache", device)
        source = dict(model_id=source.model_id, model_revision=source.model_revision, embedding=source.embedding_kind)
    else:
        embed, source = embed_factory(), dict(model_id="test-fake")
    found = embed([by_path[p] for p in sorted(images)], "query")
    return {p: np.asarray(found[p], dtype=np.float32) for p in images}, source


def fit(features, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    names = sorted(labels)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(np.stack([features[p] for p in names]), [labels[p] for p in names])
    return model


def score(rows, choices):
    """The Q1/Q2 part of run_agent_tool_pilot.score, unchanged: same functions, same invalid-output policy."""
    result = cons.consistency_row(cons.image_pairs(rows), choices)
    first = [r for r in rows if r["question_type"] == Q1]
    summary = anomaly_classification_summary(
        [r["anomaly_gt"] for r in first], [base.predicted_semantic(r, choices[r["id"]]) or "invalid" for r in first])
    result.update({"q1_" + k: summary[k] for k in ("balanced_accuracy", "precision", "recall", "tp", "fp", "tn", "fn")})
    return result


def main(argv=None, *, embed_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-qa", "qa-dir", "sup-zip", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--overlap-train-qa", type=Path, nargs="+", default=None,
                        help="the training lists the gold runner was given; they define the frame-disjoint gold photos")
    parser.add_argument("--dev-qa", type=Path, default=None, help="also score the dev cohort the agent runs use")
    parser.add_argument("--n-images", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if not args.out_dir.name.startswith("sft_label_classifier_baseline"):
        raise ValueError("use a new sft_label_classifier_baseline* output directory")
    labels, q1_text = training_labels(args.train_qa)
    rows = gold.load_gold(args.qa_dir)
    overlap, _ = gold.frame_overlap(rows, args.overlap_train_qa or [args.train_qa], args.sup_zip)
    shared = {image for image, flags in overlap.items() if flags["same_frame"] or flags["same_bytes"]}
    cohorts = [("gold", "all", rows), ("gold", gold.PRIMARY, [r for r in rows if r["image"] not in shared])]
    if args.dev_qa is not None:
        dev = pilot.select_rows(agent.read_jsonl(args.dev_qa), args.n_images, args.seed)
        cohorts.append((f"dev{args.n_images}", "all", [r for r in dev if r["question_type"] in (Q1, Q2)]))
    scored = {r["image"] for _, _, part in cohorts for r in part}
    if scored & set(labels):
        raise ValueError(f"{len(scored & set(labels))} evaluation photos are training photos")
    signature = pilot.digest(dict(
        protocol=PROTOCOL, labels=labels, classifier=CLASSIFIER, seed=args.seed, shared=sorted(shared),
        cohorts=[(name, subset, [(r["id"], r["image"], r["options"], r["answer"]) for r in part])
                 for name, subset, part in cohorts]))
    meta_path = args.out_dir / "run_metadata.json"
    old_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    if old_meta and old_meta["signature"] != signature:
        raise ValueError("labels, questions or settings changed: use a new output folder")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    support.atomic_json(args.out_dir / "run_status.json", dict(complete=False, status="running", run_id=run_id))

    features, source = embed_photos(sorted(scored | set(labels)), args.sup_zip, args.out_dir, args.device, embed_factory)
    if old_meta and old_meta["features"] != source:
        raise ValueError("feature model changed: use a new output folder")
    names = sorted(features)
    np.savez_compressed(args.out_dir / "features.npz", images=np.array(names), features=np.stack([features[p] for p in names]))
    cohort_of = {}
    for name, _, part in cohorts:
        for r in part:
            cohort_of.setdefault(r["image"], name)
    base.write_csv(args.out_dir / "feature_index.csv", [
        dict(row=i, image=p, used_for="train" if p in labels else cohort_of[p], train_class=labels.get(p, ""))
        for i, p in enumerate(names)])

    model = fit(features, labels)
    classes = list(model.classes_)
    train_names = sorted(labels)
    train_accuracy = float(np.mean(model.predict(np.stack([features[p] for p in train_names]))
                                   == np.array([labels[p] for p in train_names])))
    predicted, chance = {}, {}
    for image in sorted(scored):
        probabilities = model.predict_proba(features[image][None])[0]
        predicted[image], chance[image] = classes[int(np.argmax(probabilities))], probabilities

    def choose(row):
        """The option letter whose text is the predicted answer; None (an invalid output) if no option has it."""
        text = predicted[row["image"]] if row["question_type"] == Q2 else q1_text[predicted[row["image"]]]
        return next((k for k, v in row["options"].items() if v == text), None)

    boards, by_asset, by_defect, per_image = [], [], [], []
    for name, subset, part in cohorts:
        choices = {r["id"]: choose(r) for r in part}
        tag = dict(eval_set=name, subset=subset, policy=NAME)
        boards.append(dict(tag, **score(part, choices), calls=0, scope="Q1 and Q2 only; this baseline gives no location"))
        for asset in sorted({r["asset_category"] for r in part}):
            by_asset.append(dict(tag, asset_category=asset,
                                 **score([r for r in part if r["asset_category"] == asset], choices)))
        for defect in sorted({r["defect_gt"] for r in part}):
            second = [r for r in part if r["question_type"] == Q2 and r["defect_gt"] == defect]
            correct = sum(choices[r["id"]] == r["answer"] for r in second)
            by_defect.append(dict(tag, defect_gt=defect, images=len(second), q2_correct=correct,
                                  q2_accuracy=correct / len(second)))
        if subset == "all":
            for a, b in cons.image_pairs(part):
                per_image.append(dict(
                    eval_set=name, core_id=a["core_id"], image=a["image"], asset_category=a["asset_category"],
                    anomaly_gt=a["anomaly_gt"], defect_gt=a["defect_gt"],
                    scored_in_primary=a["image"] not in shared if name == "gold" else "",
                    predicted=predicted[a["image"]], q1_correct=choices[a["id"]] == a["answer"],
                    q2_correct=choices[b["id"]] == b["answer"],
                    **{"p_" + c: round(float(p), 6) for c, p in zip(classes, chance[a["image"]])}))
    for name, table in (("leaderboard.csv", boards), ("metrics_by_asset.csv", by_asset),
                        ("q2_by_defect.csv", by_defect), ("per_image.csv", per_image)):
        base.write_csv(args.out_dir / name, table)
    software = {}
    for package in ("scikit-learn", "numpy", "torch", "transformers"):
        try:
            software[package] = version(package)
        except PackageNotFoundError:
            software[package] = None
    support.atomic_json(meta_path, dict(
        protocol=PROTOCOL, signature=signature, run_id=run_id, benchmark_commit=pilot.source_commit(),
        features=source, classifier=CLASSIFIER, classes=classes, software=software,
        training_photos=len(labels), training_classes={c: sum(v == c for v in labels.values()) for c in classes},
        training_accuracy=train_accuracy, training_file=pilot.file_sha(args.train_qa),
        cohorts={f"{name}/{subset}": len(cons.image_pairs(part)) for name, subset, part in cohorts},
        gold_photos_sharing_a_frame_or_bytes=len(shared), primary_scope=gold.PRIMARY,
        execution="real_dinov2" if embed_factory is None else "injected_test_backend",
        gpu=agent.gpu_info() if embed_factory is None else None,
        limitations="Target photo only: no reference photo, no asset name, no location. One feature model, one "
                    "classifier, no tuning and no second seed. Gold has prior exploratory use by other methods."))
    support.atomic_json(args.out_dir / "run_status.json", dict(
        complete=True, status="scored", run_id=run_id,
        scope="gold v2 scored once under a fixed setting; never used to choose a setting"))
    print(json.dumps(boards, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
