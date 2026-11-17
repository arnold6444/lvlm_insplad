"""Score the tool-choice agent (base or LoRA) on frozen gold v2 under one fixed protocol (docs/agent_sft_gold_eval.md).

Prompts, options, the action parser and the episode loop are run_agent_tool_pilot's; nothing is tuned here and gold
chooses nothing. What is new is the scoring: the reported Q3 cell against the 133 located defects, and every metric
on the gold photos that share no drone frame and no bytes with a fine-tuning photo (the primary scope).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
import uuid
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import build_agent_sft_data as sft  # noqa: E402
import run_agent_tool_pilot as pilot  # noqa: E402
import score_gold_v2_q3_position as position  # noqa: E402

agent, base, cons, support = pilot.agent, pilot.base, pilot.cons, pilot.support
Q1, Q2, Q3 = pilot.Q1, pilot.Q2, pilot.Q3
PROTOCOL = "agent_sft_gold_eval_v1"
MODEL = "qwen3_5_4b"
POLICIES = ("direct_reference", "adaptive_tools")
PRIMARY, SUBSETS = "frame_disjoint", ("frame_disjoint", "all")
GOLD_IMAGES, LOCATED_IMAGES = 246, 133
BOOT_METRICS = ("q1_anomaly_f1", "q2_defect_macro_f1", "joint_accuracy", "false_defect_rate", "consistency_error")


def load_gold(qa_dir):
    """Frozen gold v2 Q1/Q2 pairs. The legacy four-way Q3 rows are not used; Q3 is the pilot's nine-cell question."""
    rows = [dict(r) for r in agent.v2.load_gold_qa_v2(qa_dir) if r["question_type"] in (Q1, Q2)]
    if any(r.get("protocol") == "gold_v2_dev" or r.get("split") == "supervised_train_sft" for r in rows):
        raise ValueError("expected frozen gold v2 QA, not dev or training QA")
    pairs = cons.image_pairs(rows)
    if len(pairs) != GOLD_IMAGES or len(rows) != 2 * len(pairs) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"expected {GOLD_IMAGES} gold photos with one Q1 and one Q2 each")
    for a, b in pairs:
        if any(a.get(k) != b.get(k) for k in ("core_id", "image", "asset_category", "anomaly_gt", "defect_gt")):
            raise ValueError("Q1/Q2 labels or images disagree")
        if any(r.get("answer") not in r["options"] or r["anomaly_gt"] not in ("normal", "anomalous") for r in (a, b)):
            raise ValueError("invalid gold Q1/Q2 label")
    return rows


def load_location(path, rows):
    """Cells and boxes of the frozen location set; it must be a subset of the defective gold photos."""
    records = position.read_jsonl(path)
    gt = position.location_gt(records)
    labels = {r["core_id"]: r for r in rows if r["question_type"] == Q1}
    for core, item in gt.items():
        label = labels.get(core)
        if label is None or label["anomaly_gt"] != "anomalous" or label["defect_gt"] != item["defect_gt"]:
            raise ValueError("location set does not match frozen gold v2")
    if len(gt) != LOCATED_IMAGES:
        raise ValueError(f"expected {LOCATED_IMAGES} located defects")
    return gt, {r["core_id"]: r["localization_gt"]["normalized_bboxes_xywh"] for r in records}


def grid_rows(rows, gt):
    """The pilot's nine-cell Q3 (same question, same option shuffle). The answer never enters a prompt."""
    return [{**r, "id": r["core_id"] + "-q3-gold-grid", "protocol": PROTOCOL,
             "location_gt_known": r["core_id"] in gt} for r in support.q3_rows(rows)]


def frame_overlap(rows, train_files, sup_zip):
    """Per gold photo: does it share a drone frame, or its bytes, with any fine-tuning photo?"""
    train = set()
    for path in train_files:
        records = agent.read_jsonl(path)
        if not records or any(r.get("split") != "supervised_train_sft" for r in records):
            raise ValueError(f"not an SFT training QA file: {path}")
        train.update(r["image"] for r in records)
    keys = {sft.photo_key(p) for p in train}
    if None in keys:
        raise ValueError("a training photo has no flight date or DJI frame number")
    with ZipFile(sup_zip) as archive:
        stamp = {i.filename.replace("\\", "/").lstrip("/"): (i.file_size, i.CRC) for i in archive.infolist()}
    images = sorted({r["image"] for r in rows})
    missing = [p for p in [*sorted(train), *images] if p not in stamp]
    if missing:
        raise FileNotFoundError(f"{len(missing)} photos are not in the supervised ZIP, e.g. {missing[0]}")
    train_bytes = {stamp[p] for p in train}
    out = {}
    for image in images:
        key = sft.photo_key(image)
        if key is None:
            raise ValueError(f"gold photo without a flight date or DJI frame number: {image}")
        out[image] = dict(same_frame=key in keys, same_bytes=stamp[image] in train_bytes)
    return out, sorted(train)


def reported_cells(q3, choices):
    """core_id -> the reported cell 1..9, or None for "no region" and for an invalid answer."""
    return {r["core_id"]: position.TEXT_TO_CELL[r["options"][choices[r["id"]]]] if choices.get(r["id"]) in r["options"]
            else None for r in q3}


def crop_boxes(records):
    """question id -> the box of the crop the model requested in that episode."""
    return {r["id"]: step["output"]["bbox"] for r in records for step in r["tool_actions"]
            if step["output"].get("action") == "crop"}


def covers(box, gt_boxes):
    """A crop [x1, y1, x2, y2] covers a defect when it contains the centre of one of its boxes [x, y, w, h]."""
    return any(box[0] <= x + w / 2 <= box[2] and box[1] <= y + h / 2 <= box[3] for x, y, w, h in gt_boxes)


def score(rows, q3, records, policy, gt, boxes):
    board = pilot.score(rows, q3, records, policy)
    choices = {r["id"]: r["predicted_choice"] for r in records}
    cells = reported_cells(q3, choices)
    located = sorted(i for i in cells if i in gt)
    hits = position.count(located, gt, cells)
    constant = {c: sum(c in gt[i]["cells"] for i in located) for c in range(1, 10)}
    q12 = {a["core_id"]: choices[a["id"]] == a["answer"] and choices[b["id"]] == b["answer"]
           for a, b in cons.image_pairs(rows)}
    normal = [r for r in q3 if r["anomaly_gt"] == "normal"]
    joint = (sum(q12[r["core_id"]] and choices[r["id"]] == r["none_choice"] for r in normal)
             + sum(q12[i] and cells[i] is not None and cells[i] in gt[i]["cells"] for i in located))
    crops = crop_boxes(records)
    shapes = Counter(json.dumps(b) for b in crops.values())
    q3_crop = {r["core_id"]: crops[r["id"]] for r in q3 if r["id"] in crops}
    board.update(
        q3_location_accuracy=hits["position_hit_rate"], q3_anomalous_location_labels=len(located),
        q3_position_hit=hits["position_hit"], q3_region_reported_on_located=hits["region_reported"],
        q3_hit_given_reported=hits["hit_given_reported"], q3_always_center_hit=constant[5],
        q3_best_constant_hit=max(constant.values()) if located else None,
        q1_q2_q3_location_joint=joint / (len(normal) + len(located)) if normal or located else None,
        q1_q2_q3_location_joint_images=len(normal) + len(located),
        crop_requests=len(crops), crop_distinct_boxes=len(shapes),
        crop_most_common=shapes.most_common(1)[0][1] if shapes else 0,
        q3_crops_on_located=sum(i in q3_crop for i in located),
        q3_crop_covers_defect=sum(i in q3_crop and covers(q3_crop[i], boxes[i]) for i in located),
        center60_covers_defect=sum(covers(pilot.CENTER, boxes[i]) for i in located),
        scope="gold v2, protocol fixed before the run; the location joint counts normal and located photos")
    return board


def keep(items, cores):
    return [r for r in items if r["core_id"] in cores]


def main(argv=None, *, loader=None, infer=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qa-dir", "loc-qa", "refs-csv", "sup-zip", "unsup-zip", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--train-qa", type=Path, nargs="+", required=True,
                        help="sft_train_qa.jsonl of every fine-tuning set scored here; their photos define the overlap")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="LoRA adapter; omit for the base model")
    parser.add_argument("--max-calls", type=int, choices=(2, 3), default=3)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--inputs-only", action="store_true",
                        help="check every input and print the cohort sizes; no model is loaded and nothing is written")
    args = parser.parse_args(argv)
    if not args.out_dir.name.startswith("agent_sft_gold_eval"):
        raise ValueError("use a new agent_sft_gold_eval* output directory")
    if args.n_boot < 20:
        raise ValueError("n-boot must be at least 20")
    adapter_files = None
    if args.adapter_dir is not None:
        adapter_files = {p.name: pilot.file_sha(p) for p in sorted(args.adapter_dir.glob("adapter_*")) if p.is_file()}
        if not {"adapter_config.json", "adapter_model.safetensors"} <= set(adapter_files):
            raise FileNotFoundError(f"no complete LoRA adapter in {args.adapter_dir}")
    rows = load_gold(args.qa_dir)
    gt, boxes = load_location(args.loc_qa, rows)
    q3 = grid_rows(rows, gt)
    with args.refs_csv.open(encoding="utf-8-sig", newline="") as stream:
        refs_list = list(csv.DictReader(stream))
    if len({r["image"] for r in refs_list}) != len(refs_list):
        raise ValueError("duplicate reference mapping")
    refs = {r["image"]: r["retrieved_diffdate"] for r in refs_list}
    for r in rows:
        ref = refs.get(r["image"])
        if not ref or "/train/good/" not in "/" + ref.replace("\\", "/"):
            raise ValueError("missing different-date train-normal reference")
        if pilot.flight_date(r["image"]) and pilot.flight_date(r["image"]) == pilot.flight_date(ref):
            raise ValueError("same-date reference violates the different-date rule")
    overlap, train_images = frame_overlap(rows, args.train_qa, args.sup_zip)
    shared = {image for image, flags in overlap.items() if flags["same_frame"] or flags["same_bytes"]}
    cores = {"all": {r["core_id"] for r in rows},
             "frame_disjoint": {r["core_id"] for r in rows if r["image"] not in shared}}
    located = {s: sum(i in cores[s] for i in gt) for s in SUBSETS}
    if args.inputs_only:
        with ZipFile(args.unsup_zip) as archive:
            names = {n.replace("\\", "/").lstrip("/") for n in archive.namelist()}
        missing = sorted({refs[r["image"]] for r in rows} - names)
        if missing:
            raise FileNotFoundError(f"{len(missing)} references are not in the unsupervised ZIP, e.g. {missing[0]}")
        print(json.dumps(dict(protocol=PROTOCOL, images=len(cores["all"]), frame_disjoint=len(cores[PRIMARY]),
                              located=located, same_frame=sum(f["same_frame"] for f in overlap.values()),
                              same_bytes=sum(f["same_bytes"] for f in overlap.values()),
                              training_photos=len(train_images), lora_adapter=adapter_files is not None)))
        return 0
    spec = base.MODEL_REGISTRY[MODEL]
    source_files = [Path(__file__), Path(pilot.__file__), Path(support.__file__), Path(agent.__file__),
                    Path(base.__file__), Path(position.__file__), Path(sft.__file__),
                    ROOT / "scripts/run_dev_prompt_diagnostic.py", ROOT / "scripts/q3_multiregion.py",
                    ROOT / "src/insplad_benchmark/hf_adapters.py",
                    ROOT / "scripts/run_gold_v2_consistency_metrics.py", ROOT / "src/insplad_benchmark/metrics.py"]
    signature = pilot.digest(dict(
        protocol=PROTOCOL, rows=rows, q3=q3, model=spec, max_calls=args.max_calls, refs=refs, policies=POLICIES,
        location=gt, boxes=boxes, train_images=train_images, lora_adapter=adapter_files,
        sources={p.name: pilot.file_sha(p) for p in source_files}))
    meta_path = args.out_dir / "run_metadata.json"
    old_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    if old_meta and old_meta["signature"] != signature:
        raise ValueError("code/input/settings changed: use a new output folder")
    targets = support.verified_images(args.sup_zip, [r["image"] for r in rows], args.out_dir / "_images/target")
    normals = support.verified_images(args.unsup_zip, [refs[r["image"]] for r in rows], args.out_dir / "_images/normal")
    if any(pilot.file_sha(targets[r["image"]]) == pilot.file_sha(normals[refs[r["image"]]]) for r in rows):
        raise ValueError("target and normal reference contain identical image bytes")
    manifest = [dict(role=role, member=k, sha256=pilot.file_sha(p))
                for role, paths in (("target", targets), ("reference", normals)) for k, p in sorted(paths.items())]
    identity = pilot.digest([signature, manifest])
    if old_meta and old_meta["identity"] != identity:
        raise ValueError("image bytes changed: preserve this run and use a new folder")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    software = {}
    for package in ("torch", "transformers", "accelerate", "peft", "numpy", "Pillow"):
        try:
            software[package] = version(package)
        except PackageNotFoundError:
            software[package] = None
    run_id = str(uuid.uuid4())
    meta = dict(
        protocol=PROTOCOL, signature=signature, identity=identity, model=spec, run_id=run_id, software=software,
        resolved_model_revision=(old_meta or {}).get("resolved_model_revision"),
        benchmark_commit=pilot.source_commit(), images=len(cores["all"]), max_calls=args.max_calls,
        policies=list(POLICIES), training=adapter_files is not None, lora_adapter=adapter_files,
        gpu=(agent.gpu_info() or (old_meta or {}).get("gpu")) if loader is None else None,
        execution="real_hf" if loader is None and infer is None else "injected_test_backend",
        primary_scope=PRIMARY, post_hoc_rules=None,
        frame_overlap=dict(training_photos=len(train_images), training_files=[p.name for p in args.train_qa],
                           same_frame=sum(f["same_frame"] for f in overlap.values()),
                           same_bytes=sum(f["same_bytes"] for f in overlap.values()),
                           frame_disjoint_images=len(cores["frame_disjoint"]), located_images=located),
        inputs=dict(loc_qa=pilot.file_sha(args.loc_qa), train_qa=[pilot.file_sha(p) for p in args.train_qa]),
        cell_rule="each box centre mapped to a 3x3 cell; hit = the single reported cell is one of those cells",
        invalid_policy="invalid output: anomalous rows count as FN, normal rows as FP; an invalid Q3 answer is no hit",
        limitations="One model, one training run, greedy decoding. Gold has prior exploratory use by other methods. "
                    "The reference is retrieved, not the fixed benchmark reference. Prompts differ from the paper's "
                    "gold tables, so only rows of this protocol are comparable with each other.")
    support.atomic_json(meta_path, meta)
    base.write_csv(args.out_dir / "image_manifest.csv", manifest)
    base.write_jsonl(args.out_dir / "qa_snapshot.jsonl", rows + q3)
    base.write_csv(args.out_dir / "frame_overlap.csv",
                   [dict(image=image, **flags, scored_in_primary=image not in shared) for image, flags in overlap.items()])
    support.atomic_json(args.out_dir / "run_status.json", dict(complete=False, status="running", run_id=run_id))
    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer

    def real_loader(model_spec):
        pilot.require_gpu()
        from insplad_benchmark.hf_adapters import load_hf_adapter
        adapter = load_hf_adapter(model_spec)
        if args.adapter_dir is not None:
            from peft import PeftModel
            adapter["model"] = PeftModel.from_pretrained(adapter["model"], str(args.adapter_dir)).eval()
        revision = getattr(getattr(adapter.get("model"), "config", None), "_commit_hash", None)
        if (old_meta or {}).get("resolved_model_revision") not in (None, revision):
            raise RuntimeError("downloaded model revision changed; use a new run folder")
        return adapter

    holder = base.LazyHFAdapter(spec, loader=loader or real_loader)

    def call(images, prompt):
        try:
            adapter = holder.get()
        except Exception as exc:
            raise pilot.ModelLoadError(str(exc)) from exc
        return infer(adapter, images, prompt)

    records, done = [], []
    try:
        for policy in POLICIES:
            for index, row in enumerate(rows + q3):
                records.append(pilot.episode(row, policy, targets[row["image"]], normals[refs[row["image"]]],
                                             args.out_dir, identity, call, args.max_calls))
                if (index + 1) % 20 == 0 or index == len(rows) + len(q3) - 1:
                    base.write_jsonl(args.out_dir / "answers.jsonl", records)
                    print(f"[{policy}] {index+1}/{len(rows)+len(q3)} questions checkpointed", flush=True)
            if not any(r["error"] for r in records if r["policy"] == policy):
                done.append(policy)
    except Exception as exc:
        support.atomic_json(args.out_dir / "run_status.json", dict(
            complete=False, status="fatal_error", run_id=run_id, error=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        if holder.loaded and isinstance(holder._adapter, dict):
            config = getattr(holder._adapter.get("model"), "config", None)
            meta["resolved_model_revision"] = getattr(config, "_commit_hash", None)
            if loader is None:
                meta["gpu"] = agent.gpu_info()
            support.atomic_json(meta_path, meta)
        holder.release()

    by_policy = {p: [r for r in records if r["policy"] == p] for p in done}
    choices = {p: {r["id"]: r["predicted_choice"] for r in rs} for p, rs in by_policy.items()}
    boards, by_asset, by_defect, by_position, significance = [], [], [], [], []
    for subset in SUBSETS:
        sub_rows, sub_q3 = keep(rows, cores[subset]), keep(q3, cores[subset])
        ids = sorted(i for i in gt if i in cores[subset])
        for policy in done:
            current = keep(by_policy[policy], cores[subset])
            boards.append(dict(subset=subset, **score(sub_rows, sub_q3, current, policy, gt, boxes)))
            cells = reported_cells(sub_q3, choices[policy])
            for asset in sorted({r["asset_category"] for r in sub_rows}):
                part = {r["core_id"] for r in sub_rows if r["asset_category"] == asset}
                by_asset.append(dict(subset=subset, asset_category=asset, **score(
                    keep(sub_rows, part), keep(sub_q3, part), keep(current, part), policy, gt, boxes)))
            for defect in sorted({r["defect_gt"] for r in sub_rows}):
                part = [r for r in sub_rows if r["question_type"] == Q2 and r["defect_gt"] == defect]
                correct = sum(choices[policy][r["id"]] == r["answer"] for r in part)
                by_defect.append(dict(subset=subset, policy=policy, defect_gt=defect, images=len(part),
                                      q2_correct=correct, q2_accuracy=correct / len(part)))
            for defect in sorted({gt[i]["defect_gt"] for i in ids}):
                part = [i for i in ids if gt[i]["defect_gt"] == defect]
                by_position.append(dict(subset=subset, policy=policy, defect_gt=defect,
                                        **position.count(part, gt, cells),
                                        always_center_hit=sum(5 in gt[i]["cells"] for i in part)))
            if ids:
                gain = position.bootstrap(ids, gt, cells, {i: 5 for i in ids}, n=args.n_boot)
                significance.append(dict(
                    subset=subset, baseline="always_center", condition=policy, metric="q3_position_hit_rate",
                    base=sum(5 in gt[i]["cells"] for i in ids) / len(ids), new=position.count(ids, gt, cells)["position_hit_rate"],
                    delta=gain["delta"], ci_low=gain["ci_low"], ci_high=gain["ci_high"],
                    significant=bool(gain["ci_low"] > 0 or gain["ci_high"] < 0)))
        if set(POLICIES) <= set(done):
            stats = cons.paired_bootstrap(cons.image_pairs(sub_rows), [], choices[POLICIES[0]], choices[POLICIES[1]],
                                          args.n_boot, 42)
            for metric in BOOT_METRICS:
                significance.append(dict(subset=subset, baseline=POLICIES[0], condition=POLICIES[1], metric=metric,
                                         **stats[metric]))
            if ids:
                gain = position.bootstrap(ids, gt, reported_cells(sub_q3, choices[POLICIES[1]]),
                                          reported_cells(sub_q3, choices[POLICIES[0]]), n=args.n_boot)
                significance.append(dict(
                    subset=subset, baseline=POLICIES[0], condition=POLICIES[1], metric="q3_position_hit_rate",
                    base=None, new=None, delta=gain["delta"], ci_low=gain["ci_low"], ci_high=gain["ci_high"],
                    significant=bool(gain["ci_low"] > 0 or gain["ci_high"] < 0)))
    per_image, grids = [], {r["core_id"]: r for r in q3}
    for a, b in cons.image_pairs(rows):
        core, grid = a["core_id"], grids[a["core_id"]]
        row = dict(core_id=core, asset_category=a["asset_category"], anomaly_gt=a["anomaly_gt"],
                   defect_gt=a["defect_gt"], scored_in_primary=core in cores[PRIMARY],
                   gt_cells=" ".join(map(str, gt[core]["cells"])) if core in gt else "")
        for policy in done:
            cell = reported_cells([grid], choices[policy])[core]
            crop = crop_boxes(keep(by_policy[policy], {core})).get(grid["id"])
            row.update({f"{policy}_q1_correct": choices[policy][a["id"]] == a["answer"],
                        f"{policy}_q2_correct": choices[policy][b["id"]] == b["answer"],
                        f"{policy}_q3_valid": choices[policy][grid["id"]] in grid["options"],
                        f"{policy}_q3_cell": "" if cell is None else cell,
                        f"{policy}_q3_crop": "" if crop is None else json.dumps(crop)})
        per_image.append(row)
    for name, table in (("leaderboard.csv", boards), ("metrics_by_asset.csv", by_asset),
                        ("q2_by_defect.csv", by_defect), ("position_by_defect.csv", by_position),
                        ("significance.csv", significance), ("per_image.csv", per_image)):
        base.write_csv(args.out_dir / name, table)
    errors = sum(bool(r["error"]) for r in records)
    support.atomic_json(args.out_dir / "run_status.json", dict(
        complete=errors == 0, status="inference_complete" if errors == 0 else "has_errors", run_id=run_id,
        row_errors=errors, invalid_episodes=sum(bool(r["invalid"]) for r in records),
        calls_this_run=sum(r["calls_this_run"] for r in records), reused_calls=sum(r["reused_calls"] for r in records),
        eligible_policies=done, location_evaluation_complete=errors == 0,
        scope="gold v2, scored once under agent_sft_gold_eval_v1; never used to choose a setting"))
    print(json.dumps([b for b in boards if b["subset"] == PRIMARY], indent=2))
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
