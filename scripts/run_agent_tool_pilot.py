"""Training-free dev-only tool pilot; not an AgentIAD SFT/GRPO reproduction."""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_gold_v2_q2_q3_agent_dev as support
from insplad_benchmark.metrics import anomaly_classification_summary

agent, base, cons = support.agent, support.agent.base, support.agent.cons
Q1, Q2, Q3 = agent.Q1, agent.Q2, agent.Q3
PROTOCOL = "agent_tool_pilot_dev_v1"
POLICIES = ("direct_reference", "fixed_evidence", "adaptive_tools", "matched_reask")
CENTER = [0.2, 0.2, 0.8, 0.8]


class ModelLoadError(RuntimeError):
    """Do not repeat a failed/OOM model load for every remaining question."""


def digest(value):
    return support.diagnostic.digest(value)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_gpu(torch_module=None):
    if torch_module is None:
        import torch as torch_module
    if (not torch_module.cuda.is_available()
            or torch_module.cuda.get_device_properties(0).total_memory < 20 * 1024**3):
        raise RuntimeError("This pilot requires CUDA with >=20 GiB VRAM; no quantization fallback")


def flight_date(member):
    found = re.search(r"\d{2}-\d{2}-\d{4}", Path(member).name)
    return found.group(0) if found else None


def select_rows(rows, n_images, seed):
    if not 1 <= n_images <= 240:
        raise ValueError("n-images must be 1..240 dev images")
    if any(r.get("protocol") != "gold_v2_dev" for r in rows):
        raise ValueError("only gold_v2_dev is accepted; gold evaluation is prohibited")
    pairs = cons.image_pairs(rows)
    if len(pairs) * 2 != len(rows) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("expected unique paired Q1/Q2 dev rows only")
    for a, b in pairs:
        if any(a.get(k) != b.get(k) for k in ("core_id", "image", "asset_category", "anomaly_gt", "defect_gt")):
            raise ValueError("Q1/Q2 labels or images disagree")
        if any(r.get("answer") not in r["options"] for r in (a, b)):
            raise ValueError("invalid Q1/Q2 GT")
    if n_images > len(pairs):
        raise ValueError("not enough dev images")
    # Existing deterministic stratified dev40, extended without resampling it for dev240.
    head = support.diagnostic.select_subset(rows, min(n_images, 40), seed)
    chosen = {r["core_id"] for r in head}
    remaining = sorted((p for p in pairs if p[0]["core_id"] not in chosen),
                       key=lambda p: digest([seed, p[0]["core_id"]]))
    return head + [r for p in remaining[:n_images-len(chosen)] for r in p]


def parse_action(output, options):
    text = str(output).strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)[:-3].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("action must be one JSON object")
    name = obj.get("action")
    if name == "finish" and set(obj) == {"action", "choice"} and obj["choice"] in options:
        return obj
    if name == "reference" and set(obj) == {"action"}:
        return obj
    if name == "crop" and set(obj) == {"action", "bbox"}:
        box = obj["bbox"]
        if (isinstance(box, list) and len(box) == 4
                and all(type(x) in (int, float) and math.isfinite(x) for x in box)
                and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            return obj
    raise ValueError("invalid action/schema/choice or normalized original-frame box")


def make_crop(target, bbox, path):
    from PIL import Image
    with Image.open(target) as image:
        w, h = image.size
        pixels = (math.floor(bbox[0]*w), math.floor(bbox[1]*h),
                  math.ceil(bbox[2]*w), math.ceil(bbox[3]*h))
        path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").crop(pixels).save(path)
    return path


def prompt_for(row, history, reference, crop, tools_allowed):
    # Explicit field whitelist: labels, paths, IDs and other question answers never enter inference.
    images = ["Image 1 is the original TARGET."]
    if reference:
        images.append(f"Image {len(images)+1} is a same-asset normal REFERENCE selected by the different-date rule.")
    if crop:
        images.append(f"Image {len(images)+1} is a crop of the original TARGET; bbox={crop}.")
    task = row["question"] + "\n" + "\n".join(f"{k}. {v}" for k, v in row["options"].items())
    rules = ('Return only one JSON object, with no explanation. Final answer: '
             '{"action":"finish","choice":"OPTION_LETTER"}. ')
    if tools_allowed:
        rules += ('Optional tools: {"action":"reference"} requests the pre-retrieved normal image; '
                  '{"action":"crop","bbox":[x1,y1,x2,y2]} requests a crop. Coordinates are normalized '
                  '0..1 on the ORIGINAL target, x left-to-right and y top-to-bottom. Each tool is available '
                  'only once. Use a tool only if needed; you may finish immediately. ')
    else:
        rules += "No tool request is allowed on this turn; finish now. "
    if row["question_type"] == Q3:
        rules += "Report location on the ORIGINAL full-frame 3x3 grid, not the crop. "
    return ("Inspect the target power-line asset, not the reference. " + " ".join(images)
            + "\n" + rules + "\nQuestion:\n" + task
            + "\nThis question's previous steps:\n" + json.dumps(history, ensure_ascii=False))


def episode(row, policy, target, normal, out_dir, identity, call, max_calls, matched_calls=None):
    folder = out_dir / "episodes" / policy
    # Use hash, never raw QA IDs as output paths.
    path = folder / (digest(row["id"]) + ".jsonl")
    old = agent.read_jsonl(path) if path.exists() else []
    if len(old) > max_calls or any(r.get("identity") != identity for r in old):
        raise ValueError("episode identity/budget mismatch")
    use_ref = policy != "adaptive_tools"
    bbox = CENTER if policy == "fixed_evidence" else None
    crop = make_crop(target, bbox, out_dir / "_crops" / (digest([row["id"], bbox])+".png")) if bbox else None
    history, saved = [], []
    choice, invalid, error = None, "", ""
    new_calls, reused_calls = 0, 0
    limit = matched_calls if policy == "matched_reask" else max_calls if policy == "adaptive_tools" else 1
    for index in range(limit):
        allowed = policy == "adaptive_tools" and index < limit-1
        prompt = prompt_for(row, history, use_ref, bbox, allowed)
        images = [target] + ([normal] if use_ref else []) + ([crop] if crop else [])
        stamp = digest([prompt, [file_sha(p) for p in images]])
        if index < len(old) and old[index].get("call_signature") != stamp:
            raise ValueError("saved prompt/image history changed")
        if index < len(old) and not old[index].get("error"):
            record = old[index]
            reused_calls += 1
        else:
            new_calls += 1
            start = time.monotonic()
            output, error = "", ""
            try:
                output = str(call(images, prompt))
            except ModelLoadError:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            record = dict(identity=identity, call_signature=stamp, index=index, prompt=prompt,
                          images=[str(p) for p in images], output=output, error=error,
                          latency_s=time.monotonic()-start)
        saved.append(record)
        base.write_jsonl(path, saved + old[index+1:])
        if record["error"]:
            error = record["error"]
            break
        try:
            action = parse_action(record["output"], row["options"])
            name = action["action"]
            if name != "finish" and not allowed:
                raise ValueError("tool request when final answer required")
            if name == "finish":
                choice = action["choice"]
                if policy != "matched_reask" or index == limit-1:
                    break
                history.append(dict(output=action, result="Re-examine this same question independently."))
            elif name == "reference":
                if use_ref:
                    raise ValueError("reference requested twice")
                use_ref = True
                history.append(dict(output=action, result="Normal reference attached."))
            else:
                if crop is not None:
                    raise ValueError("crop requested twice")
                bbox = action["bbox"]
                crop = make_crop(target, bbox, out_dir / "_crops" / (digest([row["id"], bbox])+".png"))
                history.append(dict(output=action, result="Requested crop attached; use original-frame coordinates."))
        except (ValueError, TypeError, KeyError) as exc:
            invalid, choice = str(exc), None
            break
    base.write_jsonl(path, saved)
    return dict(id=row["id"], core_id=row["core_id"], policy=policy, question_type=row["question_type"],
                predicted_choice=choice, calls=len(saved), tool_actions=history, error=error,
                invalid=invalid, latency_s=sum(r["latency_s"] for r in saved),
                calls_this_run=new_calls, reused_calls=reused_calls)


def score(rows, q3, records, policy):
    choices = {r["id"]: r["predicted_choice"] for r in records}
    pairs = cons.image_pairs(rows)
    result = cons.consistency_row(pairs, choices)
    first = [r for r in rows if r["question_type"] == Q1]
    summary = anomaly_classification_summary([r["anomaly_gt"] for r in first],
                                            [base.predicted_semantic(r, choices[r["id"]]) or "invalid" for r in first])
    result.update({"q1_"+k: summary[k] for k in ("balanced_accuracy", "precision", "recall", "tp", "fp", "tn", "fn")})
    q12 = {a["core_id"]: choices[a["id"]] == a["answer"] and choices[b["id"]] == b["answer"] for a, b in pairs}
    result.update(support.presence_metrics(q3, choices, q12))
    result.update(policy=policy, images=len(first), calls=sum(r["calls"] for r in records),
                  invalid_episodes=sum(bool(r["invalid"]) for r in records),
                  q1_q2_q3_location_joint=None, scope="exploratory dev; no defective spatial GT")
    return result


def main(argv=None, *, loader=None, infer=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dev-qa", "refs-csv", "sup-zip", "unsup-zip", "out-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--model", choices=support.diagnostic.dev.MODELS, default="internvl3_5_8b")
    parser.add_argument("--n-images", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-calls", type=int, choices=(2, 3), default=3)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    if not args.out_dir.name.startswith("agent_tool_pilot_dev"):
        raise ValueError("use a new agent_tool_pilot_dev* output directory")
    if args.n_boot < 20:
        raise ValueError("n-boot must be at least 20")
    rows = select_rows(agent.read_jsonl(args.dev_qa), args.n_images, args.seed)
    q3 = support.q3_rows(rows)
    with args.refs_csv.open(encoding="utf-8-sig", newline="") as stream:
        refs_list = list(csv.DictReader(stream))
    if len({r["image"] for r in refs_list}) != len(refs_list):
        raise ValueError("duplicate reference mapping")
    refs = {r["image"]: r["retrieved_diffdate"] for r in refs_list}
    if any(not refs.get(r["image"]) for r in rows):
        raise ValueError("missing different-date normal reference; run the existing reference preparation")
    for r in rows:
        if flight_date(r["image"]) and flight_date(r["image"]) == flight_date(refs[r["image"]]):
            raise ValueError("same-date reference violates the different-date rule")
    spec = base.MODEL_REGISTRY[args.model]
    source_files = [Path(__file__), Path(support.__file__), Path(agent.__file__), Path(base.__file__),
                    ROOT / "scripts/run_dev_prompt_diagnostic.py", ROOT / "src/insplad_benchmark/hf_adapters.py",
                    ROOT / "scripts/run_gold_v2_consistency_metrics.py", ROOT / "src/insplad_benchmark/metrics.py"]
    signature = digest(dict(protocol=PROTOCOL, rows=rows, q3=q3, model=spec, max_calls=args.max_calls,
                            seed=args.seed, refs=refs, sources={p.name: file_sha(p) for p in source_files}))
    meta_path = args.out_dir / "run_metadata.json"
    old_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    if old_meta and old_meta["signature"] != signature:
        raise ValueError("code/input/settings changed: use a new output folder")
    targets = support.verified_images(args.sup_zip, [r["image"] for r in rows], args.out_dir / "_images/target")
    normals = support.verified_images(args.unsup_zip, [refs[r["image"]] for r in rows], args.out_dir / "_images/normal")
    if any(file_sha(targets[r["image"]]) == file_sha(normals[refs[r["image"]]]) for r in rows):
        raise ValueError("target and normal reference contain identical image bytes")
    manifest = [dict(role=role, member=k, sha256=file_sha(p)) for role, paths in (("target", targets), ("reference", normals)) for k, p in sorted(paths.items())]
    identity = digest([signature, manifest])
    if old_meta and old_meta["identity"] != identity:
        raise ValueError("image bytes changed: preserve this run and use a new folder")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    software = {}
    for package in ("torch", "transformers", "accelerate", "numpy", "Pillow"):
        try:
            software[package] = version(package)
        except PackageNotFoundError:
            software[package] = None
    run_id = str(uuid.uuid4())
    meta = dict(protocol=PROTOCOL, signature=signature, identity=identity, model=spec, run_id=run_id, software=software,
                resolved_model_revision=(old_meta or {}).get("resolved_model_revision"),
                benchmark_commit=agent.git_commit(), images=args.n_images, max_calls=args.max_calls,
                training=False, gpu=agent.gpu_info() if loader is None else None,
                execution="real_hf" if loader is None and infer is None else "injected_test_backend",
                spatial_gt="absent; no Q3 location or location-joint claim", tile_budget=spec.get("max_total_tiles"),
                limitations="Prompt and available images differ across controls; extra images share visual budget. Dev is repeatedly explored.")
    support.atomic_json(meta_path, meta)
    base.write_csv(args.out_dir / "image_manifest.csv", manifest)
    base.write_jsonl(args.out_dir / "qa_snapshot.jsonl", rows+q3)
    support.atomic_json(args.out_dir / "run_status.json", dict(complete=False, status="running", run_id=run_id))
    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer
    def real_loader(model_spec):
        require_gpu()
        from insplad_benchmark.hf_adapters import load_hf_adapter
        adapter = load_hf_adapter(model_spec)
        revision = getattr(getattr(adapter.get("model"), "config", None), "_commit_hash", None)
        if (old_meta or {}).get("resolved_model_revision") not in (None, revision):
            raise RuntimeError("downloaded model revision changed; use a new run folder")
        return adapter
    holder = base.LazyHFAdapter(spec, loader=loader or real_loader)
    def call(images, prompt):
        try:
            adapter = holder.get()
        except Exception as exc:
            raise ModelLoadError(str(exc)) from exc
        return infer(adapter, images, prompt)
    records, boards = [], []
    try:
        for policy in POLICIES:
            for index, row in enumerate(rows+q3):
                matching = next((r["calls"] for r in records if r["policy"] == "adaptive_tools" and r["id"] == row["id"]), None)
                result = episode(row, policy, targets[row["image"]], normals[refs[row["image"]]],
                                 args.out_dir, identity, call,
                                 args.max_calls, matching)
                records.append(result)
                if (index+1) % 20 == 0 or index == len(rows)+len(q3)-1:
                    base.write_jsonl(args.out_dir / "answers.jsonl", records)
                    print(f"[{policy}] {index+1}/{len(rows)+len(q3)} questions checkpointed", flush=True)
            current = [r for r in records if r["policy"] == policy]
            if not any(r["error"] for r in current):
                boards.append(score(rows, q3, current, policy))
    except Exception as exc:
        support.atomic_json(args.out_dir / "run_status.json", dict(complete=False, status="fatal_error", run_id=run_id, error=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        if holder.loaded and isinstance(holder._adapter, dict):
            model = holder._adapter.get("model")
            config = getattr(model, "config", None)
            meta["resolved_model_revision"] = getattr(config, "_commit_hash", None)
            if loader is None:
                meta["gpu"] = agent.gpu_info()
            support.atomic_json(meta_path, meta)
        holder.release()
    base.write_csv(args.out_dir / "leaderboard.csv", boards)
    by_asset, by_defect = [], []
    for board in boards:
        current = [r for r in records if r["policy"] == board["policy"]]
        choices = {r["id"]: r["predicted_choice"] for r in current}
        for asset in sorted({r["asset_category"] for r in rows}):
            sub_rows = [r for r in rows if r["asset_category"] == asset]
            ids = {r["core_id"] for r in sub_rows}
            by_asset.append(dict(asset_category=asset, **score(sub_rows, [r for r in q3 if r["core_id"] in ids],
                                                             [r for r in current if r["core_id"] in ids], board["policy"])))
        for defect in sorted({r["defect_gt"] for r in rows}):
            sub = [r for r in rows if r["question_type"] == Q2 and r["defect_gt"] == defect]
            correct = sum(choices[r["id"]] == r["answer"] for r in sub)
            by_defect.append(dict(policy=board["policy"], defect_gt=defect, images=len(sub), q2_correct=correct, q2_accuracy=correct/len(sub)))
    base.write_csv(args.out_dir / "metrics_by_asset.csv", by_asset)
    base.write_csv(args.out_dir / "q2_by_defect.csv", by_defect)
    found = {(args.model, b["policy"]): {r["id"]: r["predicted_choice"] for r in records if r["policy"] == b["policy"]} for b in boards}
    comparisons = [(p, "adaptive_tools") for p in POLICIES if p != "adaptive_tools"]
    significance = []
    for control, policy in comparisons:
        if (args.model, control) not in found or (args.model, policy) not in found:
            continue
        intervals = cons.paired_bootstrap(cons.image_pairs(rows), [], found[args.model, control], found[args.model, policy], args.n_boot, 42)
        for metric in ("q1_anomaly_f1", "q2_defect_macro_f1", "joint_accuracy", "false_defect_rate", "consistency_error"):
            significance.append(dict(baseline=control, condition=policy, metric=metric, **intervals[metric], scope="exploratory dev, unadjusted"))
    joints = {(m, p): {a["core_id"]: c[a["id"]] == a["answer"] and c[b["id"]] == b["answer"]
                       for a, b in cons.image_pairs(rows)} for (m, p), c in found.items()}
    for row in support.q3_bootstrap(q3, found, joints, [args.model], n_boot=args.n_boot, comparisons=comparisons):
        significance.append({k: v for k, v in row.items() if k not in ("model_key", "n_boot")})
    base.write_csv(args.out_dir / "significance.csv", significance)
    errors = sum(bool(r["error"]) for r in records)
    invalid = sum(bool(r["invalid"]) for r in records)
    adaptive = [r for r in records if r["policy"] == "adaptive_tools"]
    support.atomic_json(args.out_dir / "run_status.json", dict(complete=errors == 0, status="inference_complete" if errors == 0 else "has_errors", run_id=run_id,
                        row_errors=errors, invalid_episodes=invalid,
                        calls_this_run=sum(r["calls_this_run"] for r in records), reused_calls=sum(r["reused_calls"] for r in records),
                        adaptive_valid_rate=1-sum(bool(r["invalid"] or r["error"]) for r in adaptive)/len(adaptive),
                        eligible_policies=[r["policy"] for r in boards],
                        pilot_format_gate=errors == 0 and sum(bool(r["invalid"]) for r in adaptive)/len(adaptive) <= .1,
                        location_evaluation_complete=False, scope="dev pilot only; not experiment effectiveness or publication"))
    print(json.dumps(boards, indent=2))
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
