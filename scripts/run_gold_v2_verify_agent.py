#!/usr/bin/env python3
"""Question-independent verification agent for the gold-v2 QA (choose on the dev set, run gold once).

Every question (Q1 anomaly, Q2 defect type, Q3 location) is verified on its own, the way the baselines ask it.
Re-checking a question uses only the images, that question and that question's own stage-1 answer. No other
question's answer is shown in the prompt or used to decide whether to re-check, which tool to use or what the final
answer is, and a re-check never changes another question's answer. Each re-check is a fresh single-turn call (no chat
history is shared between questions). Stage 1 reuses stored per-question answers (default:
``normal_ref_retrieved_diffdate``, the retrieved normal reference with same-flight candidates excluded).

Tools (a re-check always sees [normal reference, target], the question and its own stage-1 answer, plus):

- ``reask``      : nothing new (controls for asking again)
- ``zoom``       : enlarged central crops of the reference and the target; not applied to Q3, whose options name
                   regions of the full frame
- ``define``     : short visual definitions of the question's own options
- ``zoom_define``: both

Routes (which questions are re-checked, decided per question from its own answers only):

- ``all``          : every question
- ``disagreement`` : the question's stage-1 answer differs from the same question's image-only answer

A variant is ``<tool>@<route>``. A valid re-check answer becomes that question's final answer; an invalid one keeps
the stage-1 answer. Stage-1 and final answers are both kept (``answers.csv``). Q1-Q2 and Q1-Q3 contradictions are
computed afterwards from the stored final answers and only reported. Reported per variant: Q1 F1, Q2 macro-F1,
per-question accuracy (and Q3 on gold), contradiction (and how many appeared / disappeared vs stage 1), joint
accuracy, FDR, re-checks, LVLM calls per question, latency, answer changes and paired bootstrap CIs against stage 1.

A dev run also ranks the variants by a fixed criterion (``selection.csv``). A gold run needs ``--frozen-config``
(written from the dev run with ``make_frozen_config``) and runs that single variant.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_gold_baselines as base  # noqa: E402
import run_gold_v2_baselines as v2  # noqa: E402
import run_gold_v2_consistency_metrics as cons  # noqa: E402

Q1, Q2, Q3 = cons.Q1, cons.Q2, cons.Q3
QUESTION_TYPES = (Q1, Q2, Q3)
SHORT = {Q1: "q1", Q2: "q2", Q3: "q3"}
TOOLS = ("reask", "zoom", "define", "zoom_define")
ROUTES = ("all", "disagreement")
PROTOCOL = "gold_v2_verify_agent_independent"
PROMPT_VERSION = 2
ZOOM_BOX = (0.2, 0.2, 0.8, 0.8)            # central 60% x 60%
ZOOM_SCALE = 2.0
ZOOM_MAX_SIDE = 1344
_CELLS = {"Upper-left region.": "top-left", "Upper-center region.": "top-middle", "Upper-right region.": "top-right",
          "Middle-left region.": "middle-left", "Center region.": "central", "Middle-right region.": "middle-right",
          "Lower-left region.": "bottom-left", "Lower-center region.": "bottom-middle",
          "Lower-right region.": "bottom-right"}
# Visual definitions written from general inspection knowledge (not from gold images), one per option text.
DEFINITIONS = {
    "Normal.": "all parts are present and intact; surfaces keep their usual uniform colour without orange-brown "
               "stains, gaps or foreign material.",
    "Anomalous.": "at least one visible defect, such as rust or corrosion stains, a missing component that leaves a "
                  "gap, or foreign nesting material on the hardware.",
    "Normal / no defect.": "all parts are present and intact; surfaces keep their usual uniform colour without "
                           "orange-brown stains, gaps or foreign material.",
    "Rust / corrosion.": "orange, reddish-brown or dark-brown discoloration, scaling or pitting on metal hardware "
                         "such as clamps, shackles, bolts or caps.",
    "Missing cap / missing insulator component.": "a disc, cap or other component that belongs to the regular "
                                                  "repeated structure is absent, leaving a visible gap or an exposed pin.",
    "Bird nest / foreign nesting material.": "a mass of twigs, straw, grass or other debris lodged on or around the "
                                             "hardware.",
    cons.NONE_TEXT: "the asset shows no visible defect anywhere in the target image.",
    **{text: f"the centre of the defect area falls in the {cell} cell of a 3 x 3 grid over the target image."
       for text, cell in _CELLS.items()},
}
SELECTION_CRITERION = ("mean over the dev models of the change vs stage 1 in (Q1 anomaly F1 + Q2 defect macro-F1) / 2; "
                       "ties go to fewer LVLM calls per question. Contradiction is reported, not used.")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def variant_name(tool: str, route: str) -> str:
    return f"{tool}@{route}"


def uses_zoom(row: Mapping, tool: str) -> bool:
    return tool.startswith("zoom") and row["question_type"] != Q3


def build_prompt(row: Mapping, own_first: str | None, tool: str) -> str:
    """Re-check prompt for one question: its text, its options and its own stage-1 answer, nothing else."""
    options = dict(row["options"])
    intro = ("You are an industrial visual inspector. Image 1 is a verified NORMAL reference of the same asset "
             "category. Image 2 is the TARGET image. ")
    if uses_zoom(row, tool):
        intro += "Both images are enlarged central crops; the inspected asset is centred in both. "
    intro += "Use Image 1 only as the normal visual standard and answer about Image 2.\n\n"
    first = f'"{own_first}. {options[own_first]}"' if own_first in options else "not a valid option"
    review = (f"This is a second look at this question. In a first look, the answer to this question was {first}. "
              "That answer may be right or wrong. Re-examine the TARGET carefully before answering.\n\n")
    body = ""
    if "define" in tool:
        body = "Visual definitions of the options:\n" + "\n".join(
            f"- {text} {DEFINITIONS[text]}" for text in options.values()) + "\n\n"
    lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (intro + review + body + f"Question: {row['question']}\n{lines}\n\n"
            "Return exactly one option letter and nothing else.")


def is_routed(route: str, own_first: str | None, own_image_only: str | None) -> bool:
    """Whether one question is re-checked; sees only that question's own answers."""
    if route == "all":
        return True
    if route == "disagreement":
        return own_first != own_image_only
    raise ValueError(route)


def final_answer(own_first: str | None, own_recheck: str | None, routed: bool) -> str | None:
    """A routed question takes its own valid re-check answer; anything else keeps its stage-1 answer."""
    return own_recheck if routed and own_recheck else own_first


def zoom_path(zoom_dir: Path, kind: str, member: str) -> Path:
    return zoom_dir / kind / (hashlib.sha1(member.encode("utf-8")).hexdigest()[:16] + ".png")


def zoom_image(src: Path, out: Path) -> Path:
    if out.is_file() and out.stat().st_size > 0:
        return out
    from PIL import Image

    with Image.open(src) as h:
        img = h.convert("RGB")
    w, h_ = img.size
    crop = img.crop((round(ZOOM_BOX[0] * w), round(ZOOM_BOX[1] * h_), round(ZOOM_BOX[2] * w), round(ZOOM_BOX[3] * h_)))
    scale = min(ZOOM_SCALE, ZOOM_MAX_SIDE / max(crop.size))
    if scale > 1:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.BICUBIC)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.png")
    crop.save(tmp, format="PNG")
    tmp.replace(out)
    return out


def reusable(old: Mapping | None, model_key: str, tool: str, own_first: str | None, rule: str) -> bool:
    return bool(old and not old.get("error") and str(old.get("output", "")).strip()
                and old.get("model_key") == model_key and old.get("tool") == tool and old.get("rule") == rule
                and old.get("prompt_version") == PROMPT_VERSION and old.get("own_first_choice") == own_first)


def load_answers(path: Path) -> tuple[dict[str, str | None], dict[str, float]]:
    rows = read_jsonl(path)
    return ({str(r["id"]): (r.get("predicted_choice") or None) for r in rows},
            {str(r["id"]): float(r.get("latency_s") or 0) for r in rows})


def _contradicts(q1: Mapping, other: Mapping, choice: Mapping[str, str | None]) -> bool | None:
    a = base.predicted_semantic(q1, choice.get(str(q1["id"])))
    if other["question_type"] == Q2:
        d = base.predicted_semantic(other, choice.get(str(other["id"])))
        other_normal = None if d is None else d == "normal"
    else:
        kind = cons.q3_kind(other, choice.get(str(other["id"])))
        other_normal = None if kind is None else kind == "none"
    if a is None or other_normal is None:
        return None
    return (a == "normal") != other_normal


def contradiction_changes(rows: Sequence[Mapping], first: Mapping, final: Mapping) -> dict:
    """Contradictions present after but not before (new) and before but not after (resolved), per image."""
    by_core: dict[str, dict[str, Mapping]] = defaultdict(dict)
    for r in rows:
        by_core[str(r["core_id"])][r["question_type"]] = r
    out = {}
    for qt, key in ((Q2, "q1q2"), (Q3, "q1q3")):
        pairs = [(q[Q1], q[qt]) for q in by_core.values() if Q1 in q and qt in q]
        if not pairs:
            continue
        before = [_contradicts(a, b, first) for a, b in pairs]
        after = [_contradicts(a, b, final) for a, b in pairs]
        out[f"{key}_contradictions_new"] = sum(bool(y) and not x for x, y in zip(before, after))
        out[f"{key}_contradictions_resolved"] = sum(bool(x) and not y for x, y in zip(before, after))
    return out


def question_stats(rows: Sequence[Mapping], first: Mapping, final: Mapping, routed: set[str],
                   latency: Mapping[str, float]) -> list[dict]:
    out = []
    for qt in QUESTION_TYPES:
        qs = [r for r in rows if r["question_type"] == qt]
        if not qs:
            continue
        ids = [str(r["id"]) for r in qs]
        changed = [r for r in qs if final[str(r["id"])] != first[str(r["id"])]]
        out.append({
            "question_type": qt, "questions": len(qs),
            "accuracy": round(sum(final[str(r["id"])] == r["answer"] for r in qs) / len(qs), 6),
            "invalid_rate": round(sum(final[str(r["id"])] not in dict(r["options"]) for r in qs) / len(qs), 6),
            "rechecked": sum(i in routed for i in ids),
            "changed": len(changed),
            "changed_to_correct": sum(final[str(r["id"])] == r["answer"] for r in changed),
            "changed_to_wrong": sum(first[str(r["id"])] == r["answer"] for r in changed),
            "recheck_latency_s": round(sum(latency.get(i, 0.0) for i in ids if i in routed), 3),
        })
    return out


def summary_row(rows: Sequence[Mapping], first: Mapping, final: Mapping, routed: set[str],
                latency: Mapping[str, float]) -> dict:
    row = {**cons.consistency_row(cons.image_pairs(rows), final), **cons.q3_row(cons.q3_pairs(rows), final)}
    stats = question_stats(rows, first, final, routed, latency)
    for s in stats:
        if s["question_type"] != Q3:          # Q3 accuracy comes from q3_row
            row[f"{SHORT[s['question_type']]}_accuracy"] = s["accuracy"]
    row.update(contradiction_changes(rows, first, final))
    n_q = len(rows)
    rechecks = sum(s["rechecked"] for s in stats)
    row.update({"questions": n_q, "rechecks": rechecks,
                **{f"{SHORT[s['question_type']]}_rechecks": s["rechecked"] for s in stats},
                "avg_lvlm_calls_per_q": round(1 + rechecks / n_q, 4) if n_q else 0.0,
                "recheck_latency_s": round(sum(s["recheck_latency_s"] for s in stats), 3)})
    return row


def selection_rows(leaderboard: Sequence[Mapping], models: Sequence[str]) -> list[dict]:
    """Rank variants by SELECTION_CRITERION (dev only). Contradiction is shown for reference, not scored."""
    stage1 = {r["model_key"]: r for r in leaderboard if r["variant"] == "stage1"}
    deltas: dict[str, list[dict]] = defaultdict(list)
    for r in leaderboard:
        b = stage1.get(r["model_key"])
        if r["variant"] == "stage1" or b is None:
            continue
        deltas[r["variant"]].append({
            "q1": r["q1_anomaly_f1"] - b["q1_anomaly_f1"], "q2": r["q2_defect_macro_f1"] - b["q2_defect_macro_f1"],
            "con": r["consistency_error"] - b["consistency_error"], "calls": r["avg_lvlm_calls_per_q"]})
    out = []
    for variant, ds in deltas.items():
        if len(ds) != len(models):
            continue
        mean = lambda k: sum(d[k] for d in ds) / len(ds)  # noqa: E731
        out.append({"variant": variant, "models": len(ds), "delta_q1_anomaly_f1": round(mean("q1"), 6),
                    "delta_q2_defect_macro_f1": round(mean("q2"), 6),
                    "score": round((mean("q1") + mean("q2")) / 2, 6),
                    "avg_lvlm_calls_per_q": round(mean("calls"), 4),
                    "delta_consistency_error_not_scored": round(mean("con"), 6)})
    out.sort(key=lambda r: (-r["score"], r["avg_lvlm_calls_per_q"], r["variant"]))
    for rank, r in enumerate(out, 1):
        r["rank"] = rank
    return out


def make_frozen_config(dev_meta: Mapping, variant: str, selection: Sequence[Mapping] = ()) -> dict:
    """The single variant a gold run may use, fixed from a finished dev run."""
    tool, _, route = variant.partition("@")
    if dev_meta.get("protocol") != PROTOCOL or dev_meta.get("split") != "dev":
        raise ValueError("a frozen config must come from a dev run of this protocol")
    if variant not in dev_meta.get("variants", []):
        raise ValueError(f"{variant} was not run on dev")
    return {"protocol": PROTOCOL, "selected_on": "dev", "dev_qa": dev_meta["qa"], "dev_models": dev_meta["models"],
            "dev_benchmark_commit": dev_meta.get("benchmark_commit"), "variant": variant, "tool": tool, "route": route,
            "stage1_condition": dev_meta["stage1_condition"], "rule": dev_meta["rule"],
            "prompt_version": dev_meta["prompt_version"], "selection_criterion": dev_meta.get("selection_criterion"),
            "selection_rank": next((int(r["rank"]) for r in selection if r["variant"] == variant), None)}


def check_frozen(cfg: Mapping, stage1_condition: str, rule: str) -> tuple[str, str]:
    problems = []
    if cfg.get("protocol") != PROTOCOL or cfg.get("selected_on") != "dev":
        problems.append("not a dev-selected config of this protocol")
    if cfg.get("prompt_version") != PROMPT_VERSION:
        problems.append(f"prompt_version {cfg.get('prompt_version')} != {PROMPT_VERSION}")
    if cfg.get("stage1_condition") != stage1_condition:
        problems.append(f"stage1_condition {cfg.get('stage1_condition')} != {stage1_condition}")
    if cfg.get("rule") != rule:
        problems.append(f"rule {cfg.get('rule')} != {rule}")
    if cfg.get("tool") not in TOOLS or cfg.get("route") not in ROUTES:
        problems.append(f"unknown variant {cfg.get('tool')}@{cfg.get('route')}")
    if problems:
        raise RuntimeError("frozen config does not match this run: " + "; ".join(problems))
    return str(cfg["tool"]), str(cfg["route"])


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def gpu_info() -> dict | None:
    torch = sys.modules.get("torch")          # only if an adapter already imported it
    try:
        if torch is not None and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"name": props.name, "vram_gb": round(props.total_memory / 1024 ** 3, 1)}
    except Exception:
        pass
    return None


def main(argv: Sequence[str] | None = None, *, load_qa: Callable[[Path], list[dict]] = v2.load_gold_qa_v2,
         loader: Callable | None = None, infer: Callable | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--qa-file", type=Path, help="dev_master_qa_*.jsonl (dev: compare variants)")
    p.add_argument("--qa-dir", type=Path, help="gold_v2_final (gold: the one frozen variant)")
    p.add_argument("--stage1-dir", type=Path, required=True, help="predictions/<model>/<condition>.jsonl root")
    p.add_argument("--stage1-condition", default="normal_ref_retrieved_diffdate")
    p.add_argument("--image-only-dir", type=Path, help="predictions/<model>/image_only.jsonl root (default: stage1-dir)")
    p.add_argument("--image-only-csv", type=Path, help="alternative: gold prediction_review.csv")
    p.add_argument("--refs-csv", type=Path, required=True, help="retrieved_refs.csv (reference for the re-check)")
    p.add_argument("--rule", default="diffdate", choices=["any", "diffdate"])
    p.add_argument("--sup-zip", type=Path, required=True)
    p.add_argument("--unsup-zip", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["internvl3_5_8b", "qwen3_5_4b"])
    p.add_argument("--tools", nargs="+", choices=list(TOOLS), help="dev only (default: all)")
    p.add_argument("--routes", nargs="+", choices=list(ROUTES), help="dev only (default: all)")
    p.add_argument("--frozen-config", type=Path, help="gold: frozen_config.json written from the dev run")
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)

    if bool(args.qa_file) == bool(args.qa_dir):
        raise RuntimeError("give exactly one of --qa-file (dev) or --qa-dir (gold)")
    split = "gold" if args.qa_dir else "dev"
    frozen, frozen_sha = None, None
    if split == "gold":
        if not args.frozen_config:
            raise RuntimeError("a gold run needs --frozen-config chosen on the dev set")
        if args.tools or args.routes:
            raise RuntimeError("on gold the tool and route come only from --frozen-config")
        raw = args.frozen_config.read_bytes()
        frozen, frozen_sha = json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()
        tool, route = check_frozen(frozen, args.stage1_condition, args.rule)
        tools, routes = [tool], [route]
        old_meta = args.out_dir / "run_metadata.json"
        if old_meta.is_file():
            prev = json.loads(old_meta.read_text(encoding="utf-8")).get("frozen_config_sha256")
            if prev and prev != frozen_sha:
                raise RuntimeError(f"{args.out_dir} was already run on gold with another frozen config")
    else:
        if args.frozen_config:
            raise RuntimeError("--frozen-config is only for gold runs")
        tools, routes = list(args.tools or TOOLS), list(args.routes or ROUTES)
    variants = [variant_name(t, r) for t in tools for r in routes]

    if infer is None:
        from insplad_benchmark.hf_adapters import infer_hf as infer  # noqa: N813
    rows = read_jsonl(args.qa_file) if args.qa_file else load_qa(args.qa_dir)
    rows = [r for r in rows if r.get("question_type") in QUESTION_TYPES]
    is_dev = [r.get("protocol") == "gold_v2_dev" for r in rows]
    if not rows or (split == "dev" and not all(is_dev)) or (split == "gold" and any(is_dev)):
        raise RuntimeError(f"expected {'only gold_v2_dev' if split == 'dev' else 'gold'} questions for a {split} run")
    ids = [str(r["id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("question ids must be unique")
    with args.refs_csv.open(newline="", encoding="utf-8-sig") as h:
        refs = {r["image"]: r[f"retrieved_{args.rule}"] for r in csv.DictReader(h)}
    missing = sorted({str(r["image"]) for r in rows} - set(refs))
    if missing:
        raise RuntimeError(f"--refs-csv lacks {len(missing)} images, e.g. {missing[:2]}")
    image_only_csv: dict[tuple[str, str], str | None] = {}
    if args.image_only_csv:
        with args.image_only_csv.open(newline="", encoding="utf-8-sig") as h:
            image_only_csv = {(r["model_key"], str(r["id"])): (r.get("predicted_choice") or None) for r in csv.DictReader(h)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = base.prepare_images(rows, args.sup_zip, args.out_dir / "_images" / "target")
    normals = {m: base.safe_extract(args.unsup_zip, m, args.out_dir / "_images" / "normal")
               for m in sorted({refs[str(r["image"])] for r in rows})}
    zoom_dir = args.out_dir / "_images" / "zoom"

    def images_for(row: Mapping, tool: str) -> list[Path]:
        target_member = str(row["image"])
        ref_member = refs[target_member]
        if uses_zoom(row, tool):
            return [zoom_image(normals[ref_member], zoom_path(zoom_dir, "ref", ref_member)),
                    zoom_image(targets[target_member], zoom_path(zoom_dir, "target", target_member))]
        return [normals[ref_member], targets[target_member]]

    by_asset: dict[str, list[Mapping]] = defaultdict(list)
    for r in rows:
        by_asset[str(r["asset_category"])].append(r)
    leaderboard, per_question, per_asset, answers, statuses, route_stats = [], [], [], [], [], []
    found: dict[tuple[str, str], dict[str, str | None]] = {}

    def record(model_key: str, variant: str, first: Mapping, final: Mapping, routed: set[str],
               latency: Mapping[str, float], stage1_latency: float) -> None:
        found[(model_key, variant)] = dict(final)
        leaderboard.append({"model_key": model_key, "variant": variant,
                            **summary_row(rows, first, final, routed, latency),
                            "stage1_latency_s": round(stage1_latency, 3)})
        for s in question_stats(rows, first, final, routed, latency):
            per_question.append({"model_key": model_key, "variant": variant, **s})
        for asset, subset in sorted(by_asset.items()):
            per_asset.append({"model_key": model_key, "variant": variant, "asset_category": asset,
                              **summary_row(subset, first, final, routed, latency)})

    for model_key in args.models:
        stage1_path = args.stage1_dir / model_key / f"{args.stage1_condition}.jsonl"
        if not stage1_path.is_file():
            raise FileNotFoundError(f"stage-1 predictions missing: {stage1_path}")
        first, first_latency = load_answers(stage1_path)
        if any(i not in first for i in ids):
            raise RuntimeError(f"{stage1_path}: stage-1 predictions do not cover every question")
        image_only: dict[str, str | None] = {}
        io_path = (args.image_only_dir or args.stage1_dir) / model_key / "image_only.jsonl"
        if io_path.is_file():
            image_only = load_answers(io_path)[0]
        elif image_only_csv:
            image_only = {i: c for (m, i), c in image_only_csv.items() if m == model_key}
        if "disagreement" in routes and any(i not in image_only for i in ids):
            raise RuntimeError(f"{model_key}: the disagreement route needs image-only answers for every question")
        # per question, from that question's own stage-1 and image-only answers only
        routed = {route: {str(r["id"]) for r in rows if is_routed(route, first[str(r["id"])], image_only.get(str(r["id"])))}
                  for route in routes}
        route_stats.append({"model_key": model_key, "questions": len(rows), "image_only_available": bool(image_only),
                            "routed": {route: {SHORT[qt]: sum(str(r["id"]) in routed[route] for r in rows
                                                              if r["question_type"] == qt)
                                               for qt in QUESTION_TYPES if any(r["question_type"] == qt for r in rows)}
                                       for route in routes}})
        need = set().union(*routed.values())
        stage1_latency = sum(first_latency.get(i, 0.0) for i in ids)
        record(model_key, "stage1", first, first, set(), {}, stage1_latency)
        holder = base.LazyHFAdapter(base.MODEL_REGISTRY[model_key], loader=loader)
        try:
            for tool in tools:
                print(f"\n=== {model_key} / {tool} ({len(need)} questions) ===", flush=True)
                pred_path = args.out_dir / "predictions" / model_key / f"recheck_{tool}.jsonl"
                existing = {str(r["id"]): r for r in (read_jsonl(pred_path) if pred_path.is_file() else [])}
                done: dict[str, dict] = {}
                called = reused = errors = 0
                for row in rows:
                    qid = str(row["id"])
                    if qid not in need:
                        continue
                    own_first = first[qid]
                    old = existing.get(qid)
                    if reusable(old, model_key, tool, own_first, args.rule):
                        done[qid] = old
                        reused += 1
                        continue
                    images = images_for(row, tool)
                    prompt = build_prompt(row, own_first, tool)
                    adapter = holder.get()
                    started = time.time()
                    output, error = "", ""
                    try:
                        output = str(infer(adapter, images, prompt))
                    except Exception as exc:  # retried on the next run
                        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                        errors += 1
                    done[qid] = {"id": qid, "core_id": row["core_id"], "question_type": row["question_type"],
                                 "model_key": model_key, "tool": tool, "prompt_version": PROMPT_VERSION,
                                 "rule": args.rule, "own_first_choice": own_first, "zoomed": uses_zoom(row, tool),
                                 "images": len(images), "prompt_chars": len(prompt), "output": output,
                                 "predicted_choice": base.parse_choice(output, {str(k).upper() for k in row["options"]}),
                                 "latency_s": round(time.time() - started, 3), "error": error}
                    called += 1
                    if args.checkpoint_every > 0 and called % args.checkpoint_every == 0:
                        base.write_jsonl(pred_path, list(done.values()) + [r for i, r in existing.items() if i not in done])
                        print(f"[{model_key}/{tool}] {len(done)}/{len(need)} checkpoint", flush=True)
                base.write_jsonl(pred_path, list(done.values()) + [r for i, r in existing.items() if i not in done])
                statuses.append({"model_key": model_key, "tool": tool, "questions": len(need), "called_this_run": called,
                                 "reused_predictions": reused, "row_errors": errors,
                                 "status": "complete" if errors == 0 else "has_errors"})
                if errors:
                    msg = next(r["error"] for r in done.values() if r.get("error"))
                    print(f"[{model_key}/{tool}] {errors} errors, first: {msg[:300]}", flush=True)
                latency = {qid: float(rec.get("latency_s") or 0) for qid, rec in done.items()}
                for route in routes:
                    variant = variant_name(tool, route)
                    final = {}
                    for row in rows:
                        qid = str(row["id"])
                        on = qid in routed[route]
                        own_recheck = (done.get(qid) or {}).get("predicted_choice") if on else None
                        final[qid] = final_answer(first[qid], own_recheck, on)
                        answers.append({"model_key": model_key, "variant": variant, "id": qid,
                                        "core_id": row["core_id"], "asset_category": row["asset_category"],
                                        "question_type": row["question_type"], "answer": row["answer"],
                                        "stage1_choice": first[qid], "image_only_choice": image_only.get(qid),
                                        "rechecked": on, "recheck_choice": own_recheck, "final_choice": final[qid],
                                        "changed": final[qid] != first[qid],
                                        "stage1_correct": first[qid] == row["answer"],
                                        "final_correct": final[qid] == row["answer"]})
                    record(model_key, variant, first, final, routed[route], latency, stage1_latency)
        finally:
            adapter = None   # drop the loop's reference too, or the next model loads beside this one
            holder.release()

    pairs, loc_pairs = cons.image_pairs(rows), cons.q3_pairs(rows)
    sig = cons.significance_rows(pairs, loc_pairs, found, args.models, [("stage1", v) for v in variants],
                                 n_boot=args.n_boot)
    if not loc_pairs:
        sig = [r for r in sig if r["metric"] not in ("q3_accuracy", "q1q3_contradiction")]
    cons.write_csv(args.out_dir / "leaderboard.csv", leaderboard)
    cons.write_csv(args.out_dir / "per_question.csv", per_question)
    cons.write_csv(args.out_dir / "metrics_by_asset.csv", per_asset)
    cons.write_csv(args.out_dir / "answers.csv", answers)
    cons.write_csv(args.out_dir / "significance.csv", sig)
    selection = selection_rows(leaderboard, args.models) if split == "dev" else []
    if split == "dev":
        cons.write_csv(args.out_dir / "selection.csv", selection)
    (args.out_dir / "route_stats.json").write_text(json.dumps(route_stats, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    (args.out_dir / "run_status.json").write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    meta = {
        "protocol": PROTOCOL, "prompt_version": PROMPT_VERSION, "split": split,
        "qa": (args.qa_file or args.qa_dir).name, "questions": len(rows),
        "question_types": [qt for qt in QUESTION_TYPES if any(r["question_type"] == qt for r in rows)],
        "stage1_condition": args.stage1_condition, "rule": args.rule,
        "tools": tools, "routes": routes, "variants": variants, "models": list(args.models),
        "model_ids": {m: base.MODEL_REGISTRY[m].get("model_id") for m in args.models}, "backend": "hf",
        "benchmark_commit": git_commit(), "gpu": gpu_info(),
        "independence": "each question is re-checked on its own: the prompt holds only that question, its options and "
                        "its own stage-1 answer; routing uses only that question's own stage-1 and image-only answers; "
                        "a re-check changes only that question's answer; every call is a fresh single-turn chat",
        "decision": "routed question with a valid re-check answer -> that answer; otherwise the stage-1 answer",
        "contradiction_policy": "computed after all final answers are stored; never used for routing or editing",
        "recheck_sharing": "a re-check is computed once per question and tool (greedy decoding, same prompt and images) "
                           "and shared by the routes; a route's cost counts only the questions it re-checks",
        "cost": "rechecks = re-check calls counted for the variant; avg_lvlm_calls_per_q = 1 + rechecks / questions "
                "(stage 1 is one call per question); recheck_latency_s / stage1_latency_s = model-call seconds",
        "zoom": {"box": ZOOM_BOX, "scale": ZOOM_SCALE, "max_side": ZOOM_MAX_SIDE, "applied_to": [Q1, Q2]},
        "definitions": DEFINITIONS,
        "significance": f"image-level paired bootstrap vs stage1, {args.n_boot} resamples, two-sided 95% CI",
    }
    if split == "dev":
        meta["selection_criterion"] = SELECTION_CRITERION
        meta["purpose"] = "choose one variant on dev; gold runs only that frozen variant"
    else:
        meta["frozen_config"] = frozen
        meta["frozen_config_sha256"] = frozen_sha
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    errors = sum(s["row_errors"] for s in statuses)
    print(json.dumps(route_stats, ensure_ascii=False, indent=2))
    if errors:
        print(f"WARNING: {errors} rows failed; re-run the same cell to retry only those rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
