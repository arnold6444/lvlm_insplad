#!/usr/bin/env python3
"""Opt-in full-dev Q2 cue/Q3 disagreement experiment; retains cached Q1 exactly."""
from __future__ import annotations

import hashlib
import sys
from collections import defaultdict

import run_gold_v2_q2_q3_agent_dev as old

VERSION = 1
PROTOCOL = "gold_v2_q2_q3_refined_dev"
FOLDER_PREFIX = "gold_v2_q2_q3_refined_dev"
POLICIES = ("baseline", "previous_agent", "fresh_define", "q2_cues",
            "q2_cues_q3_position", "q2_cues_q3_review", "matched_reask")
COMPARISONS = [("baseline", p) for p in POLICIES[1:]] + [
    ("previous_agent", "q2_cues_q3_review"), ("fresh_define", "q2_cues"),
    ("q2_cues", "q2_cues_q3_position"), ("q2_cues_q3_position", "q2_cues_q3_review"),
    ("matched_reask", "q2_cues_q3_review")]
DEFINITIONS = {
    "Normal / no defect.": "no visible corrosion, missing component or attached foreign nesting material. "
        "Normal metal need not be shiny or uniformly coloured; lighting, shadows, paint and background differences alone are not defects.",
    "Rust / corrosion.": "oxidation on the METAL hardware can appear orange/red-brown OR dark brown/black, "
        "with irregular mottling, roughness, flakes, scaling or pitting. Bright orange is not required. Inspect rods, "
        "links, bolts and clamps. Dark shadow or smooth dark paint alone is insufficient; distinguish surface texture from loose material.",
    "Missing cap / missing insulator component.": "a required glass disc, skirt, cap or repeated insulator component "
        "is absent or visibly broken, interrupting the repeated structure or leaving exposed hardware. A metal cap can "
        "remain even when its glass disc/skirt is missing. Different viewpoints or a part hidden by another part alone do not prove loss.",
    "Bird nest / foreign nesting material.": "a separate cluster of intertwined twigs, straw or grass is lodged "
        "against or wrapped around the hardware. Thin grey/brown material can blend with the background. Background "
        "branches, normal cable strands and rough metal corrosion alone are not a nest.",
}


def select_rows(rows, n_images=240, seed=42):
    """Fixed design40 + additional dev; labels stratify sampling, never model outputs."""
    by_core = defaultdict(dict)
    for row in rows:
        if row.get("protocol") != "gold_v2_dev":
            raise ValueError("only gold_v2_dev questions are allowed")
        if row["question_type"] not in (old.agent.Q1, old.agent.Q2):
            continue
        qs = by_core[row["core_id"]]
        if row["question_type"] in qs:
            raise ValueError("duplicate question in a core")
        qs[row["question_type"]] = row
    for qs in by_core.values():
        if set(qs) != {old.agent.Q1, old.agent.Q2}:
            raise ValueError("each dev core needs Q1 and Q2")
        a, b = qs[old.agent.Q1], qs[old.agent.Q2]
        if any(a.get(k) != b.get(k) for k in ("image", "asset_category", "anomaly_gt", "defect_gt")):
            raise ValueError("Q1/Q2 image labels disagree")
    if not 1 <= n_images <= min(len(by_core), 240):
        raise ValueError("use 1..240 images within the available dev QA")
    # The diagnosed cohort is fixed at the original seed42, even if sampling seed changes.
    design = old.diagnostic.select_subset(rows, min(40, len(by_core)), 42)
    design_ids = list(dict.fromkeys(r["core_id"] for r in design))
    remaining = {c: qs for c, qs in by_core.items() if c not in set(design_ids)}
    strata = defaultdict(list)
    for core, qs in remaining.items():
        q = qs[old.agent.Q1]
        strata[(q["asset_category"], q["anomaly_gt"], tuple(q["options"].items()))].append(core)
    for cores in strata.values():
        cores.sort(key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    chosen = design_ids[:n_images]
    while len(chosen) < n_images:
        for key in sorted(strata):
            if strata[key]:
                chosen.append(strata[key].pop(0))
            if len(chosen) == n_images:
                break
    selected = [by_core[c][qt] for c in chosen for qt in (old.agent.Q1, old.agent.Q2)]
    cohorts = {"all_dev": set(chosen), "design_dev40": set(chosen) & set(design_ids),
               "additional_dev": set(chosen) - set(design_ids)}
    return selected, cohorts


def definition_prompt(row, revised=False):
    defs = DEFINITIONS if revised else old.agent.DEFINITIONS
    text = "Visual definitions of the options:\n" + "\n".join(f"- {v} {defs[v]}" for v in row["options"].values())
    intro = old.agent.build_prompt(row, None, "define").split("This is a second look at this question.")[0]
    options = "\n".join(f"{k}. {v}" for k, v in row["options"].items())
    return intro + text + f"\n\nQuestion: {row['question']}\n{options}\n\nReturn exactly one option letter and nothing else."


def cues():
    return ("Inspect the TARGET hardware itself. Corrosion may be dark/black mottled, rough or flaky metal; "
            "bright orange is not required and it may span a rod/link rather than a compact patch. For missing "
            "insulator material, look for an interrupted repeated glass-disc/skirt structure even if a metal cap "
            "remains; the evidence region is the gap or exposed/broken structure. Foreign nesting material may "
            "be thin grey/brown intertwined twigs attached to the hardware, not just a large nest. Do not count "
            "background vegetation, normal cable strands, viewpoint, smooth paint or lighting differences alone. "
            "Judge visible evidence; neither absence nor presence is preferred.")


def presence_task(row):
    options = old.options_for(row["core_id"], ["No visible defect evidence.", "Visible defect evidence."], "q3_refined_presence")
    task = {**row, "question": "Does the TARGET contain visible defect evidence on its hardware or insulator structure?", "options": options}
    return task, old.direct_prompt(task).replace("\n\nQuestion:", "\n\n"+cues()+"\n\nQuestion:", 1)


def position_prompt(row):
    task = {**row, "question": "Where is the visible defect evidence in the TARGET? For missing material, locate the gap or broken repeated structure. Choose no-region when there is no visible defect evidence."}
    text = old.position_prompt(task)
    return text.replace("\n\nQuestion:", "\n\n"+cues()+"\n\nQuestion:", 1)


def disagreement(row, presence, position):
    task, _ = presence_task(row)
    p = task["options"].get(presence)
    if p is None or position not in row["options"]:
        return False
    return (p == "Visible defect evidence.") != (position != row["none_choice"])


def review_prompt(row, presence, position):
    task, _ = presence_task(row)
    history = ("Two independent attempts at THIS localization question disagreed. "
               f'Its evidence check answered "{task["options"].get(presence, "INVALID")}". '
               f'Its grid attempt answered "{row["options"].get(position, "INVALID")}". '
               "Neither answer is a fact. Resolve from the TARGET image and visible defect evidence. "
               "Do not automatically follow either earlier answer. A region requires visible evidence, "
               "while a small or distributed defect need not look like a compact patch.")
    return position_prompt(row).replace("\n\nQuestion:", "\n\n"+history+"\n\nQuestion:", 1)


def audit_prompts(rows, q3, first, models):
    result = []
    for model in models:
        for r in rows:
            if r["question_type"] == old.agent.Q2:
                for step, prompt in (("q2_reask", old.agent.build_prompt(r, first[model][r["id"]], "reask")),
                                     ("q2_define", old.agent.build_prompt(r, first[model][r["id"]], "define")),
                                     ("q2_fresh_define", definition_prompt(r)), ("q2_cues", definition_prompt(r, True))):
                    result.append({"model_key": model, "id": r["id"], "step": step, "prompt": prompt})
        for r in q3:
            for step, prompt in (("q3_direct", old.direct_prompt(r)), ("q3_presence", old.presence_task(r)[1]),
                                 ("q3_position", old.position_prompt(r)), ("q3_cue_presence", presence_task(r)[1]),
                                 ("q3_cue_position", position_prompt(r))):
                result.append({"model_key": model, "id": r["id"], "step": step, "prompt": prompt})
    return result


def run_image(q2, r3, own_first, call):
    reask = call(q2, old.agent.build_prompt(q2, own_first, "reask"), "q2_reask")
    define = call(q2, old.agent.build_prompt(q2, own_first, "define"), "q2_define")
    fresh = call(q2, definition_prompt(q2), "q2_fresh_define")
    revised = call(q2, definition_prompt(q2, True), "q2_cues")
    direct = call(r3, old.direct_prompt(r3), "q3_direct")
    legacy_task, prompt = old.presence_task(r3)
    legacy_presence = call(legacy_task, prompt, "q3_presence")
    legacy_positive = legacy_task["options"].get(legacy_presence["predicted_choice"]) == "A visible localized defect region."
    legacy_position = call(r3, old.position_prompt(r3), "q3_position") if legacy_positive else None
    legacy_decision = legacy_task["options"].get(legacy_presence["predicted_choice"])
    legacy_choice = (r3["none_choice"] if legacy_decision == "No visible localized defect region." else
                     legacy_position["predicted_choice"] if legacy_position and legacy_position["predicted_choice"] else direct["predicted_choice"])
    task, prompt = presence_task(r3)
    presence = call(task, prompt, "q3_cue_presence")
    # Run position even after a negative presence check, preventing a one-answer veto.
    position = call(r3, position_prompt(r3), "q3_cue_position")
    needs_review = disagreement(r3, presence["predicted_choice"], position["predicted_choice"])
    review = call(r3, review_prompt(r3, presence["predicted_choice"], position["predicted_choice"]), "q3_cue_review") if needs_review else None
    position_choice = position["predicted_choice"] or direct["predicted_choice"]
    review_choice = review["predicted_choice"] if review and review["predicted_choice"] else position_choice
    controls = [call(r3, old.direct_prompt(r3), f"q3_reask_{i+1}") for i in range(2+int(needs_review))]
    control_choice = next((r["predicted_choice"] for r in reversed(controls) if r["predicted_choice"]), direct["predicted_choice"])
    # A policy sees only dependencies it actually uses; failure never becomes a selected configuration.
    result = {
        "baseline": (own_first, direct["predicted_choice"], [direct]),
        "previous_agent": (define["predicted_choice"] or own_first, legacy_choice, [direct, define, legacy_presence]+([legacy_position] if legacy_position else [])),
        "fresh_define": (fresh["predicted_choice"] or own_first, direct["predicted_choice"], [direct, fresh]),
        "q2_cues": (revised["predicted_choice"] or own_first, direct["predicted_choice"], [direct, revised]),
        "q2_cues_q3_position": (revised["predicted_choice"] or own_first, position_choice, [direct, revised, position]),
        "q2_cues_q3_review": (revised["predicted_choice"] or own_first, review_choice, [direct, revised, presence, position]+([review] if review else [])),
        "matched_reask": (reask["predicted_choice"] or own_first, control_choice, [direct, reask, *controls]),
    }
    assert len(result["matched_reask"][2]) == len(result["q2_cues_q3_review"][2])
    return result


def main(argv=None, *, loader=None, infer=None):
    return old.main(argv, loader=loader, infer=infer, refinement=sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
