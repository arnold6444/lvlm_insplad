#!/usr/bin/env python3
"""Build paper tables for the gold-v2 benchmark from published results (no inference).

Reads ``results/latest/gold_v2_consistency_metrics`` (every condition's Q1/Q2/Q3 scores and cross-question
consistency, plus ``significance_main.csv`` when present) and writes CSV + Markdown tables to
``reports/paper_tables/``:

- table1_main: 4 models x 4 main conditions, all metrics
- table2_reference: normal / positive / normal+positive reference ablation
- table3_vision_expert: 4 experts x 5 evidence types, mean over models
- table4_q3_baselines: Q3 against constant answers (the set is small and centre/none heavy)
- table5_significance: paired bootstrap deltas with 95% CI for the main comparisons
- table6_score_baselines: score-only Q1 baselines (PatchCore, retrieval cosine) from gold_v2_main_rerun
- all_conditions: every model x condition row (``@original`` = run replaced by the greedy re-run)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {"qwen3_5_4b": "Qwen3.5-4B", "internvl3_5_8b": "InternVL3.5-8B",
               "minicpm_v_4_5": "MiniCPM-V-4.5", "llava_onevision_7b": "LLaVA-OneVision-7B"}
MAIN = {"image_only": "Image only", "normal_reference": "+ Normal ref",
        "vx:patchcore:score": "+ Expert score", "ref+patchcore:score": "+ Normal ref + Expert score",
        "normal_ref_retrieved": "+ Similar normal ref", "normal_ref_retrieved_diffdate": "+ Similar normal ref (other flight)"}
REFERENCE = {"image_only": "Image only", "normal_reference": "+ Normal ref", "positive_reference": "+ Defect ref",
             "normal_positive_reference": "+ Normal ref + Defect ref"}
METRICS = [("q1_anomaly_f1", "Q1 F1", True), ("q2_defect_macro_f1", "Q2 macro-F1", True),
           ("q3_accuracy", "Q3 Acc", True), ("consistency_error", "Q1-Q2 contra.", False),
           ("q1q3_contradiction", "Q1-Q3 contra.", False), ("joint_accuracy", "Joint Acc", True),
           ("false_defect_rate", "FDR", False)]
LOWER_IS_BETTER = {k for k, _, up in METRICS if not up}


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


def num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt(v) -> str:
    x = num(v)
    return "–" if x is None else f"{x:.3f}"


def markdown(rows: Sequence[Mapping], columns: Sequence[tuple[str, str]], bold: Mapping[tuple[int, str], bool] = {}) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [head, sep]
    for i, r in enumerate(rows):
        cells = []
        for key, _ in columns:
            v = r.get(key)
            text = fmt(v) if isinstance(v, float) or v is None else str(v)
            cells.append(f"**{text}**" if bold.get((i, key)) else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_model_table(rows: Sequence[Mapping], conditions: Mapping[str, str]) -> tuple[list[dict], dict]:
    index = {(r["model_key"], r["condition"]): r for r in rows}
    out, bold = [], {}
    for model in MODEL_NAMES:
        block = [(c, index[(model, c)]) for c in conditions if (model, c) in index]
        if not block:
            continue
        start = len(out)
        for c, r in block:
            out.append({"Model": MODEL_NAMES[model], "Condition": conditions[c],
                        **{k: num(r.get(k)) for k, _, _ in METRICS}})
        for k, _, up in METRICS:
            vals = [(j, out[j][k]) for j in range(start, len(out)) if out[j][k] is not None]
            if vals:
                best = (max if up else min)(v for _, v in vals)
                for j, v in vals:
                    if abs(v - best) < 1e-12:
                        bold[(j, k)] = True
    return out, bold


def vision_expert_table(rows: Sequence[Mapping]) -> list[dict]:
    groups: dict[tuple[str, str], list[Mapping]] = {}
    for r in rows:
        if r["condition"].startswith("vx:"):
            _, expert, evidence = r["condition"].split(":", 2)
            groups.setdefault((expert, evidence), []).append(r)
    out = []
    for (expert, evidence), rs in sorted(groups.items()):
        item = {"Expert": expert, "Evidence": evidence, "Models": len(rs)}
        for k, _, _ in METRICS:
            vals = [num(r.get(k)) for r in rs if num(r.get(k)) is not None]
            item[k] = sum(vals) / len(vals) if vals else None
        out.append(item)
    return sorted(out, key=lambda x: -(x["q1_anomaly_f1"] or 0))


def significance_table(sig: Sequence[Mapping]) -> list[dict]:
    out = []
    keyed: dict[tuple[str, str, str], dict] = {}
    for r in sig:
        key = (r["model_key"], r["baseline"], r["condition"])
        item = keyed.setdefault(key, {"Model": MODEL_NAMES.get(r["model_key"], r["model_key"]),
                                      "Comparison": f"{MAIN.get(r['baseline'], r['baseline'])} → "
                                                    f"{MAIN.get(r['condition'], r['condition'])}"})
        mark = "*" if str(r.get("significant")).lower() == "true" else ""
        item[r["metric"]] = f"{float(r['delta']):+.3f} [{float(r['ci_low']):+.3f}, {float(r['ci_high']):+.3f}]{mark}"
    out.extend(keyed.values())
    return out


def build(results_dir: Path, out_dir: Path) -> dict:
    cons_dir = results_dir / "gold_v2_consistency_metrics"
    rows = read_csv(cons_dir / "consistency_leaderboard.csv")
    meta = json.loads((cons_dir / "run_metadata.json").read_text(encoding="utf-8"))
    sig_path = cons_dir / "significance_main.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [(k, label) for k, label, _ in METRICS]
    md = ["# Paper tables (gold v2)", "",
          f"Source: `results/latest/gold_v2_consistency_metrics` ({len(rows)} model x condition rows). "
          "Each question was asked independently (MMAD protocol). Bold = best within each model block "
          "(contradiction and FDR: lower is better). Generated by `scripts/build_paper_tables.py`.", "",
          "**Latest results (verification agent, gold246, follow-up dev experiments): "
          "[latest_results.md](latest_results.md).**", ""]

    t1, b1 = per_model_table(rows, MAIN)
    write_csv(out_dir / "table1_main.csv", t1)
    md += ["## Table 1. Main conditions", "", markdown(t1, [("Model", "Model"), ("Condition", "Condition")] + metric_cols, b1), ""]

    t2, b2 = per_model_table(rows, REFERENCE)
    write_csv(out_dir / "table2_reference.csv", t2)
    md += ["## Table 2. Reference ablation", "", markdown(t2, [("Model", "Model"), ("Condition", "Condition")] + metric_cols, b2), ""]

    t3 = vision_expert_table(rows)
    write_csv(out_dir / "table3_vision_expert.csv", t3)
    md += ["## Table 3. Vision expert evidence (mean over models, sorted by Q1 F1)", "",
           markdown(t3, [("Expert", "Expert"), ("Evidence", "Evidence")] + metric_cols), ""]

    base = meta.get("q3_constant_baselines", {})
    t4 = [{"Answer policy": "Always 'no defect region'", "Q3 Acc": base.get("always_no_defect_region")},
          {"Answer policy": "Always 'center'", "Q3 Acc": base.get("always_center")}]
    t4 += [{"Answer policy": f"{MODEL_NAMES[r['model_key']]} (image only)", "Q3 Acc": num(r.get("q3_accuracy"))}
           for r in rows if r["condition"] == "image_only" and r["model_key"] in MODEL_NAMES]
    write_csv(out_dir / "table4_q3_baselines.csv", t4)
    md += ["## Table 4. Q3 against constant answers", "",
           f"Q3 has {base.get('questions', '?')} questions; answers: {base.get('answer_regions', {})}.", "",
           markdown(t4, [("Answer policy", "Answer policy"), ("Q3 Acc", "Q3 Acc")]), ""]

    n_sig = 0
    if sig_path.is_file():
        sig = read_csv(sig_path)
        t5 = significance_table(sig)
        n_sig = sum(str(r.get("significant")).lower() == "true" for r in sig)
        write_csv(out_dir / "table5_significance.csv", t5)
        md += ["## Table 5. Paired bootstrap (delta [95% CI], * = CI excludes 0)", "",
               markdown(t5, [("Model", "Model"), ("Comparison", "Comparison")] + metric_cols), ""]
    else:
        md += ["## Table 5. Paired bootstrap", "", "_significance_main.csv not published yet_", ""]

    score_path = results_dir / "gold_v2_main_rerun" / "score_baselines.csv"
    if score_path.is_file():
        t6 = [{"Method": r["method"], "q1_anomaly_f1": num(r.get("q1_anomaly_f1")), "AUROC": num(r.get("auroc")),
               "Threshold": r.get("threshold", ""), "Threshold source": r.get("threshold_source", "")}
              for r in read_csv(score_path)]
        write_csv(out_dir / "table6_score_baselines.csv", t6)
        md += ["## Table 6. Score-only Q1 baselines (no LVLM)", "",
               "An LVLM condition should beat these; otherwise its Q1 gain may come from the score or retrieval "
               "similarity alone.", "",
               markdown(t6, [("Method", "Method"), ("q1_anomaly_f1", "Q1 F1"), ("AUROC", "AUROC"),
                             ("Threshold", "Threshold"), ("Threshold source", "Threshold source")]), ""]
    else:
        md += ["## Table 6. Score-only Q1 baselines", "", "_gold_v2_main_rerun/score_baselines.csv not published yet_", ""]

    full = [{"Model": MODEL_NAMES.get(r["model_key"], r["model_key"]), "Condition": r["condition"],
             **{k: num(r.get(k)) for k, _, _ in METRICS}} for r in rows]
    write_csv(out_dir / "all_conditions.csv", full)
    (out_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"rows": len(rows), "significance_rows": n_sig, "out_dir": str(out_dir)}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "latest")
    p.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "paper_tables")
    args = p.parse_args(argv)
    print(json.dumps(build(args.results_dir, args.out_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
