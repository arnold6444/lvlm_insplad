#!/usr/bin/env python3
"""Draw the paper figures (English and Korean) into figures/.

usage: python make_figures.py --sup-zip <supervised_fault_classification.zip>

Answers and references are read from the committed result files, never typed in:
  results/latest/gold_v2_matched_call/answers.csv     direct (baseline) and agent (previous_agent) answers
  results/latest/gold_v2_minicpm_greedy/answers.csv   MiniCPM-V-4.5 answers with a defect reference
  results/latest/gold_v2_reference/reference_manifest.csv   the retrieved defect reference of each target
  results/latest/gold_v2_final/manual_review_final.csv      gold labels and image paths
InsPLAD images: Vieira-e-Silva et al., CC BY-NC 3.0.
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
DPI = 300
# One gold image per class; normal/rust/bird nest are the same asset type, so only the condition differs.
EXAMPLES = [("normal", "core-07f9ce99fdf16056"), ("rust", "core-28a24ec7956ea697"),
            ("bird-nest", "core-3df353cc203c9ed3"), ("missing-cap", "core-bb120ed6aa3fc6a1")]
REF_EXAMPLES = ["core-bb120ed6aa3fc6a1", "core-28a24ec7956ea697"]
NONE = "No localized defect region is present."
GREEN, RED, GREY = "#1b7f3b", "#b3261e", "#555555"
FACE = {"src": "#eef3fb", "mid": "#f4f4f4", "gold": "#fff4dc", "dev": "#e9f5ec"}

T = {
    "en": dict(
        font="DejaVu Sans",
        cls={"normal": "Normal", "rust": "Rust", "bird-nest": "Bird nest", "missing-cap": "Missing cap"},
        asset={"vari-grip": "vari-grip", "glass-insulator": "glass insulator"},
        anomalous="Anomalous", normal="Normal", region="region", none="none",
        gt="Truth", direct="Direct", agent_lbl="Agent", shared="shared",
        ref_target="Target", ref_ref="Defect reference", minicpm="MiniCPM-V-4.5 Q1:",
        ref_caption="{label}, cos {sim:.2f}",
        pipe=dict(
            ad="InsPLAD anomaly-\ndetection release\n17 assets, 6,481 test",
            fc="InsPLAD fault-\nclassification release\n5 assets, 6,417 val",
            align="Exact SHA-256\nalignment\n8,990 identical images\n−6 label conflicts\nAligned Core 1,378",
            gold="Gold v2 (frozen)\n139 defective +\n107 normal images\n534 questions\nmanually reviewed",
            cond="Conditions × models\nimage only, + normal ref.,\n+ defect ref., + both,\n+ expert cues (4 × 5)\n4 open LVLMs (+ API)",
            dev="Dev set\n240 images from\nsupervised train\nleakage-checked",
            design="Agent design\nchosen on dev\n(definitions,\npresence staging)",
            run="Agent run once\non gold\nInternVL3.5-8B\n+ matched re-ask",
            evalr="One evaluator\nQ1 F1/BAcc,\nQ2 macro-F1, joint,\nconsistency, FDR,\npaired bootstrap",
        ),
        agent=dict(
            inp="Target +\nretrieved normal\nreference\n(other capture date)",
            direct="Direct\nanswers\nQ1, Q2, Q3\n(1 call each)",
            q1="Q1 kept (never re-checked)",
            q2="Q2 second look:\nown first answer +\none-line option definitions",
            q3a="Q3 presence:\nis a defect region present?",
            q3b="Q3 position:\nwhich 3 × 3 cell?",
            yes="yes", no="no → none",
            final="Final Q1, Q2, Q3 stored; contradictions computed only afterwards (no cross-question routing)",
            ctrl="Matched re-ask control\nsame Q2/Q3 calls per image,\nown-answer re-checks only,\nno definitions, no staging",
        ),
    ),
    "ko": dict(
        font="Malgun Gothic",
        cls={"normal": "정상", "rust": "녹", "bird-nest": "새집", "missing-cap": "캡 누락"},
        asset={"vari-grip": "vari-grip", "glass-insulator": "유리 애자"},
        anomalous="이상", normal="정상", region="영역", none="없음",
        gt="정답", direct="직접", agent_lbl="에이전트", shared="공통",
        ref_target="대상", ref_ref="결함 참조", minicpm="MiniCPM-V-4.5 Q1:",
        ref_caption="{label}, 코사인 {sim:.2f}",
        pipe=dict(
            ad="InsPLAD\n이상탐지 공개본\n설비 17종, test 6,481장",
            fc="InsPLAD\n결함분류 공개본\n설비 5종, val 6,417장",
            align="SHA-256\n정확 일치 정렬\n동일 이미지 8,990장\n라벨 충돌 6장 제외\nAligned Core 1,378장",
            gold="gold v2 (동결)\n결함 139장 +\n정상 107장\n534문항\n수작업 검토",
            cond="조건 × 모델\n이미지만, + 정상 참조,\n+ 결함 참조, + 둘 다,\n+ expert 단서 (4 × 5)\n공개 LVLM 4종 (+ API)",
            dev="dev 세트\nsupervised train\n240장\n누출 검사",
            design="에이전트 설계\ndev에서 선택\n(정의, 유무 먼저\n묻는 단계)",
            run="gold에서\n1회 실행\nInternVL3.5-8B\n+ 호출 수 대조",
            evalr="같은 채점기\nQ1 F1/BAcc,\nQ2 macro-F1, 동시 정답,\n모순율, 결함 오판율,\n대응 bootstrap",
        ),
        agent=dict(
            inp="대상 +\n검색한 정상 참조\n(다른 촬영일)",
            direct="직접 답\nQ1, Q2, Q3\n(각 1회 호출)",
            q1="Q1 유지 (재확인 안 함)",
            q2="Q2 재확인:\n자기 첫 답 +\n보기별 한 줄 정의",
            q3a="Q3 유무:\n결함 영역이 있는가?",
            q3b="Q3 위치:\n3 × 3 중 어느 칸?",
            yes="예", no="아니오 → 없음",
            final="최종 Q1·Q2·Q3를 저장한 뒤에만 모순을 계산 (문항 간 답 전달·덮어쓰기 없음)",
            ctrl="호출 수 대조군\n이미지마다 Q2·Q3 호출 수 동일,\n자기 답 재확인만,\n정의 없음, 단계 없음",
        ),
    ),
}


def read(rel):
    with (ROOT / rel).open(encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def setup(lang):
    plt.rcParams.update({"font.family": T[lang]["font"], "font.size": 7, "axes.unicode_minus": False})


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, kind, size=6.6, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", linewidth=0.8,
                                edgecolor="#444444", facecolor=FACE[kind], linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.25)


def arrow(ax, a, b, text=None, ls="-", text_dy=0.015):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#333333",
                                 linestyle=ls, shrinkA=0, shrinkB=0))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + text_dy, text, ha="center", va="bottom", fontsize=6.2, color=GREY)


def pipeline(lang):
    t = T[lang]["pipe"]
    fig, ax = canvas(7.0, 2.7)
    w = 0.17
    xs = [0.01, 0.215, 0.42, 0.625, 0.83]
    top_y, top_h, bot_y, bot_h = 0.55, 0.42, 0.04, 0.42
    box(ax, xs[0], 0.775, w, 0.195, t["ad"], "src")
    box(ax, xs[0], 0.55, w, 0.195, t["fc"], "src")
    box(ax, xs[1], top_y, w, top_h, t["align"], "mid")
    box(ax, xs[2], top_y, w, top_h, t["gold"], "gold")
    box(ax, xs[3], top_y, w, top_h, t["cond"], "mid")
    box(ax, xs[4], 0.29, w, 0.42, t["evalr"], "gold")
    box(ax, xs[0], bot_y, w, bot_h, t["dev"], "dev")
    box(ax, xs[1], bot_y, w, bot_h, t["design"], "dev")
    box(ax, xs[3], bot_y, w, bot_h, t["run"], "dev")
    arrow(ax, (xs[0] + w, 0.87), (xs[1], 0.82))
    arrow(ax, (xs[0] + w, 0.65), (xs[1], 0.70))
    arrow(ax, (xs[1] + w, 0.76), (xs[2], 0.76))
    arrow(ax, (xs[2] + w, 0.76), (xs[3], 0.76))
    arrow(ax, (xs[3] + w, 0.76), (xs[4] + w / 2, 0.71))
    arrow(ax, (xs[0] + w / 2, 0.55), (xs[0] + w / 2, bot_y + bot_h))      # dev comes from the fault-classification train split
    arrow(ax, (xs[0] + w, 0.25), (xs[1], 0.25))
    arrow(ax, (xs[1] + w, 0.25), (xs[3], 0.25))
    arrow(ax, (xs[2] + w / 2, top_y), (xs[3] + 0.02, bot_y + bot_h), ls="--")  # the single gold run uses the frozen gold set
    arrow(ax, (xs[3] + w, 0.25), (xs[4] + w / 2, 0.29))
    return fig


def agent(lang):
    t = T[lang]["agent"]
    fig, ax = canvas(7.0, 3.0)
    box(ax, 0.01, 0.42, 0.15, 0.30, t["inp"], "src")
    box(ax, 0.20, 0.42, 0.13, 0.30, t["direct"], "mid")
    arrow(ax, (0.16, 0.57), (0.20, 0.57))
    box(ax, 0.39, 0.83, 0.25, 0.12, t["q1"], "mid")
    box(ax, 0.39, 0.53, 0.25, 0.22, t["q2"], "gold")
    box(ax, 0.39, 0.27, 0.25, 0.18, t["q3a"], "gold")
    box(ax, 0.72, 0.27, 0.20, 0.18, t["q3b"], "gold")
    for y in (0.89, 0.64, 0.36):
        arrow(ax, (0.33, 0.57), (0.39, y))
    arrow(ax, (0.64, 0.36), (0.72, 0.36), text=t["yes"])
    box(ax, 0.01, 0.03, 0.98, 0.10, t["final"], "dev", size=6.4)
    arrow(ax, (0.515, 0.27), (0.515, 0.13))
    ax.text(0.525, 0.20, t["no"], ha="left", va="center", fontsize=6.2, color=GREY)
    arrow(ax, (0.82, 0.27), (0.82, 0.13))
    for x0, y0 in ((0.64, 0.89), (0.64, 0.64)):  # Q1 and Q2 finals go straight down the right margin
        ax.plot([x0, 0.96, 0.96], [y0, y0, 0.45], color="#333333", linewidth=0.8)
    arrow(ax, (0.96, 0.45), (0.96, 0.13))
    box(ax, 0.01, 0.17, 0.32, 0.20, t["ctrl"], "mid", size=6.0, dashed=True)
    return fig


def answers():
    return {(r["policy"], r["core_id"], r["question_type"]): r["predicted_text"]
            for r in read("results/latest/gold_v2_matched_call/answers.csv")}


def qa_examples(lang, z, labels):
    t = T[lang]
    a = answers()
    q1t = {"Anomalous.": t["anomalous"], "Normal.": t["normal"]}
    q2t = {"Normal / no defect.": "normal", "Rust / corrosion.": "rust",
           "Missing cap / missing insulator component.": "missing-cap", "Bird nest / foreign nesting material.": "bird-nest"}
    fig = plt.figure(figsize=(7.0, 2.9))
    # row label, Q1, Q2, Q3 (fraction of figure width, per panel); the Korean row label "에이전트" is wider
    cols = [0.0, 0.047, 0.118, 0.198] if lang == "en" else [0.0, 0.062, 0.118, 0.198]
    for k, (cls, core) in enumerate(EXAMPLES):
        lab = labels[core]
        left = 0.005 + k * 0.25
        ax = fig.add_axes([left, 0.33, 0.24, 0.60])
        ax.imshow(Image.open(io.BytesIO(z.read(lab["image"]))).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"({'abcd'[k]}) {t['cls'][cls]}, {t['asset'].get(lab['asset_category'], lab['asset_category'])}",
                     fontsize=7.2, pad=3)
        anom = lab["anomaly_gt"] == "anomalous"
        rows = [("", [("Q1", None), ("Q2", None), ("Q3", None)]),
                (t["gt"], [(t["anomalous"] if anom else t["normal"], None), (t["cls"][cls], None),
                           (t["region"] if anom else t["none"], None)])]
        for pol, name in (("baseline", t["direct"]), ("previous_agent", t["agent_lbl"])):
            q1 = a[(pol, core, "anomaly_detection")]
            q2 = q2t[a[(pol, core, "defect_classification")]]
            q3 = a[(pol, core, "defect_localization")] != NONE
            rows.append((name, [(q1t[q1] if pol == "baseline" else t["shared"], None if pol != "baseline" else (q1 == "Anomalous.") == anom),
                                (t["cls"][q2], q2 == cls), (t["region"] if q3 else t["none"], q3 == anom)]))
        y = 0.27
        for name, cells in rows:
            fig.text(left + 0.004 + cols[0], y, name, fontsize=5.8, va="top", weight="bold")
            for (text, ok), cx in zip(cells, cols[1:]):
                colour = "black" if ok is None and name else (GREY if ok is None else (GREEN if ok else RED))
                fig.text(left + 0.004 + cx, y, text, fontsize=5.8, va="top", color=colour,
                         weight="bold" if not name else "normal")
            y -= 0.065
    return fig


def defect_reference(lang, z, labels):
    t = T[lang]
    man = {r["core_id"]: r for r in read("results/latest/gold_v2_reference/reference_manifest.csv")}
    mc = {(r["core_id"], r["condition"], r["question_type"]): r for r in read("results/latest/gold_v2_minicpm_greedy/answers.csv")}
    fig = plt.figure(figsize=(7.0, 2.25))
    for k, core in enumerate(REF_EXAMPLES):
        lab, ref = labels[core], man[core]
        q1 = mc[(core, "positive_reference", "anomaly_detection")]
        said_normal = q1["correct"] not in ("True", "true", "1") and lab["anomaly_gt"] == "anomalous"
        ref_label = t["cls"].get(ref["positive_reference_defect_label"], ref["positive_reference_defect_label"])
        panels = ((lab["image"], f"{t['ref_target']}: {t['cls'][lab['defect_gt']]}"),
                  (ref["positive_reference_image"],
                   f"{t['ref_ref']}: " + t["ref_caption"].format(label=ref_label, sim=float(ref["positive_reference_similarity"]))))
        for j, (member, title) in enumerate(panels):
            ax = fig.add_axes([0.01 + k * 0.5 + j * 0.24, 0.17, 0.225, 0.72])
            ax.imshow(Image.open(io.BytesIO(z.read(member))).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=7.0, pad=3)
        verdict = t["normal"] if said_normal else t["anomalous"]
        fig.text(0.01 + k * 0.5 + 0.235, 0.075, f"{t['minicpm']} {verdict}", ha="center", fontsize=7.2,
                 color=RED if said_normal else GREEN, weight="bold")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sup-zip", type=Path, required=True)
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    z = zipfile.ZipFile(args.sup_zip)
    labels = {r["core_id"]: r for r in read("results/latest/gold_v2_final/manual_review_final.csv")}
    for lang in ("en", "ko"):
        setup(lang)
        for name, fig in (("pipeline", pipeline(lang)), ("agent", agent(lang)),
                          ("qa_examples", qa_examples(lang, z, labels)), ("defect_reference", defect_reference(lang, z, labels))):
            fig.savefig(OUT / f"fig_{name}_{lang}.png", dpi=DPI, facecolor="white")
            plt.close(fig)
            print(OUT / f"fig_{name}_{lang}.png")


if __name__ == "__main__":
    main()
