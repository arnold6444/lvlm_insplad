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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
DPI = 300
WIDTH = 7.0  # inches: the text width of a one-column page; still legible at one column of a two-column page
# One gold image per class; normal/rust/bird nest are the same asset type, so only the condition differs.
EXAMPLES = [("normal", "core-07f9ce99fdf16056"), ("rust", "core-28a24ec7956ea697"),
            ("bird-nest", "core-3df353cc203c9ed3"), ("missing-cap", "core-bb120ed6aa3fc6a1")]
REF_EXAMPLES = ["core-bb120ed6aa3fc6a1", "core-28a24ec7956ea697"]
NONE = "No localized defect region is present."

# Muted, colour-blind-safe palette: (fill, edge and title colour) per role.
ROLE = {"data": ("#E3EDF7", "#2C5D8A"), "gold": ("#FBEFD9", "#8A5A12"), "dev": ("#E1F1E6", "#2D6B45"),
        "model": ("#EEEFF3", "#4A5160"), "control": ("#FFFFFF", "#6B7280")}
INK, MUTED, ARROW, FRAME = "#1F2933", "#52606D", "#7B8794", "#D3D8DE"
OK, BAD = "#1E7B45", "#B42318"
OK_BG, BAD_BG = "#E6F4EA", "#FCE8E6"

T = {
    "en": dict(
        font="Arial",
        cls={"normal": "Normal", "rust": "Rust", "bird-nest": "Bird nest", "missing-cap": "Missing cap"},
        asset={"vari-grip": "vari-grip", "glass-insulator": "glass insulator"},
        anomalous="Anomalous", normal="Normal", region="Region", none="None",
        gt="Truth", direct="Direct", agent_lbl="Agent", shared="same as direct",
        legend="Green: correct.  Red: wrong.  Q3 is scored for presence (Region / None).",
        ref_target="Target", ref_ref="Retrieved defect reference", sim="cosine {sim:.2f}",
        verdict="MiniCPM-V-4.5 answers Q1: {v}",
        lanes=("Gold set and baselines", "Dev set and verification agent"),
        pipe=dict(
            ad=("Anomaly detection", "InsPLAD release\n17 assets\n6,481 test images"),
            fc=("Fault classification", "InsPLAD release\n5 assets\n6,417 val images"),
            align=("SHA-256 alignment", "8,990 identical images\n6 label conflicts out\nAligned Core: 1,378"),
            gold=("Gold v2 (frozen)", "139 defective\n107 normal\n534 questions"),
            dev=("Dev set", "240 images from\nsupervised train\nleakage-checked"),
            cond=("Conditions × models", "image only\n+ normal / defect ref.\n+ both references\n+ 20 expert cues\n4 open LVLMs + API"),
            design=("Agent design", "chosen on dev:\noption definitions,\npresence-first Q3"),
            run=("Single gold run", "InternVL3.5-8B\nagent + matched\nre-ask control"),
            evalr=("One evaluator", "Q1 F1 / BAcc\nQ2 macro-F1\njoint accuracy\nconsistency, FDR\npaired bootstrap"),
        ),
        agent=dict(
            inp=("Input", "target image +\nretrieved normal reference\n(other capture date)"),
            direct=("Direct answers", "Q1, Q2, Q3\none call each"),
            q1=("Q1 anomaly", "kept as answered\n(never re-checked)"),
            q2=("Q2 defect type", "second look: own first answer\n+ one-line option definitions"),
            q3a=("Q3 presence", "is a defect region present?"),
            q3b=("Q3 position", "which of the 3 × 3 cells?"),
            yes="yes", no="no → “none”",
            final="Final answers are stored; contradictions between Q1, Q2 and Q3 are computed only afterwards",
            ctrl=("Matched re-ask control", "same Q2 and Q3 calls per image, own-answer\nre-checks only, no definitions, no staging"),
        ),
    ),
    "ko": dict(
        font="Malgun Gothic",
        cls={"normal": "정상", "rust": "녹", "bird-nest": "새집", "missing-cap": "캡 누락"},
        asset={"vari-grip": "vari-grip", "glass-insulator": "유리 애자"},
        anomalous="이상", normal="정상", region="영역", none="없음",
        gt="정답", direct="직접", agent_lbl="에이전트", shared="직접과 같음",
        legend="초록: 정답.  빨강: 오답.  Q3는 유무(영역 / 없음)로 채점한다.",
        ref_target="대상", ref_ref="검색된 결함 참조", sim="코사인 {sim:.2f}",
        verdict="MiniCPM-V-4.5의 Q1 답: {v}",
        lanes=("gold 세트와 기준선", "dev 세트와 검증 에이전트"),
        pipe=dict(
            ad=("이상탐지 공개본", "InsPLAD\n설비 17종\ntest 6,481장"),
            fc=("결함분류 공개본", "InsPLAD\n설비 5종\nval 6,417장"),
            align=("SHA-256 정렬", "동일 이미지 8,990장\n라벨 충돌 6장 제외\nAligned Core 1,378장"),
            gold=("gold v2 (동결)", "결함 139장\n정상 107장\n534문항"),
            dev=("dev 세트", "supervised train\n240장\n누출 검사"),
            cond=("조건 × 모델", "이미지만\n+ 정상 / 결함 참조\n+ 두 참조\n+ expert 단서 20가지\n공개 LVLM 4종 + API"),
            design=("에이전트 설계", "dev에서 선택:\n보기 정의,\n유무 먼저 묻는 Q3"),
            run=("gold 1회 실행", "InternVL3.5-8B\n에이전트 +\n호출 수 대조군"),
            evalr=("같은 채점기", "Q1 F1 / BAcc\nQ2 macro-F1\n동시 정답\n모순율, 결함 오판율\n대응 bootstrap"),
        ),
        agent=dict(
            inp=("입력", "대상 이미지 +\n검색한 정상 참조\n(다른 촬영일)"),
            direct=("직접 답", "Q1, Q2, Q3\n각 1회 호출"),
            q1=("Q1 이상 여부", "처음 답 그대로\n(재확인 안 함)"),
            q2=("Q2 결함 종류", "재확인: 자기 첫 답\n+ 보기별 한 줄 정의"),
            q3a=("Q3 유무", "결함 영역이 있는가?"),
            q3b=("Q3 위치", "3 × 3 중 어느 칸인가?"),
            yes="예", no="아니오 → ‘없음’",
            final="최종 답을 모두 저장한 뒤에만 Q1·Q2·Q3 사이의 모순을 계산한다",
            ctrl=("호출 수 대조군", "이미지마다 Q2·Q3 호출 수 동일, 자기 답\n재확인만, 정의 없음, 단계 없음"),
        ),
    ),
}


def read(rel):
    with (ROOT / rel).open(encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def setup(lang):
    plt.rcParams.update({"font.family": [T[lang]["font"], "DejaVu Sans"], "font.size": 7,
                         "axes.unicode_minus": False, "text.color": INK})


def canvas(h):
    fig = plt.figure(figsize=(WIDTH, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def card(ax, x, y, w, h, title, body, role, dashed=False, title_size=7.4, body_size=6.5):
    """Rounded box with a bold title in the role colour and a body below it, centred vertically (inches)."""
    fill, edge = ROLE[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", linewidth=0.9,
                                edgecolor=edge, facecolor=fill, linestyle=(0, (3, 2)) if dashed else "-"))
    if not body:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, weight="bold", color=edge)
        return
    title_h = title_size / 72 * 1.25
    body_h = body_size / 72 * 1.3 * (body.count("\n") + 1)
    top = y + h / 2 + (title_h + 0.05 + body_h) / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=title_size, weight="bold", color=edge)
    ax.text(x + w / 2, top - title_h - 0.05, body, ha="center", va="top", fontsize=body_size, color=INK, linespacing=1.3)


def arrow(ax, a, b, label=None, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color=ARROW,
                                 linestyle=(0, (3, 2)) if dashed else "-", shrinkA=1, shrinkB=1))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.06, label, ha="center", va="bottom", fontsize=6.3,
                color=MUTED, style="italic")


def lane(ax, y, h, label):
    ax.add_patch(Rectangle((0.02, y), WIDTH - 0.04, h, facecolor="#F7F8FA", edgecolor="none", zorder=0))
    ax.text(0.12, y + h - 0.05, label, ha="left", va="top", fontsize=7.0, weight="bold", color=MUTED)


def pipeline(lang):
    t, lanes = T[lang]["pipe"], T[lang]["lanes"]
    fig, ax = canvas(2.95)
    lane(ax, 1.50, 1.42, lanes[0])
    lane(ax, 0.04, 1.42, lanes[1])
    w, gap, x0 = 1.22, 0.175, 0.12
    xs = [x0 + i * (w + gap) for i in range(5)]
    top, bot, h = 1.56, 0.10, 1.06
    card(ax, xs[0], top, w, h, *t["ad"], "data")
    card(ax, xs[0], bot, w, h, *t["fc"], "data")
    card(ax, xs[1], top, w, h, *t["align"], "data")
    card(ax, xs[1], bot, w, h, *t["dev"], "dev")
    card(ax, xs[2], top, w, h, *t["gold"], "gold")
    card(ax, xs[2], bot, w, h, *t["design"], "dev")
    card(ax, xs[3], top, w, h, *t["cond"], "model")
    card(ax, xs[3], bot, w, h, *t["run"], "dev")
    card(ax, xs[4], bot, w, top + h - bot, *t["evalr"], "gold")
    mid_t, mid_b = top + h / 2, bot + h / 2
    arrow(ax, (xs[0] + w, mid_t), (xs[1], mid_t))                    # anomaly-detection release -> alignment
    arrow(ax, (xs[0] + w, mid_b + 0.28), (xs[1], mid_t - 0.28))      # fault-classification val -> alignment
    arrow(ax, (xs[0] + w, mid_b), (xs[1], mid_b))                    # fault-classification train -> dev
    arrow(ax, (xs[1] + w, mid_t), (xs[2], mid_t))                    # alignment -> gold
    arrow(ax, (xs[1] + w, mid_b), (xs[2], mid_b))                    # dev -> agent design
    arrow(ax, (xs[2] + w, mid_t), (xs[3], mid_t))                    # gold -> baselines
    arrow(ax, (xs[2] + w, mid_b), (xs[3], mid_b))                    # design -> single gold run
    arrow(ax, (xs[2] + w, mid_t - 0.28), (xs[3], mid_b + 0.28), dashed=True)  # the frozen gold set is used once
    arrow(ax, (xs[3] + w, mid_t), (xs[4], mid_t))
    arrow(ax, (xs[3] + w, mid_b), (xs[4], mid_b))
    return fig


def agent(lang):
    t = T[lang]["agent"]
    fig, ax = canvas(2.95)
    card(ax, 0.10, 1.42, 1.18, 0.86, *t["inp"], "data")
    card(ax, 1.50, 1.42, 0.95, 0.86, *t["direct"], "model")
    arrow(ax, (1.28, 1.85), (1.50, 1.85))
    cx, cw = 2.78, 1.86
    card(ax, cx, 2.30, cw, 0.58, *t["q1"], "model")
    card(ax, cx, 1.50, cw, 0.66, *t["q2"], "gold")
    card(ax, cx, 0.72, cw, 0.58, *t["q3a"], "gold")
    card(ax, cx + cw + 0.40, 0.72, 1.60, 0.58, *t["q3b"], "gold")
    for y in (2.59, 1.83, 1.01):
        arrow(ax, (2.45, 1.85), (cx, y))
    arrow(ax, (cx + cw, 1.01), (cx + cw + 0.40, 1.01), label=t["yes"])
    card(ax, 0.10, 0.06, WIDTH - 0.20, 0.38, t["final"], None, "dev", title_size=6.8)
    arrow(ax, (cx + cw / 2, 0.72), (cx + cw / 2, 0.44))
    ax.text(cx + cw / 2 + 0.06, 0.58, t["no"], ha="left", va="center", fontsize=6.3, color=MUTED, style="italic")
    q3b_mid = cx + cw + 0.40 + 0.80
    arrow(ax, (q3b_mid, 0.72), (q3b_mid, 0.44))
    rail = 6.82
    for y in (2.59, 1.83):  # Q1 and Q2 finals go down the right margin to the stored answers
        ax.plot([cx + cw, rail], [y, y], color=ARROW, linewidth=1.0)
    ax.plot([rail, rail], [2.59, 1.36], color=ARROW, linewidth=1.0)
    arrow(ax, (rail, 1.36), (rail, 0.44))
    card(ax, 0.10, 0.62, 2.35, 0.60, *t["ctrl"], "control", dashed=True, title_size=6.8, body_size=5.9)
    return fig


def answers():
    return {(r["policy"], r["core_id"], r["question_type"]): r["predicted_text"]
            for r in read("results/latest/gold_v2_matched_call/answers.csv")}


def square(img):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def image_axes(fig, x, y, size, fig_h, img):
    iax = fig.add_axes([x / WIDTH, y / fig_h, size / WIDTH, size / fig_h])
    iax.imshow(square(img))
    iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_edgecolor(FRAME)


def qa_examples(lang, z, labels):
    """2 x 2 panels: image on the left, a Q1-Q3 by truth/direct/agent table on the right."""
    t = T[lang]
    a = answers()
    q1t = {"Anomalous.": "anomalous", "Normal.": "normal"}
    q2t = {"Normal / no defect.": "normal", "Rust / corrosion.": "rust",
           "Missing cap / missing insulator component.": "missing-cap", "Bird nest / foreign nesting material.": "bird-nest"}
    H = 3.42
    fig, ax = canvas(H)
    pw, ph = 3.38, 1.52
    for k, (cls, core) in enumerate(EXAMPLES):
        lab = labels[core]
        x = 0.08 + (k % 2) * (pw + 0.08)
        y = 0.30 + (1 - k // 2) * (ph + 0.08)
        ax.add_patch(FancyBboxPatch((x, y), pw, ph, boxstyle="round,pad=0,rounding_size=0.05", linewidth=0.8,
                                    edgecolor=FRAME, facecolor="white"))
        image_axes(fig, x + 0.08, y + 0.13, 1.26, H, Image.open(io.BytesIO(z.read(lab["image"]))).convert("RGB"))
        tx = x + 1.44
        ax.text(tx, y + ph - 0.14, f"({'abcd'[k]}) {t['cls'][cls]}", fontsize=7.6, weight="bold", va="center")
        ax.text(tx, y + ph - 0.31, t["asset"].get(lab["asset_category"], lab["asset_category"]), fontsize=6.3,
                color=MUTED, va="center")
        anom = lab["anomaly_gt"] == "anomalous"
        truth = {"Q1": t["anomalous"] if anom else t["normal"], "Q2": t["cls"][cls], "Q3": t["region"] if anom else t["none"]}
        got = {}
        for pol in ("baseline", "previous_agent"):
            q1 = q1t[a[(pol, core, "anomaly_detection")]]
            q2 = q2t[a[(pol, core, "defect_classification")]]
            q3 = a[(pol, core, "defect_localization")] != NONE
            got[pol] = {"Q1": (t["anomalous"] if q1 == "anomalous" else t["normal"], (q1 == "anomalous") == anom),
                        "Q2": (t["cls"][q2], q2 == cls), "Q3": (t["region"] if q3 else t["none"], q3 == anom)}
        cols = [tx, tx + 0.22, tx + 0.80, tx + 1.37]
        cw = [0.22, 0.58, 0.57, 0.55]
        hy = y + ph - 0.56
        for cx, head in zip(cols[1:], (t["gt"], t["direct"], t["agent_lbl"])):
            ax.text(cx + 0.04, hy, head, fontsize=6.3, weight="bold", color=MUTED, va="center")
        for i, q in enumerate(("Q1", "Q2", "Q3")):
            ry = hy - 0.25 * (i + 1)
            ax.text(cols[0], ry, q, fontsize=6.3, weight="bold", va="center")
            ax.text(cols[1] + 0.04, ry, truth[q], fontsize=6.2, va="center")
            for j, pol in enumerate(("baseline", "previous_agent")):
                cx = cols[2 + j]
                if pol == "previous_agent" and q == "Q1":
                    ax.text(cx + 0.03, ry, t["shared"], fontsize=5.5, color=MUTED, va="center", style="italic")
                    continue
                text, ok = got[pol][q]
                ax.add_patch(FancyBboxPatch((cx, ry - 0.09), cw[2 + j] - 0.03, 0.18, boxstyle="round,pad=0,rounding_size=0.03",
                                            linewidth=0, facecolor=OK_BG if ok else BAD_BG))
                ax.text(cx + 0.03, ry, text, fontsize=6.0, va="center", color=OK if ok else BAD)
    ax.text(WIDTH / 2, 0.14, t["legend"], ha="center", va="center", fontsize=6.4, color=MUTED)
    return fig


def defect_reference(lang, z, labels):
    t = T[lang]
    man = {r["core_id"]: r for r in read("results/latest/gold_v2_reference/reference_manifest.csv")}
    mc = {(r["core_id"], r["condition"], r["question_type"]): r for r in read("results/latest/gold_v2_minicpm_greedy/answers.csv")}
    H = 2.42
    fig, ax = canvas(H)
    cw, gap, x0 = 3.38, 0.08, 0.08
    for k, core in enumerate(REF_EXAMPLES):
        lab, ref = labels[core], man[core]
        x = x0 + k * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, 0.06), cw, H - 0.12, boxstyle="round,pad=0,rounding_size=0.05",
                                    linewidth=0.8, edgecolor=FRAME, facecolor="white"))
        q1 = mc[(core, "positive_reference", "anomaly_detection")]
        said_normal = q1["correct"] not in ("True", "true", "1") and lab["anomaly_gt"] == "anomalous"
        ref_label = t["cls"].get(ref["positive_reference_defect_label"], ref["positive_reference_defect_label"])
        panels = ((lab["image"], t["ref_target"], t["cls"][lab["defect_gt"]], "data"),
                  (ref["positive_reference_image"], t["ref_ref"],
                   f"{ref_label}, " + t["sim"].format(sim=float(ref["positive_reference_similarity"])), "gold"))
        for j, (member, head, sub, role) in enumerate(panels):
            px = x + 0.08 + j * 1.64
            ax.text(px + 0.78, H - 0.17, head, ha="center", va="center", fontsize=6.9, weight="bold", color=ROLE[role][1])
            ax.text(px + 0.78, H - 0.32, sub, ha="center", va="center", fontsize=6.2, color=MUTED)
            image_axes(fig, px, 0.46, 1.56, H, Image.open(io.BytesIO(z.read(member))).convert("RGB"))
        verdict = t["normal"] if said_normal else t["anomalous"]
        ax.add_patch(FancyBboxPatch((x + 0.55, 0.14), cw - 1.10, 0.22, boxstyle="round,pad=0,rounding_size=0.05",
                                    linewidth=0, facecolor=BAD_BG if said_normal else OK_BG))
        ax.text(x + cw / 2, 0.25, t["verdict"].format(v=verdict), ha="center", va="center", fontsize=6.9,
                weight="bold", color=BAD if said_normal else OK)
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
