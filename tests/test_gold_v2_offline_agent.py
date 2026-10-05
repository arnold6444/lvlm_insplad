from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gold_v2_offline_agent", ROOT / "scripts" / "run_gold_v2_offline_agent.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)

Q1, Q2, Q3 = RUNNER.Q1, RUNNER.Q2, RUNNER.Q3
DEFECT = {"A": "Normal / no defect.", "B": "Rust / corrosion.",
          "C": "Missing cap / missing insulator component.", "D": "Bird nest / foreign nesting material."}
LETTER = {"normal": "A", "rust": "B", "missing-cap": "C", "bird-nest": "D"}


def _rows(n: int = 6) -> list[dict]:
    rows = []
    for i in range(n):
        anomalous = i % 2 == 1
        defect = "rust" if anomalous else "normal"
        core = f"core-{i}"
        rows.append({"id": f"{core}-q1", "core_id": core, "question_type": Q1,
                     "options": {"A": "Normal.", "B": "Anomalous."}, "answer": "B" if anomalous else "A",
                     "anomaly_gt": "anomalous" if anomalous else "normal", "defect_gt": defect})
        rows.append({"id": f"{core}-q2", "core_id": core, "question_type": Q2, "options": dict(DEFECT),
                     "answer": LETTER[defect], "anomaly_gt": "anomalous" if anomalous else "normal",
                     "defect_gt": defect})
        if anomalous:
            rows.append({"id": f"{core}-q3", "core_id": core, "question_type": Q3,
                         "options": {"A": "left", "B": "right"}, "answer": "A",
                         "anomaly_gt": "anomalous", "defect_gt": defect})
    return rows


def _answers(rows, wrong=()):
    """Correct answer for every question except ids in ``wrong`` (flipped to another option)."""
    out = {}
    for r in rows:
        if r["id"] in wrong:
            out[r["id"]] = next(k for k in r["options"] if k != r["answer"])
        else:
            out[r["id"]] = r["answer"]
    return out


def test_score_conflict_uses_fixed_thresholds():
    q1 = _rows(2)[0]
    assert RUNNER.score_conflict(q1, "A", 95.0, 10.0, 90.0)        # said normal, score high
    assert not RUNNER.score_conflict(q1, "A", 50.0, 10.0, 90.0)
    assert RUNNER.score_conflict(q1, "B", 5.0, 10.0, 90.0)         # said anomalous, score low
    assert RUNNER.score_conflict(q1, None, 50.0, 10.0, 90.0)       # invalid output is routed


def test_q12_inconsistency():
    rows = _rows(2)
    q1, q2 = rows[0], rows[1]
    assert not RUNNER.q12_inconsistent(q1, "A", q2, "A")
    assert RUNNER.q12_inconsistent(q1, "A", q2, "B")   # normal + rust
    assert RUNNER.q12_inconsistent(q1, "B", q2, "A")   # anomalous + normal


def test_compose_reasks_only_routed_question_types_and_counts_calls():
    rows = _rows(4)
    preds = {("m", "normal_ref"): {r["id"]: "A" for r in rows}, ("m", "combined"): {r["id"]: "B" for r in rows}}
    choice, calls = RUNNER.compose(rows, preds, "m", {q: "normal_ref" for q in RUNNER.QTYPES},
                                   routed={"core-1"}, routed_source={Q1: "combined"})
    assert choice["core-1-q1"] == "B" and choice["core-1-q2"] == "A" and choice["core-0-q1"] == "A"
    assert calls == 1 + 1 / len(rows)


def test_vote_breaks_ties_with_reference_model():
    rows = _rows(2)[:1]
    key = rows[0]["id"]
    assert RUNNER.vote(rows, [{key: "A"}, {key: "A"}, {key: "B"}], {key: "B"})[key] == "A"
    assert RUNNER.vote(rows, [{key: "A"}, {key: "B"}], {key: "B"})[key] == "B"


def _write_inputs(tmp: Path, rows: list[dict], models: list[str], answers: dict) -> dict:
    img = tmp / "baseline" / "prediction_review.csv"
    img.parent.mkdir(parents=True)
    with img.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["model_key", "id", "predicted_choice"])
        w.writeheader()
        for m in models:
            for r in rows:
                w.writerow({"model_key": m, "id": r["id"], "predicted_choice": answers[(m, "image_only")][r["id"]]})

    def jsonl(path: Path, recs: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(x) for x in recs) + "\n", encoding="utf-8")

    for m in models:
        jsonl(tmp / "nr" / f"{m}.jsonl",
              [{"id": r["id"], "predicted_choice": answers[(m, "normal_ref")][r["id"]]} for r in rows])
        jsonl(tmp / "ex" / m / "patchcore__score.jsonl",
              [{"id": r["id"], "predicted_choice": answers[(m, "expert_only")][r["id"]]} for r in rows])
        jsonl(tmp / "cb" / m / "patchcore__score.jsonl",
              [{"id": r["id"], "core_id": r["core_id"], "predicted_choice": answers[(m, "combined")][r["id"]],
                "score_percentile_full": 95.0 if r["anomaly_gt"] == "anomalous" else 30.0} for r in rows])
    return {"img": img}


def test_main_end_to_end_routing_fixes_conflicts(tmp_path):
    rows = _rows(6)
    models = ["qwen3_5_4b", "internvl3_5_8b"]
    # normal_ref misses anomalous core-1 (says normal, PatchCore 95th pct) and its Q2/Q3; combined is right there.
    miss = {"core-1-q1", "core-1-q2", "core-1-q3"}
    answers = {}
    for m in models:
        answers[(m, "normal_ref")] = _answers(rows, wrong=miss)
        answers[(m, "combined")] = _answers(rows, wrong={"core-3-q3"})
        answers[(m, "expert_only")] = _answers(rows, wrong={"core-0-q1"})
        answers[(m, "image_only")] = _answers(rows, wrong={"core-5-q2"})
    paths = _write_inputs(tmp_path, rows, models, answers)
    out = tmp_path / "out"
    rc = RUNNER.main(["--qa-dir", str(tmp_path), "--image-only-csv", str(paths["img"]),
                      "--normal-ref-dir", str(tmp_path / "nr"), "--expert-dir", str(tmp_path / "ex"),
                      "--combined-dir", str(tmp_path / "cb"), "--out-dir", str(out), "--models", *models],
                     load_qa=lambda _: rows)
    assert rc == 0
    with (out / "leaderboard.csv").open(encoding="utf-8") as h:
        lb = {(r["model_key"], r["policy"]): r for r in csv.DictReader(h)}
    fixed = lb[("qwen3_5_4b", "fixed_normal_ref")]
    routed = lb[("qwen3_5_4b", "A1_all")]
    assert float(fixed["Q1_anomaly_f1"]) < 1.0
    assert float(routed["Q1_anomaly_f1"]) == 1.0 and float(routed["Q3_loc_acc"]) == 1.0
    assert routed["all3_up_vs_normal_ref"] == "True"
    assert float(routed["avg_lvlm_calls_per_q"]) > 1.0
    assert ("patchcore_only", "A2_score_only_t50") in lb
    assert ("vote_4models", "A3_model_vote_normal_ref") in lb
    with (out / "route_stats.csv").open(encoding="utf-8") as h:
        stats = {r["model_key"]: r for r in csv.DictReader(h)}
    assert float(stats["qwen3_5_4b"]["score_conflict_rate"]) == 1 / 6
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["new_lvlm_inference"] is False


def test_missing_predictions_fail_loudly(tmp_path):
    rows = _rows(2)
    preds = {("m", c): {} for c in RUNNER.CONDITIONS}
    try:
        RUNNER.check_coverage(rows, preds, {})
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_offline_agent_notebook_is_cpu_only_and_verifies_against_published_results():
    nb = json.loads((ROOT / "notebooks" / "lvlm_gold_v2_offline_agent_colab.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
             for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"offline_agent_nb:{i}", "exec")
    code = "\n".join(cells)
    assert "run_gold_v2_offline_agent.py" in code
    assert "import torch" not in code and '"pip"' not in code   # CPU 전용, 설치 없음
    assert "검증 통과" in code and "gold_v2_reference_expert/leaderboard.csv" in code
    assert '"--prefix", "gold_v2_offline_agent"' in code
    assert "CONFLICT_LOW, CONFLICT_HIGH = 10.0, 90.0" in code
