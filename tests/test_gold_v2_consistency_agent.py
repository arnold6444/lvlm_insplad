from __future__ import annotations

import csv
import importlib.util
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gold_v2_consistency_agent",
                                              ROOT / "scripts" / "run_gold_v2_consistency_agent.py")
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)

Q1, Q2, Q3 = RUNNER.Q1, RUNNER.Q2, RUNNER.Q3
DEFECT = {"A": "Normal / no defect.", "B": "Rust / corrosion.",
          "C": "Missing cap / missing insulator component.", "D": "Bird nest / foreign nesting material."}
MODELS = ["qwen3_5_4b", "internvl3_5_8b"]


def _rows(n: int = 6) -> list[dict]:
    rows = []
    for i in range(n):
        anomalous = i % 2 == 1
        core = f"core-{i}"
        common = {"core_id": core, "image": f"sup/{i}.jpg", "normal_reference_image": f"unsup/good/{i % 2}.jpg",
                  "anomaly_gt": "anomalous" if anomalous else "normal",
                  "defect_gt": "rust" if anomalous else "normal"}
        rows.append({**common, "id": f"{core}-q1", "question_type": Q1, "question": f"ask {core}-q1",
                     "options": {"A": "Normal.", "B": "Anomalous."}, "answer": "B" if anomalous else "A"})
        rows.append({**common, "id": f"{core}-q2", "question_type": Q2, "question": f"ask {core}-q2",
                     "options": dict(DEFECT), "answer": "B" if anomalous else "A"})
        if anomalous:
            rows.append({**common, "id": f"{core}-q3", "question_type": Q3, "question": f"ask {core}-q3",
                         "options": {"A": "left", "B": "right"}, "answer": "A"})
    return rows


def test_prompt_quotes_first_pass_and_score_only_in_score_variant():
    row = _rows(2)[0]
    plain = RUNNER.build_prompt(row, "Normal.", "Rust / corrosion.", inconsistent=True, score_conflict=False,
                                variant="self", percentile=95.0)
    assert '"Normal."' in plain and '"Rust / corrosion."' in plain and "contradict" in plain
    assert "percentile" not in plain and "Image 1 is a verified NORMAL reference" in plain
    scored = RUNNER.build_prompt(row, "Normal.", "Normal / no defect.", inconsistent=False, score_conflict=True,
                                 variant="self_score", percentile=95.0)
    assert "95.0th percentile" in scored and "disagrees" in scored and "contradict" not in scored
    assert scored.rstrip().endswith("Return exactly one option letter and nothing else.")


def test_routing_self_vs_self_score():
    rows = _rows(4)
    first = {r["id"]: r["answer"] for r in rows}
    first["core-0-q2"] = "B"          # core-0: normal + rust -> inconsistent
    first["core-1-q1"] = "A"          # core-1: says normal but PatchCore 95 -> score conflict (and inconsistent)
    first["core-1-q2"] = "A"          # keep core-1 consistent so it is a pure score conflict
    pct = {"core-0": 30.0, "core-1": 95.0, "core-2": 40.0, "core-3": 80.0}
    self_routed, _, _ = RUNNER.routed_images(rows, first, pct, "self", 10.0, 90.0)
    score_routed, conflict, inconsistent = RUNNER.routed_images(rows, first, pct, "self_score", 10.0, 90.0)
    assert self_routed == {"core-0"} and inconsistent == {"core-0"}
    assert conflict == {"core-1"} and score_routed == {"core-0", "core-1"}


def _setup(tmp: Path, rows: list[dict]) -> list[str]:
    sup, unsup = tmp / "sup.zip", tmp / "unsup.zip"
    with zipfile.ZipFile(sup, "w") as z:
        for name in sorted({r["image"] for r in rows}):
            z.writestr(name, b"img")
    with zipfile.ZipFile(unsup, "w") as z:
        for name in sorted({r["normal_reference_image"] for r in rows}):
            z.writestr(name, b"ref")

    wrong = lambda r: next(k for k in r["options"] if k != r["answer"])  # noqa: E731
    first = {r["id"]: r["answer"] for r in rows}
    first["core-1-q1"] = wrong(rows[[x["id"] for x in rows].index("core-1-q1")])   # anomalous said normal
    img = tmp / "baseline.csv"
    with img.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["model_key", "id", "predicted_choice"])
        w.writeheader()
        for m in MODELS:
            for r in rows:
                w.writerow({"model_key": m, "id": r["id"], "predicted_choice": r["answer"]})

    def jsonl(path: Path, recs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(x) for x in recs) + "\n", encoding="utf-8")

    for m in MODELS:
        jsonl(tmp / "nr" / f"{m}.jsonl", [{"id": r["id"], "predicted_choice": first[r["id"]]} for r in rows])
        jsonl(tmp / "ex" / m / "patchcore__score.jsonl", [{"id": r["id"], "predicted_choice": r["answer"]} for r in rows])
        jsonl(tmp / "cb" / m / "patchcore__score.jsonl",
              [{"id": r["id"], "core_id": r["core_id"], "predicted_choice": first[r["id"]],
                "score_percentile_full": 95.0 if r["anomaly_gt"] == "anomalous" else 30.0} for r in rows])
    return ["--qa-dir", str(tmp), "--sup-zip", str(sup), "--unsup-zip", str(unsup),
            "--image-only-csv", str(img), "--normal-ref-dir", str(tmp / "nr"), "--expert-dir", str(tmp / "ex"),
            "--combined-dir", str(tmp / "cb"), "--out-dir", str(tmp / "out"), "--models", *MODELS,
            "--n-boot", "100"]


def test_main_reasks_only_routed_images_and_reuses_on_rerun(tmp_path):
    rows = _rows(6)
    answer = {r["question"]: r["answer"] for r in rows}
    loads, calls = [], []

    def loader(spec):
        loads.append(spec["model_id"])
        return {"fake": True}

    def infer(adapter, images, prompt):
        calls.append(prompt)
        assert len(images) == 2 and all(Path(p).is_file() for p in images)
        question = next(line[len("Question: "):] for line in prompt.splitlines() if line.startswith("Question: "))
        return answer[question]

    argv = _setup(tmp_path, rows)
    assert RUNNER.main(argv, load_qa=lambda _: rows, loader=loader, infer=infer) == 0
    out = tmp_path / "out"
    with (out / "leaderboard.csv").open(encoding="utf-8") as h:
        lb = {(r["model_key"], r["policy"]): r for r in csv.DictReader(h)}
    # core-1 (anomalous) was answered "normal" in stage 1 while Q2 said rust -> routed and fixed by the re-ask
    fixed = lb[("qwen3_5_4b", "fixed_normal_ref")]
    live = lb[("qwen3_5_4b", "R_self_score")]
    assert float(fixed["Q1_anomaly_f1"]) < 1.0 and float(live["Q1_anomaly_f1"]) == 1.0
    assert live["source"] == "live" and float(live["avg_lvlm_calls_per_q"]) > 1.0
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert all(s["row_errors"] == 0 for s in status)
    assert all(s["reasked_questions"] == 2 * s["routed_images"] for s in status)
    with (out / "significance.csv").open(encoding="utf-8") as h:
        assert {r["policy"] for r in csv.DictReader(h)} >= {"R_self+S", "R_self_score+S"}
    first_calls, first_loads = len(calls), len(loads)
    assert first_calls > 0 and first_loads == len(MODELS)

    assert RUNNER.main(argv, load_qa=lambda _: rows, loader=loader, infer=infer) == 0
    assert len(calls) == first_calls and len(loads) == first_loads   # everything reused, no model load


def test_consistency_notebook_runs_agent_on_gpu_and_publishes():
    nb = json.loads((ROOT / "notebooks" / "lvlm_gold_v2_consistency_agent_colab.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
             for c in nb["cells"] if c["cell_type"] == "code"]
    for i, src in enumerate(cells):
        compile(src, f"consistency_nb:{i}", "exec")
    code = "\n".join(cells)
    assert "run_gold_v2_consistency_agent.py" in code
    assert "CHECKPOINT_EVERY = 20" in code and 'VARIANTS = ["self","self_score"]' in code
    assert "torch.cuda.is_available()" in code            # GPU 필수
    assert '"--prefix", "gold_v2_consistency_agent"' in code
    assert "CONFLICT_LOW, CONFLICT_HIGH = 10.0, 90.0" in code
