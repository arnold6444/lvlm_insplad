"""Gold scoring of the tool-choice agent: fixture-only, no model, no GPU, no real gold file."""
import csv
import json
from pathlib import Path
import re
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import run_agent_sft_gold_eval as gold  # noqa: E402
import test_agent_tool_pilot as pilot_tests  # noqa: E402
import test_gold_v2_verify_agent as fixtures  # noqa: E402

pilot = gold.pilot
N, GOLD_DATE, REF_DATE = 8, "01-12-2020", "02-12-2020"
# Odd photos are rust. c1 and c5 sit in the centre cell, c3 in the upper-left cell, c7 has no box.
BOXES = {1: [0.40, 0.40, 0.20, 0.20], 3: [0.05, 0.05, 0.10, 0.10], 5: [0.45, 0.45, 0.10, 0.10]}


def asset(i):
    return "vari-grip" if i < 4 else "yoke-suspension"


def image(i):
    return f"defect_supervised/{asset(i)}/val/{'rust' if i % 2 else 'good'}/Fotos {GOLD_DATE}_DJI_{i:04d}_g_{i}.png"


def reference(i):
    return f"{asset(i)}/train/good/Fotos {REF_DATE}_DJI_09{i:02d}_ref.png"


def gold_rows():
    rows = []
    for i in range(N):
        bad = i % 2 == 1
        common = dict(core_id=f"core-{i}", asset_category=asset(i), image=image(i), protocol="explainable_qa_gold_v2",
                      anomaly_gt="anomalous" if bad else "normal", defect_gt="rust" if bad else "normal")
        rows.append(dict(common, id=f"core-{i}-q1", question_type=gold.Q1, question=fixtures.QTEXT[gold.Q1],
                         options={"A": "Normal.", "B": "Anomalous."}, answer="B" if bad else "A"))
        rows.append(dict(common, id=f"core-{i}-q2", question_type=gold.Q2, question=fixtures.QTEXT[gold.Q2],
                         options=dict(fixtures.DEFECT), answer="B" if bad else "A"))
    return rows


def inputs(tmp, rows=None, train=None, boxes=BOXES):
    rows = rows or gold_rows()
    qa_dir = tmp / "gold_v2_final"
    qa_dir.mkdir(parents=True)
    (qa_dir / "master_qa_fixture.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    # One training photo is another crop of gold photo 1's drone frame, one has the bytes of gold photo 2.
    train = train or {f"defect_supervised/vari-grip/train/rust/Fotos {GOLD_DATE}_DJI_0001_other_crop.png": 200,
                      "defect_supervised/vari-grip/train/good/Fotos 05-12-2020_DJI_0500_t.png": 32,
                      "defect_supervised/yoke-suspension/train/rust/Fotos 06-12-2020_DJI_0600_t.png": 201}
    with zipfile.ZipFile(tmp / "sup.zip", "w") as z:
        for i in range(N):
            z.writestr(image(i), fixtures._png(30 + i))
        for name, shade in train.items():
            z.writestr(name, fixtures._png(shade))
    with zipfile.ZipFile(tmp / "unsup.zip", "w") as z:
        for i in range(N):
            z.writestr(reference(i), fixtures._png(100 + i))
    with (tmp / "refs.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["image", "retrieved_diffdate"])
        w.writeheader()
        w.writerows(dict(image=image(i), retrieved_diffdate=reference(i)) for i in range(N))
    loc = [dict(core_id=f"core-{i}", protocol="gold_v2_loc_mmad", defect_gt="rust", asset_category=asset(i),
                localization_gt=dict(normalized_bboxes_xywh=[box], basis="manual_box_annotation"))
           for i, box in boxes.items()]
    (tmp / "loc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in loc), encoding="utf-8")
    (tmp / "sft_train_qa.jsonl").write_text(
        "".join(json.dumps(dict(image=name, split="supervised_train_sft")) + "\n" for name in train), encoding="utf-8")
    out = tmp / "agent_sft_gold_eval_test"
    return ["--qa-dir", str(qa_dir), "--loc-qa", str(tmp / "loc.jsonl"), "--refs-csv", str(tmp / "refs.csv"),
            "--sup-zip", str(tmp / "sup.zip"), "--unsup-zip", str(tmp / "unsup.zip"),
            "--train-qa", str(tmp / "sft_train_qa.jsonl"), "--n-boot", "20", "--out-dir", str(out)], out


def backend(calls, q3_text=None, crop=(0.2, 0.2, 0.8, 0.8)):
    """Answers Q1/Q2 from the photo number; Q3 says the centre cell on every defect unless q3_text is given."""
    def infer(adapter, images, prompt):
        calls.append(prompt)
        if "Optional tools:" in prompt:
            if "normal REFERENCE" not in prompt:
                return '{"action":"reference"}'
            if "is a crop" not in prompt:
                return json.dumps(dict(action="crop", bbox=list(crop)))
        options = dict(re.findall(r"^([A-J])\. (.+)$", prompt, re.M))
        bad = int(Path(images[0]).stem.rsplit("_", 1)[1]) % 2
        if len(options) == 2:
            text = "Anomalous." if bad else "Normal."
        elif len(options) == 4:
            text = "Rust / corrosion." if bad else "Normal / no defect."
        else:
            if q3_text == "prose" and bad:
                return "The defect is in the centre."
            text = (q3_text or "Center region.") if bad else pilot.support.NONE
        return json.dumps(dict(action="finish", choice=next(k for k, v in options.items() if v == text)))
    return infer


@pytest.fixture(autouse=True)
def small_gold(monkeypatch):
    monkeypatch.setattr(gold, "GOLD_IMAGES", N)
    monkeypatch.setattr(gold, "LOCATED_IMAGES", len(BOXES))


def run(args, calls=None, **kw):
    return gold.main(args, loader=lambda spec: {}, infer=backend([] if calls is None else calls, **kw))


def table(out, name):
    with (out / name).open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def test_scores_both_scopes_position_crops_and_resumes(tmp_path):
    args, out = inputs(tmp_path)
    calls = []
    assert run(args, calls) == 0
    assert len(calls) == N * 3 * (1 + 3)                       # single call, then reference -> crop -> answer
    board = {(r["subset"], r["policy"]): r for r in table(out, "leaderboard.csv")}
    assert set(board) == {(s, p) for s in gold.SUBSETS for p in gold.POLICIES}
    for policy in gold.POLICIES:
        full, clean = board["all", policy], board["frame_disjoint", policy]
        assert (full["images"], clean["images"]) == ("8", "6")  # photo 1 shares a frame, photo 2 shares bytes
        assert float(full["joint_accuracy"]) == float(clean["joint_accuracy"]) == 1
        assert (full["q3_anomalous_location_labels"], full["q3_position_hit"], full["q3_always_center_hit"]) == ("3", "2", "2")
        assert (clean["q3_anomalous_location_labels"], clean["q3_position_hit"]) == ("2", "1")
        assert float(full["q3_location_accuracy"]) == pytest.approx(2 / 3)
        # Location joint: 4 normal photos answered "no region" + 2 located hits, over 4 normal + 3 located.
        assert float(full["q1_q2_q3_location_joint"]) == pytest.approx(6 / 7) and full["q1_q2_q3_location_joint_images"] == "7"
        assert full["center60_covers_defect"] == "2"
    tools, single = board["all", "adaptive_tools"], board["all", "direct_reference"]
    assert (single["crop_requests"], single["calls"]) == ("0", str(N * 3))
    assert (tools["crop_requests"], tools["crop_distinct_boxes"], tools["crop_most_common"]) == (str(N * 3), "1", str(N * 3))
    assert (tools["q3_crops_on_located"], tools["q3_crop_covers_defect"]) == ("3", "2")     # the corner box is missed
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["protocol"] == gold.PROTOCOL and meta["primary_scope"] == "frame_disjoint" and meta["training"] is False
    assert meta["frame_overlap"] == dict(training_photos=3, training_files=["sft_train_qa.jsonl"], same_frame=1,
                                         same_bytes=1, frame_disjoint_images=6,
                                         located_images=dict(frame_disjoint=2, all=3))
    flags = {Path(r["image"]).stem.rsplit("_", 1)[1]: r for r in table(out, "frame_overlap.csv")}
    assert flags["1"]["same_frame"] == "True" and flags["2"]["same_bytes"] == "True" and flags["3"]["scored_in_primary"] == "True"
    per_image = {r["core_id"]: r for r in table(out, "per_image.csv")}
    assert per_image["core-3"]["gt_cells"] == "1" and per_image["core-3"]["adaptive_tools_q3_cell"] == "5"
    assert per_image["core-1"]["scored_in_primary"] == "False" and per_image["core-0"]["direct_reference_q3_cell"] == ""
    assert json.loads(per_image["core-5"]["adaptive_tools_q3_crop"]) == [0.2, 0.2, 0.8, 0.8]
    position = {(r["subset"], r["policy"]): r for r in table(out, "position_by_defect.csv")}
    assert position["all", "direct_reference"]["position_hit"] == "2"
    assert {r["metric"] for r in table(out, "significance.csv")} == {*gold.BOOT_METRICS, "q3_position_hit_rate"}
    assert {r["asset_category"] for r in table(out, "metrics_by_asset.csv")} == {"vari-grip", "yoke-suspension"}
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["complete"] and status["location_evaluation_complete"] and status["calls_this_run"] == len(calls)
    calls.clear()
    assert run(args, calls) == 0 and calls == []                # a second "Run all" reuses every saved call
    assert json.loads((out / "run_status.json").read_text(encoding="utf-8"))["reused_calls"] == N * 3 * 4


def test_wrong_cell_no_region_and_invalid_answers_are_not_hits(tmp_path):
    for name, kw, reported in (("corner", dict(q3_text="Upper-left region."), 3), ("none", dict(q3_text=pilot.support.NONE), 0),
                               ("prose", dict(q3_text="prose"), 0)):
        args, out = inputs(tmp_path / name)
        assert run(args, **kw) == 0
        row = next(r for r in table(out, "leaderboard.csv") if (r["subset"], r["policy"]) == ("all", "direct_reference"))
        assert row["q3_region_reported_on_located"] == str(reported)
        assert row["q3_position_hit"] == ("1" if name == "corner" else "0")       # only photo 3 is in the corner
        if name == "prose":
            assert row["invalid_episodes"] == "4" and float(row["q3_invalid_rate"]) == 0.5


def test_gold_questions_are_the_pilot_questions_with_no_label_in_the_prompt():
    rows = gold_rows()
    grid = gold.grid_rows(rows, {"core-1": {}})
    dev = pilot.support.q3_rows(rows)
    assert [(a["question"], a["options"], a["none_choice"]) for a in grid] == [(b["question"], b["options"], b["none_choice"]) for b in dev]
    assert all(r["id"].endswith("-q3-gold-grid") and r["protocol"] == gold.PROTOCOL for r in grid)
    prompt = pilot.prompt_for(grid[1], [], True, None, False)
    assert "core-1" not in prompt and "anomalous" not in prompt.lower() and "rust" not in prompt.lower()


def test_refuses_dev_qa_other_folders_changed_inputs_and_mismatched_sets(tmp_path):
    args, out = inputs(tmp_path / "ok")
    with pytest.raises(ValueError, match="agent_sft_gold_eval"):
        run(args[:-1] + [str(tmp_path / "agent_tool_pilot_dev_x")])
    dev = [dict(r, protocol="gold_v2_dev") for r in gold_rows()]
    with pytest.raises(ValueError, match="not dev or training"):
        run(inputs(tmp_path / "dev", rows=dev)[0])
    with pytest.raises(ValueError, match="expected 8 gold photos"):
        run(inputs(tmp_path / "short", rows=gold_rows()[:-2])[0])
    with pytest.raises(ValueError, match="location set does not match"):         # a box on a normal photo
        run(inputs(tmp_path / "loc", boxes={**BOXES, 0: BOXES[1]})[0])
    with pytest.raises(ValueError, match="expected 3 located"):
        run(inputs(tmp_path / "few", boxes={1: BOXES[1]})[0])
    not_train = tmp_path / "ok" / "sft_train_qa.jsonl"
    saved = not_train.read_text(encoding="utf-8")
    not_train.write_text(json.dumps(dict(image=image(0), split="supervised_val")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an SFT training QA"):
        run(args)
    not_train.write_text(saved, encoding="utf-8")
    assert run(args) == 0
    not_train.write_text(saved.splitlines()[0] + "\n", encoding="utf-8")          # a different training set
    with pytest.raises(ValueError, match="use a new output folder"):
        run(args)


def test_adapter_is_bound_into_the_run_and_must_be_complete(tmp_path):
    args, out = inputs(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no complete LoRA adapter"):
        run(args + ["--adapter-dir", str(adapter)])
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    assert run(args + ["--adapter-dir", str(adapter)]) == 0
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["training"] is True and set(meta["lora_adapter"]) == {"adapter_config.json", "adapter_model.safetensors"}
    with pytest.raises(ValueError, match="use a new output folder"):               # the base model is another run
        run(args)


def test_pilot_runner_can_run_a_subset_of_conditions(tmp_path):
    args, out, _ = pilot_tests.inputs(tmp_path)
    calls = []
    assert pilot_tests.run(args + ["--policies", "adaptive_tools", "direct_reference"], calls) == 0
    assert [r["policy"] for r in table(out, "leaderboard.csv")] == ["direct_reference", "adaptive_tools"]
    assert len(calls) == 6 * 3 * (1 + 3)
    assert json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))["policies"] == ["direct_reference", "adaptive_tools"]
    with pytest.raises(ValueError, match="run both"):
        pilot_tests.run(pilot_tests.inputs(tmp_path / "b")[0] + ["--policies", "matched_reask"], [])


# ---- the one-run notebook (training, then gold scoring) ---------------------------------------------------------

def gpu_gate(stdout, returncode=0, missing=False):
    import textwrap
    from types import SimpleNamespace
    import build_agent_sft_notebook as builder

    class FakeSubprocess:
        @staticmethod
        def run(*args, **kwargs):
            if missing:
                raise FileNotFoundError("nvidia-smi")
            return SimpleNamespace(returncode=returncode, stdout=stdout)
    scope = dict(subprocess=FakeSubprocess, re=re)
    exec(textwrap.dedent(builder.GPU_GATE), scope)
    return scope["GPU_NOW"]


def test_gpu_gate_passes_l4_and_names_the_gpu_it_refuses():
    assert gpu_gate("NVIDIA L4, 23034 MiB\n") == "NVIDIA L4, 23034 MiB"
    assert gpu_gate("NVIDIA A100-SXM4-40GB, 40960 MiB\n").startswith("NVIDIA A100")
    with pytest.raises(RuntimeError, match="지금 연결된 GPU: Tesla T4, 15360 MiB"):
        gpu_gate("Tesla T4, 15360 MiB\n")
    for kw in (dict(missing=True), dict(returncode=9)):
        with pytest.raises(RuntimeError, match="GPU 없음"):
            gpu_gate("", **kw)


def test_notebook_cells_check_gold_inputs_before_training_and_run_the_runner_in_priority_order(tmp_path, capsys):
    import base64
    import hashlib
    import shutil
    import textwrap
    import zlib
    import build_agent_sft_notebook as builder

    drive = tmp_path / "drive"
    results = drive / "lvlm_insplad_results"
    work = drive / "lvlm_insplad_qa"
    inputs(work)                                               # gold QA folder, ZIPs, refs, location set, training list
    (work / "gold_v2_final/build_summary.json").write_text(
        json.dumps(dict(quality_status="FROZEN", final_master_images=246)), encoding="utf-8")
    data_dir, train_dir = results / "agent_sft_data_pc60_sft_v2_human", results / "agent_sft_lora_qwen3_5_4b_sft_v2_human"
    (train_dir / "adapter").mkdir(parents=True)
    (train_dir / "adapter/adapter_config.json").write_text("{}", encoding="utf-8")
    (train_dir / "adapter/adapter_model.safetensors").write_bytes(b"weights")
    data_dir.mkdir(parents=True)
    shutil.copy(work / "sft_train_qa.jsonl", data_dir / "sft_train_qa.jsonl")
    old_refs = results / "gold_v2_matched_call_v1/refs_gold/retrieved_refs.csv"
    old_refs.parent.mkdir(parents=True)
    old_refs.write_text("image,retrieved_diffdate\n" + f"{image(0)},{reference(0)}\n", encoding="utf-8")   # one photo only
    loc = (work / "loc.jsonl").read_bytes()
    commands = []

    def run_command(command, cwd=None, check=True):
        command = [str(x) for x in command]
        commands.append(command)
        script = Path(command[1]).name
        if script == "build_retrieved_refs.py":                # stands in for the DINOv2 search
            out = Path(command[command.index("--out-dir") + 1])
            out.mkdir(parents=True)
            shutil.copy(work / "refs.csv", out / "retrieved_refs.csv")
            return 0
        if script == "run_agent_tool_pilot.py":
            return 0
        assert script == "run_agent_sft_gold_eval.py"
        return gold.main(command[2:], loader=lambda spec: {}, infer=backend([]))

    scope = dict(DRIVE=drive, RESULTS=results, RUN_TAG="sft_v2_human", PER_CLASS=60, N_IMAGES=240,
                 GOLD_QA_FOLDER="lvlm_insplad_qa/gold_v2_final", LOC_QA_SHA256=hashlib.sha256(loc).hexdigest(),
                 LOC_QA_B64=base64.b64encode(zlib.compress(loc, 9)).decode("ascii"),
                 REPO_DIR=ROOT, SUP_ZIP=work / "sup.zip", UNSUP_ZIP=work / "unsup.zip", DATA_DIR=data_dir,
                 TRAIN_DIR=train_dir, SFT_RUN=results / "agent_tool_pilot_dev240_qwen3_5_4b_lora_sft_v2_human",
                 DEV_QA=work / "dev.jsonl", REFS_CSV=work / "dev_refs.csv", RUN_GOLD_BASE=True, RUN_DEV240=True,
                 RUN_GOLD_SFT_V1=True, run=run_command, verify_code=lambda: None,
                 sys=sys, json=json, csv=csv, hashlib=hashlib)
    exec(textwrap.dedent(builder.GOLD_PREP), scope)
    assert scope["LOC_QA"].read_bytes() == loc and not scope["HAS_V1"]
    assert Path(commands[0][1]).name == "build_retrieved_refs.py"          # the old CSV lacks gold photos: not reused
    assert scope["GOLD_REFS"] == results / "agent_sft_gold_sft_v2_human/inputs/refs_gold/retrieved_refs.csv"
    assert all(p.name.startswith("agent_sft_gold_eval") for p in scope["GOLD_RUNS"].values())
    # dev240 never reuses SFT_RUN: an interrupted run made by other source may sit there and cannot be resumed.
    assert scope["DEV_RUN"] == results / "agent_tool_pilot_dev240_qwen3_5_4b_lora_sft_v2_human_two_conditions"
    assert scope["DEV_RUN"] != scope["SFT_RUN"] and "이미 끝난 학습: 있음" in capsys.readouterr().out
    exec(textwrap.dedent(builder.GOLD_CHECK), scope)                        # runs the real runner with --inputs-only
    assert commands[-1][-1] == "--inputs-only" and not scope["GOLD_RUNS"]["gold_lora"].exists()
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (report["images"], report["frame_disjoint"], report["located"]) == (8, 6, dict(frame_disjoint=2, all=3))
    for source in (builder.GOLD_MAIN, builder.GOLD_BASE, builder.GOLD_DEV, builder.GOLD_V1):
        exec(textwrap.dedent(source), scope)
    assert scope["EXIT_CODES"] == dict(gold_lora=0, gold_base=0, dev_lora=0)   # no first-run adapter on this Drive
    ran = [(Path(c[1]).name, "--adapter-dir" in c) for c in commands[2:]]
    assert ran == [("run_agent_sft_gold_eval.py", True), ("run_agent_sft_gold_eval.py", False), ("run_agent_tool_pilot.py", True)]
    dev = commands[-1]
    assert dev[dev.index("--policies") + 1:dev.index("--policies") + 3] == ["direct_reference", "adaptive_tools"]
    assert dev[dev.index("--out-dir") + 1] == str(scope["DEV_RUN"])
    trained =json.loads((scope["GOLD_RUNS"]["gold_lora"] / "run_metadata.json").read_text(encoding="utf-8"))
    untrained = json.loads((scope["GOLD_RUNS"]["gold_base"] / "run_metadata.json").read_text(encoding="utf-8"))
    assert trained["training"] is True and untrained["training"] is False
    assert trained["frame_overlap"] == untrained["frame_overlap"]              # every model is scored on the same photos
    # A reused reference file must cover every gold photo; a first-run adapter on Drive switches its scoring on.
    shutil.copy(work / "refs.csv", old_refs)
    shutil.copytree(train_dir / "adapter", results / "agent_sft_lora_qwen3_5_4b_sft_v1/adapter")
    again = dict(scope)
    exec(textwrap.dedent(builder.GOLD_PREP), again)
    assert again["GOLD_REFS"] == old_refs and again["HAS_V1"]
    (work / "gold_v2_final/build_summary.json").write_text(json.dumps(dict(quality_status="DRAFT")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="FROZEN gold 246장"):
        exec(textwrap.dedent(builder.GOLD_PREP), dict(scope))


def test_gold_notebook_builds_with_the_runner_the_location_set_and_an_l4_runtime(tmp_path):
    import base64
    import hashlib
    import subprocess
    import zlib
    import build_agent_sft_notebook as builder

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "cat-file", "-e", f"{commit}:scripts/run_agent_sft_gold_eval.py"], cwd=ROOT).returncode:
        with pytest.raises(ValueError, match="lacks the gold runner"):
            builder.build(commit, builder.GOLD)
        pytest.skip("HEAD predates the gold runner; commit before building the notebook")
    notebook = builder.build(commit, builder.GOLD)
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == 9 and "시험(gold) 채점" in notebook["cells"][0]["source"]
    assert notebook["metadata"]["colab"]["gpuType"] == "L4" and notebook["metadata"]["accelerator"] == "GPU"
    config = {}
    exec(code[0], config)
    assert config["RUN_TAG"] == "sft_v2_human" and config["RUN_GOLD_BASE"] and config["RUN_DEV240"] and config["RUN_GOLD_SFT_V1"]
    loc = subprocess.check_output(["git", "show", f"{commit}:{builder.LOC_QA}"], cwd=ROOT)
    assert zlib.decompress(base64.b64decode(config["LOC_QA_B64"])) == loc
    assert config["LOC_QA_SHA256"] == hashlib.sha256(loc).hexdigest() and loc.count(b"\n") == 133
    assert code[1].index("지금 연결된 GPU") < code[1].index("drive.mount") < code[1].index('"pip", "install"')
    assert code[1].index("LABEL_DIRS = ") < code[1].index("GOLD_QA_DIR = ")
    assert '"--inputs-only"' in code[2] and "train_agent_sft_lora.py" in code[3]      # inputs are checked before training
    order = [("gold_lora", 4), ("gold_base", 5), ("dev_lora", 6), ("gold_v1", 7)]
    assert all(f'EXIT_CODES["{key}"]' in code[index] for key, index in order)
    assert '"--policies", "direct_reference", "adaptive_tools"' in code[6] and "load_in_4bit" not in "".join(code)
    assert "frame_disjoint" in code[8] and 'f"labels/{folder.name}/{name}"' in code[8]
    assert '"--out-dir", DEV_RUN' in code[6] and "SFT_RUN" not in code[6]           # a fresh dev240 folder
    assert '("dev_lora_earlier_four_condition_run", SFT_RUN)' in code[8] and '"dev_lora": DEV_RUN' in code[8]
    receipt = builder.pilot_nb.extract_source_bundle(config["SOURCE_ZIP_B64"], config["SOURCE_ZIP_SHA256"], tmp_path / "src", commit)
    assert {"scripts/run_agent_sft_gold_eval.py", "scripts/score_gold_v2_q3_position.py", "scripts/q3_multiregion.py"} <= set(receipt["files"])
    for script, flag in (("run_agent_sft_gold_eval.py", "--inputs-only"), ("run_agent_tool_pilot.py", "--policies")):
        result = subprocess.run([sys.executable, tmp_path / "src/scripts" / script, "--help"], capture_output=True, text=True)
        assert result.returncode == 0 and flag in result.stdout, result.stderr
    human = builder.build(commit, "sft_v2_human")              # the earlier notebooks keep their own cells
    assert "GOLD_QA_FOLDER" not in "".join(c["source"] for c in human["cells"]) and len(human["cells"]) == 7
