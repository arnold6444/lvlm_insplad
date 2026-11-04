"""CPU contracts for tool execution, fail-closed scoring and deterministic resume."""
import json
import hashlib
from pathlib import Path
import re
import sys
import subprocess
from types import SimpleNamespace
import zipfile

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_agent_tool_pilot as pilot
import test_gold_v2_verify_agent as fixtures


def inputs(tmp_path, n=6):
    rows = [r for r in fixtures._rows(n) if r["question_type"] != pilot.Q3]
    common, qa = fixtures._inputs(tmp_path, rows, {}, {})
    # Keep only this runner's arguments; no cached answers are used.
    out = tmp_path / "agent_tool_pilot_dev_test"
    args = ["--dev-qa", str(qa), "--refs-csv", str(tmp_path / "retrieved_refs.csv"),
            "--sup-zip", str(tmp_path / "sup.zip"), "--unsup-zip", str(tmp_path / "unsup.zip"),
            "--model", fixtures.MODEL, "--n-images", str(n), "--n-boot", "20", "--out-dir", str(out)]
    return args, out, qa


def backend(calls, failure=None, invalid=False):
    def infer(adapter, images, prompt):
        calls.append((list(images), prompt))
        core = Path(images[0]).stem
        if failure and failure(core, images, prompt):
            raise RuntimeError("test transient inference failure")
        if "Optional tools:" in prompt:
            if invalid:
                return '{"action":"crop","bbox":[-1,0,1,1]}'
            if "normal REFERENCE" not in prompt:
                return '{"action":"reference"}'
            if "is a crop" not in prompt:
                return '{"action":"crop","bbox":[0.2,0.2,0.8,0.8]}'
        options = dict(re.findall(r"^([A-J])\. (.+)$", prompt, re.M))
        anomalous = int(core[1:]) % 2
        if len(options) == 2:
            text = "Anomalous." if anomalous else "Normal."
        elif len(options) == 4:
            text = "Rust / corrosion." if anomalous else "Normal / no defect."
        else:
            text = "Center region." if anomalous else pilot.support.NONE
        return json.dumps(dict(action="finish", choice=next(k for k, v in options.items() if v == text)))
    return infer


def run(args, calls, **kwargs):
    return pilot.main(args, loader=lambda spec: {}, infer=backend(calls, **kwargs))


def test_end_to_end_tools_controls_scores_and_resume(tmp_path):
    args, out, _ = inputs(tmp_path)
    calls = []
    assert run(args, calls) == 0
    assert len(calls) == 144
    answers = pilot.agent.read_jsonl(out / "answers.jsonl")
    by_key = {(r["policy"], r["id"]): r for r in answers}
    for r in answers:
        if r["policy"] == "adaptive_tools":
            assert r["calls"] == 3
            assert [x["output"]["action"] for x in r["tool_actions"]] == ["reference", "crop"]
            assert by_key["matched_reask", r["id"]]["calls"] == r["calls"]
    board = pilot.agent.read_jsonl(out / "answers.jsonl")
    assert len(board) == 72
    with (out / "leaderboard.csv").open() as stream:
        rows = list(pilot.csv.DictReader(stream))
    assert len(rows) == 4
    assert all(float(r["joint_accuracy"]) == 1 for r in rows)
    assert all(r["q1_q2_q3_location_joint"] == r["q3_location_accuracy"] == "" for r in rows)
    assert all(float(r["q1_q2_q3_presence_joint"]) == 1 for r in rows)
    assert json.loads((out / "run_metadata.json").read_text())["execution"] == "injected_test_backend"
    assert json.loads((out / "run_status.json").read_text())["pilot_format_gate"]
    crops = list((out / "_crops").glob("*.png"))
    assert crops and all(Image.open(p).size == (24, 18) for p in crops)
    calls.clear()
    assert run(args, calls) == 0 and calls == []
    resumed = json.loads((out / "run_status.json").read_text())
    assert resumed["calls_this_run"] == 0 and resumed["reused_calls"] == 144


def test_error_excluded_and_only_failed_call_retried(tmp_path):
    args, out, _ = inputs(tmp_path)
    failed = []
    def once(core, images, prompt):
        if not failed and core == "c1" and len(images) == 3:
            failed.append(True)
            return True
        return False
    calls = []
    assert run(args, calls, failure=once) == 1
    status = json.loads((out / "run_status.json").read_text())
    assert not status["complete"] and status["row_errors"] == 1
    assert "fixed_evidence" not in status["eligible_policies"]
    calls.clear()
    assert run(args, calls) == 0 and len(calls) == 1


def test_invalid_actions_are_not_retried_or_replaced_with_a_good_answer(tmp_path):
    args, out, _ = inputs(tmp_path)
    calls = []
    assert run(args, calls, invalid=True) == 0
    status = json.loads((out / "run_status.json").read_text())
    assert status["invalid_episodes"] == 18 and not status["pilot_format_gate"]
    assert all(r["predicted_choice"] is None for r in pilot.agent.read_jsonl(out / "answers.jsonl") if r["policy"] == "adaptive_tools")
    calls.clear()
    assert run(args, calls, invalid=True) == 0 and not calls


@pytest.mark.parametrize("bad", [
    '{"action":"crop","bbox":[0,0,0,1]}', '{"action":"crop","bbox":[0,0,2,1]}',
    '{"action":"crop","bbox":[true,0,1,1]}', '{"action":"crop","bbox":[0,0,NaN,1]}',
    '{"action":"finish","choice":"Z"}', '{"action":"search"}',
    '{"action":"reference","extra":1}', '[]', 'not json',
])
def test_strict_schema(bad):
    with pytest.raises((ValueError, TypeError)):
        pilot.parse_action(bad, {"A": "Normal."})


def test_prompt_does_not_include_labels_paths_or_other_answers():
    row = fixtures._rows()[0]
    original = pilot.prompt_for(row, [], False, None, True)
    changed = dict(row, answer="DO_NOT_LEAK", anomaly_gt="SECRET", defect_gt="SECRET", image="SECRET", core_id="SECRET")
    assert original == pilot.prompt_for(changed, [], False, None, True)
    assert "SECRET" not in original


def test_dev240_contains_same_dev40_and_gold_is_rejected():
    rows = [r for r in fixtures._rows(240) if r["question_type"] != pilot.Q3]
    small, full = pilot.select_rows(rows, 40, 42), pilot.select_rows(rows, 240, 42)
    assert full[:80] == small and len(full) == 480
    with pytest.raises(ValueError, match="gold"):
        pilot.select_rows([dict(r, protocol="gold_v2_final") for r in rows], 40, 42)
    with pytest.raises(ValueError):
        pilot.select_rows(rows+[rows[0]], 40, 42)


def test_changed_inputs_and_images_refuse_resume(tmp_path):
    args, out, qa = inputs(tmp_path)
    assert run(args, []) == 0
    data = pilot.agent.read_jsonl(qa)
    data[0]["question"] += " changed"
    pilot.base.write_jsonl(qa, data)
    with pytest.raises(ValueError, match="changed"):
        run(args, [])
    data[0]["question"] = data[0]["question"].removesuffix(" changed")
    pilot.base.write_jsonl(qa, data)
    with zipfile.ZipFile(tmp_path / "sup.zip", "w") as archive:
        for name in sorted({r["image"] for r in data}):
            archive.writestr(name, fixtures._png(99))
    with pytest.raises(ValueError, match="bytes changed"):
        run(args, [])


def test_unsafe_archive_member_rejected(tmp_path):
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as archive:
        archive.writestr("../escape.png", fixtures._png(0))
    with pytest.raises(ValueError, match="unsafe"):
        pilot.support.verified_images(tmp_path / "bad.zip", ["../escape.png"], tmp_path / "images")
    assert not (tmp_path / "escape.png").exists()


def test_model_load_failure_stops_immediately(tmp_path):
    args, out, _ = inputs(tmp_path)
    loads = []
    def loader(spec):
        loads.append(spec)
        raise RuntimeError("GPU unavailable")
    with pytest.raises(pilot.ModelLoadError):
        pilot.main(args, loader=loader, infer=backend([]))
    assert len(loads) == 1
    assert json.loads((out / "run_status.json").read_text())["status"] == "fatal_error"


def test_notebook_builder_uses_full_source_pin_and_compiles_all_cells(tmp_path):
    import build_agent_tool_pilot_notebook as builder
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    notebook = builder.build(commit)
    assert notebook["nbformat"] == 4
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == 4
    for source in code:
        compile(source, "<test notebook>", "exec")
    assert 'EXECUTION_COMMIT = "' + commit + '"' in code[0]
    config = {}
    exec(code[0], config)
    assert config["N_IMAGES"] == 240 and config["RUN_TAG"] == "once_v2"
    assert '"git", "clone"' not in code[1]
    unpacked = tmp_path / "embedded"
    receipt = builder.extract_source_bundle(config["SOURCE_ZIP_B64"], config["SOURCE_ZIP_SHA256"], unpacked, commit)
    assert receipt["commit"] == commit and not (unpacked / ".git").exists()
    assert (unpacked / "scripts/run_agent_tool_pilot.py").is_file()
    # Standalone extracted source can import its helpers and expose the CLI with no GitHub access.
    result = subprocess.run([sys.executable, unpacked / "scripts/run_agent_tool_pilot.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--n-images" in result.stdout
    assert 'status.get("run_id") != metadata.get("run_id")' in code[-1]
    assert 'metadata.get("execution") != "real_hf"' in code[-1]
    with pytest.raises(ValueError):
        builder.build("main")
    with pytest.raises(ValueError, match="hash mismatch"):
        builder.extract_source_bundle(config["SOURCE_ZIP_B64"], "0" * 64, tmp_path / "bad", commit)
    (unpacked / "scripts/run_agent_tool_pilot.py").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        builder.extract_source_bundle(config["SOURCE_ZIP_B64"], config["SOURCE_ZIP_SHA256"], unpacked, commit)


def test_gpu_guard_refuses_local_6gib_and_accepts_l4_memory():
    def device(gib):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                               get_device_properties=lambda i: SimpleNamespace(total_memory=gib*1024**3)))
    with pytest.raises(RuntimeError, match="20 GiB"):
        pilot.require_gpu(device(6))
    pilot.require_gpu(device(24))


def test_adaptive_can_finish_immediately_and_matched_control_uses_one_call(tmp_path):
    args, out, _ = inputs(tmp_path)
    calls = []
    infer = backend(calls)
    def finish(adapter, images, prompt):
        return infer(adapter, images, prompt.replace("Optional tools:", "No optional tools:"))
    assert pilot.main(args, loader=lambda spec: {}, infer=finish) == 0
    answers = pilot.agent.read_jsonl(out / "answers.jsonl")
    assert len(calls) == 72
    assert all(r["calls"] == 1 and not r["tool_actions"] for r in answers if r["policy"] == "adaptive_tools")


def test_embedded_source_commit_checks_receipt_and_all_bytes(tmp_path, monkeypatch):
    script = tmp_path / "scripts/run_agent_tool_pilot.py"
    script.parent.mkdir()
    script.write_bytes(b"source fixture")
    receipt = dict(commit="a" * 40, files={"scripts/run_agent_tool_pilot.py": hashlib.sha256(script.read_bytes()).hexdigest()})
    (tmp_path / ".agent_pilot_source.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot, "__file__", str(script))
    assert pilot.source_commit() == "a" * 40
    script.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source changed"):
        pilot.source_commit()


def test_report_downloads_diagnostics_on_inference_error(tmp_path, monkeypatch):
    import types
    import build_agent_tool_pilot_notebook as builder
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    notebook = builder.build(commit)
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_status.json").write_text(json.dumps(dict(complete=False, run_id="r", pilot_format_gate=False)))
    (root / "run_metadata.json").write_text(json.dumps(dict(execution="real_hf", run_id="r", benchmark_commit=commit)))
    private = root / "_images"
    private.mkdir()
    (private / "raw.png").write_bytes(b"must not export raw images")
    downloads = []
    google, colab = types.ModuleType("google"), types.ModuleType("google.colab")
    colab.files = SimpleNamespace(download=downloads.append)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = dict(RUN_ROOT=root, EXECUTION_COMMIT=commit, INFERENCE_EXIT_CODE=1,
                     json=json, zipfile=zipfile, display=lambda x: None)
    exec(notebook["cells"][-1]["source"], namespace)
    assert not namespace["verified_complete"] and len(downloads) == 1
    with zipfile.ZipFile(downloads[0]) as archive:
        assert "run_status.json" in archive.namelist()
        assert not json.loads(archive.read("notebook_execution.json"))["verified_complete"]
        assert not any(n.startswith("_images") for n in archive.namelist())
