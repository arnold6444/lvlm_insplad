"""CPU contracts for the SFT data builder, the runner's LoRA option and the SFT Colab builder."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import build_agent_sft_data as sft  # noqa: E402
import run_agent_tool_pilot as pilot  # noqa: E402
import sft_train_location_annotation as ann  # noqa: E402
import test_agent_tool_pilot as pilot_tests  # noqa: E402

CLASSES = {"glass-insulator": ["good", "missing-cap"], "lightning-rod-suspension": ["good", "rust"],
           "polymer-insulator-upper-shackle": ["normal", "corrosão"], "vari-grip": ["good", "bird-nest", "rust"],
           "yoke-suspension": ["good", "rust"]}


def png(seed, size=(42, 28)):
    buf = io.BytesIO()
    Image.new("RGB", size, (seed % 251, (seed * 7) % 251, (seed * 13) % 251)).save(buf, format="PNG")
    return buf.getvalue()


def fixture(tmp):
    """Supervised train photos named like InsPLAD: 'Fotos DD-MM-YYYY_DJI_nnnn_..._k.png'."""
    sup, unsup = tmp / "sup.zip", tmp / "unsup.zip"
    dev_images, n = [], 0
    with zipfile.ZipFile(sup, "w") as z, zipfile.ZipFile(unsup, "w") as u:
        for a, (asset, classes) in enumerate(CLASSES.items()):
            for i in range(3):
                u.writestr(f"{asset}/train/good/Fotos 01-01-2020_DJI_{900 + a * 10 + i:04d}_ref.png", png(5000 + n)); n += 1
            for c, cls in enumerate(classes):
                for i in range(6):
                    frame = 1000 + a * 100 + c * 10 + i
                    name = f"defect_supervised/{asset}/train/{cls}/Fotos 0{2 + i % 2}-02-2020_DJI_{frame:04d}_p_{i}.png"
                    data = png(n); n += 1
                    z.writestr(name, data)
                    if i == 0:
                        dev_images.append(name)
                        # Another crop of the same drone photo must be excluded too.
                        z.writestr(name.replace("_p_0.png", "_p_9.png"), png(n)); n += 1
                    if i == 5:
                        u.writestr(f"{asset}/test/good/copy_{frame}.png", data)   # same bytes as unsup test
    dev = []
    for img in dev_images:
        dev.append(dict(id=f"d{len(dev)}-q1", core_id=f"d{len(dev)}", image=img, protocol="gold_v2_dev",
                        question_type=pilot.Q1, options={"A": "Normal.", "B": "Anomalous."}, answer="A"))
    qa = tmp / "dev.jsonl"
    pilot.base.write_jsonl(qa, dev)
    return sup, unsup, qa, set(dev_images)


def fake_embed():
    def embed(rows, which):
        out = {}
        for r in rows:
            v = np.frombuffer(hashlib.sha256(str(r["relative_path"]).encode()).digest()[:16], dtype=np.uint8).astype(np.float32)
            out[str(r["relative_path"])] = v / np.linalg.norm(v)
        return out
    return embed


def fake_patches(images, strength=None):
    """Reference = identity patches; the target differs in a lower-right 4 x 4 patch block by an image-dependent amount."""
    grid = sft.PATCH_INPUT // 14
    n = grid * grid
    ref = np.zeros((n, n + 1), dtype=np.float32)        # one spare dimension no reference patch uses
    ref[np.arange(n), np.arange(n)] = 1
    if strength is None:
        strength = 0.3 + 0.6 * (float(np.asarray(images[0]).mean()) % 7) / 7
    target = ref.copy()
    for r in range(grid - 4, grid):
        for c in range(grid - 4, grid):
            i = r * grid + c
            target[i, i], target[i, -1] = np.cos(strength), np.sin(strength)   # diff = 1 - cos(strength)
    return np.stack([target, ref])


def build(tmp, **kw):
    sup, unsup, qa, dev_images = fixture(tmp)
    out = tmp / "data"
    args = ["--dev-qa", str(qa), "--sup-zip", str(sup), "--unsup-zip", str(unsup), "--out-dir", str(out),
            "--per-class", "2", "--device", "cpu", *kw.get("extra", [])]
    assert sft.main(args, patch_encoder=fake_patches, embed_factory=fake_embed) == 0
    return out, dev_images, args


def test_builder_holds_out_dev_photos_test_copies_and_corrosion(tmp_path):
    out, dev_images, _ = build(tmp_path)
    summary = json.loads((out / "build_summary.json").read_text(encoding="utf-8"))
    rows = pilot.agent.read_jsonl(out / "sft_train_qa.jsonl")
    images = {r["image"] for r in rows}
    dev_keys = {sft.photo_key(p) for p in dev_images}
    assert images and not images & dev_images and not {sft.photo_key(p) for p in images} & dev_keys
    assert not any("_p_5" in p or "corros" in p for p in images)
    assert summary["selection"]["excluded_before_sampling"]["dev240_same_drone_photo"] == len(dev_images)
    assert summary["images"] == 20 and summary["by_asset_defect"]["polymer-insulator-upper-shackle:normal"] == 2
    assert all(r["protocol"] == sft.PROTOCOL and r["core_id"].startswith("sft-") for r in rows)
    assert summary["crop_by_label"]["normal:no_crop"] > 0 and "anomalous:no_crop" not in summary["crop_by_label"]
    assert summary["centred_fallback_boxes"] == 0


def test_trajectories_replay_through_the_eval_runner_with_identical_prompts(tmp_path):
    out, _, _ = build(tmp_path)
    samples = pilot.agent.read_jsonl(out / "sft_samples.jsonl")
    boxes = {r["core_id"]: r for r in csv.DictReader((out / "boxes.csv").open(encoding="utf-8"))}
    rows = pilot.agent.read_jsonl(out / "sft_train_qa.jsonl") + pilot.agent.read_jsonl(out / "sft_train_q3.jsonl")
    assert {r["anomaly_gt"] for r in rows} == {"normal", "anomalous"}
    by_id = {}
    for s in samples:
        by_id.setdefault(s["id"], []).append(s)
    for row in rows:
        steps = sorted(by_id[row["id"]], key=lambda s: s["sample_id"])
        info = boxes[row["core_id"]]
        assert [json.loads(s["target"])["action"] for s in steps] == (
            ["reference", "crop", "finish"] if info["crop"] == "True" else ["reference", "finish"])
        if row["anomaly_gt"] == "anomalous":
            assert info["crop"] == "True"
        final = json.loads(steps[-1]["target"])["choice"]
        if row["question_type"] == pilot.Q3:
            assert row["options"][final] == info["q3_cell"]
            assert (info["q3_cell"] == pilot.support.NONE) == (row["anomaly_gt"] == "normal")
        else:
            assert final == row["answer"]
        replies = iter(s["target"] for s in steps)
        seen = []
        def call(images, prompt):
            seen.append((len(images), prompt))
            return next(replies)
        result = pilot.episode(row, "adaptive_tools", out / steps[0]["images"][0], out / steps[1]["images"][1],
                               tmp_path / "replay", "id", call, 3)
        assert result["predicted_choice"] == final and not result["invalid"]
        assert seen == [(len(s["images"]), s["prompt"]) for s in steps]


def test_auto_box_cell_and_difference_map():
    grid = sft.PATCH_INPUT // 14
    diff = sft.difference_map(*fake_patches(None, strength=np.pi / 2), grid)
    assert diff.shape == (grid, grid) and diff[-1, -1] == pytest.approx(1) and diff[0, 0] == 0
    box = sft.normalized_box(diff, (420, 280))
    assert 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
    assert sft.cell_text(box) == "Lower-right region."
    assert sft.cell_text([0.0, 0.0, 0.2, 0.2]) == "Upper-left region." and sft.cell_text([0.4, 0.4, 0.6, 0.6]) == "Center region."
    with pytest.raises(ValueError):
        sft.difference_map(np.zeros((4, 3)), np.zeros((5, 3)), 2)


def test_builder_refuses_other_settings_and_non_dev_qa(tmp_path):
    out, _, args = build(tmp_path)
    assert sft.main(args, patch_encoder=fake_patches, embed_factory=fake_embed) == 0       # same settings: no-op
    changed = [a if a != "2" else "3" for a in args]
    with pytest.raises(ValueError, match="other settings"):
        sft.main(changed, patch_encoder=fake_patches, embed_factory=fake_embed)
    rows = pilot.agent.read_jsonl(Path(args[1]))
    pilot.base.write_jsonl(Path(args[1]), [dict(r, protocol="gold_v2_final") for r in rows])
    with pytest.raises(ValueError, match="gold_v2_dev"):
        sft.main([*args[:-6], "--out-dir", str(tmp_path / "other"), "--per-class", "2", "--device", "cpu"],
                 patch_encoder=fake_patches, embed_factory=fake_embed)


def test_runner_lora_option_binds_adapter_bytes_and_rejects_other_models(tmp_path):
    args, out, _ = pilot_tests.inputs(tmp_path)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    assert pilot.main(args + ["--adapter-dir", str(adapter)], loader=lambda spec: {}, infer=pilot_tests.backend([])) == 0
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["training"] is True and set(meta["lora_adapter"]) == {"adapter_config.json", "adapter_model.safetensors"}
    (adapter / "adapter_model.safetensors").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="changed"):
        pilot.main(args + ["--adapter-dir", str(adapter)], loader=lambda spec: {}, infer=pilot_tests.backend([]))
    with pytest.raises(ValueError, match="qwen3_5_4b only"):
        pilot.main([*[a if a != pilot_tests.fixtures.MODEL else "internvl3_5_8b" for a in args], "--adapter-dir", str(adapter)],
                   loader=lambda spec: {}, infer=pilot_tests.backend([]))
    with pytest.raises(FileNotFoundError):
        pilot.main(args + ["--adapter-dir", str(tmp_path / "missing")], loader=lambda spec: {}, infer=pilot_tests.backend([]))


def test_lr_schedule_and_resume_order_are_deterministic():
    import train_agent_sft_lora as train
    factors = [train.lr_factor(s, 2, 10) for s in range(10)]
    assert factors[:2] == [0.5, 1.0] and factors[-1] == pytest.approx(0.125) and max(factors) == 1
    assert train.epoch_order(20, 42, 0) == train.epoch_order(20, 42, 0) != train.epoch_order(20, 42, 1)


def test_label_mask_covers_only_the_action_and_end_token():
    torch = pytest.importorskip("torch")
    import train_agent_sft_lora as train

    class Tok:
        unk_token_id = 0
        def __call__(self, text, add_special_tokens, return_tensors):
            return {"input_ids": torch.tensor([[ord(c) for c in text]])}

    class Proc:
        tokenizer = Tok()
        def apply_chat_template(self, messages, **kw):
            assert kw["add_generation_prompt"] and kw["enable_thinking"] is False
            n = 7 + len(messages[0]["content"])
            return {"input_ids": torch.ones((1, n), dtype=torch.long), "attention_mask": torch.ones((1, n), dtype=torch.long),
                    "mm_token_type_ids": torch.ones((1, n), dtype=torch.long), "pixel_values": torch.zeros((3, 4))}

    sample = dict(images=[Path("a.png"), Path("b.png")], prompt="p", target='{"action":"reference"}')
    batch, n_prompt = train.encode(Proc(), sample, 99)
    labels = batch["labels"][0]
    assert n_prompt == 10 and (labels[:n_prompt] == -100).all()
    assert labels[n_prompt:].tolist() == [ord(c) for c in sample["target"]] + [99]
    assert batch["attention_mask"][0, n_prompt:].eq(1).all() and batch["mm_token_type_ids"][0, n_prompt:].eq(0).all()
    assert batch["pixel_values"].shape == (3, 4)


def test_action_loss_equals_full_length_loss_and_refuses_ignored_logits_to_keep():
    torch = pytest.importorskip("torch")
    import train_agent_sft_lora as train

    class Out:
        def __init__(self, logits):
            self.logits = logits

    class Tiny(torch.nn.Module):
        def __init__(self, honour=True):
            super().__init__()
            torch.manual_seed(0)
            self.emb, self.head, self.honour = torch.nn.Embedding(50, 8), torch.nn.Linear(8, 50), honour
        def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
            h = self.emb(input_ids)
            if self.honour and logits_to_keep:
                h = h[:, -logits_to_keep:]
            return Out(self.head(h))

    ids = torch.randint(0, 50, (1, 12))
    n_prompt = 8
    labels = torch.full_like(ids, -100)
    labels[:, n_prompt:] = ids[:, n_prompt:]
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    model = Tiny()
    full = model(ids).logits[:, :-1]
    expected = torch.nn.functional.cross_entropy(full.reshape(-1, 50), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(train.action_loss(model, batch, n_prompt), expected)
    with pytest.raises(RuntimeError, match="logits_to_keep"):
        train.action_loss(Tiny(honour=False), batch, n_prompt)


def test_sft_notebook_compiles_and_embeds_the_training_scripts(tmp_path):
    import build_agent_sft_notebook as builder
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    has = subprocess.run(["git", "cat-file", "-e", f"{commit}:scripts/train_agent_sft_lora.py"], cwd=ROOT).returncode == 0
    if not has:
        with pytest.raises(ValueError, match="lacks"):
            builder.build(commit)
        pytest.skip("HEAD predates the SFT scripts; commit before building the notebook")
    notebook = builder.build(commit)
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == 6
    config = {}
    exec(code[0], config)
    assert config["PER_CLASS"] == 60 and config["N_IMAGES"] == 240
    receipt = builder.pilot_nb.extract_source_bundle(config["SOURCE_ZIP_B64"], config["SOURCE_ZIP_SHA256"], tmp_path / "src", commit)
    assert {"scripts/build_agent_sft_data.py", "scripts/train_agent_sft_lora.py"} <= set(receipt["files"])
    for script in ("build_agent_sft_data.py", "train_agent_sft_lora.py", "run_agent_tool_pilot.py"):
        result = subprocess.run([sys.executable, tmp_path / "src/scripts" / script, "--help"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    assert '"--adapter-dir", TRAIN_DIR / "adapter"' in code[4] and "load_in_4bit" not in "".join(code)
    assert 'bool(meta.get("training")) == (label == "lora")' in code[5]
    assert '"uninstall", "-y", "-q", "torchao"' in code[1] and code[1].index("torchao") > code[1].index('"peft>=0.18"')


def test_human_box_notebook_trains_on_the_label_folders_and_evaluates_only_the_new_adapter(tmp_path):
    import build_agent_sft_notebook as builder
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if b"--human-boxes" not in subprocess.check_output(["git", "show", f"{commit}:scripts/build_agent_sft_data.py"], cwd=ROOT):
        with pytest.raises(ValueError, match="human-box"):
            builder.build(commit, "sft_v2_human")
        pytest.skip("HEAD predates the human-box builder; commit before building the notebook")
    with pytest.raises(ValueError, match="unknown variant"):
        builder.build(commit, "sft_v3")
    notebook = builder.build(commit, "sft_v2_human")
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == 6 and "사람이 그린 박스" in notebook["cells"][0]["source"]
    config = {}
    exec(code[0], config)
    assert config["RUN_TAG"] == "sft_v2_human" and config["PER_CLASS"] == 60 and config["N_IMAGES"] == 240
    assert config["LABEL_RUNS"] == builder.LABEL_RUNS and config["FIRST_LABELS_SHA256"] == builder.FIRST_LABELS_SHA256
    assert code[1].index("LABEL_DIRS = ") > code[1].index('RESULTS = DRIVE / "lvlm_insplad_results"')
    assert '"--add-polymer-corrosion", "--human-boxes", *LABEL_DIRS' in code[2] and "agent_sft_train_v2_human_boxes" in code[2]
    assert "train_agent_sft_lora.py" in code[3]
    assert code[4].count("run_agent_tool_pilot.py") == 1 and '"--adapter-dir", TRAIN_DIR / "adapter"' in code[4]
    assert "BASE_RUN" not in code[4] + code[5] and 'meta.get("training") is True' in code[5]
    assert 'f"labels/{folder.name}/{name}"' in code[5] and "load_in_4bit" not in "".join(code)
    receipt = builder.pilot_nb.extract_source_bundle(config["SOURCE_ZIP_B64"], config["SOURCE_ZIP_SHA256"], tmp_path / "src", commit)
    assert "scripts/sft_train_location_annotation.py" in receipt["files"]
    result = subprocess.run([sys.executable, tmp_path / "src/scripts/build_agent_sft_data.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--human-boxes" in result.stdout and "--add-polymer-corrosion" in result.stdout
    first = builder.build(commit)                              # the sft_v1 notebook keeps its own cells
    v1 = [c["source"] for c in first["cells"] if c["cell_type"] == "code"]
    assert 'RUN_TAG = "sft_v1"' in v1[0] and "LABEL_RUNS" not in "".join(v1) and "BASE_RUN" in v1[4]


def test_human_box_notebook_stops_before_training_while_photos_are_unlabelled(tmp_path, capsys):
    import build_agent_sft_notebook as builder
    import textwrap
    sup, _, _, first, polymer = human_fixture(tmp_path)
    runs = {label_run(tmp_path, "first", sup, first, BOXES).name: len(first),
            label_run(tmp_path, "polymer", sup, polymer, BOXES, {1: "pending"}).name: len(polymer)}
    scope = dict(RESULTS=tmp_path, LABEL_RUNS=runs, FIRST_LABELS_SHA256="0" * 64, N_IMAGES=240, json=json)
    gate = textwrap.dedent(builder.HUMAN_CHECK)
    with pytest.raises(RuntimeError, match="표시가 1/2장입니다"):
        exec(gate, dict(scope))
    with pytest.raises(RuntimeError, match="표시 결과가 없습니다"):
        exec(gate, dict(scope, LABEL_RUNS={"sft_train_location_annotation_missing": 1, **runs}))
    session = ann.prepare(ann.read(tmp_path / f"{ann.PREFIX}_polymer/manifest.json"), sup, tmp_path / f"{ann.PREFIX}_polymer")
    session.save(1, BOXES[0], "boxed", "", 0)
    done = dict(scope)
    exec(gate, done)
    out = capsys.readouterr().out
    assert "표시 2/2장" in out and "경고: 300장 표시가" in out and [p.name for p in done["LABEL_DIRS"]] == list(runs)
    assert sft.load_human_boxes(done["LABEL_DIRS"])[1][1]["status"] == {"boxed": 2}     # what the gate lets through, loads
    with pytest.raises(RuntimeError, match="표시가 2/2장입니다"):                          # a different target count
        exec(gate, dict(scope, LABEL_RUNS={**runs, f"{ann.PREFIX}_polymer": 60}))


def human_fixture(tmp):
    """The fixture selection with polymer corrosion, plus its anomalous photos split like the two real label runs."""
    sup, unsup, qa, dev_images = fixture(tmp)
    unsup_rows = sft.parse_unsupervised_manifest(sft.devset.zip_manifest(unsup, tmp / "m/unsup.csv"))
    sup_rows, _ = sft.parse_supervised_manifest(sft.devset.zip_manifest(sup, tmp / "m/sup.csv"))
    chosen, _, _ = sft.select_train_images(sup_rows, unsup_rows, dev_images, 2, 42, [sft.POLYMER_CORROSION])
    anomalous = [r for r in chosen if r["defect_class"] != "normal"]
    polymer = [r for r in anomalous if r["defect_class"] == "corrosao"]
    return sup, unsup, qa, [r for r in anomalous if r not in polymer], polymer


def label_run(tmp, name, sup, rows, boxes, status=None):
    """A run folder written by the real label tool. ``boxes[i]`` are photo i's xywh boxes; ``status`` overrides one."""
    targets = []
    with zipfile.ZipFile(sup) as z:
        for i, r in enumerate(rows):
            targets.append(dict(core_id=f"sft-{name}{i}", image=r["relative_path"], sha256=ann.sha(z.read(r["relative_path"])),
                                size=[42, 28], asset_category=r["asset_category"], defect_gt=r["defect_class"],
                                anomaly_gt="anomalous", sft_split="pending", content_fingerprint=str(r["content_fingerprint"])))
    session = ann.prepare(dict(protocol=ann.PROTOCOL, source=dict(fixture=True), cohort=name, targets=targets), sup,
                          tmp / f"{ann.PREFIX}_{name}")
    for i in range(len(rows)):
        state = (status or {}).get(i, "boxed")
        if state == "boxed":
            session.save(i, boxes[i % len(boxes)], "boxed", "", 0)
        elif state != "pending":
            session.save(i, [], state, "cannot see it", 0)
    return session.out


BOXES = [[[.70, .10, .10, .12]],                          # small: grown to 0.22 around its centre
         [[.05, .05, .10, .10], [.30, .40, .50, .50]],    # two boxes: the larger one is used
         [[.00, .00, .08, .09]]]                          # small in the corner: grown, then shifted into the frame


def human_args(tmp, qa, sup, unsup, runs, out="human"):
    return ["--dev-qa", str(qa), "--sup-zip", str(sup), "--unsup-zip", str(unsup), "--out-dir", str(tmp / out),
            "--per-class", "2", "--device", "cpu", "--add-polymer-corrosion", "--human-boxes", *map(str, runs)]


def no_patch_model(images):
    raise AssertionError("human-box data must not load the DINOv2 patch model")


def test_human_crop_box_uses_the_largest_box_and_grows_small_ones_inside_the_frame():
    box, cell, grown = sft.human_crop_box(BOXES[0][:])
    assert grown and cell == "Upper-right region." and box == [.64, .05, .86, .27]
    assert box[0] <= .75 <= box[2] and box[1] <= .16 <= box[3]            # the human box centre stays inside
    assert sft.human_crop_box(BOXES[1]) == ([.3, .4, .8, .9], "Center region.", False)
    assert sft.human_crop_box(BOXES[2]) == ([0.0, 0.0, .22, .22], "Upper-left region.", True)
    assert sft.human_crop_box([[.9, .9, .1, .1]]) == ([.78, .78, 1.0, 1.0], "Lower-right region.", True)
    assert sft.human_crop_box([[0, 0, 1, 1]]) == ([0.0, 0.0, 1.0, 1.0], "Center region.", False)
    for boxes in BOXES:                                                    # the Q3 cell is the label tool's own cell
        largest = max(boxes, key=lambda b: b[2] * b[3])
        assert sft.CELL_ORDER.index(sft.human_crop_box(boxes)[1]) + 1 == ann.box_regions([largest])[0]


def test_human_boxes_become_the_crop_and_the_q3_answer_and_normals_request_no_crop(tmp_path):
    sup, unsup, qa, first, polymer = human_fixture(tmp_path)
    runs = [label_run(tmp_path, "first", sup, first, BOXES), label_run(tmp_path, "polymer", sup, polymer, BOXES)]
    args = human_args(tmp_path, qa, sup, unsup, runs)
    assert sft.main(args, patch_encoder=no_patch_model, embed_factory=fake_embed) == 0
    assert sft.main(args, patch_encoder=no_patch_model, embed_factory=fake_embed) == 0      # same labels: no-op
    out = tmp_path / "human"
    summary = json.loads((out / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"] == sft.HUMAN_PROTOCOL and summary["images"] == 22
    assert summary["settings"]["extra_classes"] == ["polymer-insulator-upper-shackle:corrosao"]
    assert summary["settings"]["human_boxes"] == [pilot.file_sha(r / "annotations.json") for r in runs]
    assert summary["crop_by_label"] == {"anomalous:crop": 12, "normal:no_crop": 10}
    assert summary["polymer_corrosion_photos_answered_as_rust"] == 2 and summary["unlocated_photos_left_out"] == 0
    assert summary["by_asset_defect"]["polymer-insulator-upper-shackle:rust"] == 2
    assert [r["status"] for r in summary["human_box_runs"]] == [{"boxed": 10}, {"boxed": 2}]
    assert summary["grown_small_boxes"] == 8 and summary["distinct_crop_boxes"] == 3 and sum(summary["q3_cells"].values()) == 12
    label = {}
    for run in runs:
        targets, records = ann.read(run / "manifest.json")["targets"], ann.read(run / "annotations.json")["records"]
        label.update({t["image"]: records[t["core_id"]]["boxes_xywh"] for t in targets})
    rows = pilot.agent.read_jsonl(out / "sft_train_qa.jsonl") + pilot.agent.read_jsonl(out / "sft_train_q3.jsonl")
    assert all(r["protocol"] == sft.HUMAN_PROTOCOL for r in rows)
    polymer_q2 = [r for r in rows if "corros" in r["image"] and r["question_type"] == pilot.Q2]
    assert len(polymer_q2) == 2 and all(r["answer_text"] == sft.devset.DEFECT_TEXT["rust"] and r["defect_gt"] == "rust"
                                        for r in polymer_q2)
    by_id = {}
    for s in pilot.agent.read_jsonl(out / "sft_samples.jsonl"):
        by_id.setdefault(s["id"], []).append(s)
    for row in rows:
        steps = [json.loads(s["target"]) for s in sorted(by_id[row["id"]], key=lambda s: s["sample_id"])]
        if row["anomaly_gt"] == "normal":
            assert [s["action"] for s in steps] == ["reference", "finish"]
            cell = pilot.support.NONE
        else:
            box, cell, _ = sft.human_crop_box(label[row["image"]])
            assert steps[:2] == [{"action": "reference"}, {"action": "crop", "bbox": box}] and len(steps) == 3
        if row["question_type"] == pilot.Q3:
            assert row["options"][steps[-1]["choice"]] == cell
        else:
            assert steps[-1]["choice"] == row["answer"]
        replies = iter(s["target"] for s in sorted(by_id[row["id"]], key=lambda s: s["sample_id"]))
        first_step = sorted(by_id[row["id"]], key=lambda s: s["sample_id"])
        result = pilot.episode(row, "adaptive_tools", out / first_step[0]["images"][0], out / first_step[1]["images"][1],
                               tmp_path / "replay", "id", lambda images, prompt: next(replies), 3)
        assert result["predicted_choice"] == steps[-1]["choice"] and not result["invalid"]
    boxes = list(csv.DictReader((out / "boxes.csv").open(encoding="utf-8")))
    assert {r["box_source"] for r in boxes} == {"human", "none"}
    assert all((r["bbox"] == "null") == (r["anomaly_gt"] == "normal") for r in boxes)


def test_unfinished_missing_or_changed_human_boxes_are_refused_and_unlocated_photos_leave_training(tmp_path):
    sup, unsup, qa, first, polymer = human_fixture(tmp_path)
    done = label_run(tmp_path, "first", sup, first, BOXES)
    with pytest.raises(ValueError, match="1 of 2 photos are not labelled yet"):
        sft.load_human_boxes([done, label_run(tmp_path, "half", sup, polymer, BOXES, {1: "pending"})])
    with pytest.raises(ValueError, match="2 unlabelled, 0 labelled but not selected"):
        sft.main(human_args(tmp_path, qa, sup, unsup, [done], "a"), patch_encoder=no_patch_model, embed_factory=fake_embed)
    with pytest.raises(ValueError, match="labelled twice"):
        sft.load_human_boxes([done, done])
    unknown = label_run(tmp_path, "polymer", sup, polymer, BOXES, {1: "unlocatable"})
    args = human_args(tmp_path, qa, sup, unsup, [done, unknown], "b")
    assert sft.main(args, patch_encoder=no_patch_model, embed_factory=fake_embed) == 0
    summary = json.loads((tmp_path / "b/build_summary.json").read_text(encoding="utf-8"))
    assert summary["images"] == 21 and summary["unlocated_photos_left_out"] == 1
    assert summary["human_box_runs"][1]["status"] == {"boxed": 1, "unlocatable": 1}
    left_out = polymer[1]["relative_path"]
    assert left_out not in {r["image"] for r in pilot.agent.read_jsonl(tmp_path / "b/sft_train_qa.jsonl")}
    session = ann.prepare(ann.read(unknown / "manifest.json"), sup, unknown)        # the annotator changes one box
    session.save(0, [[.2, .2, .3, .3]], "boxed", "", session.item(0)["revision"])
    with pytest.raises(ValueError, match="other settings"):
        sft.main(args, patch_encoder=no_patch_model, embed_factory=fake_embed)


def test_polymer_corrosion_is_added_only_on_request_and_moves_no_other_pick(tmp_path):
    sup, unsup, _, dev_images = fixture(tmp_path)
    unsup_rows = sft.parse_unsupervised_manifest(sft.devset.zip_manifest(unsup, tmp_path / "m/unsup.csv"))
    sup_rows, _ = sft.parse_supervised_manifest(sft.devset.zip_manifest(sup, tmp_path / "m/sup.csv"))
    base, _, _ = sft.select_train_images(sup_rows, unsup_rows, dev_images, 2, 42)
    both, stats, _ = sft.select_train_images(sup_rows, unsup_rows, dev_images, 2, 42, [sft.POLYMER_CORROSION])
    base_paths = {r["relative_path"] for r in base}
    extra = [r for r in both if r["relative_path"] not in base_paths]
    assert {r["relative_path"] for r in both} - {r["relative_path"] for r in extra} == base_paths
    assert not any(r["defect_class"] == "corrosao" for r in base)
    assert len(extra) == 2 and all((r["asset_category"], r["defect_class"]) == sft.POLYMER_CORROSION for r in extra)
    assert not {r["relative_path"] for r in extra} & dev_images and "class_not_used" not in json.dumps(stats["excluded"])
    with pytest.raises(ValueError, match="already"):
        sft.select_train_images(sup_rows, unsup_rows, dev_images, 2, 42, [("vari-grip", "rust")])
