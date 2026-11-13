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
