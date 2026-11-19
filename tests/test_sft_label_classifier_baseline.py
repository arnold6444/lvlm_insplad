"""The small supervised baseline on the fine-tuning labels: fixture-only, fake features, no GPU, no real gold file."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import run_sft_label_classifier_baseline as baseline  # noqa: E402
import test_agent_sft_gold_eval as gold_tests  # noqa: E402
import test_gold_v2_verify_agent as fixtures  # noqa: E402

gold, pilot = baseline.gold, baseline.pilot
N = gold_tests.N
NORMAL, RUST = fixtures.DEFECT["A"], fixtures.DEFECT["B"]


def photo(kind, date, frame, tag):
    return f"defect_supervised/vari-grip/{kind}/Fotos {date}_DJI_{frame:04d}_{tag}.png"


# Twelve training photos. One is another crop of gold photo 1's drone frame, one has the bytes of gold photo 2.
TRAIN = {photo("train/rust", gold_tests.GOLD_DATE, 1, "other_crop"): 200,
         photo("train/good", "05-12-2020", 500, "t"): 32}
TRAIN.update({photo("train/rust" if k % 2 else "train/good", "07-12-2020", 700 + k, f"t{k}"): 201 + k for k in range(10)})
DEV = {photo("train/rust" if k % 2 else "train/good", "08-12-2020", 800 + k, f"d{k}"): 220 + k for k in range(4)}


def qa_pair(core, image, split=None, protocol=None):
    bad = "/rust/" in image
    common = dict(core_id=core, asset_category="vari-grip", image=image, anomaly_gt="anomalous" if bad else "normal",
                  defect_gt="rust" if bad else "normal")
    common.update({k: v for k, v in (("split", split), ("protocol", protocol)) if v})
    return [dict(common, id=core + "-q1", question_type=gold.Q1, question=fixtures.QTEXT[gold.Q1],
                 options={"A": "Normal.", "B": "Anomalous."}, answer="B" if bad else "A"),
            dict(common, id=core + "-q2", question_type=gold.Q2, question=fixtures.QTEXT[gold.Q2],
                 options=dict(fixtures.DEFECT), answer="B" if bad else "A")]


def inputs(tmp, train=TRAIN):
    args, _ = gold_tests.inputs(tmp, train={**train, **DEV})      # gold QA, sup.zip with every photo, refs, location set
    train_qa = tmp / "sft_train_qa.jsonl"
    pilot.base.write_jsonl(train_qa, [r for i, name in enumerate(train) for r in qa_pair(f"sft-{i}", name, "supervised_train_sft")])
    dev_qa = tmp / "dev.jsonl"
    pilot.base.write_jsonl(dev_qa, [r for i, name in enumerate(DEV) for r in qa_pair(f"dev-{i}", name, protocol="gold_v2_dev")])
    out = tmp / "sft_label_classifier_baseline_test"
    return ["--train-qa", str(train_qa), "--qa-dir", args[args.index("--qa-dir") + 1],
            "--sup-zip", args[args.index("--sup-zip") + 1], "--out-dir", str(out)], out, train_qa, dev_qa


def fake_embed(flip=()):
    """Rust and normal photos are linearly separable; a photo named in flip gets the other class's feature."""
    def factory():
        def embed(rows, which):
            out = {}
            for r in rows:
                path = str(r["relative_path"])
                noise = np.frombuffer(hashlib.sha256(path.encode()).digest()[:4], dtype=np.uint8) / 2550
                bad = ("/rust/" in path) != (path in flip)
                out[path] = np.array([1.0, 0.0] if bad else [0.0, 1.0], dtype=np.float32) + noise[:2].astype(np.float32)
            return out
        return embed
    return factory


@pytest.fixture(autouse=True)
def small_gold(monkeypatch):
    monkeypatch.setattr(gold, "GOLD_IMAGES", N)
    monkeypatch.setattr(gold, "LOCATED_IMAGES", len(gold_tests.BOXES))
    monkeypatch.setattr(pilot, "select_rows", lambda rows, n, seed: rows[:2 * n])     # the stratified pick needs dev240


def table(out, name):
    with (out / name).open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def test_trains_on_the_sft_labels_and_scores_gold_and_dev_like_the_agent_runs(tmp_path):
    args, out, train_qa, dev_qa = inputs(tmp_path)
    wrong = gold_tests.image(3)                                 # a rust gold photo whose feature looks normal
    assert baseline.main(args + ["--dev-qa", str(dev_qa), "--n-images", "4"], embed_factory=fake_embed({wrong})) == 0
    board = {(r["eval_set"], r["subset"]): r for r in table(out, "leaderboard.csv")}
    assert set(board) == {("gold", "all"), ("gold", "frame_disjoint"), ("dev4", "all")}
    full, clean, dev = board["gold", "all"], board["gold", "frame_disjoint"], board["dev4", "all"]
    assert (full["images"], clean["images"], dev["images"]) == ("8", "6", "4")    # photo 1 shares a frame, photo 2 bytes
    assert (full["q1_tp"], full["q1_fn"], full["q1_tn"], full["q1_fp"]) == ("3", "1", "4", "0")
    assert float(full["joint_accuracy"]) == pytest.approx(7 / 8) and float(clean["joint_accuracy"]) == pytest.approx(5 / 6)
    assert float(dev["joint_accuracy"]) == 1 and float(full["consistency_error"]) == 0 and full["policy"] == "dinov2_linear"
    assert {full["q1_invalid_rate"], full["q2_invalid_rate"]} == {"0.0"}
    per_image = {(r["eval_set"], r["core_id"]): r for r in table(out, "per_image.csv")}
    assert len(per_image) == 12 and per_image["gold", "core-3"]["predicted"] == NORMAL
    assert per_image["gold", "core-3"]["q2_correct"] == "False" and per_image["gold", "core-5"]["predicted"] == RUST
    assert per_image["gold", "core-1"]["scored_in_primary"] == "False" and per_image["dev4", "dev-0"]["scored_in_primary"] == ""
    assert float(per_image["gold", "core-5"]["p_" + RUST]) > 0.5
    defects = {(r["eval_set"], r["subset"], r["defect_gt"]): r for r in table(out, "q2_by_defect.csv")}
    assert (defects["gold", "all", "rust"]["q2_correct"], defects["gold", "all", "rust"]["images"]) == ("3", "4")
    assert {r["asset_category"] for r in table(out, "metrics_by_asset.csv")} == {"vari-grip", "yoke-suspension"}
    saved = np.load(out / "features.npz")
    index = table(out, "feature_index.csv")
    assert saved["features"].shape == (12 + 8 + 4, 2) and list(saved["images"]) == [r["image"] for r in index]
    assert sorted({r["used_for"] for r in index}) == ["dev4", "gold", "train"]
    assert sum(r["train_class"] == RUST for r in index) == 6 and sum(r["train_class"] == NORMAL for r in index) == 6
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["protocol"] == baseline.PROTOCOL and meta["training_photos"] == 12 and meta["training_accuracy"] == 1
    assert meta["training_classes"] == {NORMAL: 6, RUST: 6} and meta["execution"] == "injected_test_backend"
    assert meta["cohorts"] == {"gold/all": 8, "gold/frame_disjoint": 6, "dev4/all": 4}
    assert meta["gold_photos_sharing_a_frame_or_bytes"] == 2
    assert json.loads((out / "run_status.json").read_text(encoding="utf-8"))["complete"] is True
    # The gold runner, given the same training list, scores the same frame-disjoint photos.
    gold_args, gold_out = gold_tests.inputs(tmp_path / "agent", train={**TRAIN, **DEV})
    gold_args[gold_args.index("--train-qa") + 1] = str(train_qa)
    assert gold_tests.run(gold_args) == 0
    flags = {r["image"]: r["scored_in_primary"] for r in gold_tests.table(gold_out, "frame_overlap.csv")}
    assert {r["image"]: r["scored_in_primary"] for r in per_image.values() if r["eval_set"] == "gold"} == flags
    # The same inputs give the same result again; other labels need another folder.
    first = (out / "leaderboard.csv").read_bytes()
    assert baseline.main(args + ["--dev-qa", str(dev_qa), "--n-images", "4"], embed_factory=fake_embed({wrong})) == 0
    assert (out / "leaderboard.csv").read_bytes() == first
    with pytest.raises(ValueError, match="use a new output folder"):
        baseline.main(args, embed_factory=fake_embed())           # without dev: other questions


def test_refuses_leaked_photos_bad_training_files_and_other_folders(tmp_path):
    args, out, train_qa, dev_qa = inputs(tmp_path / "ok")
    with pytest.raises(ValueError, match="sft_label_classifier_baseline"):
        baseline.main(args[:-1] + [str(tmp_path / "somewhere_else")], embed_factory=fake_embed())
    rows = pilot.agent.read_jsonl(train_qa)
    pilot.base.write_jsonl(train_qa, [dict(r, split="supervised_val") for r in rows])
    with pytest.raises(ValueError, match="not an SFT training QA"):
        baseline.main(args, embed_factory=fake_embed())
    pilot.base.write_jsonl(train_qa, rows[:-1])                 # a photo without its Q2
    with pytest.raises(ValueError, match="one Q1 and one Q2"):
        baseline.main(args, embed_factory=fake_embed())
    pilot.base.write_jsonl(train_qa, [r for r in rows if r["defect_gt"] == "normal"])
    with pytest.raises(ValueError, match="single class"):
        baseline.main(args, embed_factory=fake_embed())
    leaked = rows + qa_pair("sft-leak", gold_tests.image(0), "supervised_train_sft")
    pilot.base.write_jsonl(train_qa, leaked)
    with pytest.raises(ValueError, match="1 evaluation photos are training photos"):
        baseline.main(args, embed_factory=fake_embed())
    assert not out.exists()
