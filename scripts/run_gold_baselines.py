from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_REGISTRY = {
    "qwen3_5_4b": {
        "backend": "hf", "provider": "huggingface",
        "model_id": "Qwen/Qwen3.5-4B",
        "trust_remote_code": False, "adapter": "pipeline",
        "role": "qwen_family",
    },
    "internvl3_5_8b": {
        "backend": "hf", "provider": "huggingface",
        "model_id": "OpenGVLab/InternVL3_5-8B",
        "trust_remote_code": True, "adapter": "internvl_chat",
        "max_total_tiles": 6, "role": "internvl_family",
    },
    "minicpm_v_4_5": {
        "backend": "hf", "provider": "huggingface",
        "model_id": "openbmb/MiniCPM-V-4_5",
        "trust_remote_code": True, "adapter": "minicpm_chat",
        "role": "minicpm_family",
    },
    "llava_onevision_7b": {
        "backend": "hf", "provider": "huggingface",
        "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "trust_remote_code": False, "adapter": "llava_processor",
        "role": "llava_family",
    },
    "luxia": {
        "backend": "luxia", "provider": "luxia",
        "model_id": "document-ai + luxia3-llm-32b-0731",
        "role": "luxia_api_pipeline",
    },
}

DEFECT_TEXT_TO_LABEL = {
    "Normal / no defect.": "normal",
    "Rust / corrosion.": "rust",
    "Missing cap / missing insulator component.": "missing-cap",
    "Bird nest / foreign nesting material.": "bird-nest",
}
QUESTION_ORDER = {
    "anomaly_detection": 1,
    "defect_classification": 2,
    "defect_localization": 3,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def qa_identity(rows: list[dict]) -> str:
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_gold_qa(qa_dir: Path) -> list[dict]:
    master_candidates = sorted(qa_dir.glob("master_qa_*.jsonl"))
    if not master_candidates:
        raise FileNotFoundError(f"master_qa_*.jsonl not found under {qa_dir}")
    master = read_jsonl(master_candidates[-1])

    if len(master) != 500:
        raise RuntimeError(f"expected 500 Q1/Q2 rows, got {len(master)}")
    types = Counter(str(r.get("question_type")) for r in master)
    if types != {"anomaly_detection": 250, "defect_classification": 250}:
        raise RuntimeError(f"unexpected master question counts: {dict(types)}")

    q3: list[dict] = []
    loc_candidates = sorted(qa_dir.glob("localization_qa_*.jsonl"))
    if loc_candidates:
        all_loc = read_jsonl(loc_candidates[-1])
        q3 = [r for r in all_loc if r.get("question_type") == "defect_localization"]

    rows = [dict(r) for r in master] + [dict(r) for r in q3]
    rows.sort(
        key=lambda r: (
            str(r.get("core_id", "")),
            QUESTION_ORDER.get(str(r.get("question_type", "")), 99),
            str(r.get("id", "")),
        )
    )
    ids = [str(r.get("id", "")) for r in rows]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("QA IDs must be present and unique")
    return rows


def prompt_for(row: dict) -> str:
    options = row.get("options")
    if not isinstance(options, dict) or not options:
        raise ValueError(f"QA row {row.get('id')} has no options")
    option_lines = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        "You are an industrial visual inspector. Answer the multiple-choice question using only "
        "the target image. Do not use hidden metadata or external knowledge.\n\n"
        f"Question: {row['question']}\n"
        f"{option_lines}\n\n"
        "Return exactly one option letter and nothing else."
    )


def parse_choice(output: str, valid_letters: set[str]) -> str | None:
    text = str(output or "").strip()
    if not text:
        return None
    upper = text.upper()
    if upper in valid_letters:
        return upper

    patterns = [
        r'"(?:answer|choice)"\s*:\s*"([A-Z])"',
        r"\b(?:ANSWER|CHOICE|OPTION)\s*[:=\-]?\s*([A-Z])\b",
        r"^\s*\(?([A-Z])\)?[\.\):\-]?\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match and match.group(1) in valid_letters:
            return match.group(1)

    candidates = re.findall(r"(?<![A-Z0-9])([A-Z])(?![A-Z0-9])", upper)
    candidates = [x for x in candidates if x in valid_letters]
    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None


def safe_extract(zip_path: Path, member: str, out_root: Path) -> Path:
    target = out_root / member
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        try:
            payload = archive.read(member)
        except KeyError:
            normalized = member.replace("\\", "/").lstrip("/")
            matches = [
                name
                for name in archive.namelist()
                if name.replace("\\", "/").lstrip("/") == normalized
            ]
            if not matches:
                raise
            payload = archive.read(matches[0])
    target.write_bytes(payload)
    return target


def prepare_images(rows: list[dict], sup_zip: Path, image_root: Path) -> dict[str, Path]:
    cache: dict[str, Path] = {}
    for row in rows:
        member = str(row["image"])
        if member not in cache:
            cache[member] = safe_extract(sup_zip, member, image_root)
    return cache


def reusable_prediction(row: dict | None) -> bool:
    return bool(row and not row.get("error") and str(row.get("output", "")).strip())


class LazyHFAdapter:
    """Load an HF adapter on first inference so fully reused runs skip model loading."""

    def __init__(self, spec: dict, loader=None) -> None:
        self.spec = spec
        self._loader = loader
        self._adapter = None

    @property
    def loaded(self) -> bool:
        return self._adapter is not None

    def get(self):
        if self._adapter is None:
            loader = self._loader
            if loader is None:
                from insplad_benchmark.hf_adapters import load_hf_adapter

                loader = load_hf_adapter
            print(f"[{self.spec.get('model_id', '')}] loading adapter", flush=True)
            self._adapter = loader(self.spec)
        return self._adapter

    def release(self) -> None:
        was_loaded = self._adapter is not None
        self._adapter = None
        if not was_loaded:
            return
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def infer_luxia_mcqa(image_path: Path, prompt: str, api_key: str) -> tuple[str, float]:
    from insplad_benchmark.luxia_adapter import (
        LUXIA_CHAT_MODEL,
        call_luxia_chat,
        call_luxia_document_ai,
    )

    started = time.time()
    vision = call_luxia_document_ai(image_path, api_key)
    chat_prompt = (
        "You are evaluating one industrial image from a visual description generated by Luxia "
        "Vision. Do not invent details absent from the description.\n\n"
        f"Luxia Vision description:\n{vision['content']}\n\n"
        f"Benchmark instruction:\n{prompt}"
    )
    chat = call_luxia_chat(
        chat_prompt,
        api_key,
        model=LUXIA_CHAT_MODEL,
        max_completion_tokens=32,
    )
    return str(chat["content"]), round(time.time() - started, 3)


def run_model(
    model_key: str,
    rows: list[dict],
    image_paths: dict[str, Path],
    out_dir: Path,
    checkpoint_every: int,
) -> dict:
    spec = MODEL_REGISTRY[model_key]
    pred_path = out_dir / "predictions" / f"{model_key}.jsonl"
    existing_rows = read_jsonl(pred_path) if pred_path.is_file() else []
    existing = {str(r["id"]): r for r in existing_rows if r.get("id")}
    predictions: list[dict] = []

    holder = LazyHFAdapter(spec) if spec["backend"] == "hf" else None

    api_key = os.environ.get("LUXIA_API_KEY", "")
    if spec["backend"] == "luxia" and not api_key:
        raise RuntimeError("LUXIA_API_KEY is required for Luxia")

    called = 0
    reused = 0
    errors = 0
    total_latency = 0.0

    try:
        for index, row in enumerate(rows, 1):
            old = existing.get(str(row["id"]))
            if reusable_prediction(old):
                predictions.append(old)
                reused += 1
                continue

            prompt = prompt_for(row)
            image_path = image_paths[str(row["image"])]
            # Load outside the per-row try so a load failure fails the model, not every row.
            adapter = holder.get() if holder is not None else None
            started = time.time()
            error = None
            output = ""
            try:
                if spec["backend"] == "hf":
                    from insplad_benchmark.hf_adapters import infer_hf

                    output = infer_hf(adapter, [image_path], prompt)
                    latency = round(time.time() - started, 3)
                else:
                    output, latency = infer_luxia_mcqa(image_path, prompt, api_key)
            except Exception as exc:
                latency = round(time.time() - started, 3)
                error = f"{type(exc).__name__}: {exc}"
                errors += 1

            valid_letters = {str(k).upper() for k in row["options"]}
            choice = parse_choice(output, valid_letters)
            prediction = {
                "id": row["id"],
                "core_id": row["core_id"],
                "asset_category": row["asset_category"],
                "question_type": row["question_type"],
                "output": output,
                "predicted_choice": choice,
                "latency_s": latency,
                "error": error,
            }
            predictions.append(prediction)
            called += 1
            total_latency += latency

            if checkpoint_every > 0 and called % checkpoint_every == 0:
                write_jsonl(pred_path, predictions)
                print(f"[{model_key}] {index}/{len(rows)} checkpoint")

        write_jsonl(pred_path, predictions)
    finally:
        adapter = None
        if holder is not None:
            holder.release()

    return {
        "model_key": model_key,
        "called_this_run": called,
        "reused_predictions": reused,
        "row_errors": errors,
        "called_latency_s": round(total_latency, 3),
    }


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_f1(gold: list[str], pred: list[str | None], positive: str) -> tuple[float, float, float]:
    tp = sum(g == positive and p == positive for g, p in zip(gold, pred))
    fp = sum(g != positive and p == positive for g, p in zip(gold, pred))
    fn = sum(g == positive and p != positive for g, p in zip(gold, pred))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def macro_f1(gold: list[str], pred: list[str | None]) -> float:
    labels = sorted(set(gold))
    values = []
    for label in labels:
        _, _, f1 = binary_f1(gold, pred, label)
        values.append(f1)
    return sum(values) / len(values) if values else 0.0


def predicted_semantic(row: dict, choice: str | None) -> str | None:
    if not choice:
        return None
    text = str((row.get("options") or {}).get(choice, ""))
    qtype = row.get("question_type")
    if qtype == "anomaly_detection":
        if text == "Normal.":
            return "normal"
        if text == "Anomalous.":
            return "anomalous"
    if qtype == "defect_classification":
        return DEFECT_TEXT_TO_LABEL.get(text)
    if qtype == "defect_localization":
        return choice
    return None


def evaluate_model(model_key: str, rows: list[dict], pred_path: Path) -> tuple[dict, list[dict]]:
    predictions = read_jsonl(pred_path)
    by_id = {str(r["id"]): r for r in predictions}
    if len(by_id) != len(rows):
        raise RuntimeError(
            f"{model_key}: predictions={len(by_id)} do not match QA rows={len(rows)}"
        )

    scored = []
    for row in rows:
        pred = by_id[str(row["id"])]
        choice = pred.get("predicted_choice")
        scored.append(
            {
                "row": row,
                "choice": choice,
                "valid": choice is not None,
                "correct": choice == row["answer"],
                "semantic": predicted_semantic(row, choice),
            }
        )

    total = len(scored)
    valid_n = sum(x["valid"] for x in scored)
    correct_n = sum(x["correct"] for x in scored)

    q1 = [x for x in scored if x["row"]["question_type"] == "anomaly_detection"]
    q2 = [x for x in scored if x["row"]["question_type"] == "defect_classification"]
    q3 = [x for x in scored if x["row"]["question_type"] == "defect_localization"]

    anomaly_gold = [x["row"]["anomaly_gt"] for x in q1]
    anomaly_pred = [x["semantic"] for x in q1]
    ap, ar, af1 = binary_f1(anomaly_gold, anomaly_pred, "anomalous")

    defect_gold = [x["row"]["defect_gt"] for x in q2]
    defect_pred = [x["semantic"] for x in q2]

    row = {
        "model_key": model_key,
        "model_id": MODEL_REGISTRY[model_key]["model_id"],
        "backend": MODEL_REGISTRY[model_key]["backend"],
        "questions": total,
        "valid_output_rate": round(_safe_div(valid_n, total), 6),
        "overall_accuracy": round(_safe_div(correct_n, total), 6),
        "anomaly_accuracy": round(_safe_div(sum(x["correct"] for x in q1), len(q1)), 6),
        "anomaly_precision": round(ap, 6),
        "anomaly_recall": round(ar, 6),
        "anomaly_f1": round(af1, 6),
        "defect_accuracy": round(_safe_div(sum(x["correct"] for x in q2), len(q2)), 6),
        "defect_macro_f1": round(macro_f1(defect_gold, defect_pred), 6),
        "localization_questions": len(q3),
        "localization_accuracy": round(
            _safe_div(sum(x["correct"] for x in q3), len(q3)), 6
        ),
    }

    by_asset_rows: list[dict] = []
    for asset in sorted({str(r["asset_category"]) for r in rows}):
        aset = [x for x in scored if x["row"]["asset_category"] == asset]
        by_asset_rows.append(
            {
                "model_key": model_key,
                "asset_category": asset,
                "questions": len(aset),
                "accuracy": round(_safe_div(sum(x["correct"] for x in aset), len(aset)), 6),
                "q1_accuracy": round(
                    _safe_div(
                        sum(
                            x["correct"]
                            for x in aset
                            if x["row"]["question_type"] == "anomaly_detection"
                        ),
                        sum(
                            x["row"]["question_type"] == "anomaly_detection"
                            for x in aset
                        ),
                    ),
                    6,
                ),
                "q2_accuracy": round(
                    _safe_div(
                        sum(
                            x["correct"]
                            for x in aset
                            if x["row"]["question_type"] == "defect_classification"
                        ),
                        sum(
                            x["row"]["question_type"] == "defect_classification"
                            for x in aset
                        ),
                    ),
                    6,
                ),
                "q3_accuracy": round(
                    _safe_div(
                        sum(
                            x["correct"]
                            for x in aset
                            if x["row"]["question_type"] == "defect_localization"
                        ),
                        sum(
                            x["row"]["question_type"] == "defect_localization"
                            for x in aset
                        ),
                    ),
                    6,
                ),
            }
        )
    return row, by_asset_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--sup-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY),
        choices=list(MODEL_REGISTRY),
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()

    rows = load_gold_qa(args.qa_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = prepare_images(rows, args.sup_zip, args.out_dir / "_images")

    snapshot = args.out_dir / "run_qa.jsonl"
    if snapshot.is_file():
        existing = read_jsonl(snapshot)
        if qa_identity(existing) != qa_identity(rows):
            raise RuntimeError("out-dir contains a different frozen QA snapshot")
    else:
        write_jsonl(snapshot, rows)

    run_status: list[dict] = []
    leaderboard: list[dict] = []
    by_asset_all: list[dict] = []

    for model_key in args.models:
        print(f"\n=== {model_key} ===")
        try:
            status = run_model(
                model_key,
                rows,
                image_paths,
                args.out_dir,
                args.checkpoint_every,
            )
            metric, by_asset = evaluate_model(
                model_key,
                rows,
                args.out_dir / "predictions" / f"{model_key}.jsonl",
            )
            status["status"] = "completed"
            leaderboard.append(metric)
            by_asset_all.extend(by_asset)
        except Exception as exc:
            status = {
                "model_key": model_key,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(status["error"])
        run_status.append(status)
        write_csv(args.out_dir / "leaderboard.csv", leaderboard)
        write_csv(args.out_dir / "metrics_by_asset.csv", by_asset_all)
        (args.out_dir / "run_status.json").write_text(
            json.dumps(run_status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    metadata = {
        "qa_dir": str(args.qa_dir),
        "qa_count": len(rows),
        "qa_identity_sha256": qa_identity(rows),
        "master_q1_q2_count": 500,
        "q3_count": sum(r["question_type"] == "defect_localization" for r in rows),
        "models_requested": args.models,
        "model_registry": MODEL_REGISTRY,
        "condition": "image_only_final_gold_v1",
    }
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
