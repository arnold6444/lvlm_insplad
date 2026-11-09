#!/usr/bin/env python3
"""LoRA SFT of Qwen3.5-4B (BF16, no quantization) on tool-use trajectories from build_agent_sft_data.py.

Each sample is one agent step: the exact eval-runner prompt and images for that step, and the JSON action the
policy should emit. The loss covers only the action tokens plus the end-of-turn token; prompt and image tokens are
masked. Input ids come from the same ``processor.apply_chat_template`` call the eval adapter uses, so the trained
prompt is byte-identical to the evaluated one. LoRA targets every Linear layer of the language model; the vision
tower is frozen. Checkpoints go to ``<out-dir>/checkpoint`` so a Colab disconnect resumes instead of restarting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_agent_tool_pilot as pilot  # noqa: E402
from insplad_benchmark.hf_adapters import _make_chat  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
END_TOKEN = "<|im_end|>"


def read_samples(data_dir: Path):
    rows = pilot.agent.read_jsonl(data_dir / "sft_samples.jsonl")
    for r in rows:
        r["images"] = [data_dir / p for p in r["images"]]
        pilot.parse_action(r["target"], {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1, "G": 1, "H": 1, "I": 1, "J": 1})
    return rows


def lora_targets(model):
    """Full names of language-model Linear layers (vision tower and lm_head excluded)."""
    import torch
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and "language_model" in n and "visual" not in n and not n.endswith("lm_head")]
    if not names:
        raise RuntimeError("no language-model Linear layers found for LoRA")
    return names


def encode(processor, sample, end_ids):
    """Prompt part = the eval adapter's apply_chat_template output; labels only on the target action."""
    import torch
    prompt = processor.apply_chat_template(_make_chat(sample["images"], sample["prompt"]), add_generation_prompt=True,
                                           enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt")
    target = processor.tokenizer(sample["target"], add_special_tokens=False, return_tensors="pt")["input_ids"]
    target = torch.cat([target, torch.tensor([end_ids])], dim=1)
    n_prompt = prompt["input_ids"].shape[1]
    batch = {}
    for key, value in prompt.items():
        if key == "input_ids":
            batch[key] = torch.cat([value, target], dim=1)
        elif hasattr(value, "dim") and value.dim() == 2 and value.shape == prompt["input_ids"].shape:
            fill = torch.ones_like(target) if key == "attention_mask" else torch.zeros_like(target)
            batch[key] = torch.cat([value, fill.to(value.dtype)], dim=1)
        else:
            batch[key] = value
    labels = torch.full_like(batch["input_ids"], -100)
    labels[:, n_prompt:] = target
    batch["labels"] = labels
    return batch, n_prompt


def to_device(batch, model):
    import torch
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(model.device, dtype=torch.bfloat16) if v.is_floating_point() else v.to(model.device)
        else:
            out[k] = v
    return out


def lr_factor(step, warmup, total):
    """Linear warmup, then linear decay to zero."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


_ORDERS = {}


def epoch_order(n, seed, epoch):
    """Deterministic per-epoch shuffle, so a resumed run continues the same sequence."""
    if (n, seed, epoch) not in _ORDERS:
        order = list(range(n))
        random.Random(f"{seed}:{epoch}").shuffle(order)
        _ORDERS[n, seed, epoch] = order
    return _ORDERS[n, seed, epoch]


def signature(args, samples):
    payload = dict(model=MODEL_ID, data=hashlib.sha256((args.data_dir / "sft_samples.jsonl").read_bytes()).hexdigest(),
                   epochs=args.epochs, lr=args.lr, grad_accum=args.grad_accum, r=args.lora_r, alpha=args.lora_alpha,
                   dropout=args.lora_dropout, seed=args.seed, max_steps=args.max_steps, n=len(samples))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=0, help="optimizer steps cap for smoke runs; 0 = full")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    out = args.out_dir
    status_path = out / "train_status.json"
    samples = read_samples(args.data_dir)
    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] == "val"]
    sig, settings = signature(args, samples)
    if status_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("signature") != sig:
            raise ValueError("training settings/data changed: use a new output folder")
        if old.get("complete"):
            print("LoRA adapter already trained:", out / "adapter")
            return 0
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    pilot.require_gpu(torch)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                     device_map="auto")
    revision = getattr(model.config, "_commit_hash", None)
    end_id = processor.tokenizer.convert_tokens_to_ids(END_TOKEN)
    if end_id is None or end_id == processor.tokenizer.unk_token_id:
        raise RuntimeError(f"{END_TOKEN} is not a token of {MODEL_ID}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = lora_targets(model)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Only the action tokens and the end-of-turn token are supervised.
    first, n_prompt = encode(processor, train[0], end_id)
    supervised = first["labels"][0][first["labels"][0] != -100].tolist()
    if supervised[-1] != end_id or processor.tokenizer.decode(supervised[:-1]) != train[0]["target"]:
        raise AssertionError("label mask does not cover exactly the target action")

    total_micro = len(train) * args.epochs
    total_steps = math.ceil(total_micro / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(0.03 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, warmup, total_steps))
    ckpt = out / "checkpoint"
    step, micro = 0, 0
    if (ckpt / "state.json").exists():
        state = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
        if state["signature"] != sig:
            raise ValueError("checkpoint belongs to other settings")
        set_peft_model_state_dict(model, load_file(str(ckpt / "adapter_model.safetensors")))
        optim = torch.load(ckpt / "optim.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(optim["optimizer"])
        scheduler.load_state_dict(optim["scheduler"])
        step, micro = state["step"], state["micro"]
        print(f"resumed at optimizer step {step}", flush=True)

    def write_status(**kw):
        status_path.write_text(json.dumps(dict(signature=sig, settings=settings, **kw), indent=2), encoding="utf-8")

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for s in val:
                batch, _ = encode(processor, s, end_id)
                losses.append(float(model(**to_device(batch, model)).loss))
        model.train()
        return sum(losses) / len(losses) if losses else None

    write_status(complete=False, status="training", step=step, total_steps=total_steps)
    log = (out / "train_log.jsonl").open("a", encoding="utf-8")
    model.train()
    start, running, seen = time.monotonic(), 0.0, 0
    val_before = evaluate() if step == 0 else None
    if val_before is not None:
        log.write(json.dumps(dict(step=0, val_loss=val_before)) + "\n")
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        epoch, pos = divmod(micro, len(train))
        sample = train[epoch_order(len(train), args.seed, epoch)[pos]]
        batch, _ = encode(processor, sample, end_id)
        loss = model(**to_device(batch, model)).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at sample {sample['sample_id']}")
        (loss / args.grad_accum).backward()
        running += float(loss)
        seen += 1
        micro += 1
        if micro % args.grad_accum == 0 or micro == total_micro:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                rate = (time.monotonic() - start) / max(1, seen)
                log.write(json.dumps(dict(step=step, train_loss=running / seen, lr=scheduler.get_last_lr()[0],
                                          sec_per_sample=round(rate, 3))) + "\n")
                log.flush()
                print(f"step {step}/{total_steps} loss {running / seen:.4f} ({rate:.2f}s/sample)", flush=True)
                running, seen = 0.0, 0
            if step % args.checkpoint_every == 0 and step < total_steps:
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt)
                torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), ckpt / "optim.pt")
                (ckpt / "state.json").write_text(json.dumps(dict(signature=sig, step=step, micro=micro)), encoding="utf-8")
                write_status(complete=False, status="training", step=step, total_steps=total_steps)
    val_after = evaluate()
    log.write(json.dumps(dict(step=step, val_loss=val_after)) + "\n")
    log.close()
    model.save_pretrained(out / "adapter")
    software = {}
    for package in ("torch", "transformers", "peft", "accelerate"):
        try:
            software[package] = version(package)
        except PackageNotFoundError:
            software[package] = None
    meta = dict(model_id=MODEL_ID, resolved_model_revision=revision, dtype="bfloat16", quantization=None,
                lora_targets=len(targets), trainable_parameters=trainable, train_samples=len(train), val_samples=len(val),
                optimizer_steps=step, val_loss_before=val_before, val_loss_after=val_after, software=software,
                gpu=pilot.agent.gpu_info(), benchmark_commit=pilot.source_commit(), settings=settings,
                loss_mask="action JSON + end-of-turn token only")
    (out / "train_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_status(complete=True, status="trained", step=step, total_steps=total_steps)
    print(json.dumps(dict(val_loss_before=val_before, val_loss_after=val_after, steps=step), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
