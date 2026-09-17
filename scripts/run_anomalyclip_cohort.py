#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.evidence_ablation import write_csv_atomic  # noqa: E402
from insplad_benchmark.vision_expert import (  # noqa: E402
    save_roi_artifacts,
    validate_roi_manifest,
)

ANOMALYCLIP_COMMIT = "3911738c0867544f545a076ad78f3f11d9ecbfdf"


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _artifact_exists(root: Path, row: Mapping[str, str]) -> bool:
    for field in ("roi_path", "anomaly_map_path", "overlay_path"):
        value = str(row.get(field, ""))
        if not value:
            return False
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            return False
    return True


def ensure_checkout(checkout: Path) -> dict:
    valid = False
    if checkout.is_dir() and (checkout / ".git").is_dir():
        try:
            current = subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
            ).strip()
            valid = current == ANOMALYCLIP_COMMIT
        except Exception:
            valid = False

    if not valid:
        if checkout.exists():
            shutil.rmtree(checkout)
        subprocess.run(
            [
                "git", "clone", "--filter=blob:none", "--no-checkout",
                "https://github.com/zqhang/AnomalyCLIP.git", str(checkout),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "sparse-checkout", "init", "--no-cone"],
            check=True,
        )
        sparse = checkout / ".git/info/sparse-checkout"
        sparse.write_text(
            "/*.py\n/AnomalyCLIP_lib/\n/checkpoints/9_12_4_multiscale/epoch_15.pth\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "git", "-C", str(checkout), "fetch", "--depth", "1",
                "origin", ANOMALYCLIP_COMMIT,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--detach", ANOMALYCLIP_COMMIT],
            check=True,
        )

    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != ANOMALYCLIP_COMMIT:
        raise RuntimeError(f"AnomalyCLIP commit mismatch: {actual}")

    from insplad_benchmark.anomalyclip_compat import patch_official_anomalyclip_checkout
    return patch_official_anomalyclip_checkout(checkout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the pinned official AnomalyCLIP checkpoint on a prepared full InsPLAD cohort."
    )
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, default=Path("/content/AnomalyCLIP"))
    parser.add_argument("--model-cache", type=Path, default=Path("/content/anomalyclip_model_cache"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from scipy.ndimage import gaussian_filter

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    manifest_path = args.workspace_root / "workspace_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    workspace = pd.DataFrame(read_csv(manifest_path))
    query_workspace = workspace.loc[workspace.role == "query"].copy()
    if query_workspace.empty:
        raise RuntimeError("workspace has no query rows")
    query_ids = [str(value) for value in query_workspace.query_id]
    query_by_id = {str(row.query_id): row for row in query_workspace.itertuples(index=False)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roi_manifest = args.out_dir / "roi_manifest.csv"
    existing_rows = [] if args.force or not roi_manifest.is_file() else read_csv(roi_manifest)
    existing = {
        (str(row.get("query_id", "")), str(row.get("expert", ""))): row
        for row in existing_rows
        if row.get("expert") == "anomalyclip"
    }
    current_rows = {
        qid: dict(existing[(qid, "anomalyclip")])
        for qid in query_ids
        if (qid, "anomalyclip") in existing
        and _artifact_exists(args.out_dir, existing[(qid, "anomalyclip")])
    }
    if len(current_rows) == len(query_ids) and not args.force:
        validate_roi_manifest(
            list(current_rows.values()),
            query_ids=query_ids,
            experts=("anomalyclip",),
            manifest_path=roi_manifest,
            artifact_root=args.out_dir,
            require_files=True,
        )
        print("[anomalyclip] all queries already complete; reuse")
        return 0

    compat = ensure_checkout(args.checkout)
    for module_name in list(sys.modules):
        if (
            module_name == "AnomalyCLIP_lib"
            or module_name.startswith("AnomalyCLIP_lib.")
            or module_name in {"prompt_ensemble", "utils"}
        ):
            sys.modules.pop(module_name, None)
    if str(args.checkout) not in sys.path:
        sys.path.insert(0, str(args.checkout))

    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    cfg = SimpleNamespace(
        image_size=518,
        features_list=[6, 12, 18, 24],
        feature_map_layer=[0, 1, 2, 3],
        depth=9,
        n_ctx=12,
        t_n_ctx=4,
        sigma=4,
    )
    design = {
        "Prompt_length": 12,
        "learnabel_text_embedding_depth": 9,
        "learnabel_text_embedding_length": 4,
    }
    args.model_cache.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px",
        device=device,
        design_details=design,
        download_root=str(args.model_cache),
    )
    model.eval()
    preprocess, _ = get_transform(cfg)
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    checkpoint_path = args.checkout / "checkpoints/9_12_4_multiscale/epoch_15.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized, compound = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized, compound).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    kept = [row for row in existing_rows if row.get("expert") != "anomalyclip"]
    for position, query_id in enumerate(query_ids, start=1):
        if query_id in current_rows:
            continue
        query = query_by_id[query_id]
        image_path = (args.workspace_root / str(query.local_path)).resolve()
        image = preprocess(Image.open(image_path).convert("RGB")).reshape(
            1, 3, cfg.image_size, cfg.image_size
        ).to(device)

        with torch.inference_mode():
            image_features, patch_features = model.encode_image(
                image, cfg.features_list, DPAM_layer=20
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_logits = 100.0 * image_features @ text_features[0].transpose(-1, -2)
            image_probability = image_logits.softmax(dim=-1)
            image_anomaly_score = float(image_probability[0, 1].detach().cpu())

            anomaly_maps = []
            for feature_index, patch_feature in enumerate(patch_features):
                if feature_index < cfg.feature_map_layer[0]:
                    continue
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(
                    patch_feature, text_features[0]
                )
                similarity_map = AnomalyCLIP_lib.get_similarity_map(
                    similarity[:, 1:, :], cfg.image_size
                )
                anomaly_maps.append(
                    (similarity_map[..., 1] + 1.0 - similarity_map[..., 0]) / 2.0
                )
            anomaly_map = torch.stack(anomaly_maps).sum(dim=0)[0].detach().cpu().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=cfg.sigma)

        row = save_roi_artifacts(
            image_path=image_path,
            anomaly_map=anomaly_map,
            out_dir=(
                args.out_dir / "vision_experts" / "anomalyclip" / str(query.asset_category)
            ),
            query_id=query_id,
            expert="anomalyclip",
            asset_category=str(query.asset_category),
            query_image=str(query.source_path),
            metadata={
                "expert_protocol": "official pretrained zero-shot AnomalyCLIP",
                "model_id": "zqhang/AnomalyCLIP ViT-L/14@336px",
                "model_revision": ANOMALYCLIP_COMMIT,
                "compat_patch_schema_version": compat["schema_version"],
                "compat_source_sha256": compat["sha256_after"],
                "checkpoint": "checkpoints/9_12_4_multiscale/epoch_15.pth",
                "checkpoint_training_source": "VisA",
                "model_pred_score": image_anomaly_score,
            },
            artifact_root=args.out_dir,
        )
        current_rows[query_id] = row
        ordered = [current_rows[qid] for qid in query_ids if qid in current_rows]
        write_csv_atomic(roi_manifest, [*kept, *ordered])
        print(
            f"[AnomalyCLIP] {position}/{len(query_ids)} "
            f"remaining={len(query_ids) - len(current_rows)}"
        )

    ordered = [current_rows[qid] for qid in query_ids]
    write_csv_atomic(roi_manifest, [*kept, *ordered])
    validate_roi_manifest(
        ordered,
        query_ids=query_ids,
        experts=("anomalyclip",),
        manifest_path=roi_manifest,
        artifact_root=args.out_dir,
        require_files=True,
    )

    summary = {
        "query_count": len(query_ids),
        "commit": ANOMALYCLIP_COMMIT,
        "checkpoint": "checkpoints/9_12_4_multiscale/epoch_15.pth",
        "compat_sha256": compat["sha256_after"],
    }
    (args.out_dir / "anomalyclip_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del model, prompt_learner, text_features, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
