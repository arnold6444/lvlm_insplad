#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insplad_benchmark.reconpatch_lite import (
    ReConPatchLiteConfig,
    ReConPatchLiteProjector,
    WideResNetPatchExtractor,
    nearest_memory_map,
    random_memory,
    sample_rows,
)
from insplad_benchmark.vision_expert import read_manifest, save_roi_artifacts, write_manifest


def read_workspace(root: Path) -> list[dict[str, str]]:
    with (root / "workspace_manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Screening-only paper-faithful ReConPatch-lite expert.")
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--max-train-patches", type=int, default=6000)
    parser.add_argument("--memory-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not 0 < args.memory_ratio <= 1:
        parser.error("memory-ratio must be in (0,1]")

    workspace = read_workspace(args.workspace_root)
    queries = [row for row in workspace if row["role"] == "query"]
    trains = [row for row in workspace if row["role"] == "train_normal"]
    assets = sorted({row["asset_category"] for row in queries})
    manifest_path = args.out_dir / "roi_manifest_reconpatch_lite.csv"
    existing = read_manifest(manifest_path)
    existing_keys = {(row["query_id"], row["expert"]) for row in existing}
    all_rows = list(existing)

    extractor = WideResNetPatchExtractor(device="cuda", pretrained=True, image_size=256)
    for asset_index, asset in enumerate(assets):
        query_asset = [row for row in queries if row["asset_category"] == asset]
        train_asset = [row for row in trains if row["asset_category"] == asset]
        if all((row["query_id"], "reconpatch_lite") in existing_keys for row in query_asset):
            print(f"[RESUME] reconpatch_lite {asset} complete")
            continue
        if not train_asset or not query_asset:
            raise RuntimeError(f"empty ReConPatch-lite data for asset={asset}")

        train_paths = [args.workspace_root / row["local_path"] for row in train_asset]
        patch_chunks = []
        # Do not materialize every 32x32 patch from up to 200 images in RAM.
        # Keep a deterministic per-image sample whose total stays near max_train_patches.
        per_image_cap = max(1, (args.max_train_patches + len(train_paths) - 1) // len(train_paths))
        generator = torch.Generator(device="cpu").manual_seed(args.seed + asset_index)
        for start in range(0, len(train_paths), 4):
            patches, _, _ = extractor.extract_batch(train_paths[start:start+4])
            for image_patches in patches:
                if image_patches.shape[0] > per_image_cap:
                    indices = torch.randperm(image_patches.shape[0], generator=generator)[:per_image_cap]
                    image_patches = image_patches[indices]
                patch_chunks.append(image_patches)
        train_features = torch.cat(patch_chunks, dim=0)
        train_features = sample_rows(train_features, args.max_train_patches, args.seed + asset_index)
        config = ReConPatchLiteConfig(
            train_steps=args.train_steps,
            max_train_patches=args.max_train_patches,
            memory_ratio=args.memory_ratio,
            batch_size=args.batch_size,
            seed=args.seed + asset_index,
        )
        projector = ReConPatchLiteProjector(train_features.shape[1], config, device="cuda")
        losses = projector.fit(train_features)
        transformed_train = projector.transform(train_features)
        memory = random_memory(transformed_train, config.memory_ratio, config.seed)
        print(
            f"[reconpatch_lite] {asset}: train_images={len(train_asset)} patches={len(train_features)} "
            f"memory={len(memory)} loss={losses[-1]:.4f}"
        )

        asset_rows = []
        for query in query_asset:
            path = args.workspace_root / query["local_path"]
            patches, h, w = extractor.extract_batch([path])
            query_features = projector.transform(patches[0])
            anomaly_map = nearest_memory_map(query_features, memory, h, w)
            score = float(anomaly_map.max())
            artifact_dir = args.out_dir / "reconpatch_lite" / asset
            row = save_roi_artifacts(
                image_path=path,
                anomaly_map=anomaly_map,
                out_dir=artifact_dir,
                query_id=query["query_id"],
                expert="reconpatch_lite",
                asset_category=asset,
                query_image=query["source_path"],
                artifact_root=args.out_dir,
                metadata={
                    "expert_protocol": "screening-only paper-faithful ReConPatch-lite; not official release",
                    "model_id": "local/ReConPatch-lite-WRN50_2-layer2+3",
                    "model_revision": "paper-faithful-screening-v1",
                    "model_pred_score": score,
                    "train_steps": args.train_steps,
                    "max_train_patches": args.max_train_patches,
                    "memory_ratio": args.memory_ratio,
                    "implementation_warning": "If this expert wins, revalidate with a publication-grade implementation before paper claims.",
                },
            )
            asset_rows.append(row)

        all_rows = [
            row for row in all_rows
            if not (row.get("expert") == "reconpatch_lite" and row.get("asset_category") == asset)
        ] + asset_rows
        write_manifest(manifest_path, all_rows)
        existing_keys.update((row["query_id"], row["expert"]) for row in asset_rows)
        del projector, transformed_train, memory, train_features, patch_chunks
        gc.collect(); torch.cuda.empty_cache()

    expected = {(row["query_id"], "reconpatch_lite") for row in queries}
    actual = {(row["query_id"], row["expert"]) for row in read_manifest(manifest_path)}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"ReConPatch-lite manifest incomplete: {missing[:10]}")
    print("ReConPatch-lite complete:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
