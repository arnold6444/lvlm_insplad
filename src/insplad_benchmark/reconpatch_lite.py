from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReConPatchLiteConfig:
    embedding_dim: int = 384
    projection_dim: int = 128
    alpha: float = 0.5
    margin: float = 1.0
    contextual_k: int = 5
    ema_decay: float = 0.9
    train_steps: int = 120
    batch_size: int = 192
    lr: float = 1e-3
    max_train_patches: int = 6000
    memory_ratio: float = 0.05
    seed: int = 42


def pairwise_similarity(z, sigma: float = 1.0):
    import torch

    dist2 = torch.cdist(z, z, p=2).pow(2)
    return torch.exp(-dist2 / max(float(sigma), 1e-6))


def contextual_similarity(z, k: int = 5):
    import torch

    n = z.shape[0]
    k = max(1, min(int(k), n))
    dist = torch.cdist(z, z, p=2).pow(2)
    indices = torch.topk(dist, k=k, dim=1, largest=False).indices
    mask = torch.zeros((n, n), device=z.device, dtype=torch.float32)
    mask.scatter_(1, indices, 1.0)
    shared = (mask @ mask.T) / float(k)
    local = shared * mask
    reciprocal = mask * mask.T
    denom = reciprocal.sum(dim=1, keepdim=True).clamp_min(1.0)
    expanded = (reciprocal @ local) / denom
    return 0.5 * (expanded + expanded.T)


def relaxed_contrastive_loss(z, similarity, margin: float = 1.0):
    import torch

    distance = torch.cdist(z, z, p=2)
    denom = distance.mean(dim=1, keepdim=True).clamp_min(1e-6)
    delta = distance / denom
    attractive = similarity * delta.pow(2)
    repulsive = (1.0 - similarity) * torch.relu(float(margin) - delta).pow(2)
    return (attractive + repulsive).mean()


class ReConPatchLiteProjector:
    """Paper-faithful screening projector, explicitly not an official ReConPatch release."""

    def __init__(self, input_dim: int, config: ReConPatchLiteConfig, device: str = "cuda"):
        import copy
        import torch
        from torch import nn

        self.torch = torch
        self.device = torch.device(device)
        self.config = config
        self.f = nn.Linear(input_dim, config.embedding_dim).to(self.device)
        self.g = nn.Linear(config.embedding_dim, config.projection_dim).to(self.device)
        self.f_ema = copy.deepcopy(self.f).eval()
        self.g_ema = copy.deepcopy(self.g).eval()
        for module in (self.f_ema, self.g_ema):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            [*self.f.parameters(), *self.g.parameters()], lr=config.lr
        )

    def _ema_update(self) -> None:
        decay = self.config.ema_decay
        with self.torch.no_grad():
            for target, source in zip(self.f_ema.parameters(), self.f.parameters()):
                target.mul_(decay).add_(source, alpha=1.0 - decay)
            for target, source in zip(self.g_ema.parameters(), self.g.parameters()):
                target.mul_(decay).add_(source, alpha=1.0 - decay)

    def fit(self, features) -> list[float]:
        torch = self.torch
        features = features.to(self.device, dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        losses: list[float] = []
        self.f.train(); self.g.train()
        n = features.shape[0]
        for _ in range(self.config.train_steps):
            if n <= self.config.batch_size:
                indices = torch.arange(n)
            else:
                indices = torch.randperm(n, generator=generator)[: self.config.batch_size]
            batch = features[indices.to(features.device)]
            with torch.no_grad():
                z_ema = self.g_ema(self.f_ema(batch))
                z_ema = torch.nn.functional.normalize(z_ema, dim=1)
                pair = pairwise_similarity(z_ema, sigma=1.0)
                context = contextual_similarity(z_ema, self.config.contextual_k)
                similarity = self.config.alpha * pair + (1.0 - self.config.alpha) * context
            z = self.g(self.f(batch))
            z = torch.nn.functional.normalize(z, dim=1)
            loss = relaxed_contrastive_loss(z, similarity, self.config.margin)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self._ema_update()
            losses.append(float(loss.detach().cpu()))
        self.f.eval(); self.g.eval()
        return losses

    def transform(self, features):
        torch = self.torch
        with torch.no_grad():
            value = self.f(features.to(self.device, dtype=torch.float32))
            return torch.nn.functional.normalize(value, dim=1)


def sample_rows(features, limit: int, seed: int):
    import torch

    if features.shape[0] <= limit:
        return features
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(features.shape[0], generator=generator)[:limit]
    return features[indices]


def random_memory(features, ratio: float, seed: int):
    import torch

    count = max(32, int(round(features.shape[0] * float(ratio))))
    count = min(count, features.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(features.shape[0], generator=generator)[:count]
    return features[indices.to(features.device)]


def nearest_memory_map(query_features, memory, height: int, width: int, chunk: int = 512) -> np.ndarray:
    import torch

    scores = []
    for start in range(0, query_features.shape[0], chunk):
        part = query_features[start : start + chunk]
        distances = torch.cdist(part, memory, p=2)
        scores.append(distances.min(dim=1).values.detach().cpu())
    flat = torch.cat(scores).numpy().astype(np.float32, copy=False)
    if flat.size != height * width:
        raise ValueError(f"query feature count {flat.size} != {height}x{width}")
    return flat.reshape(height, width)


class WideResNetPatchExtractor:
    def __init__(self, device: str = "cuda", pretrained: bool = True, image_size: int = 256):
        import torch
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
        from torchvision.models.feature_extraction import create_feature_extractor

        self.torch = torch
        self.device = torch.device(device)
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = wide_resnet50_2(weights=weights).eval().to(self.device)
        self.model = create_feature_extractor(
            backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
        ).eval()
        self.image_size = int(image_size)

    def _tensor(self, path: Path):
        from PIL import Image
        from torchvision.transforms import v2

        transform = v2.Compose([
            v2.Resize((self.image_size, self.image_size)),
            v2.ToImage(),
            v2.ToDtype(self.torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        return transform(image)

    def extract_batch(self, paths: Sequence[Path]):
        torch = self.torch
        tensors = torch.stack([self._tensor(Path(path)) for path in paths]).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensors)
            layer2 = outputs["layer2"]
            layer3 = torch.nn.functional.interpolate(
                outputs["layer3"], size=layer2.shape[-2:], mode="bilinear", align_corners=False
            )
            merged = torch.cat([layer2, layer3], dim=1)
            merged = torch.nn.functional.avg_pool2d(merged, kernel_size=3, stride=1, padding=1)
            b, c, h, w = merged.shape
            patches = merged.permute(0, 2, 3, 1).reshape(b, h * w, c)
            return patches.detach().cpu(), h, w
