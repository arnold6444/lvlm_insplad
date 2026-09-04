from __future__ import annotations

import pytest


def test_reconpatch_lite_similarity_and_loss_are_finite() -> None:
    torch = pytest.importorskip("torch")
    from insplad_benchmark.reconpatch_lite import contextual_similarity, pairwise_similarity, relaxed_contrastive_loss

    torch.manual_seed(1)
    z = torch.nn.functional.normalize(torch.randn(12, 8), dim=1)
    pair = pairwise_similarity(z)
    context = contextual_similarity(z, 3)
    similarity = 0.5 * pair + 0.5 * context
    loss = relaxed_contrastive_loss(z, similarity, 1.0)
    assert pair.shape == context.shape == (12, 12)
    assert torch.isfinite(loss)
    assert float(context.min()) >= 0.0
    assert float(context.max()) <= 1.0 + 1e-6
