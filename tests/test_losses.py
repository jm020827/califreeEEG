from __future__ import annotations

import torch

from cfeg.losses import kl_normal, representation_consistency_loss, symmetric_kl_logits


def test_losses_finite():
    h1 = torch.randn(4, 8)
    h2 = torch.randn(4, 8)
    logits1 = torch.randn(4, 3)
    logits2 = torch.randn(4, 3)
    mu = torch.zeros(4, 2)
    logvar = torch.zeros(4, 2)
    for loss in [
        representation_consistency_loss(h1, h2),
        symmetric_kl_logits(logits1, logits2),
        kl_normal(mu, logvar),
    ]:
        assert torch.isfinite(loss)

