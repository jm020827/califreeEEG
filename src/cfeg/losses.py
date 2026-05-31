from __future__ import annotations

import torch
import torch.nn.functional as F


def representation_consistency_loss(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    h1 = F.normalize(h1, dim=-1)
    h2 = F.normalize(h2, dim=-1)
    return F.mse_loss(h1, h2)


def symmetric_kl_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    log_pa = F.log_softmax(logits_a / temperature, dim=-1)
    log_pb = F.log_softmax(logits_b / temperature, dim=-1)
    pa = log_pa.exp()
    pb = log_pb.exp()
    return 0.5 * (
        F.kl_div(log_pa, pb, reduction="batchmean") + F.kl_div(log_pb, pa, reduction="batchmean")
    ) * (temperature**2)


def kl_normal(mu: torch.Tensor | None, logvar: torch.Tensor | None) -> torch.Tensor:
    if mu is None or logvar is None:
        return torch.tensor(0.0)
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

