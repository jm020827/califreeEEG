"""Full-case temporal-score batch; no legacy mean correction or anchor use."""

from __future__ import annotations

import torch

from cfeg.analysis import task_trca_shape_operator as operator
from cfeg.analysis import task_trca_shape_signfree as signfree


class TaskBatch:
    """Mathematically equivalent full-batch loss for temporal source cases."""

    def __init__(self, cases):
        from cfeg.analysis.task_trca_temporal_learning import _validate_cases

        # Keep caller order: head logits use precisely that order. Validation's
        # sorted return is only for canonical learner pipelines, not this API.
        cases = tuple(cases)
        _validate_cases(cases)
        self.s = torch.stack([c.s for c in cases])
        self.c = torch.stack([c.c for c in cases])
        self.weights = torch.stack([c.weights for c in cases])
        self.labels = torch.stack([c.labels for c in cases])
        self.stats = {
            name: torch.stack([getattr(c.statistics, name) for c in cases])
            for name in ("query_gram", "template_gram", "cross_gram")
        }
        self.samples = tuple(c.samples for c in cases)
        self.cases = len(cases)
        # Validate the temporal sufficient-statistic boundary once. Identity
        # projectors only test nonzero projected variance; no native W is used.
        identity = torch.eye(8, dtype=torch.float64, device=self.s.device).expand(5, 12, 8, 8)
        for c in cases:
            signfree.score_temporal_gram(identity, c.statistics, c.weights)

    def scores(self, logits):
        if not isinstance(logits, torch.Tensor) or logits.shape != (self.cases, 5, 8):
            raise ValueError("Temporal batch logits must be [cases,5,8]")
        if logits.dtype != torch.float64 or logits.device != self.s.device:
            raise ValueError("Temporal batch logits require matching float64 device")
        r = operator.shape_prior(logits)
        projectors = signfree.bounded_projectors(self.s, self.c, r[:, :, None, :])
        j = projectors.sum(dim=2)
        dot = torch.einsum("pbij,pncbij->pnbc", j, self.stats["cross_gram"])
        xx = torch.einsum("pbij,pnbij->pnb", j, self.stats["query_gram"])
        tt = torch.einsum("pbij,pcbij->pbc", j, self.stats["template_gram"])
        if (
            not all(torch.isfinite(v).all() for v in (dot, xx, tt))
            or (xx <= 0).any()
            or (tt <= 0).any()
        ):
            raise ValueError("VALIDITY_FAILURE: invalid temporal batch projected variance")
        corr = dot / xx.sqrt().unsqueeze(-1) / tt.sqrt().unsqueeze(1)
        if not torch.isfinite(corr).all():
            raise ValueError("VALIDITY_FAILURE: invalid temporal batch correlation")
        scores = torch.einsum("pnbc,pb->pnc", corr.clamp(-1, 1), self.weights)
        if not torch.isfinite(scores).all():
            raise ValueError("VALIDITY_FAILURE: nonfinite temporal batch scores")
        return scores

    def loss(self, logits):
        scores = self.scores(logits)
        scaled = scores / self.weights.sum(-1)[:, None, None] / 0.1
        return torch.nn.functional.cross_entropy(scaled.flatten(0, 1), self.labels.flatten())
