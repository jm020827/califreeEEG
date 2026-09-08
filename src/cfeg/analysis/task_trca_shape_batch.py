"""Semantics-preserving batch of the fixed task-shape loss, no file access.

Each case retains its sample count, weights and labels. Complete participant
grids are checked by the caller. No scientific constants or optimizer change.
"""

from __future__ import annotations

import torch

from cfeg.analysis import task_trca_shape_operator as op


class TaskBatch:
    def __init__(self, cases):
        if any(type(c.samples) is not int or c.samples < 2 or c.samples != c.statistics.samples for c in cases):
            raise ValueError("Case and Gram sample counts must agree")
        self.s = torch.stack([c.s for c in cases])
        self.c = torch.stack([c.c for c in cases])
        self.anchors = torch.stack([c.anchors for c in cases])
        self.weights = torch.stack([c.weights for c in cases])
        self.labels = torch.stack([c.labels for c in cases])
        self.stats = {
            name: torch.stack([getattr(c.statistics, name) for c in cases])
            for name in ("query_gram", "template_gram", "cross_gram", "query_mean", "template_mean")
        }
        self.samples = torch.tensor(
            [c.samples for c in cases], dtype=torch.float64, device=self.s.device
        )
        # Validate all immutable statistics once, before any optimization step.
        for value in (self.s, self.c, self.anchors, self.weights, *self.stats.values()):
            if value.dtype != torch.float64 or value.requires_grad or not torch.isfinite(value).all():
                raise ValueError("Invalid detached float64 batch constant")
        if (self.weights <= 0).any():
            raise ValueError("Native weights must be positive")
        if self.labels.shape != (len(cases), 12):
            raise ValueError("Task batch requires twelve source labels per case")
        for c in cases:
            # Also applies reference shape/sample-count checks at the boundary.
            op.score_gram(c.anchors.transpose(-1, -2), c.statistics, c.weights)

    def scores(self, logits):
        r = op.shape_prior(logits)
        filters = op.bounded_filters(
            self.s, self.c, self.anchors, r[:, :, None, :]
        ).transpose(-1, -2)
        j = filters @ filters.transpose(-1, -2)
        centered = filters - filters.mean(dim=-1, keepdim=True)
        h = centered @ centered.transpose(-1, -2)
        mx, mt = self.stats["query_mean"], self.stats["template_mean"]
        dot = torch.einsum("pbij,pncbij->pnbc", j, self.stats["cross_gram"])
        dot = dot + self.samples[:, None, None, None] * torch.einsum("pnbi,pbij,pcbj->pnbc", mx, h, mt)
        xx = torch.einsum("pbij,pnbij->pnb", j, self.stats["query_gram"])
        xx = xx + self.samples[:, None, None] * torch.einsum("pnbi,pbij,pnbj->pnb", mx, h, mx)
        tt = torch.einsum("pbij,pcbij->pbc", j, self.stats["template_gram"])
        tt = tt + self.samples[:, None, None] * torch.einsum("pcbi,pbij,pcbj->pbc", mt, h, mt)
        if not all(torch.isfinite(v).all() for v in (dot, xx, tt)) or (xx <= 0).any() or (tt <= 0).any():
            raise ValueError("VALIDITY_FAILURE: invalid batch projected variance")
        corr = dot / xx.sqrt().unsqueeze(-1) / tt.sqrt().unsqueeze(1)
        if not torch.isfinite(corr).all():
            raise ValueError("VALIDITY_FAILURE: invalid batch correlation")
        return torch.einsum("pnbc,pb->pnc", corr.clamp(-1, 1), self.weights)

    def loss(self, logits):
        scores = self.scores(logits)
        scaled = scores / self.weights.sum(-1)[:, None, None] / 0.1
        return torch.nn.functional.cross_entropy(scaled.flatten(0, 1), self.labels.flatten())
