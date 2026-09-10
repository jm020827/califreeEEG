"""Batched, deterministic source-trained priors for the frozen MobileBCI candidate."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def projection_scores(support_cov, support_cross, query_cov, query_cross, gram, diagonal, ridge):
    scale = torch.diagonal(support_cov, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    weights = torch.linalg.solve(
        support_cov + ridge * scale[..., None, None] * torch.diag_embed(diagonal)[:, None],
        support_cross,
    )
    cross = torch.einsum("bcdi,bncdj->bncij", weights, query_cross)
    numerator = (
        cross * torch.linalg.solve(gram[:, None], cross.transpose(-1, -2)).transpose(-1, -2)
    ).sum((-1, -2))
    energy = torch.einsum("bcdi,bnde,bcei->bnc", weights, query_cov, weights)
    scores = torch.where(energy > 1e-12, numerator / energy.clamp_min(1e-12), energy * 0)
    require(bool(torch.isfinite(scores).all()), "nonfinite_projection_scores")
    require(bool((scores >= -1e-8).all() and (scores <= 1 + 1e-8).all()), "score_range")
    return scores.clamp(0, 1), int((energy <= 1e-12).sum())


def cca_scores(query_cov, query_cross, gram):
    channels = query_cov.shape[-1]
    scale = torch.diagonal(query_cov, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    cov = query_cov + 0.001 * scale[..., None, None] * torch.eye(channels, dtype=query_cov.dtype)
    lx = torch.linalg.cholesky(cov)[:, :, None]
    ly = torch.linalg.cholesky(gram)[:, None]
    a = torch.linalg.solve_triangular(lx, query_cross, upper=False)
    b = torch.linalg.solve_triangular(ly, a.transpose(-1, -2), upper=False)
    scores = torch.linalg.svdvals(b)[..., 0].square()
    require(bool(torch.isfinite(scores).all() and (scores <= 1 + 1e-8).all()), "cca_score_range")
    return scores.clamp(0, 1)


def balanced_loss(scores, labels):
    losses = functional.cross_entropy(
        (10 * scores).reshape(-1, 3), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)
    counts = torch.stack([(labels == c).sum(-1) for c in range(3)], dim=-1)
    require(bool((counts > 0).all()), "query_class_missing")
    weights = 1 / (3 * torch.gather(counts, 1, labels))
    return (losses * weights).sum(-1).mean()


def balanced_accuracy(prediction, labels):
    return np.array(
        [np.mean([np.mean(p[y == c] == c) for c in range(3)]) for p, y in zip(prediction, labels)]
    )


def donor_indices(source, recipients, subjects, speeds, common):
    donors, distances = [], []
    for i in recipients:
        choices = [j for j in source if subjects[j] != subjects[i] and speeds[j] == speeds[i]]
        require(bool(choices), "no_nonself_source_donor")
        ranked = sorted(
            (
                abs(common[j, 3] - common[i, 3]) + abs(common[j, 4] - common[i, 4]) / 10,
                subjects[j],
                j,
            )
            for j in choices
        )
        distance, _, donor = ranked[0]
        donors.append(donor)
        distances.append(float(distance))
    return np.asarray(donors), distances


@dataclass
class Head:
    layer: torch.nn.Linear
    mean: torch.Tensor
    std: torch.Tensor
    constant_features: int

    def logits(self, inputs):
        return self.layer((inputs - self.mean) / self.std)

    def serialize(self):
        return {
            "weight": self.layer.weight.detach().tolist(),
            "bias": self.layer.bias.detach().tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "constant_features": self.constant_features,
        }


def new_head(training_inputs, channels=9):
    mean = training_inputs.mean(0)
    raw_std = training_inputs.std(0, correction=0)
    head = torch.nn.Linear(training_inputs.shape[-1], channels, dtype=torch.float64)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    return Head(head, mean, raw_std.clamp_min(1e-12), int((raw_std < 1e-12).sum()))


def prior(head, inputs, frozen=None):
    logits = head.logits(inputs)
    if frozen is None:
        return 0.25 + 0.75 * logits.shape[-1] * torch.softmax(logits, -1)
    v = torch.tanh(logits)
    return frozen.detach() + 0.05 * (v - v.mean(-1, keepdim=True))


def fit_head(inputs, data, ridge, frozen=None, steps=100, on_step=None):
    head = new_head(inputs, data[0].shape[-1])
    optimizer = torch.optim.AdamW(
        head.layer.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    first_loss, max_grad = None, 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        diagonal = prior(head, inputs, frozen)
        scores, _ = projection_scores(*data[:5], diagonal, ridge)
        loss = balanced_loss(scores, data[5])
        require(bool(torch.isfinite(loss)), "nonfinite_training_loss")
        if first_loss is None:
            first_loss = float(loss.detach())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.layer.parameters(), 1.0, error_if_nonfinite=True)
        max_grad = max(max_grad, float(norm))
        optimizer.step()
        if on_step:
            on_step()
    for parameter in head.layer.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        diagonal = prior(head, inputs, frozen)
        scores, floor_count = projection_scores(*data[:5], diagonal, ridge)
        final_loss = float(balanced_loss(scores, data[5]))
        logs = {
            "steps": steps,
            "first_loss": first_loss,
            "final_loss": final_loss,
            "max_gradient_norm_before_clip": max_grad,
            "constant_features": head.constant_features,
            "energy_floor_count": floor_count,
            "prior_min": float(diagonal.min()),
            "prior_trace_max_error": float((diagonal.sum(-1) - diagonal.shape[-1]).abs().max()),
            "residual_max_abs": 0.0 if frozen is None else float((diagonal - frozen).abs().max()),
        }
        if frozen is not None:
            baseline, _ = projection_scores(*data[:5], frozen, ridge)
            logs["source_score_actuation_max_abs"] = float((scores - baseline).abs().max())
            logs["tanh_saturation_fraction"] = float(
                (torch.tanh(head.logits(inputs)).abs() > 0.99).to(torch.float64).mean()
            )
    return head, logs
