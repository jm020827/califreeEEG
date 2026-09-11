"""Query-independent routing over class-specific source spatial-metric experts."""

import torch

from cfeg.analysis.mobilebci_prior_learning import require


def validate_source_subjects(experts, allowed_sources, recipients):
    require(len(experts) == len(set(experts)), "duplicate_source_expert")
    require(set(experts) <= set(allowed_sources), "source_outside_training_pool")
    require(not set(experts).intersection(recipients), "recipient_in_source_bank")


def expert_scores(weights, query_cov, query_cross, gram):
    """W[E,class,channel,harmonic]; output[episode,E,query,class]."""
    require(weights.ndim == 4 and query_cov.ndim == 4 and query_cross.ndim == 5, "expert_shapes")
    require(
        all(bool(torch.isfinite(a).all()) for a in [weights, query_cov, query_cross, gram]),
        "nonfinite_expert_input",
    )
    cross = torch.einsum("ecdi,bncdj->bencij", weights, query_cross)
    numerator = (
        cross
        * torch.linalg.solve(gram[None, None, None], cross.transpose(-1, -2)).transpose(-1, -2)
    ).sum((-1, -2))
    energy = torch.einsum("ecdh,bndf,ecfh->benc", weights, query_cov, weights)
    score = torch.where(energy > 1e-12, numerator / energy.clamp_min(1e-12), energy * 0)
    require(
        bool(torch.isfinite(score).all() and (score >= -1e-8).all() and (score <= 1 + 1e-8).all()),
        "expert_score_range",
    )
    return score.clamp(0, 1), int((energy <= 1e-12).sum())


def support_compatibility(weights, support_cov, support_cross, gram, samples):
    """Coherent pooled own-class compatibility from averaged support statistics."""
    cov, xy = support_cov * samples, support_cross * samples
    cross = torch.einsum("ecdi,cdj->ecij", weights, xy)
    numerator = (
        cross * torch.linalg.solve(gram[None], cross.transpose(-1, -2)).transpose(-1, -2)
    ).sum((-1, -2))
    energy = torch.einsum("ecdh,cdf,ecfh->ec", weights, cov, weights)
    score = torch.where(energy > 1e-12, numerator / energy.clamp_min(1e-12), energy * 0)
    require(
        bool(torch.isfinite(score).all() and (score >= -1e-8).all() and (score <= 1 + 1e-8).all()),
        "compatibility_range",
    )
    return score.clamp(0, 1)


def pair_features(weights, target_cov, target_cross, bank_cov, gram, samples):
    compatibility = support_compatibility(weights, target_cov, target_cross, gram, samples)
    target_norm = target_cov / torch.diagonal(target_cov, dim1=-2, dim2=-1).sum(-1)[
        ..., None, None
    ].clamp_min(1e-12)
    bank_norm = bank_cov / torch.diagonal(bank_cov, dim1=-2, dim2=-1).sum(-1)[
        ..., None, None
    ].clamp_min(1e-12)
    distance = (bank_norm - target_norm).square().sum((-1, -2)).mean(-1)
    self_flag = torch.zeros(weights.shape[0], dtype=weights.dtype)
    self_flag[0] = 1
    base = torch.stack((self_flag, compatibility.mean(-1), distance), dim=-1)
    q2 = torch.stack((compatibility.std(-1, correction=0), compatibility.min(-1).values), dim=-1)
    q2[0] = 0
    return base, q2


def metadata_differences(target_metadata, source_metadata):
    """Return zero self difference followed by pairwise source differences."""
    require(
        target_metadata.ndim == 2
        and source_metadata.ndim == 2
        and target_metadata.shape[1] == source_metadata.shape[1],
        "metadata_shapes",
    )
    require(
        bool(torch.isfinite(target_metadata).all() and torch.isfinite(source_metadata).all()),
        "nonfinite_metadata",
    )
    source = (target_metadata[:, None] - source_metadata[None]).abs()
    return torch.cat((torch.zeros_like(source[:, :1]), source), dim=1)


def mix_probabilities(probabilities, weights, no_transfer=False):
    require(probabilities.ndim == 4 and weights.ndim == 2, "router_must_not_have_query_axis")
    require(probabilities.shape[:2] == weights.shape, "router_shape")
    require(
        bool(torch.isfinite(probabilities).all() and torch.isfinite(weights).all()),
        "nonfinite_mixture",
    )
    require(bool((probabilities >= 0).all() and (weights >= 0).all()), "negative_mixture")
    require(
        bool(
            torch.allclose(
                probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), atol=1e-9, rtol=0
            )
        ),
        "expert_probability_sum",
    )
    require(
        bool(torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]), atol=1e-9, rtol=0)),
        "router_weight_sum",
    )
    if no_transfer:
        return probabilities[:, 0]
    return torch.einsum("benk,be->bnk", probabilities, weights)


class Router:
    def __init__(self, inputs):
        require(inputs.ndim == 3 and bool(torch.isfinite(inputs).all()), "router_input")
        self.mean = inputs.mean((0, 1)).detach()
        self.std = inputs.std((0, 1), correction=0).clamp_min(1e-12).detach()
        self.layer = torch.nn.Linear(inputs.shape[-1], 1, dtype=torch.float64)
        with torch.no_grad():
            self.layer.weight.zero_()
            self.layer.bias.zero_()

    def weights(self, inputs):
        require(inputs.ndim == 3 and bool(torch.isfinite(inputs).all()), "router_input")
        return torch.softmax(self.layer((inputs - self.mean) / self.std).squeeze(-1), dim=-1)

    def state(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weight": self.layer.weight.detach().tolist(),
            "bias": self.layer.bias.detach().tolist(),
        }


def fit_router(inputs, probabilities, labels, steps=100, on_step=None):
    router = Router(inputs)
    optimizer = torch.optim.AdamW(router.layer.parameters(), lr=0.05, weight_decay=0.01)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        p = mix_probabilities(probabilities, router.weights(inputs))
        selected = torch.gather(p, -1, labels[..., None]).squeeze(-1).clamp_min(1e-12)
        loss = -selected.log().mean()
        require(bool(torch.isfinite(loss)), "nonfinite_router_loss")
        loss.backward()
        require(
            all(bool(torch.isfinite(p.grad).all()) for p in router.layer.parameters()),
            "nonfinite_router_gradient",
        )
        optimizer.step()
        if on_step is not None:
            on_step()
        losses.append(float(loss.detach()))
    for parameter in router.layer.parameters():
        parameter.requires_grad_(False)
    return router, {"steps": steps, "initial_loss": losses[0], "last_update_loss": losses[-1]}
