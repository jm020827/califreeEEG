from __future__ import annotations

import math

import torch

from cfeg.models.metadata_calibration_prior import BoundedMetadataCalibrationPrior


def _head() -> BoundedMetadataCalibrationPrior:
    torch.manual_seed(4)
    return BoundedMetadataCalibrationPrior(
        embedding_dim=6,
        n_classes=3,
        q_feature_dim=4,
        metadata_feature_dim=5,
        hidden_dim=8,
    )


def _inputs():
    torch.manual_seed(9)
    query = torch.randn(4, 6)
    support = torch.randn(6, 6)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    query_q = torch.randn(4, 4)
    support_q = torch.randn(6, 4)
    query_m = torch.randn(4, 5)
    support_m = torch.randn(6, 5)
    query_missing = torch.zeros_like(query_m, dtype=torch.bool)
    support_missing = torch.zeros_like(support_m, dtype=torch.bool)
    return (
        query,
        support,
        labels,
        query_q,
        support_q,
        query_m,
        support_m,
        query_missing,
        support_missing,
    )


def _run(
    head, *, mode="off", query_m=None, support_m=None, query_missing=None, support_missing=None
):
    (
        query,
        support,
        labels,
        query_q,
        support_q,
        default_qm,
        default_sm,
        default_qmiss,
        default_smiss,
    ) = _inputs()
    return head(
        query,
        query_q,
        support_embeddings=support,
        support_labels=labels,
        support_q_features=support_q,
        query_metadata_values=default_qm if query_m is None else query_m,
        query_metadata_missing=default_qmiss if query_missing is None else query_missing,
        support_metadata_values=default_sm if support_m is None else support_m,
        support_metadata_missing=default_smiss if support_missing is None else support_missing,
        metadata_mode=mode,
    )


def test_precision_is_positive_and_metadata_ratio_is_bounded() -> None:
    head = _head()
    with torch.no_grad():
        head.metadata_precision_encoder[-1].weight.fill_(2.0)
        head.metadata_precision_encoder[-1].bias.fill_(0.5)
    off = _run(head, mode="off")
    observed = _run(head, mode="observed")

    assert torch.isfinite(observed.query_precision).all()
    assert (observed.query_precision > 0.0).all()
    ratio = observed.query_precision / off.query_precision
    assert float(ratio.min().detach()) >= 0.8 - 1e-6
    assert float(ratio.max().detach()) <= 1.25 + 1e-6


def test_exact_off_is_invariant_to_even_nonfinite_metadata() -> None:
    head = _head().eval()
    clean = _run(head, mode="off")
    poison_query = torch.full((4, 5), float("nan"))
    poison_support = torch.full((6, 5), float("inf"))
    poison = _run(
        head,
        mode="off",
        query_m=poison_query,
        support_m=poison_support,
    )

    assert torch.equal(clean.logits, poison.logits)
    assert torch.equal(clean.probabilities, poison.probabilities)
    assert torch.equal(clean.query_precision, poison.query_precision)
    assert torch.equal(clean.support_precision, poison.support_precision)


def test_q_exact_off_enables_same_graph_a0_and_am_roles() -> None:
    head = _head().eval()
    (
        query,
        support,
        labels,
        _query_q,
        _support_q,
        query_m,
        support_m,
        query_missing,
        support_missing,
    ) = _inputs()
    poison_query_q = torch.full((4, 4), float("nan"))
    poison_support_q = torch.full((6, 4), float("inf"))
    a0 = head(
        query,
        poison_query_q,
        support_embeddings=support,
        support_labels=labels,
        support_q_features=poison_support_q,
        q_mode="off",
        metadata_mode="off",
    )
    am = head(
        query,
        poison_query_q,
        support_embeddings=support,
        support_labels=labels,
        support_q_features=poison_support_q,
        query_metadata_values=query_m,
        query_metadata_missing=query_missing,
        support_metadata_values=support_m,
        support_metadata_missing=support_missing,
        q_mode="off",
        metadata_mode="observed",
    )
    assert a0.logits.shape == am.logits.shape == (4, 3)
    assert torch.isfinite(a0.logits).all()


def test_all_missing_observed_is_bitwise_equal_to_exact_off() -> None:
    head = _head().eval()
    with torch.no_grad():
        head.metadata_precision_encoder[-1].weight.fill_(1.5)
        head.metadata_precision_encoder[-1].bias.fill_(0.25)
    off = _run(head, mode="off")
    missing = _run(
        head,
        mode="observed",
        query_m=torch.full((4, 5), float("nan")),
        support_m=torch.full((6, 5), float("nan")),
        query_missing=torch.ones(4, 5, dtype=torch.bool),
        support_missing=torch.ones(6, 5, dtype=torch.bool),
    )

    for field in ("logits", "probabilities", "query_precision", "support_precision"):
        assert torch.equal(getattr(off, field), getattr(missing, field))
    assert torch.count_nonzero(missing.query_metadata_log_residual) == 0
    assert torch.count_nonzero(missing.support_metadata_log_residual) == 0


def test_support_update_is_permutation_invariant_and_k0_never_reads_labels() -> None:
    head = _head().eval()
    (
        query,
        support,
        labels,
        query_q,
        support_q,
        query_m,
        support_m,
        query_missing,
        support_missing,
    ) = _inputs()
    first = head(
        query,
        query_q,
        support_embeddings=support,
        support_labels=labels,
        support_q_features=support_q,
        query_metadata_values=query_m,
        query_metadata_missing=query_missing,
        support_metadata_values=support_m,
        support_metadata_missing=support_missing,
        metadata_mode="observed",
    )
    order = torch.tensor([5, 2, 4, 0, 3, 1])
    reordered = head(
        query,
        query_q,
        support_embeddings=support[order],
        support_labels=labels[order],
        support_q_features=support_q[order],
        query_metadata_values=query_m,
        query_metadata_missing=query_missing,
        support_metadata_values=support_m[order],
        support_metadata_missing=support_missing[order],
        metadata_mode="observed",
    )
    assert torch.allclose(first.logits, reordered.logits, rtol=1e-6, atol=1e-7)

    empty = head(
        query,
        query_q,
        support_embeddings=torch.empty(0, 6),
        support_labels=torch.empty(0, dtype=torch.long),
        support_q_features=torch.empty(0, 4),
        query_metadata_values=query_m,
        query_metadata_missing=query_missing,
        support_metadata_values=torch.empty(0, 5),
        support_metadata_missing=torch.empty(0, 5, dtype=torch.bool),
        metadata_mode="observed",
    )
    assert empty.posterior_mean.shape == (3, 6)
    assert empty.support_precision.shape == (0, 6)
    assert torch.allclose(empty.probabilities.sum(dim=-1), torch.ones(4))


def test_stage2_freezes_every_common_parameter_and_only_m_receives_gradients() -> None:
    head = _head()
    head.freeze_common_path_for_stage2()
    common = list(head.common_parameters())
    metadata = list(head.metadata_parameters())
    assert common and metadata
    assert not any(parameter.requires_grad for parameter in common)
    assert all(parameter.requires_grad for parameter in metadata)

    output = _run(head, mode="observed")
    loss = -torch.log(output.probabilities[:, 0].clamp_min(1e-8)).mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in common)
    assert any(parameter.grad is not None for parameter in metadata)


def test_query_metadata_can_change_k0_predictive_metric_without_moving_anchor() -> None:
    head = _head().eval()
    (
        query,
        _support,
        _labels,
        query_q,
        _support_q,
        query_m,
        _support_m,
        query_missing,
        _support_missing,
    ) = _inputs()
    empty_embedding = torch.empty(0, 6)
    empty_q = torch.empty(0, 4)
    empty_m = torch.empty(0, 5)
    empty_missing = torch.empty(0, 5, dtype=torch.bool)
    off = head(
        query,
        query_q,
        support_embeddings=empty_embedding,
        support_labels=torch.empty(0, dtype=torch.long),
        support_q_features=empty_q,
        query_metadata_values=query_m,
        query_metadata_missing=query_missing,
        support_metadata_values=empty_m,
        support_metadata_missing=empty_missing,
        metadata_mode="off",
    )
    with torch.no_grad():
        last = head.metadata_precision_encoder[-1]
        last.weight.copy_(torch.arange(last.weight.numel()).reshape_as(last.weight) / 20.0)
        last.bias.copy_(torch.linspace(-1.0, 1.0, 6))
    observed = head(
        query,
        query_q,
        support_embeddings=empty_embedding,
        support_labels=torch.empty(0, dtype=torch.long),
        support_q_features=empty_q,
        query_metadata_values=query_m,
        query_metadata_missing=query_missing,
        support_metadata_values=empty_m,
        support_metadata_missing=empty_missing,
        metadata_mode="observed",
    )

    assert torch.equal(off.posterior_mean, observed.posterior_mean)
    assert not torch.equal(off.query_precision, observed.query_precision)
    assert not torch.equal(off.logits, observed.logits)
    ratio = observed.query_precision / off.query_precision
    assert float(ratio.min().detach()) >= math.exp(-head.metadata_log_ratio_bound) - 1e-6
    assert float(ratio.max().detach()) <= math.exp(head.metadata_log_ratio_bound) + 1e-6


def test_stage1_support_update_has_a_valid_backward_graph() -> None:
    head = _head()
    (
        query,
        support,
        labels,
        query_q,
        support_q,
        _query_m,
        _support_m,
        _query_missing,
        _support_missing,
    ) = _inputs()
    query.requires_grad_(True)
    support.requires_grad_(True)
    output = head(
        query,
        query_q,
        support_embeddings=support,
        support_labels=labels,
        support_q_features=support_q,
        metadata_mode="off",
    )
    torch.nn.functional.cross_entropy(output.logits, torch.tensor([0, 1, 2, 0])).backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert support.grad is not None and torch.isfinite(support.grad).all()
    assert head.source_anchor.grad is not None
