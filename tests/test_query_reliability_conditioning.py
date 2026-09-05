from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from cfeg.constants import CATEGORICAL_VOCABS
from cfeg.models.full_model import ConditionedEEGDecoder
from cfeg.models.query_reliability_conditioning import (
    QueryReliabilityOperator,
    _project_frobenius_norm,
    extract_query_reliability_features_v3,
)


def _config(query_mode: str) -> dict:
    return {
        "protocol": {
            "metadata_contract_version": "0.4-dev",
            "query_qc_extractor_version": ("filtered_cropped_pre_zscore_channel_std_median_v1"),
            "external_continuous_schema": "impedance_mean_max_v1",
            "reliability_query_feature_schema": (
                "query_window_reliability_features_v3_waveform_6_60_qc_scale_6_90"
            ),
        },
        "model": {
            "n_classes": 3,
            "c_max": 4,
            "t_len": 64,
            "target_sfreq": 200.0,
            "d_model": 16,
            "depth": 1,
            "n_heads": 4,
            "backbone": {
                "name": "spectral_transformer",
                "min_frequency_hz": 6.0,
                "max_frequency_hz": 60.0,
                "dropout": 0.0,
            },
            "conditioning": {
                "architecture": "query_reliability_spatial_v1",
                "hidden_dim": 8,
                "spatial_reliability": {
                    "placement": "pre_backbone_waveform",
                    "operator_family": "diagonal_low_rank",
                    "operator_rank": 2,
                    "max_operator_norm": 0.20,
                    "query_qc_mode": query_mode,
                },
            },
            "condition_encoder": {
                "enabled": True,
                "n_prompt_tokens": 0,
                "fields": [],
                "include_continuous": False,
                "include_channels": False,
                "force_missing": False,
                "external_metadata_mode": "null",
            },
            "adapter": {"enabled": False},
            "latent": {"enabled": False},
        },
    }


def _condition(batch: int = 2, channels: int = 4) -> dict:
    cond = {
        "continuous": torch.zeros(batch, 5),
        "continuous_missing": torch.ones(batch, 5, dtype=torch.bool),
        "channel_ids": torch.arange(1, channels + 1).repeat(batch, 1),
        "channel_mask": torch.ones(batch, channels, dtype=torch.bool),
        "sfreq_processed_float": torch.full((batch,), 200.0),
        "external_continuous": torch.zeros(batch, 2),
        "external_continuous_missing": torch.ones(batch, 2, dtype=torch.bool),
        "query_qc": torch.full((batch, 1), 0.3),
        "query_qc_missing": torch.zeros(batch, 1, dtype=torch.bool),
        "channel_query_qc": torch.linspace(0.1, 0.7, channels).repeat(batch, 1),
        "channel_query_qc_missing": torch.zeros(batch, channels, dtype=torch.bool),
        "channel_impedance": torch.zeros(batch, channels),
        "channel_impedance_missing": torch.ones(batch, channels, dtype=torch.bool),
        "metadata_contract_version": "0.4-dev",
        "external_metadata_mode": "null",
        "query_qc_extractor_version": ("filtered_cropped_pre_zscore_channel_std_median_v1"),
        "external_continuous_schema": "impedance_mean_max_v1",
        "development_control": "none",
    }
    for name in CATEGORICAL_VOCABS:
        cond[name] = torch.zeros(batch, dtype=torch.long)
    return cond


def _vocab_sizes() -> dict[str, int]:
    return {name: len(values) for name, values in CATEGORICAL_VOCABS.items()}


def _activate(operator: QueryReliabilityOperator) -> None:
    with torch.no_grad():
        operator.query_diagonal_score.weight.normal_(std=0.3)
        operator.query_diagonal_score.bias.fill_(0.1)
        operator.query_left_factor.weight.normal_(std=0.2)
        operator.query_left_factor.bias.normal_(std=0.1)


def test_operator_norm_projection_stays_strict_after_half_precision_input() -> None:
    torch.manual_seed(3)
    projected = _project_frobenius_norm(torch.randn(16, 64, 64, dtype=torch.float16), maximum=0.20)
    assert projected.dtype == torch.float32
    norms = torch.linalg.vector_norm(projected.flatten(start_dim=1), dim=-1)
    assert torch.all(norms <= 0.20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is unavailable")
def test_query_operator_cuda_amp_forward_preserves_the_hard_norm_bound() -> None:
    torch.manual_seed(5)
    model = ConditionedEEGDecoder(_config("observed"), _vocab_sizes()).cuda().eval()
    assert model.query_reliability_conditioner is not None
    _activate(model.query_reliability_conditioner)
    x = torch.randn(2, 4, 64, device="cuda")
    cond = {
        key: value.cuda() if torch.is_tensor(value) else value
        for key, value in _condition().items()
    }
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(x, cond)
    assert torch.isfinite(output.logits).all()
    assert torch.all(output.aux["spatial_operator_frobenius_norm"] <= 0.20)


def test_q0_q1_share_graph_and_start_as_exact_same_function() -> None:
    models = {}
    for mode in ("null", "observed"):
        torch.manual_seed(7)
        models[mode] = ConditionedEEGDecoder(_config(mode), _vocab_sizes()).eval()

    left = models["null"].state_dict()
    right = models["observed"].state_dict()
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0.0, atol=0.0)

    x = torch.randn(2, 4, 64)
    cond = _condition()
    with torch.no_grad():
        q0 = models["null"](x, cond)
        q1 = models["observed"](x, cond)
    torch.testing.assert_close(q0.logits, q1.logits, rtol=0.0, atol=0.0)
    assert q0.aux["spatial_operator_frobenius_norm"].eq(0.0).all()
    assert q1.aux["spatial_operator_frobenius_norm"].eq(0.0).all()


def test_q0_short_circuits_nan_qc_and_has_no_q_gradient() -> None:
    model = ConditionedEEGDecoder(_config("null"), _vocab_sizes()).eval()
    x = torch.randn(2, 4, 64)
    cond = _condition()
    cond["channel_query_qc"].fill_(float("nan"))

    out = model(x, cond)
    assert torch.isfinite(out.logits).all()
    out.logits.sum().backward()
    assert model.query_reliability_conditioner is not None
    for parameter in model.query_reliability_conditioner.parameters():
        assert parameter.grad is None or parameter.grad.eq(0.0).all()


def test_q1_fails_closed_on_nonfinite_active_signal_or_available_qc() -> None:
    operator = QueryReliabilityOperator(hidden_dim=8, operator_rank=2).eval()
    x = torch.randn(1, 4, 64)
    cond = _condition(batch=1)
    cond["channel_query_qc"][0, 0] = float("nan")
    try:
        operator(x, cond)
    except ValueError as exc:
        assert "finite available channel QC" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Observed Q accepted NaN QC.")

    cond = _condition(batch=1)
    x[0, 0, 0] = float("inf")
    try:
        operator(x, cond)
    except ValueError as exc:
        assert "finite active EEG" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Observed Q accepted an infinite EEG sample.")


def test_q1_has_a_nonzero_first_step_gradient_but_q0_stays_identity_after_step() -> None:
    x = torch.randn(2, 4, 64)
    cond = _condition()
    labels = torch.tensor([0, 1])

    torch.manual_seed(31)
    q1 = ConditionedEEGDecoder(_config("observed"), _vocab_sizes()).train()
    F.cross_entropy(q1(x, cond).logits, labels).backward()
    assert q1.query_reliability_conditioner is not None
    q1_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in q1.query_reliability_conditioner.parameters()
        if parameter.grad is not None
    )
    assert q1_gradient > 0.0

    torch.manual_seed(31)
    q0 = ConditionedEEGDecoder(_config("null"), _vocab_sizes()).train()
    assert q0.query_reliability_conditioner is not None
    before = {
        name: value.detach().clone()
        for name, value in q0.query_reliability_conditioner.named_parameters()
    }
    optimizer = torch.optim.AdamW(q0.parameters(), lr=1e-3, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(q0(x, cond).logits, labels).backward()
    optimizer.step()
    for name, value in q0.query_reliability_conditioner.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)
    state = q0.query_reliability_conditioner(x, cond)
    assert state.operator_delta.eq(0.0).all()


def test_q1_is_external_metadata_bitwise_invariant_after_activation() -> None:
    torch.manual_seed(13)
    model = ConditionedEEGDecoder(_config("observed"), _vocab_sizes()).eval()
    assert model.query_reliability_conditioner is not None
    _activate(model.query_reliability_conditioner)
    x = torch.randn(2, 4, 64)
    clean = _condition()
    mutated = copy.deepcopy(clean)
    mutated["external_continuous"].normal_()
    mutated["external_continuous_missing"].fill_(False)
    mutated["channel_impedance"].normal_()
    mutated["channel_impedance_missing"].fill_(False)
    for name in CATEGORICAL_VOCABS:
        mutated[name].fill_(1)

    with torch.no_grad():
        expected = model(x, clean)
        observed = model(x, mutated)
    torch.testing.assert_close(expected.logits, observed.logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        expected.aux["channel_spatial_self_gain"],
        observed.aux["channel_spatial_self_gain"],
        rtol=0.0,
        atol=0.0,
    )


def test_same_checkpoint_q1_identity_intervention_is_exact_and_eval_only() -> None:
    torch.manual_seed(23)
    q1 = ConditionedEEGDecoder(_config("observed"), _vocab_sizes()).eval()
    assert q1.query_reliability_conditioner is not None
    _activate(q1.query_reliability_conditioner)
    q0 = ConditionedEEGDecoder(_config("null"), _vocab_sizes()).eval()
    q0.load_state_dict(q1.state_dict(), strict=True)
    x = torch.randn(2, 4, 64)
    cond = _condition()
    with torch.no_grad():
        active = q1(x, cond)
        intervened = q1(
            x,
            cond,
            query_reliability_identity_intervention=True,
        )
        identity = q0(x, cond)
    assert not torch.equal(active.logits, intervened.logits)
    torch.testing.assert_close(intervened.logits, identity.logits, rtol=0.0, atol=0.0)
    assert intervened.aux["query_reliability_identity_intervention"].all()

    q1.train()
    with pytest.raises(ValueError, match="evaluation-only"):
        q1(x, cond, query_reliability_identity_intervention=True)


def test_same_checkpoint_wrong_query_intervention_is_explicit_and_eval_only() -> None:
    torch.manual_seed(29)
    q1 = ConditionedEEGDecoder(_config("observed"), _vocab_sizes()).eval()
    assert q1.query_reliability_conditioner is not None
    _activate(q1.query_reliability_conditioner)
    x = torch.randn(2, 4, 64)
    cond = _condition()
    donor_x = x.flip(0).clone()
    donor_cond = copy.deepcopy(cond)
    donor_cond["channel_query_qc"] = torch.tensor([[0.9, 0.7, 0.5, 0.3], [0.2, 0.4, 0.6, 0.8]])

    with torch.no_grad():
        observed = q1(x, cond)
        wrong_query = q1(
            x,
            cond,
            query_reliability_wrong_query_x=donor_x,
            query_reliability_wrong_query_cond=donor_cond,
        )
    assert not torch.equal(observed.logits, wrong_query.logits)
    assert wrong_query.aux["query_reliability_wrong_query_intervention"].all()
    assert not wrong_query.aux["query_reliability_identity_intervention"].any()

    with pytest.raises(ValueError, match="requires both donor x and cond"):
        q1(x, cond, query_reliability_wrong_query_x=donor_x)

    mismatched = copy.deepcopy(donor_cond)
    mismatched["channel_mask"][0, -1] = False
    with pytest.raises(ValueError, match="field channel_mask differs"):
        q1(
            x,
            cond,
            query_reliability_wrong_query_x=donor_x,
            query_reliability_wrong_query_cond=mismatched,
        )

    q1.train()
    with pytest.raises(ValueError, match="evaluation-only"):
        q1(
            x,
            query_reliability_wrong_query_x=donor_x,
            query_reliability_wrong_query_cond=donor_cond,
            cond=cond,
        )


def test_operator_is_symmetric_bounded_padding_safe_and_equivariant() -> None:
    torch.manual_seed(17)
    operator = QueryReliabilityOperator(hidden_dim=8, operator_rank=2).eval()
    _activate(operator)
    x = torch.randn(2, 4, 64)
    cond = _condition()
    cond["channel_mask"][:, 3] = False
    cond["channel_query_qc_missing"][:, 3] = True
    state = operator(x, cond)
    delta = state.operator_delta

    torch.testing.assert_close(delta, delta.transpose(1, 2), rtol=1e-6, atol=1e-7)
    assert delta[:, 3].eq(0.0).all()
    assert delta[:, :, 3].eq(0.0).all()
    norms = torch.linalg.vector_norm(delta.float().flatten(start_dim=1), dim=-1)
    assert torch.all(norms <= 0.200001)

    padded_x = x.clone()
    padded_x[:, 3] = 1e6
    padded_cond = copy.deepcopy(cond)
    padded_cond["channel_query_qc"][:, 3] = 1e6
    padded_cond["channel_query_qc_missing"][:, 3] = False
    padded = operator(padded_x, padded_cond)
    torch.testing.assert_close(
        padded.operator_delta[:, :3, :3], delta[:, :3, :3], rtol=1e-6, atol=1e-7
    )

    permutation = torch.tensor([2, 0, 1, 3])
    permuted_cond = copy.deepcopy(cond)
    for name in ("channel_mask", "channel_query_qc", "channel_query_qc_missing"):
        permuted_cond[name] = permuted_cond[name][:, permutation]
    permuted = operator(x[:, permutation], permuted_cond)
    expected = delta[:, permutation][:, :, permutation]
    torch.testing.assert_close(permuted.operator_delta, expected, rtol=1e-5, atol=1e-6)

    alone = operator(
        x[:1],
        {
            k: v[:1] if torch.is_tensor(v) and v.ndim and v.shape[0] == 2 else v
            for k, v in cond.items()
        },
    )
    torch.testing.assert_close(alone.operator_delta, delta[:1], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("mode", ["null", "observed"])
@pytest.mark.parametrize("padded_value", [1e6, float("nan"), float("inf")])
def test_padded_channel_values_cannot_change_end_to_end_logits(
    mode: str, padded_value: float
) -> None:
    torch.manual_seed(19)
    model = ConditionedEEGDecoder(_config(mode), _vocab_sizes()).eval()
    if mode == "observed":
        assert model.query_reliability_conditioner is not None
        _activate(model.query_reliability_conditioner)
    cond = _condition()
    cond["channel_mask"][:, -1] = False
    cond["channel_query_qc_missing"][:, -1] = True
    clean = torch.randn(2, 4, 64)
    mutated = clean.clone()
    mutated[:, -1] = padded_value
    mutated_cond = copy.deepcopy(cond)
    mutated_cond["channel_query_qc"][:, -1] = float("nan")
    with torch.no_grad():
        expected = model(clean, cond).logits
        observed = model(mutated, mutated_cond).logits
    torch.testing.assert_close(expected, observed, rtol=0.0, atol=0.0)


def test_waveform_band_q_features_and_spatial_operator_add_no_frequency_support() -> None:
    time = torch.arange(400, dtype=torch.float32) / 200.0
    high_only = torch.sin(2.0 * torch.pi * 80.0 * time).reshape(1, 1, -1)
    zero = torch.zeros_like(high_only)
    kwargs = {
        "channel_mask": torch.ones(1, 1, dtype=torch.bool),
        "channel_query_qc": torch.ones(1, 1),
        "channel_query_missing": torch.zeros(1, 1, dtype=torch.bool),
        "target_sfreq": 200.0,
        "min_frequency_hz": 6.0,
        "max_frequency_hz": 60.0,
    }
    high_features = extract_query_reliability_features_v3(high_only, **kwargs)
    zero_features = extract_query_reliability_features_v3(zero, **kwargs)
    torch.testing.assert_close(high_features, zero_features, atol=1e-6, rtol=0.0)

    low = torch.sin(2.0 * torch.pi * 10.0 * time).reshape(1, 1, -1)
    low_plus_high = low + 5.0 * high_only
    low_features = extract_query_reliability_features_v3(low, **kwargs)
    mixed_features = extract_query_reliability_features_v3(low_plus_high, **kwargs)
    torch.testing.assert_close(low_features, mixed_features, atol=2e-5, rtol=1e-5)

    x = torch.stack(
        [
            torch.sin(2.0 * torch.pi * 10.0 * time),
            torch.sin(2.0 * torch.pi * 15.0 * time),
            torch.sin(2.0 * torch.pi * 10.0 * time + 0.4),
            torch.sin(2.0 * torch.pi * 15.0 * time + 0.7),
        ]
    ).unsqueeze(0)
    cond = _condition(batch=1)
    operator = QueryReliabilityOperator(hidden_dim=8, operator_rank=2).eval()
    _activate(operator)
    transformed = operator(x, cond).apply(x)
    power = torch.fft.rfft(transformed, dim=-1).abs().square()
    frequencies = torch.fft.rfftfreq(400, d=1.0 / 200.0)
    outside = ~((frequencies == 10.0) | (frequencies == 15.0))
    assert power[..., outside].amax() <= power.amax() * 1e-9
