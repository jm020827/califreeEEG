from __future__ import annotations

import copy

import torch

from cfeg.constants import CATEGORICAL_VOCABS
from cfeg.models.full_model import ConditionedEEGDecoder
from cfeg.models.physical_conditioning import (
    FactorizedPhysicalConditioner,
    ZeroInitResidualFiLM,
)


def _vocab_sizes() -> dict[str, int]:
    return {name: len(values) for name, values in CATEGORICAL_VOCABS.items()}


def _condition(*, mode: str = "observed", batch: int = 2, channels: int = 4):
    cond = {
        "continuous": torch.zeros(batch, 5),
        "continuous_missing": torch.ones(batch, 5, dtype=torch.bool),
        "channel_ids": torch.arange(1, channels + 1).repeat(batch, 1),
        "channel_mask": torch.ones(batch, channels, dtype=torch.bool),
        "sfreq_processed_float": torch.full((batch,), 200.0),
        "external_continuous": torch.tensor([[0.4, 0.7]]).repeat(batch, 1),
        "external_continuous_missing": torch.zeros(batch, 2, dtype=torch.bool),
        "query_qc": torch.tensor([[0.3]]).repeat(batch, 1),
        "query_qc_missing": torch.zeros(batch, 1, dtype=torch.bool),
        "channel_query_qc": torch.zeros(batch, channels),
        "channel_query_qc_missing": torch.ones(batch, channels, dtype=torch.bool),
        "channel_impedance": torch.linspace(0.1, 0.8, channels).repeat(batch, 1),
        "channel_impedance_missing": torch.zeros(batch, channels, dtype=torch.bool),
        "metadata_contract_version": "0.4-dev",
        "external_metadata_mode": mode,
        "query_qc_extractor_version": (
            "filtered_cropped_pre_zscore_channel_std_median_v1"
        ),
        "external_continuous_schema": "impedance_mean_max_v1",
        "development_control": "none",
    }
    for name in CATEGORICAL_VOCABS:
        cond[name] = torch.zeros(batch, dtype=torch.long)
    cond["electrode_type"].fill_(1)
    cond["reference"].fill_(1)
    cond["cap_type"].fill_(1)
    return cond


def _config(mode: str) -> dict:
    return {
        "protocol": {
            "metadata_contract_version": "0.4-dev",
            "query_qc_extractor_version": (
                "filtered_cropped_pre_zscore_channel_std_median_v1"
            ),
            "external_continuous_schema": "impedance_mean_max_v1",
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
                "architecture": "physical_hybrid_v1",
                "hidden_dim": 8,
                "common_query_film": {
                    "enabled": True,
                    "max_scale_delta": 0.25,
                    "max_shift": 0.25,
                },
                "external_global_film": {
                    "enabled": True,
                    "max_scale_delta": 0.25,
                    "max_shift": 0.25,
                },
                "channel_quality": {
                    "enabled": True,
                    "source": "channel_impedance",
                    "placement": "pre_channel_embedding",
                    "max_gain_delta": 0.25,
                },
            },
            "condition_encoder": {
                "enabled": True,
                "n_prompt_tokens": 0,
                "fields": ["electrode_type"],
                "include_continuous": True,
                "include_channels": False,
                "external_metadata_mode": mode,
            },
            "adapter": {"enabled": False},
            "latent": {"enabled": False},
        },
    }


def _all_missing(cond: dict) -> dict:
    out = {key: value.clone() if torch.is_tensor(value) else value for key, value in cond.items()}
    for name in ("reference", "electrode_type", "cap_type"):
        out[name].zero_()
    out["external_continuous"].zero_()
    out["external_continuous_missing"].fill_(True)
    out["channel_impedance"].zero_()
    out["channel_impedance_missing"].fill_(True)
    return out


def _activate_external_modules(model: ConditionedEEGDecoder) -> None:
    assert model.physical_conditioner is not None
    assert model.external_film is not None
    with torch.no_grad():
        model.physical_conditioner.channel_score.weight.normal_(mean=0.2, std=0.1)
        model.physical_conditioner.channel_score.bias.fill_(0.1)
        model.external_film.to_affine.weight.normal_(mean=0.0, std=0.1)
        model.external_film.to_affine.bias.normal_(mean=0.0, std=0.1)


def test_physical_conditioner_null_unknown_and_missing_are_neutral() -> None:
    observed = FactorizedPhysicalConditioner(
        d_model=16,
        vocab_sizes=_vocab_sizes(),
        fields=["electrode_type"],
        external_metadata_mode="observed",
        hidden_dim=8,
    ).eval()
    null = FactorizedPhysicalConditioner(
        d_model=16,
        vocab_sizes=_vocab_sizes(),
        fields=["electrode_type"],
        external_metadata_mode="null",
        hidden_dim=8,
    ).eval()
    null.load_state_dict(observed.state_dict())
    cond = _condition()

    initial = observed(cond)
    assert initial.channel_gain.eq(1.0).all()
    assert initial.external_available.all()

    null_state = null(cond)
    assert null_state.external_vec.eq(0.0).all()
    assert not null_state.external_available.any()
    assert null_state.channel_gain.eq(1.0).all()
    assert not null_state.channel_available.any()

    unknown = _all_missing(cond)
    unknown["electrode_type"].fill_(999)
    unknown_state = observed(unknown)
    missing_state = observed(_all_missing(cond))
    torch.testing.assert_close(unknown_state.external_vec, missing_state.external_vec)
    torch.testing.assert_close(unknown_state.channel_gain, missing_state.channel_gain)


def test_channel_gain_is_bounded_and_missing_or_inactive_channels_are_neutral() -> None:
    conditioner = FactorizedPhysicalConditioner(
        d_model=16,
        vocab_sizes=_vocab_sizes(),
        fields=["electrode_type"],
        external_metadata_mode="observed",
        hidden_dim=8,
        max_channel_gain_delta=0.25,
    ).eval()
    with torch.no_grad():
        conditioner.channel_score.weight.fill_(100.0)
        conditioner.channel_score.bias.fill_(100.0)
    cond = _condition()
    cond["channel_impedance_missing"][:, 1] = True
    cond["channel_mask"][:, 2] = False

    state = conditioner(cond)

    assert state.channel_gain.min() >= 0.75
    assert state.channel_gain.max() <= 1.25
    assert state.channel_gain[:, 1].eq(1.0).all()
    assert state.channel_gain[:, 2].eq(1.0).all()


def test_zero_init_film_is_identity_and_absence_remains_identity_after_training() -> None:
    film = ZeroInitResidualFiLM(8, max_scale_delta=0.25, max_shift=0.25)
    h = torch.randn(3, 8)
    context = torch.randn(3, 8)
    available = torch.ones(3, 1, dtype=torch.bool)

    torch.testing.assert_close(film(h, context, available), h, rtol=0.0, atol=0.0)

    with torch.no_grad():
        film.to_affine.weight.normal_()
        film.to_affine.bias.normal_()
    absent = torch.zeros_like(available)
    torch.testing.assert_close(film(h, context, absent), h, rtol=0.0, atol=0.0)
    assert not torch.allclose(film(h, context, available), h)


def test_physical_a0_a2_share_schema_and_start_from_identical_signal_function() -> None:
    torch.manual_seed(7)
    a0 = ConditionedEEGDecoder(_config("null"), _vocab_sizes()).eval()
    torch.manual_seed(7)
    a2 = ConditionedEEGDecoder(_config("observed"), _vocab_sizes()).eval()
    assert a0.state_dict().keys() == a2.state_dict().keys()
    assert {
        name: tuple(value.shape) for name, value in a0.state_dict().items()
    } == {name: tuple(value.shape) for name, value in a2.state_dict().items()}

    x = torch.randn(2, 4, 64)
    with torch.no_grad():
        out_a0 = a0(x, _condition(mode="null"))
        out_a2 = a2(x, _condition(mode="observed"))

    assert out_a0.prompt_tokens is None
    assert out_a2.prompt_tokens is None
    torch.testing.assert_close(out_a0.logits, out_a2.logits, rtol=0.0, atol=0.0)


def test_trained_external_modules_still_fall_back_exactly_when_all_missing() -> None:
    torch.manual_seed(11)
    a2 = ConditionedEEGDecoder(_config("observed"), _vocab_sizes()).eval()
    _activate_external_modules(a2)
    a0 = ConditionedEEGDecoder(_config("null"), _vocab_sizes()).eval()
    a0.load_state_dict(a2.state_dict())
    x = torch.randn(2, 4, 64)

    null_cond = _condition(mode="null")
    missing_cond = _all_missing(_condition(mode="observed"))
    with torch.no_grad():
        null_logits = a0(x, null_cond).logits
        missing_logits = a2(x, missing_cond).logits

    torch.testing.assert_close(null_logits, missing_logits, rtol=0.0, atol=0.0)


def test_allowed_physical_fields_reach_logits_but_forbidden_fields_do_not() -> None:
    torch.manual_seed(13)
    model = ConditionedEEGDecoder(_config("observed"), _vocab_sizes()).eval()
    _activate_external_modules(model)
    x = torch.randn(2, 4, 64)
    baseline = _condition()

    forbidden = copy.deepcopy(baseline)
    forbidden["reference"].fill_(2)
    forbidden["cap_type"].fill_(2)
    forbidden["hardware_id"].fill_(2)

    electrode = copy.deepcopy(baseline)
    electrode["electrode_type"].fill_(2)
    global_impedance = copy.deepcopy(baseline)
    global_impedance["external_continuous"].add_(0.5)
    channel_impedance = copy.deepcopy(baseline)
    channel_impedance["channel_impedance"][:, 0].add_(0.5)

    with torch.no_grad():
        base_logits = model(x, baseline).logits
        forbidden_logits = model(x, forbidden).logits
        electrode_logits = model(x, electrode).logits
        global_logits = model(x, global_impedance).logits
        channel_logits = model(x, channel_impedance).logits

    torch.testing.assert_close(base_logits, forbidden_logits, rtol=0.0, atol=0.0)
    assert not torch.allclose(base_logits, electrode_logits)
    assert not torch.allclose(base_logits, global_logits)
    assert not torch.allclose(base_logits, channel_logits)


def test_physical_channel_path_is_permutation_equivariant() -> None:
    torch.manual_seed(17)
    model = ConditionedEEGDecoder(_config("observed"), _vocab_sizes()).eval()
    _activate_external_modules(model)
    x = torch.randn(2, 4, 64)
    cond = _condition()
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = copy.deepcopy(cond)
    for field in (
        "channel_ids",
        "channel_mask",
        "channel_impedance",
        "channel_impedance_missing",
        "channel_query_qc",
        "channel_query_qc_missing",
    ):
        permuted[field] = permuted[field][:, permutation]

    with torch.no_grad():
        baseline = model(x, cond).logits
        changed = model(x[:, permutation], permuted).logits

    torch.testing.assert_close(baseline, changed, rtol=1e-5, atol=1e-6)


def test_zero_init_external_paths_receive_gradient_only_when_observed() -> None:
    x = torch.randn(2, 4, 64)
    observed = ConditionedEEGDecoder(_config("observed"), _vocab_sizes()).train()
    observed(x, _condition(mode="observed")).logits.square().mean().backward()
    assert observed.physical_conditioner is not None
    assert observed.external_film is not None
    assert observed.physical_conditioner.channel_score.weight.grad is not None
    assert observed.physical_conditioner.channel_score.weight.grad.abs().sum() > 0
    assert observed.external_film.to_affine.weight.grad is not None
    assert observed.external_film.to_affine.weight.grad.abs().sum() > 0

    null = ConditionedEEGDecoder(_config("null"), _vocab_sizes()).train()
    null(x, _condition(mode="null")).logits.square().mean().backward()
    assert null.physical_conditioner is not None
    assert null.external_film is not None
    channel_grad = null.physical_conditioner.channel_score.weight.grad
    film_grad = null.external_film.to_affine.weight.grad
    assert channel_grad is None or channel_grad.abs().sum() == 0
    assert film_grad is None or film_grad.abs().sum() == 0
    assert null.backbone.spectral_projection[1].weight.grad is not None
    assert null.backbone.spectral_projection[1].weight.grad.abs().sum() > 0


def test_physical_metadata_only_control_uses_external_vector_not_eeg_or_query() -> None:
    cfg = _config("observed")
    cfg["protocol"]["development_control"] = "metadata_only"
    model = ConditionedEEGDecoder(cfg, _vocab_sizes()).eval()
    cond = _condition(mode="observed")
    cond["development_control"] = "metadata_only"
    changed = copy.deepcopy(cond)
    changed["query_qc"].fill_(1.7)
    changed["query_qc_missing"].logical_not_()
    changed["channel_ids"] = torch.flip(changed["channel_ids"], dims=[1])
    changed["channel_mask"].zero_()

    with torch.no_grad():
        first = model(torch.randn(2, 4, 64), cond).logits
        second = model(torch.randn(2, 4, 64), changed).logits

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
