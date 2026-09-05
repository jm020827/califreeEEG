from __future__ import annotations

import copy

import pytest
import torch

from cfeg.constants import CATEGORICAL_VOCABS
from cfeg.models.full_model import ConditionedEEGDecoder
from cfeg.models.reliability_conditioning import ResidualizedReliabilityOperator
from cfeg.train_loop import _trainable_state_sha256


def test_initial_state_hash_supports_trainable_scalar_parameters() -> None:
    model = torch.nn.Module()
    model.register_parameter("scalar", torch.nn.Parameter(torch.tensor(0.0)))
    model.register_parameter("vector", torch.nn.Parameter(torch.tensor([1.0, 2.0])))

    first = _trainable_state_sha256(model)
    second = _trainable_state_sha256(model)
    with torch.no_grad():
        model.scalar.fill_(0.5)

    assert first == second
    assert _trainable_state_sha256(model) != first


def _vocab_sizes() -> dict[str, int]:
    return {name: len(values) for name, values in CATEGORICAL_VOCABS.items()}


def _condition(*, metadata_mode: str = "observed", batch: int = 2, channels: int = 4):
    cond = {
        "continuous": torch.zeros(batch, 5),
        "continuous_missing": torch.ones(batch, 5, dtype=torch.bool),
        "channel_ids": torch.arange(1, channels + 1).repeat(batch, 1),
        "channel_mask": torch.ones(batch, channels, dtype=torch.bool),
        "sfreq_processed_float": torch.full((batch,), 200.0),
        "external_continuous": torch.tensor([[0.35, 0.75]]).repeat(batch, 1),
        "external_continuous_missing": torch.zeros(batch, 2, dtype=torch.bool),
        "query_qc": torch.tensor([[0.3]]).repeat(batch, 1),
        "query_qc_missing": torch.zeros(batch, 1, dtype=torch.bool),
        "channel_query_qc": torch.linspace(0.1, 0.7, channels).repeat(batch, 1),
        "channel_query_qc_missing": torch.zeros(batch, channels, dtype=torch.bool),
        "channel_impedance": torch.linspace(0.1, 0.8, channels).repeat(batch, 1),
        "channel_impedance_missing": torch.zeros(batch, channels, dtype=torch.bool),
        "metadata_contract_version": "0.4-dev",
        "external_metadata_mode": metadata_mode,
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


def _config(query_mode: str, metadata_mode: str) -> dict:
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
                "architecture": "reliability_spatial_v1",
                "hidden_dim": 8,
                "spatial_reliability": {
                    "placement": "pre_backbone_waveform",
                    "operator_family": "diagonal_low_rank",
                    "operator_rank": 2,
                    "max_operator_norm": 0.25,
                    "query_operator_norm": 0.20,
                    "metadata_operator_norm": 0.05,
                    "metadata_alpha_limit": 1.0,
                    "query_qc_mode": query_mode,
                    "metadata_residual_enabled": True,
                },
            },
            "condition_encoder": {
                "enabled": True,
                "n_prompt_tokens": 0,
                "fields": ["electrode_type"],
                "include_continuous": True,
                "include_channels": False,
                "force_missing": False,
                "external_metadata_mode": metadata_mode,
            },
            "adapter": {"enabled": False},
            "latent": {"enabled": False},
        },
    }


def _operator(query_mode: str, metadata_mode: str) -> ResidualizedReliabilityOperator:
    return ResidualizedReliabilityOperator(
        d_model=16,
        vocab_sizes=_vocab_sizes(),
        fields=["electrode_type"],
        external_metadata_mode=metadata_mode,
        hidden_dim=8,
        operator_rank=2,
        max_operator_norm=0.25,
        query_operator_norm=0.20,
        metadata_operator_norm=0.05,
        query_qc_mode=query_mode,
    )


def _activate_query(operator: ResidualizedReliabilityOperator) -> None:
    with torch.no_grad():
        operator.query_diagonal_score.weight.normal_(mean=0.0, std=0.3)
        operator.query_diagonal_score.bias.fill_(0.1)
        operator.query_left_factor.weight.normal_(mean=0.0, std=0.2)
        operator.query_left_factor.bias.normal_(mean=0.0, std=0.1)


def _activate_metadata(operator: ResidualizedReliabilityOperator) -> None:
    with torch.no_grad():
        operator.metadata_alpha_raw.fill_(0.8)


def _all_metadata_missing(cond: dict) -> dict:
    out = {key: value.clone() if torch.is_tensor(value) else value for key, value in cond.items()}
    for name in ("reference", "electrode_type", "cap_type"):
        out[name].zero_()
    out["external_continuous"].zero_()
    out["external_continuous_missing"].fill_(True)
    out["channel_impedance"].zero_()
    out["channel_impedance_missing"].fill_(True)
    return out


def _slice_batch(cond: dict, start: int, stop: int) -> dict:
    batch = int(cond["channel_mask"].shape[0])
    return {
        key: value[start:stop].clone()
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch
        else value
        for key, value in cond.items()
    }


def _branch_grad(operator: ResidualizedReliabilityOperator, prefix: str) -> float:
    return float(
        sum(
            parameter.grad.abs().sum().item()
            for name, parameter in operator.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        )
    )


def test_four_arms_share_parameters_and_start_as_the_same_signal_function() -> None:
    arms = {
        "A0": ("null", "null"),
        "A_M": ("null", "observed"),
        "A_Q": ("observed", "null"),
        "A_QM": ("observed", "observed"),
    }
    models = {}
    for name, modes in arms.items():
        torch.manual_seed(7)
        models[name] = ConditionedEEGDecoder(_config(*modes), _vocab_sizes()).eval()

    reference_state = models["A0"].state_dict()
    for model in models.values():
        state = model.state_dict()
        assert state.keys() == reference_state.keys()
        for key in state:
            torch.testing.assert_close(state[key], reference_state[key], rtol=0.0, atol=0.0)

    x = torch.randn(2, 4, 64)
    outputs = {}
    with torch.no_grad():
        for name, (_query_mode, metadata_mode) in arms.items():
            outputs[name] = models[name](x, _condition(metadata_mode=metadata_mode))

    reference_logits = outputs["A0"].logits
    for output in outputs.values():
        torch.testing.assert_close(output.logits, reference_logits, rtol=0.0, atol=0.0)
        assert output.cond_vec is None
        assert output.aux["spatial_operator_frobenius_norm"].eq(0.0).all()


def test_operator_is_pre_backbone_symmetric_bounded_and_padding_safe() -> None:
    torch.manual_seed(11)
    operator = _operator("observed", "observed").eval()
    _activate_query(operator)
    _activate_metadata(operator)
    x = torch.randn(2, 4, 64)
    cond = _condition()
    cond["channel_mask"][:, 3] = False
    cond["channel_ids"][:, 3] = 0
    cond["channel_query_qc_missing"][:, 3] = True
    cond["channel_impedance_missing"][:, 3] = True

    state = operator(x, cond)
    delta = state.operator_delta
    torch.testing.assert_close(delta, delta.transpose(1, 2), rtol=1e-6, atol=1e-7)
    assert delta[:, 3].eq(0.0).all()
    assert delta[:, :, 3].eq(0.0).all()
    norms = torch.linalg.vector_norm(delta.float().flatten(start_dim=1), dim=-1)
    assert torch.all(norms <= 0.250001)
    query_norms = torch.linalg.vector_norm(
        state.query_operator_delta.float().flatten(start_dim=1), dim=-1
    )
    metadata_norms = torch.linalg.vector_norm(
        state.metadata_operator_delta.float().flatten(start_dim=1), dim=-1
    )
    assert torch.all(query_norms <= 0.200001)
    assert torch.all(metadata_norms <= 0.050001)
    eigenvalues = torch.linalg.eigvalsh(
        torch.eye(4).unsqueeze(0) + delta.float()
    )
    assert eigenvalues.min() >= 0.74999
    assert eigenvalues.max() <= 1.25001

    masked_x = x * cond["channel_mask"].unsqueeze(-1)
    distortion = torch.linalg.vector_norm((state.apply(x) - x).flatten(start_dim=1), dim=-1)
    source_norm = torch.linalg.vector_norm(masked_x.flatten(start_dim=1), dim=-1)
    assert torch.all(distortion <= 0.25001 * source_norm)

    padded_x = x.clone()
    padded_x[:, 3] = 1e6
    padded_cond = copy.deepcopy(cond)
    padded_cond["channel_query_qc"][:, 3] = 1e6
    padded_cond["channel_query_qc_missing"][:, 3] = False
    padded_cond["channel_impedance"][:, 3] = 1e6
    padded_cond["channel_impedance_missing"][:, 3] = False
    padded_state = operator(padded_x, padded_cond)
    torch.testing.assert_close(
        padded_state.operator_delta[:, :3, :3],
        delta[:, :3, :3],
        rtol=1e-6,
        atol=1e-7,
    )


def test_query_and_metadata_branches_have_separate_access_contracts() -> None:
    torch.manual_seed(13)
    query_only = _operator("observed", "null").eval()
    _activate_query(query_only)
    x = torch.randn(2, 4, 64)
    cond = _condition(metadata_mode="null")
    changed_x = x.clone()
    changed_x[:, 0].zero_()
    baseline = query_only(x, cond)
    corrupted = query_only(changed_x, cond)
    assert not torch.allclose(baseline.query_noise_logit, corrupted.query_noise_logit)
    assert not torch.allclose(baseline.operator_delta, corrupted.operator_delta)

    changed_metadata = copy.deepcopy(cond)
    changed_metadata["channel_impedance"].flip(dims=[1])
    changed_metadata["electrode_type"].fill_(2)
    unchanged_query = query_only(x, changed_metadata)
    torch.testing.assert_close(
        baseline.operator_delta,
        unchanged_query.operator_delta,
        rtol=0.0,
        atol=0.0,
    )

    metadata_only = _operator("null", "observed").eval()
    _activate_metadata(metadata_only)
    metadata_baseline = metadata_only(x, _condition())
    metadata_changed_x = metadata_only(changed_x, _condition())
    torch.testing.assert_close(
        metadata_baseline.operator_delta,
        metadata_changed_x.operator_delta,
        rtol=0.0,
        atol=0.0,
    )
    changed = _condition()
    changed["channel_impedance"] = torch.flip(changed["channel_impedance"], dims=[1])
    metadata_changed = metadata_only(x, changed)
    assert not torch.allclose(
        metadata_baseline.metadata_noise_logit,
        metadata_changed.metadata_noise_logit,
    )
    assert not torch.allclose(metadata_baseline.operator_delta, metadata_changed.operator_delta)


def test_metadata_is_a_separately_bounded_additive_residual() -> None:
    torch.manual_seed(15)
    operator = _operator("observed", "observed").eval()
    _activate_query(operator)
    _activate_metadata(operator)
    x = torch.randn(2, 4, 64)
    observed = operator(x, _condition())
    missing = operator(x, _all_metadata_missing(_condition()))

    torch.testing.assert_close(
        observed.query_operator_delta,
        missing.query_operator_delta,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        observed.operator_delta,
        observed.query_operator_delta + observed.metadata_operator_delta,
        rtol=0.0,
        atol=0.0,
    )
    metadata_norm = torch.linalg.vector_norm(
        observed.metadata_operator_delta.float().flatten(start_dim=1), dim=-1
    )
    assert torch.all(metadata_norm <= 0.050001)


def test_trained_all_missing_metadata_falls_back_exactly_to_query_only() -> None:
    torch.manual_seed(17)
    aqm = ConditionedEEGDecoder(_config("observed", "observed"), _vocab_sizes()).eval()
    assert aqm.reliability_conditioner is not None
    _activate_query(aqm.reliability_conditioner)
    _activate_metadata(aqm.reliability_conditioner)
    aq = ConditionedEEGDecoder(_config("observed", "null"), _vocab_sizes()).eval()
    aq.load_state_dict(aqm.state_dict())
    x = torch.randn(2, 4, 64)

    null_cond = _condition(metadata_mode="null")
    missing_cond = _all_metadata_missing(_condition(metadata_mode="observed"))
    with torch.no_grad():
        query_only = aq(x, null_cond)
        missing = aqm(x, missing_cond)

    torch.testing.assert_close(query_only.logits, missing.logits, rtol=0.0, atol=0.0)
    for key in (
        "channel_spatial_self_gain",
        "spatial_offdiagonal_row_l1",
        "channel_query_noise_logit",
        "channel_combined_noise_logit",
    ):
        torch.testing.assert_close(query_only.aux[key], missing.aux[key], rtol=0.0, atol=0.0)
    assert missing.aux["channel_metadata_available_count"].eq(0).all()


def test_operator_and_model_are_channel_permutation_equivariant() -> None:
    torch.manual_seed(19)
    model = ConditionedEEGDecoder(_config("observed", "observed"), _vocab_sizes()).eval()
    assert model.reliability_conditioner is not None
    _activate_query(model.reliability_conditioner)
    _activate_metadata(model.reliability_conditioner)
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
        state = model.reliability_conditioner(x, cond)
        permuted_state = model.reliability_conditioner(x[:, permutation], permuted)
        baseline_logits = model(x, cond).logits
        permuted_logits = model(x[:, permutation], permuted).logits

    expected_delta = state.operator_delta[:, permutation][:, :, permutation]
    torch.testing.assert_close(
        permuted_state.operator_delta,
        expected_delta,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(baseline_logits, permuted_logits, rtol=1e-5, atol=1e-6)


def test_prediction_is_independent_of_batch_partners() -> None:
    torch.manual_seed(23)
    model = ConditionedEEGDecoder(_config("observed", "observed"), _vocab_sizes()).eval()
    assert model.reliability_conditioner is not None
    _activate_query(model.reliability_conditioner)
    _activate_metadata(model.reliability_conditioner)
    x = torch.randn(2, 4, 64)
    cond = _condition()
    cond["channel_impedance"][1] = torch.tensor([0.9, 0.2, 0.6, 0.1])

    with torch.no_grad():
        batched = model(x, cond)
        alone = model(x[:1], _slice_batch(cond, 0, 1))

    torch.testing.assert_close(batched.logits[:1], alone.logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        batched.aux["channel_spatial_self_gain"][:1],
        alone.aux["channel_spatial_self_gain"],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("query_mode", "metadata_mode", "expect_query", "expect_metadata"),
    [
        ("null", "null", False, False),
        ("null", "observed", False, True),
        ("observed", "null", True, False),
        ("observed", "observed", True, True),
    ],
)
def test_access_modes_gate_first_step_gradients(
    query_mode: str,
    metadata_mode: str,
    expect_query: bool,
    expect_metadata: bool,
) -> None:
    torch.manual_seed(29)
    model = ConditionedEEGDecoder(_config(query_mode, metadata_mode), _vocab_sizes()).train()
    x = torch.randn(2, 4, 64)
    model(x, _condition(metadata_mode=metadata_mode)).logits.square().mean().backward()
    operator = model.reliability_conditioner
    assert operator is not None

    query_grad = _branch_grad(operator, "query_")
    alpha_grad = operator.metadata_alpha_raw.grad
    observed_alpha_grad = 0.0 if alpha_grad is None else float(alpha_grad.abs().item())
    assert (query_grad > 0.0) is expect_query
    assert (observed_alpha_grad > 0.0) is expect_metadata
    # Alpha is zero at step 0, so metadata encoder/head gradients may open on step 1.
    assert _branch_grad(operator, "metadata_encoder") == 0.0


def test_time_invariant_spatial_operator_does_not_create_temporal_fft_bins() -> None:
    torch.manual_seed(31)
    operator = _operator("observed", "observed").eval()
    _activate_query(operator)
    _activate_metadata(operator)
    n_time = 64
    index = torch.arange(n_time, dtype=torch.float32)
    x = torch.stack(
        [
            torch.sin(2.0 * torch.pi * 5.0 * index / n_time),
            torch.cos(2.0 * torch.pi * 5.0 * index / n_time),
            torch.sin(2.0 * torch.pi * 7.0 * index / n_time),
            torch.cos(2.0 * torch.pi * 7.0 * index / n_time),
        ]
    ).unsqueeze(0)
    cond = _condition(batch=1)
    transformed = operator(x, cond).apply(x)
    spectrum = torch.fft.rfft(transformed, dim=-1).abs()
    absent = torch.ones(spectrum.shape[-1], dtype=torch.bool)
    absent[[5, 7]] = False
    assert spectrum[..., absent].max() < 1e-4


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model", "latent", "enabled"), True, "spatial-operator-only"),
        (("model", "condition_encoder", "n_prompt_tokens"), 1, "n_prompt_tokens=0"),
        (
            ("model", "conditioning", "spatial_reliability", "placement"),
            "token",
            "pre_backbone_waveform",
        ),
        (("model", "condition_encoder", "force_missing"), True, "force_missing=true"),
        (
            ("model", "conditioning", "spatial_reliability", "metadata_operator_norm"),
            0.0,
            "metadata_operator_norm",
        ),
        (
            ("model", "conditioning", "spatial_reliability", "metadata_alpha_limit"),
            0.0,
            "metadata_alpha_limit",
        ),
        (("model", "backbone", "name"), "tiny_transformer", "spectral_transformer"),
    ],
)
def test_invalid_reliability_routes_are_rejected(path: tuple[str, ...], value, message: str) -> None:
    cfg = _config("observed", "observed")
    target = cfg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path == ("model", "backbone", "name"):
        cfg["model"]["patch_size"] = 16

    with pytest.raises(ValueError, match=message):
        ConditionedEEGDecoder(cfg, _vocab_sizes())
