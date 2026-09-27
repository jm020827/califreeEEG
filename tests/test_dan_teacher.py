"""Generated-only core checks; no participant, manifest or EEG-cache IO.

One complete invocation uses 120 optimizer updates: CPU10 + CUDA10 + profile100.
The profile is partial engineering evidence, not a full-run feasibility gate.
"""

from __future__ import annotations

import copy
import io
import json
import time

import numpy as np
import pytest
import torch

from cfeg.analysis.dan_teacher import (
    SourceChannelScaler,
    logged_impedance,
    support_quality,
    teacher_block_weights,
)
from cfeg.models.dan_alignment import DanAlignment, alignment_loss, weighted_teacher


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def example():
    torch.manual_seed(17)
    return DanAlignment(2, 24).double(), torch.randn(3, 3, 2, 24, dtype=torch.float64)


def test_impedance_zero_missing_and_invalid():
    value, mask = logged_impedance(np.array([[0, np.nan], [3, 7]]))
    np.testing.assert_array_equal(mask, [[True, False], [True, True]])
    np.testing.assert_allclose(value, [[0, 0], [np.log(4), np.log(8)]])
    for invalid in (-1, np.inf, -np.inf):
        with pytest.raises(ValueError):
            logged_impedance(np.array([[invalid]]))


def test_source_scaler_no_target_fit_and_layout_roundtrip():
    rng = np.random.default_rng(37)
    original = rng.normal(size=(20, 3))
    original[:, 2] = 4
    flags = np.ones_like(original, dtype=bool)
    flags[0, 1] = False
    original[0, 1] = np.nan
    ids = [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
    kwargs = {"row_subject_ids": ids, "allowed_fit_ids": [1, 2, 3, 4]}
    a = SourceChannelScaler.fit(original, flags, **kwargs)
    buffer = io.BytesIO()
    np.savez(buffer, x=np.asfortranarray(original))
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as packet:
        b = SourceChannelScaler.fit(packet["x"], flags, **kwargs)
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.scale, b.scale)
    assert a.scale[2] == 1 and not a.mean.flags.writeable
    np.testing.assert_allclose(a.scale[0], original[:, 0].std(ddof=0))
    before = a.mean.copy()
    target = np.full((2, 3), 1e6)
    transformed = a.transform(target, np.ones_like(target, dtype=bool))
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(before, a.mean)
    with pytest.raises(ValueError, match="exactly the allowed"):
        SourceChannelScaler.fit(original, flags, row_subject_ids=ids[:-1] + [99],
                                allowed_fit_ids=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="Distinct"):
        SourceChannelScaler.fit(original, flags, row_subject_ids=ids,
                                allowed_fit_ids=[1, 2, 3, 3])


def test_all_missing_scaler_is_explicit_zero_feature():
    x = np.full((4, 2), np.nan)
    flags = np.zeros_like(x, dtype=bool)
    scaler = SourceChannelScaler.fit(x, flags, row_subject_ids=[1, 1, 2, 2],
                                     allowed_fit_ids=[1, 2])
    np.testing.assert_array_equal(scaler.mean, [0, 0])
    np.testing.assert_array_equal(scaler.scale, [1, 1])
    np.testing.assert_array_equal(scaler.transform(x, flags), np.zeros_like(x))


@pytest.mark.parametrize("k", [2, 3, 5])
@pytest.mark.parametrize("mode", ["missing", "constant", "partial_constant"])
def test_degenerate_metadata_is_bit_exact_q(k, mode):
    q = np.random.default_rng(k).normal(size=(k, 4))
    auxiliary = np.tile([0.1, 1e12, -5.3, 0], (k, 1))
    mask = np.ones_like(q, dtype=bool)
    if mode == "missing":
        mask[:] = False
        auxiliary[:] = np.nan
    elif mode == "partial_constant":
        mask[0] = False
        auxiliary[0] = np.nan
    expected = teacher_block_weights(q, "Q")
    for arm in ("Q2", "QM", "SHAM"):
        actual = teacher_block_weights(q, arm, auxiliary=auxiliary, observed=mask)
        np.testing.assert_array_equal(actual, expected)


def test_q_and_u_ignore_auxiliary_and_match_direct_formula():
    q = np.array([[1, 4], [2, -2], [3, 5]], dtype=float)
    got = teacher_block_weights(q, "Q", auxiliary=np.array([np.inf]), observed="bad")
    logits = np.clip(-(q - q.mean(0)), -3, 3)
    mass = np.exp(logits)
    expected = 0.5 / 3 + 0.5 * mass / mass.sum(0)
    np.testing.assert_allclose(got, expected, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(teacher_block_weights(q, "U"), np.full_like(q, 1 / 3))
    assert np.all(got >= 0.5 / 3)
    np.testing.assert_allclose(got.sum(0), 1, rtol=0, atol=2e-16)


def test_partial_missing_auxiliary_uses_only_observed_center():
    q = np.zeros((3, 1))
    auxiliary = np.array([[1], [np.nan], [5]])
    mask = np.isfinite(auxiliary)
    got = teacher_block_weights(q, "QM", auxiliary=auxiliary, observed=mask)
    logits = np.array([[1.0], [0.0], [-1.0]])
    expected = 1 / 6 + 0.5 * np.exp(logits) / np.exp(logits).sum(0)
    np.testing.assert_allclose(got, expected, rtol=0, atol=2e-16)


def test_weights_follow_block_permutation_and_channel_shift():
    rng = np.random.default_rng(29)
    q, aux = rng.normal(size=(2, 5, 3))
    flags = np.ones_like(q, dtype=bool)
    order = np.array([2, 4, 0, 1, 3])
    a = teacher_block_weights(q, "QM", auxiliary=aux, observed=flags)
    b = teacher_block_weights(q[order] + [10, -10, 20], "QM",
                              auxiliary=aux[order] + [1, 2, 3], observed=flags)
    np.testing.assert_allclose(b, a[order], rtol=0, atol=1e-15)


@pytest.mark.parametrize("arm", ["Q2", "QM", "SHAM"])
def test_auxiliary_arms_require_valid_explicit_packet(arm):
    q = np.zeros((2, 3))
    with pytest.raises(ValueError):
        teacher_block_weights(q, arm)
    with pytest.raises(ValueError):
        teacher_block_weights(q, arm, auxiliary=np.full_like(q, np.nan),
                              observed=np.ones_like(q, dtype=bool))
    with pytest.raises(ValueError):
        teacher_block_weights(q[:1], arm, auxiliary=q[:1], observed=q[:1])


def test_quality_stimulus_locking_gain_offset_and_label_order():
    rng = np.random.default_rng(19)
    frequencies = np.array([10, 12, 14])
    time_grid = np.arange(250) / 250
    pure = np.cos(2 * np.pi * frequencies[:, None] * time_grid + 0.31)
    support = np.tile(pure[None, :, None, :], (3, 1, 2, 1))
    support += np.array([0.02, 0.3, 2.0])[:, None, None, None] * rng.normal(
        size=support.shape)
    q, q2 = support_quality(support, frequencies=frequencies, sfreq=250)
    assert q.shape == q2.shape == (3, 2)
    assert np.all(q[0] < q[1]) and np.all(q[1] < q[2])
    scaled = support_quality(7 * support + 23, frequencies=frequencies, sfreq=250)
    for expected, actual in zip((q, q2), scaled):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    order = [2, 0, 1]
    permuted = support_quality(support[:, order], frequencies=frequencies[order], sfreq=250)
    for expected, actual in zip((q, q2), permuted):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    q_prefix, _ = support_quality(support[:2], frequencies=frequencies, sfreq=250)
    np.testing.assert_array_equal(q_prefix, q[:2])
    with pytest.raises(ValueError, match="zero-power"):
        support_quality(np.zeros_like(support), frequencies=frequencies, sfreq=250)


@pytest.mark.parametrize("frequencies,sfreq,harmonics", [
    ([0, 12], 250, 3), ([10, np.nan], 250, 3),
    ([10, 50], 250, 3), ([10, 12], 0, 3), ([10, 12], 250, 0),
])
def test_invalid_stimulus_spec(frequencies, sfreq, harmonics):
    with pytest.raises(ValueError):
        support_quality(np.ones((2, 2, 3, 50)), frequencies=frequencies,
                        sfreq=sfreq, n_harmonics=harmonics)


def test_teacher_direct_formula_class_equivariance_and_original_support():
    _, support = example()
    before = support.clone()
    weights = torch.tensor([[0.2, 0.5], [0.3, 0.3], [0.5, 0.2]], dtype=torch.float64)
    actual = weighted_teacher(support, weights)
    manual = sum(support[block] * weights[block, None, :, None] for block in range(3))
    torch.testing.assert_close(actual, manual, rtol=0, atol=1e-15)
    torch.testing.assert_close(support, before, rtol=0, atol=0)
    order = [2, 0, 1]
    torch.testing.assert_close(weighted_teacher(support[:, order], weights), actual[order])
    uniform = torch.full((3, 2), 1 / 3, dtype=torch.float64)
    torch.testing.assert_close(weighted_teacher(support, uniform), support.mean(0))
    for bad in (weights * 2, -weights, torch.full_like(weights, float("nan"))):
        with pytest.raises(ValueError):
            weighted_teacher(support, bad)


def test_metadata_changes_teacher_and_learned_gradient_not_query_score():
    model, support = example()
    q = np.zeros((3, 2))
    m = np.array([[0, 4], [2, 2], [4, 0]], dtype=float)
    wq = teacher_block_weights(q, "Q")
    wm = teacher_block_weights(q, "QM", auxiliary=m, observed=np.ones_like(m, dtype=bool))
    tq = weighted_teacher(support, torch.from_numpy(wq))
    tm = weighted_teacher(support, torch.from_numpy(wm))
    assert not torch.allclose(tq, tm)
    source = torch.randn(6, 2, 24, dtype=torch.float64)
    labels = torch.arange(6) % 3
    losses, gradients = [], []
    for teacher in (tq, tm):
        replica = copy.deepcopy(model)
        loss = alignment_loss(replica, source, labels, teacher)
        loss.backward()
        losses.append(float(loss.detach()))
        gradients.append(torch.cat([p.grad.flatten() for p in replica.parameters()]))
    assert abs(losses[0] - losses[1]) > 1e-6
    assert torch.linalg.vector_norm(gradients[0] - gradients[1]) > 1e-6


def test_mse_is_source_label_selected_teacher():
    model, support = example()
    model.eval()
    source = torch.randn(7, 2, 24, dtype=torch.float64)
    labels = torch.arange(7) % 3
    teacher = support.mean(0)
    actual = alignment_loss(model, source, labels, teacher)
    expected = ((model(source) - teacher[labels]) ** 2).mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for bad in (labels.float(), labels + 3, labels[:-1]):
        with pytest.raises(ValueError):
            alignment_loss(model, source, bad, teacher)


def test_eval_has_no_time_mixing_batch_dependence_or_bn_updates():
    model, _ = example()
    x = torch.randn(8, 2, 24, dtype=torch.float64)
    model(x)  # One train forward, zero optimizer updates.
    with pytest.raises(RuntimeError):
        model.transform_frozen(x)
    model.eval()
    before = {key: val.clone() for key, val in model.state_dict().items()}
    whole = model.transform_frozen(x)
    separate = torch.cat([model.transform_frozen(row[None]) for row in x])
    torch.testing.assert_close(whole, separate, rtol=0, atol=2e-15)
    changed = x.clone()
    changed[:, :, 5] += 10
    difference = model.transform_frozen(changed) - whole
    other_times = list(range(5)) + list(range(6, 24))
    torch.testing.assert_close(difference[:, :, other_times],
                               torch.zeros_like(difference[:, :, other_times]), rtol=0, atol=0)
    assert difference[:, :, 5].abs().max() > 1e-6
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_model_saved_roundtrip_and_direct_numpy_eval():
    model, _ = example()
    x = torch.randn(5, 2, 24, dtype=torch.float64)
    model(x)
    model.eval()
    packet = io.BytesIO()
    torch.save(model.state_dict(), packet)
    packet.seek(0)
    restored = DanAlignment(2, 24).double().eval()
    restored.load_state_dict(torch.load(packet, weights_only=True))
    expected = model.transform_frozen(x)
    torch.testing.assert_close(restored.transform_frozen(x), expected, rtol=0, atol=0)
    state = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    spatial = x.numpy().transpose(0, 2, 1) @ state["spatial.weight"].T
    flat = spatial.reshape(len(x), -1)
    bn = (flat - state["normalization.running_mean"]) / np.sqrt(
        state["normalization.running_var"] + 1e-5)
    bn = bn * state["normalization.weight"] + state["normalization.bias"]
    hidden = bn.reshape(len(x), 24, 2) @ state["hidden.weight"].T + state["hidden.bias"]
    output = np.tanh(hidden) @ state["output.weight"].T + state["output.bias"]
    np.testing.assert_allclose(output.transpose(0, 2, 1), expected.numpy(), rtol=0, atol=1e-14)


def test_cuda_cpu_forward_and_gradient_agree():
    assert torch.cuda.is_available(), "CUDA qualification must not silently skip."
    model, support = example()
    model = model.float().eval()
    source = support[0].float()
    teacher = support.mean(0).float()
    labels = torch.arange(3)
    gpu = copy.deepcopy(model).cuda()
    cpu_loss = alignment_loss(model, source, labels, teacher)
    gpu_loss = alignment_loss(gpu, source.cuda(), labels.cuda(), teacher.cuda())
    cpu_loss.backward()
    gpu_loss.backward()
    torch.testing.assert_close(cpu_loss, gpu_loss.cpu(), rtol=2e-5, atol=2e-6)
    for cpu_param, gpu_param in zip(model.parameters(), gpu.parameters()):
        torch.testing.assert_close(cpu_param.grad, gpu_param.grad.cpu(), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_generated_optimizer_ten_steps(device):
    model, support = example()
    model = model.float().to(device)
    source = support.reshape(9, 2, 24).float().to(device)
    teacher = support.mean(0).float().to(device)
    labels = (torch.arange(9) % 3).to(device)
    initial = model.spatial.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = alignment_loss(model, source, labels, teacher)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    assert not torch.equal(initial, model.spatial.weight)
    model.eval()
    assert torch.isfinite(model.transform_frozen(source)).all()


def test_generated_cuda_partial_resource_profile():
    assert torch.cuda.is_available()
    torch.manual_seed(20260923)
    model = DanAlignment(8, 375).cuda()
    source = torch.randn(96, 8, 375, device="cuda")
    labels = torch.arange(96, device="cuda") % 12
    teacher = torch.randn(12, 8, 375, device="cuda")
    validation = torch.randn(72, 8, 375, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
    torch.cuda.reset_peak_memory_stats()
    step_seconds, validation_seconds = [], []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = alignment_loss(model, source, labels, teacher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - start)
        start = time.perf_counter()
        model.eval()
        with torch.inference_mode():
            value = alignment_loss(model, validation, labels[:72], teacher)
            assert torch.isfinite(value)
        torch.cuda.synchronize()
        validation_seconds.append(time.perf_counter() - start)
    # Warmup is included in the 100 updates budget; excluded only from timing quantiles.
    step_p95 = float(np.quantile(step_seconds[10:], 0.95))
    val_p95 = float(np.quantile(validation_seconds[10:], 0.95))
    projection = 2 * (step_p95 * 15795000 + val_p95 * 8775000)
    record = {
        "scope": "generated DAN train96/val72 C8 T375 float32 sequential kernel path",
        "optimizer_updates": 100,
        "validation_forwards": 100,
        "gpu": torch.cuda.get_device_name(),
        "step_p95_seconds": step_p95,
        "validation_p95_seconds": val_p95,
        "twofold_path_projection_seconds": projection,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "does_not_qualify_whole_run": True,
        "excluded": ["cache IO", "CPU eTRCA", "checkpoint cloning and serialization",
                     "whole-program scheduling", "independent audit"],
    }
    print("DAN_PARTIAL_PROFILE=" + json.dumps(record, sort_keys=True))
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
