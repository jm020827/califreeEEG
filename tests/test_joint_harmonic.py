"""Generated-only tests. No source39/cache/manifest files are opened."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from cfeg.analysis.joint_harmonic import (
    ARMS,
    ContextScaler,
    direct_impedance_gain,
    impedance_summary,
    make_context,
    observed_threshold_cost,
    participant_splits,
    raw_prototype_scores,
    reference_bank,
    regularized_cca_scores,
    select_learning_rate,
    select_policy,
    sham_donors,
    spectral_features,
)
from cfeg.models.joint_harmonic import JointHarmonicPrototype


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def episode():
    torch.manual_seed(17)
    model = JointHarmonicPrototype(n_channels=2, n_classes=3, n_harmonics=2)
    support = torch.randn(2, 6, 2, 12)
    labels = torch.tensor([[0, 1, 2, 0, 1, 2]]).expand(2, -1)
    query = torch.randn(2, 5, 2, 12)
    context = torch.randn(2, 2, 10)
    return model, support, labels, query, context


def test_harmonic_phase_and_amplitude_are_retained():
    t = np.arange(250) / 250
    x = np.stack((np.cos(2 * np.pi * 10 * t + 0.4),
                  2 * np.cos(2 * np.pi * 14 * t - 0.2)))[None]
    result = spectral_features(x, frequencies=[10, 14], sfreq=250, n_harmonics=2)
    z = result["spectra"].reshape(1, 2, 2, 2, 2)
    z = z[..., 0] + 1j * z[..., 1]
    np.testing.assert_allclose(np.angle(z[0, 0, 0, 0]), 0.4, atol=1e-12)
    np.testing.assert_allclose(np.angle(z[0, 1, 1, 0]), -0.2, atol=1e-12)
    np.testing.assert_allclose(abs(z[0, 1, 1, 0]) / abs(z[0, 0, 0, 0]), 2)
    assert result["q"].shape == (1, 2, 2)
    assert result["q2"].shape == (1, 2, 2)


def test_spectral_extraction_is_trial_local_and_global_scale_invariant():
    x = np.random.default_rng(8).normal(size=(7, 3, 200))
    kwargs = {"frequencies": [10, 14], "sfreq": 200, "n_harmonics": 2}
    all_rows = spectral_features(x, **kwargs)
    one = spectral_features(x[:1], **kwargs)
    scaled = spectral_features(11 * x + 90, **kwargs)
    for key in all_rows:
        np.testing.assert_allclose(all_rows[key][:1], one[key])
    np.testing.assert_allclose(scaled["spectra"], all_rows["spectra"], atol=1e-12)
    np.testing.assert_allclose(scaled["q"][..., 0] - all_rows["q"][..., 0], np.log(11))


@pytest.mark.parametrize("frequencies,sfreq,n,h", [
    ([0, 10], 250, 100, 2), ([10, 10], 250, 100, 2),
    ([50], 250, 100, 3), ([np.nan], 250, 100, 1),
    ([10], 0, 100, 1), ([10], 250, 1, 1), ([10], 250, 100, 0),
])
def test_bad_reference_spec_fails(frequencies, sfreq, n, h):
    with pytest.raises(ValueError):
        reference_bank(frequencies, sfreq, n, h)


@pytest.mark.parametrize("value", [0.0, np.nan, np.inf])
def test_invalid_trials_not_silently_removed(value):
    with pytest.raises(ValueError):
        spectral_features(np.full((1, 2, 100), value), frequencies=[10], sfreq=250)


def test_impedance_missing_zero_and_population_sd():
    values, flags = impedance_summary(np.array([[0, np.nan, 3], [3, np.nan, 3]]))
    np.testing.assert_allclose(values[0], [np.log(4) / 2, np.log(4) / 2])
    assert not flags[1].any()
    assert flags[[0, 2]].all()
    assert values[2, 1] == 0
    one, one_flags = impedance_summary(np.array([[0.0, 5.0]]))
    assert one_flags.all() and np.all(one[:, 1] == 0)


@pytest.mark.parametrize("bad", [-1.0, np.inf, -np.inf])
def test_bad_impedance_fails(bad):
    with pytest.raises(ValueError):
        impedance_summary(np.array([[bad, 1]]))


def test_q_context_ignores_poison_metadata_and_duplicates_q():
    q = np.arange(6).reshape(3, 2)
    common = np.array([0, 1, 1, np.log(3)])
    first = make_context(q, common=common, arm="Q")
    poison = make_context(q, common=common, arm="Q", metadata=np.array([np.inf]),
                          available=np.array([777]))
    np.testing.assert_array_equal(first, poison)
    np.testing.assert_array_equal(first[:, :2], first[:, 2:4])
    assert (first[:, 4:6] == 1).all()


def test_observed_metadata_checked_missing_ignored_and_q2_separate():
    q, common = np.ones((3, 2)), np.array([0, 1, 1, 0])
    available = np.array([[True, False]] * 3)
    m = np.array([[2, np.nan]] * 3)
    context = make_context(q, common=common, arm="QM", metadata=m, available=available)
    assert (context[:, 2] == 2).all() and (context[:, 3] == 0).all()
    with pytest.raises(ValueError):
        make_context(q, common=common, arm="QM", metadata=m, available=np.ones_like(m, bool))
    q2 = make_context(q, common=common, arm="Q2", q2=3 * q, metadata=np.array([np.nan]))
    assert (q2[:, 2:4] == 3).all()


def test_normalizer_fit_role_excludes_target_and_retains_missing_flags():
    q = np.ones((2, 2))
    missing = np.array([[True, False], [False, False]])
    common = np.array([0, 1, 1, 0])
    a = make_context(q, common=common, arm="QM", metadata=3 * q, available=missing)
    b = make_context(2 * q, common=common, arm="QM", metadata=5 * q, available=missing)
    x = np.stack([a, b])
    scaler = ContextScaler.fit(x, subject_ids=[1, 2], allowed_fit_ids=[1, 2])
    transformed = scaler.transform(x)
    np.testing.assert_allclose(transformed[:, :, 0].mean(axis=0), 0)
    assert np.all(transformed[:, 0, 3] == 0) and np.all(transformed[:, 1, 2:4] == 0)
    np.testing.assert_array_equal(transformed[..., 4:], x[..., 4:])
    previous = scaler.mean.copy()
    scaler.transform(100 * x[..., :4].mean() * np.ones_like(x) * 0 + x)
    np.testing.assert_array_equal(scaler.mean, previous)
    with pytest.raises(ValueError, match="fit role"):
        ContextScaler.fit(x, subject_ids=[1, 99], allowed_fit_ids=[1, 2])
    with pytest.raises(ValueError):
        ContextScaler.fit(x, subject_ids=[1, 1], allowed_fit_ids=[1, 2])


def test_participant_roles_disjoint_and_outer_coverage_once():
    splits = participant_splits(list(range(1, 40)))
    outer = []
    for fold in splits:
        train, test = set(fold["outer_fit"]), set(fold["outer_query"])
        inner, validation = set(fold["inner_fit"]), set(fold["inner_validation"])
        assert len(train) == 26 and len(test) == 13
        assert not train & test and not inner & validation
        assert inner | validation == train
        outer.extend(test)
    assert sorted(outer) == list(range(1, 40))


def test_sham_derangement_preserves_role_and_headband_order():
    ids = [2, 5, 8, 10, 11, 12]
    order = {s: "dry" if s < 10 else "wet" for s in ids}
    donors = sham_donors(ids, headband_orders=order, seed=11)
    assert set(donors.values()) == set(ids)
    assert all(s != d and order[s] == order[d] for s, d in donors.items())
    assert donors == sham_donors(ids, headband_orders=order, seed=11)
    with pytest.raises(ValueError, match="exact role"):
        sham_donors(ids, headband_orders={**order, 99: "wet"}, seed=11)
    with pytest.raises(ValueError, match="Singleton"):
        sham_donors([1, 2, 3], headband_orders={1: "dry", 2: "dry", 3: "wet"}, seed=11)


def test_fixed_prototype_manual_cosine_and_gain():
    support = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=float)
    query = np.array([[[1, 0], [0, 1]]], dtype=float)
    baseline = raw_prototype_scores(support, np.arange(2), query, n_classes=2)
    weighted = raw_prototype_scores(support, np.arange(2), query, n_classes=2,
                                    channel_gain=np.array([2, 1]))
    np.testing.assert_allclose(baseline, [[1 / np.sqrt(2), 1 / np.sqrt(2)]])
    np.testing.assert_allclose(weighted, [[2 / np.sqrt(5), 1 / np.sqrt(5)]])


def test_direct_impedance_gain_missing_and_bounds():
    gains = direct_impedance_gain(np.array([[0, 9999, np.nan], [0, 9999, np.nan]]))
    np.testing.assert_array_equal(gains, [2, 0.5, 1])
    np.testing.assert_array_equal(direct_impedance_gain(np.full((3, 8), np.nan)), np.ones(8))


def test_common_cca_recognizes_exact_frequency_without_labels():
    t = np.arange(500) / 250
    rng = np.random.default_rng(1)
    trials = np.stack([np.cos(2 * np.pi * f * t) for f in [9, 11, 13]])
    trials = np.stack([trials, 0.5 * trials + rng.normal(0, 0.02, trials.shape)], axis=1)
    scores = regularized_cca_scores(trials, frequencies=[9, 11, 13], sfreq=250)
    np.testing.assert_array_equal(scores.argmax(axis=-1), [0, 1, 2])
    np.testing.assert_allclose(scores.max(axis=-1), 1, atol=1e-5)
    np.testing.assert_allclose(scores[:1], regularized_cca_scores(
        trials[:1], frequencies=[9, 11, 13], sfreq=250))


def test_model_query_batch_independence_and_support_order_invariance():
    model, support, labels, query, context = episode()
    initial = model(support, labels, query, context)
    one = model(support, labels, query[:, :1], context)
    order = torch.tensor([2, 5, 4, 0, 1, 3])
    shuffled = model(support[:, order], labels[:, order], query, context)
    torch.testing.assert_close(initial[:, :1], one)
    torch.testing.assert_close(initial, shuffled)


def test_model_support_label_permutation_equivariance():
    model, support, labels, query, context = episode()
    before = model(support, labels, query, context)
    permutation = torch.tensor([2, 0, 1])
    after = model(support, permutation[labels], query, context)
    torch.testing.assert_close(after[..., permutation], before)


def test_joint_gradients_and_conditioning_actuation_without_efficacy_gate():
    model, support, labels, query, context = episode()
    support.requires_grad_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    target = torch.tensor([[0, 1, 2, 0, 1]]).expand(2, -1)
    # Exactly 8 optimizer updates per invocation. No human data or M-utility test.
    for _ in range(8):
        optimizer.zero_grad()
        logits = model(support, labels, query, context)
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten()).backward()
        optimizer.step()
    for module in [model.channel_encoder, model.conditioner[0], model.conditioner[-1],
                   model.spatial_encoder]:
        assert module.weight.grad is not None and module.weight.grad.abs().sum() > 0
    assert support.grad.abs().sum() > 0
    changed = context.clone()
    changed[..., 2:4] += 3
    difference = model(support, labels, query, changed) - model(support, labels, query, context)
    assert difference.abs().max() > 1e-7


def test_equal_model_capacity_and_initialization_all_arms():
    models = []
    for _ in ARMS:
        torch.manual_seed(20260923)
        models.append(JointHarmonicPrototype())
    sizes = [sum(p.numel() for p in m.parameters()) for m in models]
    assert len(set(sizes)) == 1
    for key, tensor in models[0].state_dict().items():
        for model in models[1:]:
            torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)


def test_checkpoint_roundtrip():
    model, support, labels, query, context = episode()
    expected = model(support, labels, query, context)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    copy = JointHarmonicPrototype(n_channels=2, n_classes=3, n_harmonics=2)
    copy.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(copy(support, labels, query, context), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_forward_and_gradient_agreement():
    model, support, labels, query, context = episode()
    with torch.no_grad():
        model.conditioner[-1].weight.uniform_(-0.03, 0.03)
    other = JointHarmonicPrototype(n_channels=2, n_classes=3, n_harmonics=2).cuda()
    other.load_state_dict(model.state_dict())
    cpu = model(support, labels, query, context)
    gpu = other(support.cuda(), labels.cuda(), query.cuda(), context.cuda())
    torch.testing.assert_close(cpu, gpu.cpu(), atol=1e-5, rtol=1e-5)
    cpu.square().mean().backward()
    gpu.square().mean().backward()
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad.cpu(), atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("mode", ["missing_class", "wrong_dtype", "bad_label", "bad_context"])
def test_model_rejects_invalid_episode(mode):
    model, support, labels, query, context = episode()
    if mode == "missing_class":
        labels = labels * 0
    elif mode == "wrong_dtype":
        labels = labels.float()
    elif mode == "bad_label":
        labels = labels + 1
    else:
        context[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(support, labels, query, context)


def test_policy_accuracy_cost_and_unattained_are_distinct():
    assert select_learning_rate({0.001: 0.8, 0.0003: 0.8}) == 0.0003
    assert select_learning_rate({0.001: 0.81, 0.0003: 0.8}) == 0.001
    assert select_policy({0: 0.8, 1: 0.8, 3: 0.9, 5: 1}) == {
        "k": 0, "acquired_trials": 0, "fallback": False}
    assert select_policy({0: 0.5, 1: 0.7, 3: 0.79, 5: 0.79}) == {
        "k": 5, "acquired_trials": 60, "fallback": True}
    assert select_policy({0: 0.5, 1: 0.79, 3: 0.8, 5: 0.9})["acquired_trials"] == 36
    actual = observed_threshold_cost(np.array([[0.8, 1, 1, 1], [0.5, 0.81, 0.79, 0.9],
                                                [0.5, 0.6, 0.7, 0.79]]))
    np.testing.assert_allclose(actual, [0, 12, np.nan], equal_nan=True)


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_policy_rejects_invalid_accuracy(bad):
    with pytest.raises(ValueError):
        select_policy({0: 0.1, 1: bad, 3: 0.7, 5: 0.8})
    with pytest.raises(ValueError):
        observed_threshold_cost(np.array([0.1, bad, 0.7, 0.8]))
