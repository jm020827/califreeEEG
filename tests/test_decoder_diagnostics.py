import numpy as np
import pandas as pd
import pytest
import torch

from cfeg.decoder_diagnostics import (
    EEGNetDecoder,
    ReferenceCCA,
    calibration_partition,
    subject_protocols,
)


def test_pooled_protocol_retains_single_dataset_test_subjects():
    manifest = pd.DataFrame(
        [
            {"dataset_id": dataset, "subject_id": f"s{i}", "label": label}
            for dataset in ("wang", "beta")
            for i in range(10)
            for label in range(4)
        ]
    )
    protocols = subject_protocols(manifest, 42)
    for name in ("train", "val", "test"):
        expected = set(getattr(protocols["wang"], name)) | set(getattr(protocols["beta"], name))
        assert set(getattr(protocols["pooled"], name)) == expected
    assert set(manifest.iloc[protocols["wang_to_beta"].train].dataset_id) == {"wang"}
    assert set(manifest.iloc[protocols["wang_to_beta"].test].dataset_id) == {"beta"}


def test_calibration_support_has_one_per_class_and_fixed_disjoint_query():
    labels = np.repeat(np.arange(40), 4)
    support, query = calibration_partition(labels, 42)
    assert len(support) == 40 and len(query) == 120
    assert len(set(labels[support])) == 40
    assert not set(support) & set(query)
    assert set(support) | set(query) == set(range(160))


def test_reference_cca_agrees_with_sklearn_canonical_correlation():
    CCA = pytest.importorskip("sklearn.cross_decomposition").CCA
    rng = np.random.default_rng(4)
    time = np.arange(400) / 200
    x = rng.normal(size=(8, 400)) + rng.normal(size=(8, 1)) * np.sin(2 * np.pi * 10 * time)
    scorer = ReferenceCCA([8, 10], 200, 400, harmonics=3)
    scores = scorer(torch.from_numpy(x[None])).numpy()[0]
    expected = []
    for frequency in (8, 10):
        ref = np.stack(
            [fn(2 * np.pi * frequency * h * time) for h in range(1, 4) for fn in (np.sin, np.cos)],
            axis=1,
        )
        left, right = CCA(n_components=1, max_iter=5000, tol=1e-10).fit_transform(x.T, ref)
        expected.append(np.corrcoef(left[:, 0], right[:, 0])[0, 1])
    np.testing.assert_allclose(scores, expected, atol=1e-5)
    assert scores.argmax() == 1
    assert torch.equal(scorer(torch.zeros(1, 8, 400)), torch.zeros(1, 2))


def test_eegnet_can_update_weights_and_apply_maxnorm():
    torch.manual_seed(42)
    model = EEGNetDecoder(channels=8)
    previous = model.temporal.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.randn(4, 8, 400)
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    optimizer.step()
    model.constrain_weights()
    assert torch.isfinite(loss)
    assert not torch.equal(previous, model.temporal.weight)
    assert model.head.weight.flatten(1).norm(dim=1).max() <= 0.25001
