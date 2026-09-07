"""Main-environment tests of the fixture adapter; no external toolbox import."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "author_etrca_fixture", Path(__file__).resolve().parents[1] / "scripts/check_author_etrca.py"
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


@pytest.mark.parametrize("n_samples", [125, 188, 250])
def test_preprocess_exact_slice_and_no_input_mutation(n_samples):
    raw = np.arange(8 * 710).reshape(8, 710).astype(float)
    before = raw.copy()
    calls = []

    def preprocess(descriptor, x):
        assert descriptor.srate == 250
        calls.append(x.copy())
        x += 1
        return x

    def filterbank(descriptor, x):
        assert descriptor.srate == 250
        return np.stack([x * (i + 1) for i in range(5)])

    native = SimpleNamespace(preprocess=preprocess, filterbank=filterbank)
    result = FIXTURE.preprocess_trial(raw, n_samples, native)
    np.testing.assert_array_equal(calls[0], raw[:, 125 : 160 + n_samples])
    expected = np.stack([(raw[:, 160 : 160 + n_samples] + 1) * (i + 1) for i in range(5)])
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(raw, before)
    assert result.shape == (5, 8, n_samples)
    assert not np.shares_memory(result, raw)


@pytest.mark.parametrize("shape,n_samples", [((7, 710), 125), ((8, 709), 125), ((8, 710), 500)])
def test_reject_undeclared_fixture_shape_and_window(shape, n_samples):
    with pytest.raises(ValueError):
        FIXTURE.preprocess_trial(np.zeros(shape), n_samples, None)


@pytest.mark.parametrize(
    "event,args",
    [
        ("open", ("/nonexistent/example.mat", "r", 0)),
        ("open", (b"/nonexistent/example.NPZ", "r", 0)),
        ("open", (Path("/nonexistent/example.parquet"), "r", 0)),
        ("import", ("SSVEPAnalysisToolbox.datasets.wearabledataset",)),
        ("import", ("SSVEPAnalysisToolbox.evaluator",)),
        ("socket.connect", (None,)),
        ("socket.getaddrinfo", (None,)),
        ("subprocess.Popen", (None,)),
    ],
)
def test_fixture_guard_rejects_forbidden_events_without_opening_files(event, args):
    with pytest.raises(RuntimeError):
        FIXTURE.fixture_guard(event, args)


@pytest.mark.parametrize(
    "event,args",
    [
        ("open", ("/nonexistent/code.py", "r", 0)),
        ("open", (3, "r", 0)),
        ("import", ("SSVEPAnalysisToolbox.algorithms.trca",)),
    ],
)
def test_fixture_guard_allows_code_and_file_descriptors(event, args):
    FIXTURE.fixture_guard(event, args)
