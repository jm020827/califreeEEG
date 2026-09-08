import numpy as np
import pytest
import torch

from cfeg.analysis import task_trca_shape_learning as learn
from cfeg.analysis import task_trca_shape_evaluation as ev


def test_separate_no_label_inference_and_native_exact():
    torch.set_num_threads(1)
    rng = np.random.default_rng(20260910)
    x = rng.normal(size=(3, 12, 5, 8, 17))
    y = rng.normal(size=(12, 5, 8, 17))
    packet = rng.uniform(0, 20, (3, 8))
    weights = np.array([1.25, .67, .50, .43, .38])
    freq = np.linspace(9, 14.5, 12)
    train = learn.make_task_case(11001, 0, 0, x, packet, freq, y, weights=weights)
    pipeline = learn.fit_pipeline((train,), .001, backend="batch")
    restored = ev.pipeline_from_record(pipeline.record())
    assert restored.q_hash == pipeline.q_hash
    state = ev.support_state(11002, 0, 0, x, packet, freq, weights)
    assert not hasattr(state, "labels") and not hasattr(state, "statistics")
    query = np.tile(y, (4, 1, 1, 1))
    result = ev.evaluate(restored, state, query, donor=state)
    expected, _ = learn.native.score_trca(state.model, query, weights)
    np.testing.assert_array_equal(result["scores"][0], expected)
    validation = learn.make_task_case(11002, 0, 0, x, packet, freq, y, weights=weights, role="validation")
    for arm in ev.ARMS[1:]:
        ref = learn.predict(pipeline, (validation,), arm)[validation.key]
        np.testing.assert_allclose(result["scores"][ev.ARMS.index(arm)], np.tile(ref, (4, 1)), atol=1e-10, rtol=0)
    same = ev.support_state(11001, 0, 0, x, packet, freq, weights)
    with pytest.raises(PermissionError):
        ev.evaluate(pipeline, same, query, donor=same)
    packed = ev.pack_cases((train,))
    assert packed["packet5"].shape == (1, 5, 8)
    assert np.isnan(packed["packet5"][0, 3:]).all()
