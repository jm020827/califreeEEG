import numpy as np
import pytest

from cfeg.mamem_shrinkage_v1 import (
    ARMS,
    build_fold,
    normalized_psd,
    oracle_lambda,
    prepare,
    score_query,
    sham_cycle,
)


def test_oracle_endpoints_and_degenerate():
    t = np.array([np.diag([1.0, 0.0]), np.diag([1.0, 0.0])])
    p = np.array([np.diag([0.0, 1.0]), np.diag([0.0, 1.0])])
    np.testing.assert_allclose(oracle_lambda(t, p, 0.25 * t + 0.75 * p)[0], 0.75)
    np.testing.assert_allclose(oracle_lambda(t, t, p)[0], 0)
    assert oracle_lambda(t, t, p)[1] == 2
    tiny_prior = t.copy()
    tiny_prior[:, 0, 0] += 1e-13
    np.testing.assert_allclose(oracle_lambda(t, tiny_prior, t + 1)[0], 0)
    assert oracle_lambda(t, tiny_prior, t + 1)[1] == 2


def test_psd_and_score_unit_invariance():
    rng = np.random.default_rng(5)
    b = rng.normal(size=(5, 2, 4, 2))
    c = np.eye(4)
    h, count = normalized_psd(c, b)
    assert count == 0
    np.testing.assert_allclose(np.trace(h, axis1=-2, axis2=-1), 1)
    assert np.linalg.eigvalsh(h).min() > -1e-12
    np.testing.assert_allclose(score_query(c, b, h), score_query(c * 1e8, b * 1e4, h))
    fallback, count = normalized_psd(c, b * 0)
    np.testing.assert_allclose(fallback, np.broadcast_to(np.eye(4) / 4, fallback.shape))
    assert count == 10


def test_sham_joint_strata_and_singletons():
    rows = [
        {"subject": s, "label": 0, "k": 1, "event_count": n}
        for s, n in [("A", 4), ("B", 4), ("C", 5)]
    ]
    m = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    out, audit = sham_cycle(rows, m)
    np.testing.assert_array_equal(out, m[[1, 0, 2]])
    assert audit["changed_fraction"] == 2 / 3
    assert audit["singleton_rows"] == 1


def test_protected_development_subject_is_rejected():
    with pytest.raises(ValueError, match="development_subject"):
        prepare({"S001": {}})


def test_no_label_in_score_signature():
    import inspect

    assert tuple(inspect.signature(score_query).parameters) == ("covariance", "factors", "metric")
    assert ARMS == ("Q", "Q2", "QM", "SHAM")


def fixture_data():
    rng = np.random.default_rng(19)
    data = {}
    for i in range(4):
        data[f"FAKE{i}"] = {}
        for run in "ab":
            data[f"FAKE{i}"][run] = [
                {
                    "label": n // 3,
                    "start0": 1000 * n,
                    "trial_end0": 1000 * n + 1250,
                    "event_count": 40,
                    "metadata": np.array([i * 0.1, i * -0.1]) if run == "a" else None,
                    "q": rng.normal(size=(5, 2, 4)),
                    "q2": rng.normal(size=(5, 2)),
                    "factors": rng.normal(size=(5, 2, 4, 2)),
                    "covariance": np.eye(4),
                    "zero_shot": np.zeros(5),
                }
                for n in range(15)
            ]
    return data


def test_target_query_label_and_features_do_not_enter_gate_or_prior():
    data = fixture_data()
    before = build_fold(prepare(data), "FAKE0", 1)
    for row in data["FAKE0"]["b"]:
        row["label"] = (row["label"] + 1) % 5
        row["factors"] *= 200
        row["q"] *= 200
        row["event_count"] = 999
    after = build_fold(prepare(data), "FAKE0", 1)
    for key in ("prior", "template", "y", "class_mean_m"):
        np.testing.assert_array_equal(before[key], after[key])
    for arm in ARMS:
        np.testing.assert_array_equal(before["x"][arm], after["x"][arm])
        np.testing.assert_array_equal(before["eval_x"][arm], after["eval_x"][arm])
    assert all(
        "FAKE0" not in p["contributors"] and p["pseudo_target"] not in p["contributors"]
        for p in after["prior_audit"]
    )


def test_later_target_support_trials_do_not_change_k1_gate_or_template():
    data = fixture_data()
    before = build_fold(prepare(data), "FAKE0", 1)
    for n, row in enumerate(data["FAKE0"]["a"]):
        if n % 3:
            row["metadata"] += 1000
            row["q"] += 1000
            row["factors"] += 1000
    after = build_fold(prepare(data), "FAKE0", 1)
    np.testing.assert_array_equal(before["template"], after["template"])
    np.testing.assert_array_equal(before["eval_x"]["QM"], after["eval_x"]["QM"])
    np.testing.assert_array_equal(before["y"], after["y"])


def test_unordered_trials_rejected_before_folds():
    data = fixture_data()
    data["FAKE0"]["a"][0], data["FAKE0"]["a"][1] = data["FAKE0"]["a"][1], data["FAKE0"]["a"][0]
    with pytest.raises(ValueError, match="trial_sample_order"):
        prepare(data)


def test_valid_gram_query_scores_and_ties():
    rng = np.random.default_rng(34)
    x = rng.normal(size=(4, 30))
    c = x @ x.T / 30
    basis = np.linalg.qr(rng.normal(size=(30, 2)))[0]
    factors = np.broadcast_to(x @ basis / np.sqrt(30), (5, 2, 4, 2)).copy()
    metric, _ = normalized_psd(c, factors)
    values = score_query(c, factors, metric)
    assert (values >= 0).all() and (values <= 1.5 + 1e-12).all()
    assert np.argmax(values) == 0


def test_sham_roundoff_is_not_meaningful_change():
    rows = [{"subject": s, "label": 0, "k": 1, "event_count": 4} for s in ("A", "B")]
    _, audit = sham_cycle(rows, np.array([[0.0, 0.0], [1e-14, 1e-14]]))
    assert audit["changed_fraction"] == 0
    assert audit["exact_changed_fraction"] == 1
