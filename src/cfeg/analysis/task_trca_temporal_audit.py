"""Independent NumPy/SciPy audit for the explicitly new temporal-score study.

No producer imports or I/O.  Saved sufficient statistics, raw prefix metadata,
scalers, donors and validation losses are checked independently.  This does not
reproduce Q15 EEG features, raw preprocessing, Adam or access provenance, and
does not authorize a human experiment. The native-score study remains separate.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import linalg, special

from cfeg.analysis import task_trca_shape_audit as legacy

SCHEMA = "task-trca-temporal-v1"
SCORE_SCHEMA = "component-time-centered-ensemble-pearson-v1"
ARMS = (
    "FULL_NATIVE",
    "FULL_CENTERED",
    "ISO",
    "Q",
    "Q2",
    "QM",
    "SHAM_REFIT",
    "PERMUTED",
    "STALE",
    "MISSING",
)
LAMBDAS = (0.0001, 0.001, 0.01)
STATS = ("query_gram", "template_gram", "cross_gram")


def _require(condition, message):
    if not condition:
        raise ValueError("TEMPORAL_AUDIT_FAILURE: " + message)


def _schema(record):
    _require(
        isinstance(record, Mapping)
        and record.get("schema") == SCHEMA
        and record.get("score_schema") == SCORE_SCHEMA,
        "new temporal pipeline schema and score_schema required",
    )


def independent_prior(
    q, m, available, pipeline_record, arm, donor_m=None, stale_m=None, stale_available=None
):
    """Same frozen 80/20 shape head, guarded against legacy model reuse."""
    _schema(pipeline_record)
    _require(arm in ARMS[2:], "expected positive-mass temporal arm")
    return legacy.independent_prior(
        q,
        m,
        available,
        pipeline_record,
        arm,
        donor_m=donor_m,
        stale_m=stale_m,
        stale_available=stale_available,
    )


def independent_projectors(s, c, r):
    """C-normalized vv^T from independent generalized SciPy eigensystems.

    Inputs s/c[5,12,8,8], r[5,8]; output[5,12,8,8]. No native anchor is
    consulted. The top root must be simple; lower multiplicities are permitted.
    """
    s = legacy._symmetric(s, "s", (5, 12, 8, 8))
    c = legacy._symmetric(c, "c", (5, 12, 8, 8))
    r = legacy._check_prior(r)
    result = np.empty_like(s)
    for band in range(5):
        for label in range(12):
            metric = c[band, label]
            try:
                minimum = float(linalg.eigvalsh(metric)[0])
                _require(np.isfinite(minimum) and minimum > 0, "C must be positive definite")
                tau = 0.1 * minimum / (16 / 9)
                _require(np.isfinite(tau) and tau > 0, "positive penalty unrepresentable")
                denominator = metric + tau * np.diag(r[band])
                roots, vectors = linalg.eigh(s[band, label], denominator, type=1, driver="gvd")
            except linalg.LinAlgError as error:
                raise ValueError("TEMPORAL_AUDIT_FAILURE: eigensystem; no rescue") from error
            _require(np.isfinite(roots).all() and np.isfinite(vectors).all(), "finite eigensystem")
            _require(
                roots[-1] - roots[-2] > 1e-10 * max(1.0, np.max(np.abs(roots))),
                "degenerate top generalized eigenspace",
            )
            vector = vectors[:, -1]  # SciPy gives B-normalization, with arbitrary sign.
            norm = float(vector @ metric @ vector)
            _require(
                np.isfinite(norm) and 1 / 1.1 - 1e-10 <= norm <= 1 + 1e-10,
                "C trace violates bounded denominator identity",
            )
            result[band, label] = np.outer(vector, vector) / norm
    return legacy._array(result, "C-normalized projectors", (5, 12, 8, 8))


def independent_scores(projectors, statistics, weights):
    """Temporal-centered ensemble score and correlations; no global mean term.

    statistics has EXACT keys query_gram, template_gram, cross_gram, samples.
    Existing global-statistics dictionaries are deliberately rejected rather than
    silently dropping their mean correction. Projectors need not be rank one for
    this contraction, but must be nonzero PSD [5,12,8,8].
    """
    _require(
        isinstance(statistics, Mapping) and set(statistics) == {*STATS, "samples"},
        "explicit temporal statistics schema required; no global means",
    )
    f = legacy._symmetric(projectors, "projectors", (5, 12, 8, 8))
    roots = np.linalg.eigvalsh(f)
    _require(
        np.isfinite(roots).all()
        and np.all(roots[..., -1] > 0)
        and np.all(roots[..., 0] >= -1e-12 * np.max(np.abs(roots), axis=-1)),
        "projectors must be nonzero positive semidefinite",
    )
    shape = np.asarray(statistics["query_gram"]).shape
    _require(len(shape) == 4 and shape[0] > 0, "query Gram trial axis required")
    count = shape[0]
    samples = statistics["samples"]
    _require(type(samples) is int and samples >= 2, "integer samples >=2 required")
    xx = legacy._symmetric(statistics["query_gram"], "query_gram", (count, 5, 8, 8))
    tt = legacy._symmetric(statistics["template_gram"], "template_gram", (12, 5, 8, 8))
    xt = legacy._array(statistics["cross_gram"], "cross_gram", (count, 12, 5, 8, 8))
    weights = legacy._array(weights, "weights", (5,))
    _require(np.all(weights > 0) and np.isfinite(weights.sum()), "positive native weights required")
    correlations = np.empty((count, 5, 12))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            # Independent explicit traces, not the producer's einsum contraction.
            for band in range(5):
                total = np.sum(f[band], axis=0)
                query_energy = np.trace(total @ xx[:, band], axis1=-2, axis2=-1)
                template_energy = np.trace(total @ tt[:, band], axis1=-2, axis2=-1)
                _require(
                    np.all(query_energy > 0) and np.all(template_energy > 0),
                    "nonpositive temporal projected variance; no floor",
                )
                numerator = np.trace(total @ xt[:, :, band], axis1=-2, axis2=-1)
                corr = numerator / np.sqrt(query_energy[:, None]) / np.sqrt(template_energy[None])
                correlations[:, band] = np.clip(corr, -1, 1)
            values = np.sum(correlations * weights[None, :, None], axis=1)
        except FloatingPointError as error:
            raise ValueError(
                "TEMPORAL_AUDIT_FAILURE: nonfinite temporal score arithmetic"
            ) from error
    return (
        legacy._array(values, "temporal scores", (count, 12)),
        legacy._array(correlations, "temporal correlations", (count, 5, 12)),
    )


def _ids(values, name, *, allow_empty=False):
    values = tuple(values)
    _require(
        (bool(values) or allow_empty)
        and all(type(pid) is int and pid > 0 for pid in values)
        and list(values) == sorted(set(values)),
        name + " must be sorted unique positive integers",
    )
    return values


def _validate_data(data, outer_ids):
    _require(isinstance(data, Mapping), "case artifact mapping required")
    _require(
        {"keys", "orders", "q", "m", "available", "packet5", "labels", "weights", "s", "c", *STATS}
        <= set(data),
        "complete saved case fields required",
    )
    _require(not {"query_mean", "template_mean"} & set(data), "legacy global statistics rejected")
    for name, expected in (("schema", SCHEMA), ("score_schema", SCORE_SCHEMA)):
        _require(
            name in data
            and np.asarray(data[name]).shape == ()
            and str(np.asarray(data[name]).item()) == expected,
            "case artifact " + name,
        )
    keys = np.asarray(data["keys"])
    _require(
        keys.dtype.kind in "iu" and keys.ndim == 2 and keys.shape[1] == 4 and len(keys) > 0,
        "integer nonempty case keys required",
    )
    _require(
        keys.tolist() == sorted(keys.tolist()) and len({tuple(v) for v in keys}) == len(keys),
        "ordered unique case keys required",
    )
    _require(sorted(set(keys[:, 0].tolist())) == list(outer_ids), "source participant set")
    _require(
        np.isin(keys[:, 1], (0, 1)).all()
        and (keys[:, 2] >= 2).all()
        and np.isin(keys[:, 3], (3, 5)).all(),
        "valid interface/sample/support grid",
    )
    grids = [set(map(tuple, keys[keys[:, 0] == pid, 1:].tolist())) for pid in outer_ids]
    _require(all(grid == grids[0] for grid in grids), "complete common participant condition grid")
    count = len(keys)
    orders = np.asarray(data["orders"])
    _require(
        orders.dtype.kind in "iu" and orders.shape == (count,) and np.isin(orders, (0, 1)).all(),
        "binary integer acquisition orders",
    )
    q = legacy._array(data["q"], "q", (count, 5, 8, 15))
    m = legacy._array(data["m"], "m", (count, 8, 2))
    available = legacy._mask(data["available"], (count, 8))
    packet = legacy._array(data["packet5"], "packet5", (count, 5, 8), finite=False)
    labels = np.asarray(data["labels"])
    _require(
        labels.dtype.kind in "iu"
        and labels.shape == (count, 12)
        and np.all(np.sort(labels, axis=-1) == np.arange(12)),
        "labels must be recorded class permutations",
    )
    weights = legacy._array(data["weights"], "weights", (count, 5))
    _require(
        np.all(weights > 0) and np.isfinite(weights.sum(-1)).all(), "positive finite native weights"
    )
    for name in ("s", "c"):
        legacy._symmetric(data[name], name, (count, 5, 12, 8, 8))
    for name, shape in (
        ("query_gram", (count, 12, 5, 8, 8)),
        ("template_gram", (count, 12, 5, 8, 8)),
        ("cross_gram", (count, 12, 12, 5, 8, 8)),
    ):
        legacy._array(data[name], name, shape)
    metadata_identity, acquisition_orders, prefixes = {}, {}, {}
    for index, (pid, interface, _, k) in enumerate(keys):
        _require(np.isnan(packet[index, k:]).all(), "k3 future metadata must not be stored")
        expected_m, observed = legacy.independent_metadata(packet[index, :k])
        np.testing.assert_allclose(m[index], expected_m, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(available[index], observed)
        np.testing.assert_allclose(
            q[index, ..., 4],
            np.broadcast_to(np.isfinite(packet[index, :k]).mean(0), (5, 8)),
            atol=1e-12,
            rtol=0,
        )
        identity = (int(pid), int(interface), int(k))
        if int(pid) in acquisition_orders:
            _require(
                acquisition_orders[int(pid)] == orders[index],
                "participant acquisition order differs",
            )
        acquisition_orders[int(pid)] = orders[index]
        prefix_key = (int(pid), int(interface))
        if prefix_key in prefixes:
            previous = prefixes[prefix_key]
            common = min(int(k), int(keys[previous, 3]))
            _require(
                np.array_equal(packet[previous, :common], packet[index, :common], equal_nan=True),
                "support metadata prefixes differ across budgets/windows",
            )
        if prefix_key not in prefixes or k > keys[prefixes[prefix_key], 3]:
            prefixes[prefix_key] = index
        if identity in metadata_identity:
            previous = metadata_identity[identity]
            _require(
                orders[previous] == orders[index]
                and np.array_equal(packet[previous], packet[index], equal_nan=True),
                "metadata identity differs across windows",
            )
        metadata_identity[identity] = index
    return keys, labels


def donor_positions(data, positions):
    """Independent partition-local exact-mask/order/condition cyclic donors."""
    groups = {}
    for j in positions:
        _, interface, samples, k = data["keys"][j]
        key = (
            int(interface),
            int(samples),
            int(k),
            int(data["orders"][j]),
            tuple(np.isfinite(data["packet5"][j, :k]).ravel().tolist()),
        )
        groups.setdefault(key, []).append(int(j))
    donors = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda j: data["keys"][j, 0])
        donors.update({j: ordered[(index + 1) % len(ordered)] for index, j in enumerate(ordered)})
    return donors


def _record(data, pipeline, ids):
    _schema(pipeline)
    _require(pipeline["fit_ids"] == list(ids), "pipeline fit IDs")
    _require(
        type(pipeline["lambda"]) in (int, float) and pipeline["lambda"] in LAMBDAS,
        "frozen lambda grid required",
    )
    positions = np.flatnonzero(np.isin(data["keys"][:, 0], ids))
    available, q = data["available"][positions], data["q"][positions]
    for name, values, mask in (
        ("q_scaler", q, np.ones(q.shape[:-1], dtype=bool)),
        ("m_scaler", data["m"][positions], available),
        ("q2_scaler", q[..., 1:3], np.broadcast_to(available[:, None], q.shape[:-1])),
    ):
        rows = values[mask]
        mean = rows.mean(0) if len(rows) else np.zeros(values.shape[-1])
        scale = rows.std(0, ddof=0) if len(rows) else np.ones(values.shape[-1])
        scale[scale < 1e-12] = 1
        _require(pipeline[name]["fit_ids"] == list(ids), "scaler fit IDs")
        np.testing.assert_allclose(pipeline[name]["mean"], mean, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(pipeline[name]["scale"], scale, atol=1e-12, rtol=1e-12)
    entries = pipeline["donors"]
    actual = {tuple(row["case"]): row["donor_id"] for row in entries}
    expected = {
        tuple(data["keys"][j]): int(data["keys"][d, 0])
        for j, d in donor_positions(data, positions).items()
    }
    _require(len(actual) == len(entries) and actual == expected, "fit donor map/partition")
    _require(set(pipeline["residuals"]) == {"Q2", "QM", "SHAM_REFIT"}, "exact residual arm set")
    for name, head in (("Q", pipeline["Q"]), *pipeline["residuals"].items()):
        legacy._array(head["coefficients"], name + " coefficients", (16 if name == "Q" else 3,))
        _require(head["steps"] == 200 and len(head["trace"]) == 200, "fixed200 optimizer steps")
        _require(
            [row["step"] for row in head["trace"]] == list(range(1, 201)), "optimizer trace order"
        )
        trace = legacy._array(
            [
                [row[field] for field in ("loss_before_step", "ce_before_step", "gradient_norm")]
                for row in head["trace"]
            ],
            "head trace",
            (200, 3),
        )
        _require(np.all(trace[:, 2] >= 0), "nonnegative gradient norms")
        legacy._array([head["initial_loss"], head["final_loss"]], "head loss endpoints", (2,))
        _require(
            head["initial_loss"] == head["trace"][0]["loss_before_step"], "initial loss binding"
        )
        if name != "Q" and not available.any():
            _require(
                np.all(np.asarray(head["coefficients"]) == 0) and np.all(trace[:, 2] == 0),
                "empty metadata requires zero residual coefficients/gradients",
            )
    first = int(positions[0])
    # The independent prior checks the frozen Q byte hash, even for final refit.
    independent_prior(data["q"][first], None, None, pipeline, "Q")


def _case_scores(data, j, pipeline, arm, donors):
    donor_m = data["m"][donors[j]] if arm == "SHAM_REFIT" else None
    r = independent_prior(
        data["q"][j], data["m"][j], data["available"][j], pipeline, arm, donor_m=donor_m
    )
    f = independent_projectors(data["s"][j], data["c"][j], r)
    statistics = {name: data[name][j] for name in STATS}
    statistics["samples"] = int(data["keys"][j, 2])
    return independent_scores(f, statistics, data["weights"][j])[0]


def audit_training(data, model, outer_ids, evaluation_ids):
    """Audit a saved complete COMMON grid, including tiny generated nested runs.

    N=17/23 and one-condition engineering grids are allowed. This is deliberately
    not a human39/16-condition execution manifest. Source labels are read from
    the artifact and may be ordered permutations, never inferred from argmax.
    """
    outer_ids = _ids(outer_ids, "outer_ids")
    evaluation_ids = _ids(evaluation_ids, "evaluation_ids", allow_empty=True)
    _require(
        len(outer_ids) >= 6 and not set(outer_ids) & set(evaluation_ids), "outer role separation"
    )
    keys, labels = _validate_data(data, outer_ids)
    _schema(model["pipeline"])
    selection = model["selection"]
    _schema(selection)
    _require(selection["outer_evaluation_ids"] == list(evaluation_ids), "selection evaluation IDs")
    if "selection_signal" in selection:
        _require(selection["selection_signal"] == "Q CE only", "Q-only selection signal")
    _require(len(selection["inner"]) == 9, "nine inner fold/lambda records")
    measured, recorded, errors, seen = {}, {}, [], set()
    for row in selection["inner"]:
        inner, value = row["inner_fold"], row["lambda"]
        _require(
            type(inner) is int
            and inner in (0, 1, 2)
            and type(value) in (int, float)
            and value in LAMBDAS
            and (inner, value) not in seen,
            "unique fixed inner grid",
        )
        seen.add((inner, value))
        validation_ids = list(outer_ids[inner::3])
        training_ids = [pid for pid in outer_ids if pid not in validation_ids]
        _require(
            row["validation_ids"] == validation_ids and row["fit_ids"] == training_ids,
            "inner participant role split",
        )
        pipeline = row["pipeline"]
        _require(pipeline["lambda"] == value, "lambda record identity")
        _record(data, pipeline, training_ids)
        positions = np.flatnonzero(np.isin(keys[:, 0], validation_ids))
        donors = donor_positions(data, positions)
        _require(set(row["validation_ce"]) == {"Q", "Q2", "QM", "SHAM_REFIT"}, "validation arms")
        for arm in ("Q", "Q2", "QM", "SHAM_REFIT"):
            losses = []
            for j in positions:
                values = _case_scores(data, j, pipeline, arm, donors)
                logits = values / np.sum(data["weights"][j]) / 0.1
                losses.append(
                    np.mean(special.logsumexp(logits, axis=-1) - logits[np.arange(12), labels[j]])
                )
            loss = float(np.mean(losses))
            original = float(row["validation_ce"][arm])
            error = abs(loss - original)
            _require(
                np.isfinite(original) and np.isfinite(loss) and error <= 1e-8,
                "independent validation CE mismatch",
            )
            errors.append(error)
            if arm == "Q":
                measured.setdefault(value, []).append((loss, len(validation_ids)))
                recorded.setdefault(value, []).append((original, len(validation_ids)))

    def choose(rows):
        means = {
            value: float(
                np.average([x[0] for x in rows[value]], weights=[x[1] for x in rows[value]])
            )
            for value in LAMBDAS
        }
        best = min(means.values())
        return max(value for value, loss in means.items() if loss <= best + 1e-12), means

    selected, weighted = choose(measured)
    recorded_selected, _ = choose(recorded)
    _require(
        selected
        == recorded_selected
        == selection["selected_lambda"]
        == model["pipeline"]["lambda"],
        "independent and recorded Q-only lambda selection",
    )
    _record(data, model["pipeline"], outer_ids)
    return {
        "status": "TEMPORAL_SOURCE_SELECTION_AUDIT_PASS",
        "schema": SCHEMA,
        "score_schema": SCORE_SCHEMA,
        "cases": len(keys),
        "selected_lambda": selected,
        "weighted_q_validation_ce": {str(k): v for k, v in weighted.items()},
        "max_validation_ce_abs_error": max(errors),
        "validation_ce_comparisons": len(errors),
        "scope": "independent prefix M/scalers/donors/temporal validation/selection; not raw preprocessing, Q15 EEG, Adam or provenance",
    }


def summarize(scores, a0, coverage, actuation):
    """Fixed ten-arm endpoint audit; centered guard can only DEMOTE a candidate.

    actuation must contain explicit r_max_abs_qm_minus_q,
    projector_max_abs_qm_minus_q and j_max_abs_qm_minus_q[39,2,4,2] (or flattened
    624 values), plus positive band_weights/a0_band_weights and qm_coefficients.
    Original native FULL, metadata and observed-grid cost requirements are kept.
    QM3-FULL_CENTERED3 lower descriptive participant t-CI must additionally exceed
    -1pp. Neither nonattainment nulls nor old failure branches are reinterpreted.
    """
    scores = legacy._array(scores, "temporal scores", (39, 2, 4, 2, 10, 48, 12))
    _require(isinstance(actuation, Mapping), "explicit temporal actuation evidence required")
    _require(
        {"projector_max_abs_qm_minus_q", "j_max_abs_qm_minus_q"} <= set(actuation),
        "explicit projector and J case maxima required",
    )
    maxima = {}
    for field in ("projector_max_abs_qm_minus_q", "j_max_abs_qm_minus_q"):
        value = legacy._array(actuation[field], field)
        _require(
            value.shape in ((39, 2, 4, 2), (624,)) and np.all(value >= 0),
            field + " requires624 nonnegative case maxima",
        )
        maxima[field] = value.reshape(39, 2, 4, 2)
    old_actuation = {
        **actuation,
        "filter_max_abs_qm_minus_q": maxima["projector_max_abs_qm_minus_q"],
    }
    native_projection = scores[..., (0, 2, 3, 4, 5, 6, 7, 8, 9), :, :]
    result = legacy.summarize(native_projection, a0, coverage, old_actuation)
    original_terminal = result["terminal"]
    truth = np.tile(np.arange(12), 4)
    counts = (scores.argmax(-1) == truth).sum(-1, dtype=np.int64)
    centered = legacy._paired_counts(
        (counts[..., 0, 5] - counts[..., 0, 1]).sum((1, 2), dtype=np.int64)
    )
    centered_ok = centered["ci_low_pp"] > -1
    checks = result["calibration_additional_checks"]
    checks["qm3_minus_full_native3_ci_low_gt_minus_1pp"] = checks.pop(
        "qm3_minus_full3_ci_low_gt_minus_1pp"
    )
    checks["qm3_minus_full_centered3_ci_low_gt_minus_1pp"] = centered_ok
    result["calibration_additional_passed"] = all(checks.values())
    result["comparisons"]["QM3_minus_FULL_NATIVE3"] = result["comparisons"].pop("QM3_minus_FULL3")
    result["comparisons"]["QM3_minus_FULL_CENTERED3"] = centered
    result["comparisons"]["FULL_CENTERED3_minus_FULL_NATIVE3"] = legacy._paired_counts(
        (counts[..., 0, 1] - counts[..., 0, 0]).sum((1, 2), dtype=np.int64)
    )
    if original_terminal == "DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE" and not centered_ok:
        result["terminal"] = "CLASSIFICATION_INCREMENT_ONLY"
    weights = legacy._weights(actuation["band_weights"], "band_weights")
    nll = np.empty((39, 2, 4, 2, 10))
    for interface in range(2):
        logits = scores[:, interface] / weights[interface].sum() / 0.1
        true = np.take_along_axis(logits, np.broadcast_to(truth, logits.shape[:-1])[..., None], -1)[
            ..., 0
        ]
        nll[:, interface] = (special.logsumexp(logits, axis=-1) - true).mean(-1)
    _require(np.isfinite(nll).all(), "finite temporal NLL")
    for index, condition in enumerate(result["conditions"]):
        interface, window = divmod(index, 4)
        condition["accuracy"] = (counts[:, interface, window].mean(0) / 48).tolist()
        condition["nll"] = nll[:, interface, window].mean(0).tolist()
    for budget, k in enumerate((3, 5)):
        diagnosis = result["actuation"]["by_k"][str(k)]
        diagnosis["projector_max_abs_qm_minus_q"] = diagnosis.pop("filter_max_abs_qm_minus_q")
        for name in (
            "q_top_two_native_margin_min",
            "q_top_two_native_margin_median",
            "q_top_two_native_margins_at_most_1e_12",
        ):
            diagnosis[name.replace("native", "temporal")] = diagnosis.pop(name)
        value = maxima["j_max_abs_qm_minus_q"][..., budget]
        diagnosis["j_max_abs_qm_minus_q"] = {
            "maximum": float(value.max()),
            "changed_cells_exact": int((value != 0).sum()),
            "changed_cells_above_1e_12": int((value > 1e-12).sum()),
        }
    result["actuation"].pop("r_filter_maxima_are_descriptive_not_independent_proof")
    result["actuation"]["r_projector_j_maxima_are_descriptive_not_independent_proof"] = True
    result.update(
        schema=SCHEMA,
        score_schema=SCORE_SCHEMA,
        artifact_kind="independent_endpoint_summary",
        arms=list(ARMS),
        correct_counts=counts.tolist(),
        accuracy_by_budget_arm=(counts.mean((0, 1, 2)) / 48).tolist(),
        nll_by_budget_arm=nll.mean((0, 1, 2)).tolist(),
        terminal_before_centered_guard=original_terminal,
        centered_control_can_only_demote=True,
        native_reference_preserved=True,
        nll_note="each arm's score / explicit native interface band-weight sum / .1; FULL_NATIVE global, other arms temporal; not calibrated probabilities",
        evidence_scope="fixed39 endpoint schema only; data origin/provenance must be established externally; no execution authorization",
    )
    return result
