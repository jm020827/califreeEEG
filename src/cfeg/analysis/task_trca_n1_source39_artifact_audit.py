"""Independent persisted temporal source39 checks, downstream of saved Q/S/C.

Only NumPy/SciPy and independent audit code; no producer, native fitting, raw
archive, metadata projection, optimizer or execution-authority imports.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from cfeg.analysis import task_trca_n1_audit as independent
from cfeg.analysis import task_trca_shape_audit as legacy

STUDY_ID = "task-trca-n1-source39-v1"
SCHEMA, SCORE_SCHEMA = independent.SCHEMA, independent.SCORE_SCHEMA
ARMS = independent.ARMS
SAMPLES = (125, 188, 250, 500)
SOURCE_IDS = (
    4,
    6,
    8,
    11,
    14,
    21,
    22,
    25,
    28,
    29,
    30,
    31,
    32,
    33,
    37,
    41,
    42,
    43,
    44,
    46,
    54,
    55,
    56,
    61,
    63,
    65,
    67,
    73,
    74,
    77,
    80,
    82,
    83,
    84,
    89,
    92,
    97,
    100,
    102,
)
STATS = independent.STATS
HEADERS = {"schema", "score_schema", "arms"}
ROW_FIELDS = {
    "keys",
    "orders",
    "q",
    "m",
    "available",
    "packet5",
    "s",
    "c",
    "anchors",
    "weights",
    "donor_id",
    "a0_correlations",
    "cached_full_correlations",
    "scores",
    "r",
    "projectors",
    "native_filters",
    "native_full_correlations",
    *STATS,
    *("native_" + name for name in (*STATS, "query_mean", "template_mean")),
}
donor_positions = independent.donor_positions


def profile_record(profile="human"):
    """Only the two prospectively fixed profiles; never infer a grid from data."""
    require(type(profile) is str and profile in ("human", "generated"), "fixed runtime profile")
    generated = profile == "generated"
    return {
        "source_ids": list(SOURCE_IDS[:9] if generated else SOURCE_IDS),
        "interfaces": [0, 1],
        "samples": [17] if generated else list(SAMPLES),
        "budgets": [3, 5],
        "generated": generated,
        "seed": 20260914 if generated else None,
    }


def require(value, message):
    if not value:
        raise ValueError("TEMPORAL_ARTIFACT_AUDIT_FAILURE: " + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _headers(data):
    require(isinstance(data, Mapping), "artifact mapping required")
    for name, value in (("schema", SCHEMA), ("score_schema", SCORE_SCHEMA)):
        item = np.asarray(data.get(name))
        require(
            item.dtype.kind == "U" and item.shape == () and item.item() == value,
            "scalar Unicode " + name,
        )
    arms = np.asarray(data.get("arms"))
    require(
        arms.dtype.kind == "U" and arms.shape == (10,) and arms.tolist() == list(ARMS),
        "exact ten-arm Unicode header",
    )


def pack_evaluation(records):
    """Flatten enriched evaluator returns; scalar headers stay scalar across rows."""
    records = tuple(records)
    require(bool(records), "nonempty evaluation records")
    rows = []
    for record in records:
        _headers(record)
        row = {name: record[name] for name in ROW_FIELDS if name in record}
        for group, prefix, names in (
            ("statistics", "", STATS),
            ("native_statistics", "native_", (*STATS, "query_mean", "template_mean")),
        ):
            require(set(record[group]) == {*names, "samples"}, "exact " + group + " fields")
            require(
                type(record[group]["samples"]) is int
                and record[group]["samples"] == int(np.asarray(record["keys"])[2]),
                "statistics/key samples binding",
            )
            for name in names:
                require(prefix + name not in row, "ambiguous flat/nested statistic")
                row[prefix + name] = record[group][name]
        require(set(row) == ROW_FIELDS, "complete flat evaluation row fields")
        require(
            all(np.asarray(value).dtype.kind in "biuf" for value in row.values()),
            "no object/string arrays in evaluation rows",
        )
        rows.append(row)
    result = {name: np.stack([row[name] for row in rows]) for name in ROW_FIELDS}
    result.update(schema=np.array(SCHEMA), score_schema=np.array(SCORE_SCHEMA), arms=np.array(ARMS))
    return result


def combine_evaluation(parts):
    """Concatenate only row arrays, retaining exact single-file headers."""
    parts = tuple(parts)
    require(bool(parts), "nonempty evaluation parts")
    for part in parts:
        _headers(part)
        require(set(part) == ROW_FIELDS | HEADERS, "exact evaluation fields")
        count = len(part["keys"])
        require(count > 0, "nonempty evaluation part")
        for name in ROW_FIELDS:
            value = np.asarray(part[name])
            require(
                value.dtype.kind in "biuf" and value.ndim > 0 and len(value) == count,
                "numeric aligned row field: " + name,
            )
    result = {name: np.concatenate([p[name] for p in parts]) for name in ROW_FIELDS}
    result.update(schema=np.array(SCHEMA), score_schema=np.array(SCORE_SCHEMA), arms=np.array(ARMS))
    return result


def _grid(data, ids, profile="human"):
    contract = profile_record(profile)
    ids = tuple(ids)
    require(
        ids
        and all(type(pid) is int for pid in ids)
        and tuple(sorted(set(ids))) == ids
        and set(ids) <= set(contract["source_ids"]),
        "fixed source39 participant IDs",
    )
    expected = [
        [p, i, n, k]
        for p in ids
        for i in contract["interfaces"]
        for n in contract["samples"]
        for k in contract["budgets"]
    ]
    keys = np.asarray(data["keys"])
    require(
        keys.dtype.kind in "iu" and keys.shape == (len(expected), 4) and keys.tolist() == expected,
        "exact ordered participant/condition Cartesian grid",
    )
    return keys


def _outer(fit_ids, evaluation_ids, profile="human"):
    source_ids = tuple(profile_record(profile)["source_ids"])
    fitting, evaluation = tuple(fit_ids), tuple(evaluation_ids)
    matches = [
        fold
        for fold in range(3)
        if evaluation == source_ids[fold::3]
        and fitting == tuple(p for p in source_ids if p not in evaluation)
    ]
    require(len(matches) == 1, "exact frozen outer fit/evaluation partition")
    return matches[0]


def audit_training(data, model, outer_ids, evaluation_ids, *, profile="human"):
    """Human-contract wrapper around the independently computed temporal fit audit."""
    _outer(outer_ids, evaluation_ids, profile)
    keys = _grid(data, outer_ids, profile)
    np.testing.assert_array_equal(data["labels"], np.broadcast_to(np.arange(12), (len(keys), 12)))
    result = independent.audit_training(data, model, outer_ids, evaluation_ids)
    return {
        **result,
        "fixed_source39_grid_verified": profile == "human",
        "fixed_profile_grid_verified": True,
        "profile": profile,
    }


def _features(data, keys):
    count = len(keys)
    q = legacy._array(data["q"], "q", (count, 5, 8, 15))
    m = legacy._array(data["m"], "m", (count, 8, 2))
    available = legacy._mask(data["available"], (count, 8))
    packet = legacy._array(data["packet5"], "packet5", (count, 5, 8), finite=False)
    orders = np.asarray(data["orders"])
    require(
        orders.dtype.kind in "iu" and orders.shape == (count,) and np.isin(orders, (0, 1)).all(),
        "binary acquisition orders",
    )
    identities, prefixes, subject_orders, weights = {}, {}, {}, {}
    for j, (pid, interface, _, k) in enumerate(keys):
        require(np.isnan(packet[j, k:]).all(), "future support metadata padding")
        expected, observed = legacy.independent_metadata(packet[j, :k])
        np.testing.assert_allclose(m[j], expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(available[j], observed)
        np.testing.assert_array_equal(
            q[j, ..., 4], np.broadcast_to(np.isfinite(packet[j, :k]).mean(0), (5, 8))
        )
        for slot, value in ((3, np.log(k)), (5, orders[j]), (6, interface != orders[j])):
            np.testing.assert_allclose(q[j, ..., slot], value, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(q[j, ..., 7:], np.broadcast_to(np.eye(8), (5, 8, 8)))
        require(
            pid not in subject_orders or orders[j] == subject_orders[pid],
            "participant order across conditions",
        )
        subject_orders[pid] = orders[j]
        identity = (int(pid), int(interface), int(k))
        if identity in identities:
            np.testing.assert_array_equal(packet[j], packet[identities[identity]])
        identities[identity] = j
        prefix = (int(pid), int(interface))
        if prefix in prefixes:
            previous = prefixes[prefix]
            common = min(k, keys[previous, 3])
            np.testing.assert_array_equal(packet[j, :common], packet[previous, :common])
        if prefix not in prefixes or k > keys[prefixes[prefix], 3]:
            prefixes[prefix] = j
        if interface in weights:
            np.testing.assert_array_equal(data["weights"][j], weights[interface])
        weights[interface] = data["weights"][j]
    return q, m, available, packet


def _native_correlations(w, statistics):
    """Band-resolved native Pearson, also ruling out weight-cancelling corruption."""
    result = np.empty((len(statistics["query_mean"]), 5, 12))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for band in range(5):
            filters = w[band]
            mx = statistics["query_mean"][:, band] @ filters
            mt = statistics["template_mean"][:, band] @ filters
            mx, mt = mx - mx.mean(-1, keepdims=True), mt - mt.mean(-1, keepdims=True)
            xx = np.einsum("ik,nij,jk->n", filters, statistics["query_gram"][:, band], filters)
            tt = np.einsum("ik,cij,jk->c", filters, statistics["template_gram"][:, band], filters)
            xx += statistics["samples"] * np.square(mx).sum(-1)
            tt += statistics["samples"] * np.square(mt).sum(-1)
            require(np.all(xx > 0) and np.all(tt > 0), "native projected variance")
            dot = np.einsum(
                "ik,ncij,jk->nc", filters, statistics["cross_gram"][:, :, band], filters
            )
            dot += statistics["samples"] * mx @ mt.T
            result[:, band] = np.clip(dot / np.sqrt(xx[:, None]) / np.sqrt(tt[None]), -1, 1)
    return result


def audit_evaluation(data, pipeline, evaluation_ids, *, profile="human"):
    """Replay all ten arms and exact integer counts for the externally fixed fold."""
    _headers(data)
    require(set(data) == ROW_FIELDS | HEADERS, "exact evaluation artifact fields")
    independent._schema(pipeline)
    _outer(pipeline["fit_ids"], evaluation_ids, profile)
    keys = _grid(data, evaluation_ids, profile)
    count = len(keys)
    q, m, available, packet = _features(data, keys)
    shapes = {
        "s": (5, 12, 8, 8),
        "c": (5, 12, 8, 8),
        "anchors": (5, 12, 8),
        "weights": (5,),
        "scores": (10, 48, 12),
        "r": (8, 5, 8),
        "projectors": (8, 5, 12, 8, 8),
        "native_filters": (5, 8, 12),
        "native_full_correlations": (48, 5, 12),
        "cached_full_correlations": (48, 5, 12),
        "a0_correlations": (48, 5, 12),
        "query_gram": (48, 5, 8, 8),
        "template_gram": (12, 5, 8, 8),
        "cross_gram": (48, 12, 5, 8, 8),
        "native_query_gram": (48, 5, 8, 8),
        "native_template_gram": (12, 5, 8, 8),
        "native_cross_gram": (48, 12, 5, 8, 8),
        "native_query_mean": (48, 5, 8),
        "native_template_mean": (12, 5, 8),
    }
    for name, shape in shapes.items():
        legacy._array(data[name], name, (count, *shape))
    require(np.all(data["weights"] > 0), "positive native weights")
    for name in ("a0_correlations", "native_full_correlations", "cached_full_correlations"):
        require(np.max(np.abs(data[name])) <= 1 + 1e-12, "bounded " + name)
    donor_ids = np.asarray(data["donor_id"])
    require(donor_ids.dtype.kind in "iu" and donor_ids.shape == (count,), "integer donor IDs")
    donors = donor_positions(data, range(count))
    require(all(donor_ids[j] == keys[d, 0] for j, d in donors.items()), "exact frozen donor map")
    errors = {name: 0.0 for name in ("r", "projectors", "scores", "native", "centered", "cached")}
    truth = np.tile(np.arange(12), 4)
    integer_checks = 0

    def compare(name, expected, actual, tolerance=1e-8):
        np.testing.assert_allclose(expected, actual, atol=tolerance, rtol=0)
        errors[name] = max(errors[name], float(np.max(np.abs(expected - actual))))

    def score_check(name, expected, actual, tolerance=1e-8):
        nonlocal integer_checks
        compare(name, expected, actual, tolerance)
        np.testing.assert_array_equal(expected.argmax(-1), actual.argmax(-1))
        require(
            int(np.count_nonzero(expected.argmax(-1) == truth))
            == int(np.count_nonzero(actual.argmax(-1) == truth)),
            "integer correct counts",
        )
        integer_checks += 1

    for j, (_, _, n, k) in enumerate(keys):
        weight, w = data["weights"][j], data["native_filters"][j]
        np.testing.assert_array_equal(data["anchors"][j], w.transpose(0, 2, 1))
        temporal = {name: data[name][j] for name in STATS}
        temporal["samples"] = int(n)
        native = {
            name: data["native_" + name][j] for name in (*STATS, "query_mean", "template_mean")
        }
        native["samples"] = int(n)
        for name in STATS:
            # Temporal construction subtracts sample zero before centering;
            # native construction centers directly. Same algebra, not bit identity.
            np.testing.assert_allclose(temporal[name], native[name], atol=1e-10, rtol=0)
        full = legacy.independent_scores(w, native, weight)
        score_check("native", full, data["scores"][j, 0])
        compare("native", _native_correlations(w, native), data["native_full_correlations"][j])
        exact_full = np.einsum("nbc,b->nc", data["native_full_correlations"][j], weight)
        np.testing.assert_array_equal(exact_full, data["scores"][j, 0])
        compare(
            "cached", data["native_full_correlations"][j], data["cached_full_correlations"][j], 1e-9
        )
        cached = np.einsum("nbc,b->nc", data["cached_full_correlations"][j], weight)
        score_check("cached", cached, exact_full)
        full_projectors = np.einsum("bik,bjk->bkij", w, w)
        centered, _ = independent.independent_scores(full_projectors, temporal, weight)
        score_check("centered", centered, data["scores"][j, 1])
        stale_m, stale_available = legacy.independent_metadata(np.repeat(packet[j, :1], k, 0))
        for index, arm in enumerate(ARMS[2:]):
            r = independent.independent_prior(
                q[j],
                m[j],
                available[j],
                pipeline,
                arm,
                donor_m=m[donors[j]],
                stale_m=stale_m,
                stale_available=stale_available,
            )
            f = independent.independent_projectors(data["s"][j], data["c"][j], r)
            scores, _ = independent.independent_scores(f, temporal, weight)
            compare("r", r, data["r"][j, index])
            compare("projectors", f, data["projectors"][j, index])
            score_check("scores", scores, data["scores"][j, index + 2])
        np.testing.assert_array_equal(data["scores"][j, 3], data["scores"][j, 9])
    return {
        "status": "TEMPORAL_EVALUATION_AUDIT_PASS",
        "cases": count,
        "max_abs_errors": errors,
        "integer_count_comparisons": integer_checks,
        "ten_arm_argmax_exact": True,
        "missing_exact_q": True,
        "both_full_same_native_filters": True,
        "fixed_source39_grid_verified": profile == "human",
        "fixed_profile_grid_verified": True,
        "profile": profile,
        "scope": "saved Q/S/C/prefix M/Grams/native filters through all ten scores/counts; "
        "not independent raw preprocessing, Q15, native fitting, Adam or access provenance",
    }
