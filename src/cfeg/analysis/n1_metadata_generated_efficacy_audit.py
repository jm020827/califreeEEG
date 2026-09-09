"""Read-only, independent generated-capacity artifact audit.

Raw waveforms, native geometry, scalers, donors, bounded priors, generalized
eigenvectors and both Pearson scores are reconstructed here without importing
the producer, fixture generator, learner, evaluator or Torch. Q15 and M use the
unchanged feature constructors: that shared implementation is an explicit limit
of independence. Model traces are checked, not independently trained/replayed.
PASS means artifact consistency, not a positive efficacy outcome or human access.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import linalg, stats

from cfeg.analysis import task_trca_shape_features as features

SCHEMA = "n1-metadata-generated-efficacy-audit-v1"
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
REGIMES = ("coupled", "null")
ARTIFACTS = frozenset(
    {
        "primary_start.json",
        "fixture.json",
        "fit.npz",
        "support.npz",
        "query.npz",
        "fit_features.npz",
        "models.json",
        "freeze.json",
        "events.jsonl",
        "evaluation.npz",
    }
)
LIMITATIONS = [
    "Q15 and M reuse unchanged feature constructors; not an independent feature implementation.",
    "Stored optimizer traces/counts are checked; Adam training and gradients are not independently replayed.",
    "Noise is checked by waveform and orthogonality invariants, not byte-exact regeneration of its QR draw.",
    "Chronology is the pinned program event record, not OS/cryptographic non-access evidence.",
    "Local artifacts do not authenticate an external launch/code-authority envelope or resource supervisor.",
    "Generated constructive controls and conditional descriptive intervals do not establish human efficacy.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class _Checks:
    def __init__(self, atol):
        self.atol = atol
        self.checks = {}
        self.differences = {}

    def close(self, name, actual, expected, *, exact=False):
        actual, expected = np.asarray(actual), np.asarray(expected)
        _require(actual.shape == expected.shape, name + ": shape mismatch")
        _require(
            actual.dtype.kind in "biuf" and expected.dtype.kind in "biuf", name + ": numeric only"
        )
        _require(np.isfinite(actual).all() and np.isfinite(expected).all(), name + ": nonfinite")
        difference = float(np.max(np.abs(actual.astype(float) - expected.astype(float)), initial=0))
        self.differences[name] = max(difference, self.differences.get(name, 0.0))
        _require(
            np.array_equal(actual, expected) if exact else difference <= self.atol,
            name + ": values differ (max_abs=" + repr(difference) + ")",
        )

    def done(self, name):
        self.checks[name] = True


def _descriptor(path):
    """Hash only immutable, unaliased, single-link regular artifacts."""
    path = Path(path)
    _require(
        path.is_absolute() and path.resolve() == path and not path.is_symlink(),
        "unaliased absolute artifact path required",
    )
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            "single-link regular artifact required",
        )
        _require(before.st_mode & 0o222 == 0, "artifact must have no write bits")
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
        after = os.fstat(stream.fileno())
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "artifact changed while hashing",
        )
    return {"sha256": digest, "bytes": before.st_size}


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON constant " + value)

    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=invalid)


def _timestamp(value):
    result = datetime.fromisoformat(value)
    _require(
        result.tzinfo is not None and result.utcoffset().total_seconds() == 0,
        "UTC timestamp required",
    )
    return result


def _config(config):
    # Nonregistered toy seeds/group counts are allowed by this pure audit API;
    # production config equality is authenticated by its caller and all artifacts.
    fixed = {
        "schema": "n1-metadata-generated-efficacy-config-v1",
        "samples": 256,
        "sampling_rate": 250,
        "support_blocks": 3,
        "classes": 12,
        "bands": 5,
        "channels": 8,
        "fit_id_base": 11000,
        "evaluation_id_base": 21000,
        "noise_amplitude": 0.2,
        "delta_halfwidth": 0.01,
        "regularization": 0.001,
        "weights": [1, 1, 1, 1, 1],
        "oracle_coefficients": [2, 0, 0],
        "updates_per_head": 200,
        "updates_total": 1600,
        "primary_seconds": 900,
        "audit_seconds": 600,
        "rss_limit_bytes": 8589934592,
        "address_limit_bytes": 17179869184,
        "output_limit_bytes": 4294967296,
        "numeric_atol": 1e-8,
        "oracle_min_gain_pp": 5,
        "learner_min_gain_pp": 1,
        "null_equivalence_pp": 1,
        "q_accuracy_range_percent": [10, 90],
    }
    _require(
        isinstance(config, dict)
        and set(config) == {*fixed, "seed", "fit_groups", "evaluation_groups"},
        "exact config fields required",
    )
    _require(
        all(config[key] == value for key, value in fixed.items()), "frozen configuration mismatch"
    )
    _require(
        type(config["seed"]) is int and config["seed"] >= 0, "integer nonnegative seed required"
    )
    for key, maximum in (("fit_groups", 8), ("evaluation_groups", 32)):
        _require(
            type(config[key]) is int and 2 <= config[key] <= maximum,
            "bounded paired group count required",
        )
    if config["seed"] == 20260916:
        _require(
            config["fit_groups"] == 8 and config["evaluation_groups"] == 32,
            "registered seed requires the complete frozen group grid",
        )


def _shapes(config):
    f, e, n = 2 * config["fit_groups"], 2 * config["evaluation_groups"], config["samples"]

    def base(count):
        return {
            "ids": (count,),
            "groups": (count,),
            "members": (count,),
            "support": (count, 3, 12, 5, 8, n),
            "packet_coupled": (count, 3, 8),
            "packet_null": (count, 3, 8),
            "bad": (count, 8),
            "null_mask": (count // 2, 8),
            "phases": (count // 2, 5, 12),
        }

    return {
        "fit.npz": dict(base(f), source=(f, 12, 5, 8, n), source_delta=(f // 2, 12)),
        "support.npz": base(e),
        "query.npz": {
            "ids": (e,),
            "groups": (e,),
            "members": (e,),
            "query": (e, 48, 5, 8, n),
            "query_delta": (e // 2, 4, 12),
        },
        "fit_features.npz": {
            "q": (2, f, 5, 8, 15),
            "m": (2, f, 8, 2),
            "s": (2, f, 5, 12, 8, 8),
            "c": (2, f, 5, 12, 8, 8),
        },
        "evaluation.npz": {
            "scores": (2, e, 10, 48, 12),
            "r": (2, e, 8, 5, 8),
            "projectors": (2, e, 8, 5, 12, 8, 8),
            "oracle_scores": (2, e, 48, 12),
            "oracle_r": (2, e, 5, 8),
            "oracle_projectors": (2, e, 5, 12, 8, 8),
        },
    }


def _arrays(path, shapes):
    """Validate exact ZIP members and byte lengths before allocating arrays."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        _require(
            len(entries) == len(shapes)
            and {x.filename for x in entries} == {k + ".npy" for k in shapes},
            path.name + ": exact NPZ member inventory required",
        )
        for entry in entries:
            name = entry.filename[:-4]
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                _require(version in ((1, 0), (2, 0)), "unsupported NPY header version")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, fortran, dtype = reader(stream)
                _require(
                    shape == shapes[name] and not fortran,
                    path.name + "/" + name + ": header shape/order",
                )
                if name in ("ids", "groups", "members"):
                    good_type = dtype.kind in "iu" and dtype.itemsize <= 8
                elif name in ("bad", "null_mask"):
                    good_type = dtype.kind in "biu" and dtype.itemsize <= 8
                else:
                    good_type = dtype.kind == "f" and dtype.itemsize == 8
                _require(
                    good_type and not dtype.hasobject, "numeric float64/identity array required"
                )
                _require(
                    entry.file_size == stream.tell() + math.prod(shape) * dtype.itemsize,
                    "NPY payload byte length mismatch",
                )
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in shapes}
    _require(
        all(np.isfinite(value).all() for value in result.values()),
        path.name + ": finite arrays required",
    )
    return result


def _rng(config, split, group, purpose):
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([config["seed"], split, group, purpose]))
    )


def _signal(phases, samples):
    angle = 2 * np.pi * np.arange(6, 18)[:, None] * np.arange(samples)[None] / samples
    return np.sqrt(2) * np.sin(angle[None] + phases[:, :, None])


def _basis(samples):
    time_axis = np.arange(samples) / samples
    columns = [np.full(samples, 1 / np.sqrt(samples))]
    for frequency in sorted({harmonic * base for harmonic in range(1, 6) for base in range(6, 18)}):
        for function in (np.sin, np.cos):
            columns.append(np.sqrt(2 / samples) * function(2 * np.pi * frequency * time_axis))
    return np.stack(columns, axis=1)


def _waveform(signal, bad, delta):
    values = np.asarray(delta).reshape(-1)
    labels = np.arange(len(values)) % 12
    true = signal[:, labels].transpose(1, 0, 2)
    decoy = signal[:, (labels + 1) % 12].transpose(1, 0, 2)
    return np.where(
        bad[None, None, :, None],
        (1 - values[:, None, None, None]) * decoy[:, :, None],
        (1 + values[:, None, None, None]) * true[:, :, None],
    )


def _raw_split(data, query, config, split, checks):
    count = config["fit_groups" if split == 0 else "evaluation_groups"]
    base = config["fit_id_base" if split == 0 else "evaluation_id_base"]
    n, sigma = config["samples"], config["noise_amplitude"]
    prefix = "fit" if split == 0 else "evaluation"
    identities = {
        "ids": base + 1 + np.arange(2 * count),
        "groups": np.tile(np.arange(count), 2),
        "members": np.repeat(np.arange(2), count),
    }
    for key, expected in identities.items():
        checks.close(prefix + "_" + key, data[key], expected, exact=True)
        if query is not None:
            checks.close("query_" + key, query[key], expected, exact=True)
    basis = _basis(n)
    checks.close("Fourier_basis_orthogonality", basis.T @ basis, np.eye(basis.shape[1]))
    for group in range(count):
        bad = np.zeros(8, dtype=bool)
        null = np.zeros(8, dtype=bool)
        bad[_rng(config, split, group, 1).choice(8, 4, replace=False)] = True
        null[_rng(config, split, group, 2).choice(8, 4, replace=False)] = True
        phases = _rng(config, split, group, 3).uniform(0, 2 * np.pi, (5, 12))
        checks.close(prefix + "_phases", data["phases"][group], phases, exact=True)
        checks.close(prefix + "_null_mask", data["null_mask"][group], null, exact=True)
        signal = _signal(phases, n)
        support = data["support"][group]
        # Every class carries exactly the same group/band/block/channel noise.
        noise = (support[:, 0] - signal[:, 0][None, :, None]) / sigma
        expected = signal.transpose(1, 0, 2)[None, :, :, None] + sigma * noise[:, None]
        checks.close(prefix + "_support_formula", support, expected)
        for band in range(5):
            columns = noise[:, band].reshape(24, n).T
            checks.close(prefix + "_noise_orthogonality", columns.T @ columns, n * np.eye(24))
            checks.close(
                prefix + "_noise_harmonics", basis.T @ columns, np.zeros((basis.shape[1], 24))
            )
        checks.close(prefix + "_signal_norm", np.sum(signal * signal, axis=-1), np.full((5, 12), n))
        checks.close(prefix + "_signal_mean", signal.mean(axis=-1), np.zeros((5, 12)))
        delta = _rng(config, split, group, 5 if split == 0 else 6).uniform(
            -config["delta_halfwidth"], config["delta_halfwidth"], (12,) if split == 0 else (4, 12)
        )
        checks.close(
            prefix + "_delta",
            (data if split == 0 else query)["source_delta" if split == 0 else "query_delta"][group],
            delta,
            exact=True,
        )
        for member in range(2):
            index = group + member * count
            member_bad = bad if member == 0 else ~bad
            checks.close(prefix + "_bad", data["bad"][index], member_bad, exact=True)
            checks.close(prefix + "_member_support", data["support"][index], support, exact=True)
            for regime, mask in (("coupled", member_bad), ("null", null)):
                packet = np.repeat(np.expm1(1 + 2 * mask.astype(float))[None], 3, axis=0)
                checks.close(prefix + "_packet_" + regime, data["packet_" + regime][index], packet)
            values = (data if split == 0 else query)["source" if split == 0 else "query"][index]
            checks.close(prefix + "_waveform", values, _waveform(signal, member_bad, delta))
    checks.done(prefix + "_raw_formula_rng_and_noise_geometry")


def _matrices(support):
    s, c = np.empty((5, 12, 8, 8)), np.empty((5, 12, 8, 8))
    for band in range(5):
        for label in range(12):
            trials = support[:, label, band]
            total = trials.sum(axis=0)
            stacked = trials.transpose(0, 2, 1).reshape(-1, 8)
            s[band, label] = total @ total.T - stacked.T @ stacked
            centered = stacked - stacked.mean(axis=0)
            c[band, label] = centered.T @ centered
    return s, c


def _case_features(data, config, checks, prefix):
    frequencies = np.arange(6, 18) * config["sampling_rate"] / config["samples"]
    q, m, matrices = [], [[], []], []
    geometry_max = 0.0
    ideal_s = np.full((5, 12, 8, 8), 6 * config["samples"], dtype=float)
    ideal_c = np.broadcast_to(
        3 * config["samples"] * (np.ones((8, 8)) + 0.04 * np.eye(8)), ideal_s.shape
    )
    for index, support in enumerate(data["support"]):
        q.append(features.support_q_mask(support, np.ones((3, 8), dtype=bool), 0, 0, frequencies))
        s, c = _matrices(support)
        checks.close(prefix + "_analytic_S", s, ideal_s)
        checks.close(prefix + "_analytic_C", c, ideal_c)
        geometry_max = max(
            geometry_max, float(np.max(np.abs(s - ideal_s))), float(np.max(np.abs(c - ideal_c)))
        )
        matrices.append((s, c))
        for regime_index, regime in enumerate(REGIMES):
            packet = data["packet_" + regime][index]
            value, available = features.metadata_features(packet)
            _require(available.all(), "all channels must be available")
            logarithm = np.log1p(packet)
            expected = np.stack(
                (logarithm.mean(axis=0) - logarithm.mean(), logarithm.std(axis=0)), axis=-1
            )
            checks.close(prefix + "_M_formula", value, expected)
            checks.close(prefix + "_M2_zero", value[:, 1], np.zeros(8))
            stale, observed = features.metadata_features(np.repeat(packet[:1], 3, axis=0))
            checks.close(prefix + "_stale_M", stale, value, exact=True)
            _require(observed.all(), "full stale availability")
            m[regime_index].append(value)
    q, m = np.stack(q), np.asarray(m)
    s = np.stack([pair[0] for pair in matrices])
    c = np.stack([pair[1] for pair in matrices])
    checks.done(prefix + "_raw_feature_and_native_geometry_bridge")
    return q, m, s, c, geometry_max


def _scaler(values):
    values = values.reshape(-1, values.shape[-1])
    mean, scale = values.mean(axis=0), values.std(axis=0, ddof=0)
    return mean, np.where(scale < 1e-12, 1.0, scale)


def _pipeline(record, ids, q, m, config, checks):
    _require(
        record["schema"] == "task-trca-n1-integration-v1"
        and record["score_schema"] == "component-time-centered-ensemble-pearson-v1",
        "original N1 algorithm schema required",
    )
    _require(
        record["fit_ids"] == ids.tolist() and record["lambda"] == config["regularization"],
        "pipeline fit IDs/lambda",
    )
    for name, values in (("q_scaler", q), ("m_scaler", m), ("q2_scaler", q[..., 1:3])):
        scaler = record[name]
        _require(
            set(scaler) == {"mean", "scale", "fit_ids"} and scaler["fit_ids"] == ids.tolist(),
            "scaler fit ownership",
        )
        mean, scale = _scaler(values)
        checks.close(name + "_mean", scaler["mean"], mean)
        checks.close(name + "_scale", scaler["scale"], scale)
        _require(np.all(np.asarray(scaler["scale"]) > 0), "positive scaler scale")
    _require(set(record["residuals"]) == {"Q2", "QM", "SHAM_REFIT"}, "exact three residual heads")
    for name, head, size in [("Q", record["Q"], 16)] + [
        (k, record["residuals"][k], 3) for k in ("Q2", "QM", "SHAM_REFIT")
    ]:
        coefficient = np.asarray(head["coefficients"], dtype=float)
        _require(
            coefficient.shape == (size,) and np.isfinite(coefficient).all(),
            "finite coefficient dimensions",
        )
        trace = head["trace"]
        _require(
            head["steps"] == 200
            and len(trace) == 200
            and [row["step"] for row in trace] == list(range(1, 201)),
            "exact 200-step trace per head",
        )
        values = np.asarray(
            [
                [row[key] for key in ("loss_before_step", "ce_before_step", "gradient_norm")]
                for row in trace
            ],
            dtype=float,
        )
        _require(
            np.isfinite(values).all() and (values >= 0).all(), "finite nonnegative trace values"
        )
        _require(
            np.isfinite([head["initial_loss"], head["final_loss"]]).all()
            and head["final_loss"] >= 0,
            "finite model losses",
        )
        checks.close("initial_trace_loss_" + name, head["initial_loss"], values[0, 0], exact=True)
        checks.close(
            "zero_initialized_" + name + "_penalty", values[0, 0], values[0, 1], exact=True
        )
    expected_donors = [
        {"case": [int(pid), 0, config["samples"], 3], "donor_id": int(ids[(i + 1) % len(ids)])}
        for i, pid in enumerate(ids)
    ]
    _require(
        record["donors"] == expected_donors, "partition-local member-major cyclic donor mismatch"
    )
    digest = hashlib.sha256()
    for values in (
        record["q_scaler"]["mean"],
        record["q_scaler"]["scale"],
        record["Q"]["coefficients"],
    ):
        digest.update(np.asarray(values, dtype="<f8").tobytes())
    _require(record["q_hash"] == digest.hexdigest(), "frozen Q content hash mismatch")
    return digest.hexdigest()


def _prior(q, m, donor_m, record, arm, *, oracle=False):
    def transform(value, name):
        scaler = record[name]
        return (value - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])

    def head(value, coefficient, fraction):
        coefficient = np.asarray(coefficient)
        return fraction * np.log(2) / 2 * np.tanh(value @ coefficient[:-1] + coefficient[-1])

    if arm == "ISO" and not oracle:
        return np.ones((5, 8))
    logits = head(transform(q, "q_scaler"), record["Q"]["coefficients"], 0.8)
    if oracle:
        logits += head(transform(m, "m_scaler"), [2, 0, 0], 0.2)
    elif arm not in ("Q", "MISSING"):
        if arm == "Q2":
            value = transform(q[..., 1:3], "q2_scaler")
            coefficient = record["residuals"]["Q2"]["coefficients"]
        else:
            value = transform(donor_m if arm in ("SHAM_REFIT", "PERMUTED") else m, "m_scaler")
            coefficient = record["residuals"]["SHAM_REFIT" if arm == "SHAM_REFIT" else "QM"][
                "coefficients"
            ]
        logits += head(value, coefficient, 0.2)
    exponential = np.exp(logits - logits.max(axis=-1, keepdims=True))
    result = 8 * exponential / exponential.sum(axis=-1, keepdims=True)
    _require(
        np.isfinite(result).all()
        and np.all(result > 0)
        and np.all(result.max(-1) / result.min(-1) <= 2 + 1e-12)
        and np.all(result <= 16 / 9 + 1e-12),
        "bounded prior violated",
    )
    return result


def _projectors(s, c, r=None):
    """Independent generalized-eigh forward; r=None is the unregularized native control."""
    for value in (s, c):
        norm = np.linalg.norm(value, axis=(-2, -1))
        skew = np.linalg.norm(value - value.swapaxes(-1, -2), axis=(-2, -1))
        _require(
            np.isfinite(value).all()
            and np.all(skew / np.maximum(norm, np.finfo(float).tiny) <= 1e-12),
            "raw matrix symmetry guard failed",
        )
    s, c = (s + s.swapaxes(-1, -2)) / 2, (c + c.swapaxes(-1, -2)) / 2
    projectors = np.empty_like(s)
    for band in range(5):
        for label in range(12):
            metric = c[band, label]
            minimum = float(linalg.eigvalsh(metric)[0])
            _require(np.isfinite(minimum) and minimum > 0, "C is not SPD")
            denominator = (
                metric if r is None else metric + np.diag(0.1 * minimum / (16 / 9) * r[band])
            )
            roots, vectors = linalg.eigh(s[band, label], denominator, driver="gvd")
            _require(
                np.isfinite(roots).all()
                and np.isfinite(vectors).all()
                and roots[-1] - roots[-2] > 1e-10 * max(1.0, float(np.max(np.abs(roots)))),
                "nonfinite/degenerate top eigensystem",
            )
            vector = vectors[:, -1]
            norm = float(vector @ metric @ vector)
            _require(1 / 1.1 - 1e-10 <= norm <= 1 + 1e-10, "bounded C/B identity failed")
            projectors[band, label] = np.outer(vector, vector) / norm
    return projectors


def _statistics(query, templates):
    query = query - query.mean(axis=-1, keepdims=True)
    templates = templates - templates.mean(axis=-1, keepdims=True)
    return (
        query @ query.swapaxes(-1, -2),
        templates @ templates.swapaxes(-1, -2),
        np.einsum("qbct,lbkt->qlbck", query, templates),
    )


def _scores(projectors, statistics, weights):
    xx, tt, xt = statistics
    scores = np.zeros((len(xx), len(tt)))
    for band in range(5):
        ensemble = projectors[band].sum(axis=0)
        xnorm = np.trace(ensemble @ xx[:, band], axis1=-2, axis2=-1)
        tnorm = np.trace(ensemble @ tt[:, band], axis1=-2, axis2=-1)
        _require(np.all(xnorm > 0) and np.all(tnorm > 0), "zero projected variance; no floor")
        numerator = np.trace(ensemble @ xt[:, :, band], axis1=-2, axis2=-1)
        correlation = numerator / np.sqrt(xnorm[:, None] * tnorm[None])
        _require(np.isfinite(correlation).all(), "nonfinite score arithmetic")
        scores += weights[band] * np.clip(correlation, -1, 1)
    return scores


def _native_scores(projectors, query, templates, weights):
    # Sign-free native contraction is exact only when each channel is already
    # temporally centered. The frozen Fourier/noise DGP enforces that property;
    # validate it explicitly, rather than silently use temporal centering for FULL.
    _require(
        np.max(np.abs(query.mean(-1))) <= 1e-12 and np.max(np.abs(templates.mean(-1))) <= 1e-12,
        "generated native/global equivalence requires zero temporal means",
    )
    xx, tt = query @ query.swapaxes(-1, -2), templates @ templates.swapaxes(-1, -2)
    xt = np.einsum("qbct,lbkt->qlbck", query, templates)
    return _scores(projectors, (xx, tt, xt), weights)


def _replay(data, query, q, m, s, c, models, saved, config, checks):
    count = len(data["ids"])
    scores = np.empty((2, count, 10, 48, 12))
    oracle_scores = np.empty((2, count, 48, 12))
    actuation = {}
    weights = np.asarray(config["weights"], dtype=float)
    for regime_index, regime in enumerate(REGIMES):
        record = models[regime]
        values = {key: 0.0 for key in ("max_abs_R", "max_abs_F", "max_abs_J", "max_abs_score")}
        values["argmax_changed"] = 0
        values["coefficient_max_abs"] = float(
            np.max(np.abs(record["residuals"]["QM"]["coefficients"]))
        )
        values["gradient_max"] = float(
            max(row["gradient_norm"] for row in record["residuals"]["QM"]["trace"])
        )
        for index in range(count):
            templates = data["support"][index].mean(axis=0)
            raw_query = query["query"][index]
            statistics = _statistics(raw_query, templates)
            native = _projectors(s[index], c[index])
            scores[regime_index, index, 0] = _native_scores(native, raw_query, templates, weights)
            scores[regime_index, index, 1] = _scores(native, statistics, weights)
            priors, projectors = [], []
            for arm_index, arm in enumerate(ARMS[2:]):
                prior = _prior(
                    q[index],
                    m[regime_index, index],
                    m[regime_index, (index + 1) % count],
                    record,
                    arm,
                )
                projector = _projectors(s[index], c[index], prior)
                priors.append(prior)
                projectors.append(projector)
                checks.close("R_" + arm, saved["r"][regime_index, index, arm_index], prior)
                checks.close(
                    "F_" + arm, saved["projectors"][regime_index, index, arm_index], projector
                )
                scores[regime_index, index, arm_index + 2] = _scores(projector, statistics, weights)
            prior = _prior(
                q[index], m[regime_index, index], m[regime_index, index], record, "QM", oracle=True
            )
            projector = _projectors(s[index], c[index], prior)
            checks.close("R_ORACLE", saved["oracle_r"][regime_index, index], prior)
            checks.close("F_ORACLE", saved["oracle_projectors"][regime_index, index], projector)
            oracle_scores[regime_index, index] = _scores(projector, statistics, weights)
            qindex, mindex = 1, 3
            for key, difference in (
                ("max_abs_R", priors[mindex] - priors[qindex]),
                ("max_abs_F", projectors[mindex] - projectors[qindex]),
                ("max_abs_J", projectors[mindex].sum(1) - projectors[qindex].sum(1)),
                ("max_abs_score", scores[regime_index, index, 5] - scores[regime_index, index, 3]),
            ):
                values[key] = max(values[key], float(np.max(np.abs(difference))))
            values["argmax_changed"] += int(
                np.count_nonzero(
                    scores[regime_index, index, 5].argmax(-1)
                    != scores[regime_index, index, 3].argmax(-1)
                )
            )
        actuation[regime] = values
    for index, arm in enumerate(ARMS):
        checks.close("scores_" + arm, saved["scores"][:, :, index], scores[:, :, index])
    checks.close("scores_ORACLE", saved["oracle_scores"], oracle_scores)
    checks.close("all_arm_argmax", saved["scores"].argmax(-1), scores.argmax(-1), exact=True)
    checks.close(
        "oracle_argmax", saved["oracle_scores"].argmax(-1), oracle_scores.argmax(-1), exact=True
    )
    checks.close("MISSING_equals_Q", saved["scores"][:, :, 9], saved["scores"][:, :, 3], exact=True)
    checks.close("STALE_equals_QM", saved["scores"][:, :, 8], saved["scores"][:, :, 5], exact=True)
    checks.done("all_ten_arms_oracle_R_F_native_temporal_scores_and_argmax")
    return scores, oracle_scores, actuation


def _summary(scores, oracle, config, validity, actuation):
    groups = config["evaluation_groups"]
    combined = np.concatenate((scores, oracle[:, :, None]), axis=2)
    correct = combined.argmax(axis=-1) == np.arange(48) % 12
    # Explicit member-major reshape keeps counterfactual partners together.
    accuracy = correct.reshape(2, 2, groups, 11, 48).mean(axis=(1, 4)) * 100
    gain = accuracy[0, :, 5] - accuracy[0, :, 3]
    null = accuracy[1, :, 5] - accuracy[1, :, 3]
    raw = {
        "coupled_QM_minus_Q": gain,
        "coupled_QM_minus_Q2": accuracy[0, :, 5] - accuracy[0, :, 4],
        "coupled_QM_minus_SHAM_REFIT": accuracy[0, :, 5] - accuracy[0, :, 6],
        "coupled_ORACLE_minus_Q": accuracy[0, :, 10] - accuracy[0, :, 3],
        "null_QM_minus_Q": null,
        "interaction": gain - null,
    }
    contrasts = {}
    for name, value in raw.items():
        mean = float(np.mean(value))
        se = float(np.sqrt(np.sum((value - mean) ** 2) / (groups - 1) / groups))
        half = float(stats.t.ppf(0.975, groups - 1)) * se
        contrasts[name] = {
            "mean_pp": mean,
            "mcse_pp": se,
            "low_pp": mean - half,
            "high_pp": mean + half,
            "n_groups": groups,
        }
    low, high = config["q_accuracy_range_percent"]
    headroom, negative = contrasts["coupled_ORACLE_minus_Q"], contrasts["null_QM_minus_Q"]
    checks = dict(validity)
    checks["positive_headroom"] = bool(
        low <= accuracy[0, :, 3].mean() <= high
        and headroom["mean_pp"] >= config["oracle_min_gain_pp"]
        and headroom["low_pp"] > 0
    )
    checks["null_equivalence"] = bool(
        negative["low_pp"] >= -config["null_equivalence_pp"]
        and negative["high_pp"] <= config["null_equivalence_pp"]
    )
    checks["positive_improvement"] = all(
        contrasts[name]["mean_pp"] >= config["learner_min_gain_pp"]
        and contrasts[name]["low_pp"] > 0
        for name in (
            "coupled_QM_minus_Q",
            "coupled_QM_minus_Q2",
            "coupled_QM_minus_SHAM_REFIT",
            "interaction",
        )
    )
    checks["metadata_learned_and_actuated"] = all(
        actuation["coupled"][key] > 0
        for key in (
            "coefficient_max_abs",
            "gradient_max",
            "max_abs_R",
            "max_abs_score",
            "argmax_changed",
        )
    )
    if not all(validity.values()):
        terminal = "EFFICACY_NOT_EVALUATED"
    elif not checks["positive_headroom"]:
        terminal = "POSITIVE_CONTROL_NOT_QUALIFIED"
    elif not checks["null_equivalence"]:
        terminal = "NEGATIVE_CONTROL_NOT_QUALIFIED"
    elif not (checks["positive_improvement"] and checks["metadata_learned_and_actuated"]):
        terminal = "LEARNER_CAPACITY_NOT_ESTABLISHED"
    else:
        terminal = "GENERATED_M_CAPACITY_PASS"
    return {
        "terminal": terminal,
        "group_accuracy_percent": accuracy.tolist(),
        "contrasts": contrasts,
        "checks": checks,
    }


def _chronology(folder, result, fixture, models, freeze, descriptors, config, checks):
    start = _json(folder / "primary_start.json")
    _require(
        start["schema"] == "n1-metadata-generated-efficacy-primary-start-v1"
        and start["config"] == config
        and start["registered_attempt"] == 1
        and start["charged_updates"] == 1600,
        "single-attempt start/config/update binding",
    )
    _require(
        start["prohibited"] == ["human_data", "held60", "old_models", "outreach", "paid", "gpu"],
        "start prohibited authority differs",
    )
    _require(
        freeze["schema"] == "n1-metadata-generated-efficacy-freeze-v1"
        and freeze["models_sha256"] == descriptors["models.json"]["sha256"]
        and freeze["query_sha256"] == descriptors["query.npz"]["sha256"]
        and freeze["completed_updates"] == 1600
        and freeze["event"] == "both_models_frozen_before_query_decode",
        "global model/query freeze binding",
    )
    rows = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    expected_names = [
        "generation_started",
        "generation_completed",
        "fit_started",
        "fit_completed",
        "fit_started",
        "fit_completed",
        "both_models_frozen_before_query_decode",
        "query_decode_started",
        "regime_evaluation_completed",
        "regime_evaluation_completed",
        "evaluation_completed",
    ]
    _require(
        [row["event"] for row in rows] == expected_names
        and [row["sequence"] for row in rows] == list(range(1, 12)),
        "exact event order/sequence; freeze before sole query decode",
    )
    moments = [_timestamp(row["utc"]) for row in rows]
    _require(
        moments == sorted(moments)
        and _timestamp(start["utc"]) <= moments[0]
        and moments[5] <= _timestamp(freeze["utc"]) <= moments[6],
        "start/fit/freeze/query UTC chronology",
    )
    _require(
        (moments[-1] - _timestamp(start["utc"])).total_seconds() <= config["primary_seconds"],
        "recorded primary elapsed exceeds budget",
    )
    _require(rows[1]["artifacts"] == fixture["artifacts"], "generation event fixture pins")
    for index, regime in enumerate(REGIMES):
        begin, end = rows[2 + index * 2], rows[3 + index * 2]
        _require(
            begin["regime"] == end["regime"] == regime
            and begin["charged_updates"] == 800
            and begin["completed_updates"] == index * 800
            and end["completed_updates"] == (index + 1) * 800
            and end["q_hash"] == models[regime]["q_hash"],
            "fit event regime/count/frozen Q binding",
        )
        _require(rows[8 + index]["regime"] == regime, "regime evaluation order")
    _require(
        rows[6]["models_sha256"] == freeze["models_sha256"]
        and rows[6]["completed_updates"] == 1600,
        "event global freeze binding",
    )
    _require(
        rows[-1]["terminal"] == result["terminal"] and rows[-1]["completed_updates"] == 1600,
        "terminal event/result binding",
    )
    checks.done("hash_bound_start_single_attempt_counts_and_freeze_before_query_chronology")


def audit(folder: Path, config: dict) -> dict:
    """Audit already completed generated artifacts; no writes or optimizer calls.

    Integrity errors stop before decoding inputs. Numerical mismatches stop their
    downstream interpretation. An internally verified negative result may PASS;
    any audit FAIL overrides efficacy interpretation with NOT_EVALUATED.
    """
    started = time.monotonic()
    report = {
        "schema": SCHEMA,
        "status": "FAIL",
        "errors": [],
        "checks": {},
        "max_differences": {},
        "recomputed_summary": None,
        "limitations": list(LIMITATIONS),
    }
    checks = _Checks(1e-8)
    try:
        _config(config)
        folder = Path(folder)
        _require(
            folder.is_absolute()
            and folder.resolve() == folder
            and folder.is_dir()
            and not folder.is_symlink(),
            "unaliased absolute completed folder required",
        )
        _require(
            not any((folder / name).exists() for name in ("failure.json", "audit_failure.json")),
            "failure artifact takes precedence",
        )
        _require(
            {p.name for p in folder.iterdir()} == ARTIFACTS | {"result.json"},
            "exact completed artifact inventory required",
        )
        descriptors = {
            name: _descriptor(folder / name) for name in sorted(ARTIFACTS | {"result.json"})
        }
        _require(
            sum(value["bytes"] for value in descriptors.values()) <= config["output_limit_bytes"],
            "artifact output byte budget exceeded",
        )
        result, fixture, models, freeze = (
            _json(folder / name)
            for name in ("result.json", "fixture.json", "models.json", "freeze.json")
        )
        _require(
            result["schema"] == "n1-metadata-generated-efficacy-result-v1"
            and result["config"] == fixture["config"] == config
            and result["arms"] == list(ARMS)
            and result["updates_completed"] == 1600,
            "result config/schema/arms/update binding",
        )
        _require(
            result["artifacts"] == {name: descriptors[name] for name in ARTIFACTS},
            "result artifact hash/byte inventory mismatch",
        )
        _require(
            fixture["schema"] == "n1-metadata-generated-efficacy-fixture-v1"
            and fixture["counts"]
            == {key: config[key] for key in ("fit_groups", "evaluation_groups")}
            and fixture["artifacts"]
            == {name: descriptors[name] for name in ("fit.npz", "support.npz", "query.npz")},
            "fixture exact input inventory/config/count binding",
        )
        _require(
            models["schema"] == "n1-metadata-generated-efficacy-models-v1"
            and set(models["pipelines"]) == set(REGIMES)
            and models["oracle"] == {"coefficients": [2, 0, 0], "trained": False},
            "model/oracle schema",
        )
        models = models["pipelines"]
        _chronology(folder, result, fixture, models, freeze, descriptors, config, checks)
        checks.done("all_artifact_hashes_bound_before_any_numeric_decode")
        shapes = _shapes(config)
        fit = _arrays(folder / "fit.npz", shapes["fit.npz"])
        _raw_split(fit, None, config, 0, checks)
        q, m, s, c, geometry = _case_features(fit, config, checks, "fit")
        saved_features = _arrays(folder / "fit_features.npz", shapes["fit_features.npz"])
        for key, expected in (
            ("q", np.stack((q, q))),
            ("m", m),
            ("s", np.stack((s, s))),
            ("c", np.stack((c, c))),
        ):
            checks.close("fit_features_" + key, saved_features[key], expected)
        hashes = {
            regime: _pipeline(models[regime], fit["ids"], q, m[index], config, checks)
            for index, regime in enumerate(REGIMES)
        }
        _require(
            result["pipeline_q_hashes"] == hashes and len(set(hashes.values())) == 1,
            "same frozen Q required in both regimes",
        )
        checks.close("geometry_max_abs", result["geometry_max_abs"], geometry)
        checks.done("raw_fit_features_scalers_donors_model_traces_and_frozen_Q")
        del fit, saved_features, q, m, s, c
        data = _arrays(folder / "support.npz", shapes["support.npz"])
        query = _arrays(folder / "query.npz", shapes["query.npz"])
        _raw_split(data, query, config, 1, checks)
        q, m, s, c, _ = _case_features(data, config, checks, "evaluation")
        saved = _arrays(folder / "evaluation.npz", shapes["evaluation.npz"])
        scores, oracle, actuation = _replay(data, query, q, m, s, c, models, saved, config, checks)
        _require(set(result["actuation"]) == set(REGIMES), "exact actuation regime inventory")
        for regime in REGIMES:
            _require(
                set(result["actuation"][regime]) == set(actuation[regime]),
                "actuation field inventory",
            )
            for key, value in actuation[regime].items():
                checks.close(
                    "actuation_" + regime + "_" + key,
                    result["actuation"][regime][key],
                    value,
                    exact=key in ("argmax_changed", "coefficient_max_abs", "gradient_max"),
                )
        validity = {
            "finite_arrays": True,
            "common_q_hashes": True,
            "constant_support_geometry": True,
            "donors_cross_group": True,
        }
        summary = _summary(scores, oracle, config, validity, actuation)
        report["recomputed_summary"] = summary
        _require(
            result["terminal"] == summary["terminal"] and result["checks"] == summary["checks"],
            "terminal/check decision precedence mismatch",
        )
        checks.close(
            "group_accuracy_percent",
            result["group_accuracy_percent"],
            summary["group_accuracy_percent"],
            exact=True,
        )
        _require(
            set(result["contrasts"]) == set(summary["contrasts"]),
            "six exact contrast fields required",
        )
        for name, expected in summary["contrasts"].items():
            _require(
                set(result["contrasts"][name]) == set(expected),
                "contrast statistic fields mismatch",
            )
            for key, value in expected.items():
                checks.close(
                    "contrast_" + name + "_" + key,
                    result["contrasts"][name][key],
                    value,
                    exact=key == "n_groups",
                )
        checks.done("paired_group_descriptive_intervals_actuation_and_terminal_precedence")
        _require(
            {p.name for p in folder.iterdir()} == ARTIFACTS | {"result.json"},
            "artifact inventory changed during audit",
        )
        for name, descriptor in descriptors.items():
            _require(
                _descriptor(folder / name) == descriptor, "artifact changed during audit: " + name
            )
        _require(
            time.monotonic() - started <= config["audit_seconds"], "audit elapsed budget exceeded"
        )
        checks.done("all_artifacts_rebound_before_return")
        report.update(
            status="PASS",
            result_sha256=descriptors["result.json"]["sha256"],
            fixture_sha256=descriptors["fixture.json"]["sha256"],
            models_sha256=descriptors["models.json"]["sha256"],
            updates_completed_from_records=1600,
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        IndexError,
        ArithmeticError,
        zipfile.BadZipFile,
    ) as error:
        report["errors"].append(type(error).__name__ + ": " + str(error))
        report["terminal_override"] = "EFFICACY_NOT_EVALUATED"
    report["checks"] = checks.checks
    report["max_differences"] = checks.differences
    report["elapsed_seconds"] = time.monotonic() - started
    return report
