"""Frozen generated-only waveform fixture; no learner, scorer or human inputs.

The public functions consume an explicit configuration.  This module does not
authorize a registered generation: the new primary owns that one-shot boundary.
Only the seed and smaller group counts may differ for component tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import numpy as np

SCHEMA = "n1-metadata-generated-efficacy-fixture-v1"
_FIXED = {
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
_VARIABLE = {"seed", "fit_groups", "evaluation_groups"}
_TARGETS = ("fit.npz", "support.npz", "query.npz", "fixture.json")
_PREEXISTING = frozenset(("primary_start.json", "events.jsonl"))


def _same_value(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _same_value(x, y) for x, y in zip(value, expected, strict=True)
        )
    return value == expected


def _checked_config(config):
    if not isinstance(config, dict) or set(config) != set(_FIXED) | _VARIABLE:
        raise ValueError("Exact generated-efficacy configuration fields required")
    for key, expected in _FIXED.items():
        if not _same_value(config[key], expected):
            raise ValueError("Frozen configuration differs: " + key)
    for key, minimum, maximum in (
        ("seed", 0, 2**32 - 1),
        ("fit_groups", 2, 8),
        ("evaluation_groups", 2, 32),
    ):
        if type(config[key]) is not int or not minimum <= config[key] <= maximum:
            raise ValueError("Invalid generated configuration: " + key)
    # Detach the manifest from caller mutation and reject non-JSON values.
    return json.loads(json.dumps(config, allow_nan=False))


def _rng(config, split, group, purpose, band=None):
    entropy = [config["seed"], split, group, purpose]
    if band is not None:
        entropy.append(band)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _fourier_basis(samples):
    """Constant, then ascending unique harmonic bins, each sine before cosine.

    These analytically normalized Fourier columns are orthonormal; no extra QR
    of U (and hence no implementation-dependent basis rotation) is performed.
    """
    time = np.arange(samples, dtype=np.float64)
    bins = sorted({harmonic * m for harmonic in range(1, 6) for m in range(6, 18)})
    columns = [np.ones(samples, dtype=np.float64) / np.sqrt(samples)]
    for frequency_bin in bins:
        angle = 2 * np.pi * frequency_bin * time / samples
        columns.extend((np.sqrt(2 / samples) * np.sin(angle), np.sqrt(2 / samples) * np.cos(angle)))
    basis = np.stack(columns, axis=1)
    if samples - basis.shape[1] < 24:
        raise ValueError("Fourier complement cannot hold all 24 noise columns")
    return basis


def generate_group(config, split: int, group: int) -> dict:
    """Generate one counterfactual pair, with independent purpose-keyed streams.

    split0 is source fitting and split1 evaluation.  No score, label agreement,
    rank or performance criterion ever causes a retry or replacement draw.
    """
    config = _checked_config(config)
    if type(split) is not int or split not in (0, 1):
        raise ValueError("split must be the integer0 or1")
    groups = config["fit_groups" if split == 0 else "evaluation_groups"]
    if type(group) is not int or not 0 <= group < groups:
        raise ValueError("group is outside the frozen split")
    samples = config["samples"]
    bad = np.zeros((2, 8), dtype=bool)
    bad[0, _rng(config, split, group, 1).choice(8, 4, replace=False)] = True
    bad[1] = ~bad[0]
    null_mask = np.zeros(8, dtype=bool)
    null_mask[_rng(config, split, group, 2).choice(8, 4, replace=False)] = True
    phases = _rng(config, split, group, 3).uniform(0, 2 * np.pi, size=(5, 12))
    time = np.arange(samples, dtype=np.float64)
    phase_angle = 2 * np.pi * np.arange(6, 18)[:, None] * time / samples
    signals = np.sqrt(2) * np.sin(phase_angle[None] + phases[..., None])
    basis = _fourier_basis(samples)
    noise = np.empty((5, 3, 8, samples), dtype=np.float64)
    for band in range(5):
        residual = _rng(config, split, group, 4, band).normal(size=(samples, 24))
        for _ in range(2):
            residual -= basis @ (basis.T @ residual)
        q, r = np.linalg.qr(residual, mode="reduced")
        diagonal = np.diag(r)
        if not np.isfinite(q).all() or not np.isfinite(diagonal).all() or np.any(diagonal == 0):
            raise ValueError("Generated Fourier-complement QR failed; no redraw")
        q *= np.where(diagonal < 0, -1.0, 1.0)[None]
        noise[band] = (np.sqrt(samples) * q.T).reshape(3, 8, samples)
    signal = signals.transpose(1, 0, 2)  # class, band, time
    support = (
        signal[None, :, :, None] + config["noise_amplitude"] * noise.transpose(1, 0, 2, 3)[:, None]
    )
    halfwidth = config["delta_halfwidth"]
    source_delta = _rng(config, split, group, 5).uniform(-halfwidth, halfwidth, size=12)
    query_delta = _rng(config, split, group, 6).uniform(-halfwidth, halfwidth, size=(4, 12))
    decoy = np.roll(signal, -1, axis=0)
    source_good = (1 + source_delta[:, None, None]) * signal
    source_bad = (1 - source_delta[:, None, None]) * decoy
    source = np.where(
        bad[:, None, None, :, None], source_bad[None, :, :, None], source_good[None, :, :, None]
    )
    query_good = ((1 + query_delta[..., None, None]) * signal[None]).reshape(48, 5, samples)
    query_bad = ((1 - query_delta[..., None, None]) * decoy[None]).reshape(48, 5, samples)
    query = np.where(
        bad[:, None, None, :, None], query_bad[None, :, :, None], query_good[None, :, :, None]
    )
    packet_coupled = np.repeat(np.expm1(1 + 2 * bad.astype(np.float64))[:, None], 3, axis=1)
    packet_null = np.broadcast_to(np.expm1(1 + 2 * null_mask.astype(np.float64)), (2, 3, 8)).copy()
    result = {
        "support": support,
        "source": source,
        "query": query,
        "packet_coupled": packet_coupled,
        "packet_null": packet_null,
        "bad": bad,
        "null_mask": null_mask,
        "phases": phases,
        "source_delta": source_delta,
        "query_delta": query_delta,
    }
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite generated fixture; no redraw")
    return result


def _split_arrays(config, split):
    count = config["fit_groups" if split == 0 else "evaluation_groups"]
    base = config["fit_id_base" if split == 0 else "evaluation_id_base"]
    n = config["samples"]
    common = {
        "ids": np.arange(base + 1, base + 2 * count + 1, dtype=np.int64),
        "groups": np.tile(np.arange(count, dtype=np.int64), 2),
        "members": np.repeat(np.arange(2, dtype=np.int64), count),
        "support": np.empty((2 * count, 3, 12, 5, 8, n), dtype=np.float64),
        "packet_coupled": np.empty((2 * count, 3, 8), dtype=np.float64),
        "packet_null": np.empty((2 * count, 3, 8), dtype=np.float64),
        "bad": np.empty((2 * count, 8), dtype=bool),
        "null_mask": np.empty((count, 8), dtype=bool),
        "phases": np.empty((count, 5, 12), dtype=np.float64),
    }
    if split == 0:
        common["source"] = np.empty((2 * count, 12, 5, 8, n), dtype=np.float64)
        common["source_delta"] = np.empty((count, 12), dtype=np.float64)
        query = None
    else:
        query = {name: common[name].copy() for name in ("ids", "groups", "members")}
        query["query"] = np.empty((2 * count, 48, 5, 8, n), dtype=np.float64)
        query["query_delta"] = np.empty((count, 4, 12), dtype=np.float64)
    for group in range(count):
        generated = generate_group(config, split, group)
        common["null_mask"][group] = generated["null_mask"]
        common["phases"][group] = generated["phases"]
        if split == 0:
            common["source_delta"][group] = generated["source_delta"]
        else:
            query["query_delta"][group] = generated["query_delta"]
        for member in range(2):
            index = member * count + group
            common["support"][index] = generated["support"]
            for name in ("packet_coupled", "packet_null", "bad"):
                common[name][index] = generated[name][member]
            if split == 0:
                common["source"][index] = generated["source"][member]
            else:
                query["query"][index] = generated["query"][member]
    return common, query


def _regular(info):
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1


def _folder(folder):
    path = Path(folder).absolute()
    if path.resolve() != path:
        raise ValueError("Fixture folder must be an unaliased path, with no symlink ancestors")
    if not path.exists():
        path.mkdir(mode=0o700)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        names = os.listdir(descriptor)
        if set(names) - _PREEXISTING:
            raise ValueError("Fixture targets must be new; unexpected existing entries")
        for name in names:
            if not _regular(os.stat(name, dir_fd=descriptor, follow_symlinks=False)):
                raise ValueError("Existing primary journal/start must be regular single-link files")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _publish(directory_fd, name, writer):
    """Exclusive fd-relative publication; partial files remain sealed on failure."""
    if name not in _TARGETS:
        raise ValueError("Unexpected fixture artifact name")
    descriptor = os.open(
        name, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd
    )
    with os.fdopen(descriptor, "w+b") as stream:
        try:
            if not _regular(os.fstat(stream.fileno())):
                raise ValueError("Fixture artifact is not a regular single-link file")
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            info = os.fstat(stream.fileno())
            if not _regular(info):
                raise ValueError("Fixture artifact link identity changed")
            result = {"sha256": digest.hexdigest(), "bytes": info.st_size}
        finally:
            # Do not delete/retry a partial artifact or swallow disk/sealing errors.
            os.fchmod(stream.fileno(), 0o400)
            os.fsync(stream.fileno())
            os.fsync(directory_fd)
    return result


def _npz(directory_fd, name, arrays):
    def write(stream):
        with zipfile.ZipFile(
            stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            for key, value in arrays.items():
                if (
                    not key.isidentifier()
                    or not isinstance(value, np.ndarray)
                    or value.dtype.kind not in "biuf"
                ):
                    raise ValueError("Only named numeric/boolean arrays are allowed; no pickle")
                with archive.open(key + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, value, allow_pickle=False)

    return _publish(directory_fd, name, write)


def write_fixture(folder: Path, config: dict) -> dict:
    """Publish three stored numeric NPZs, then the exact immutable JSON manifest.

    A new folder or an existing empty folder is accepted.  The primary may have
    already created only its regular single-link primary_start.json/events.jsonl.
    These files are not read or modified.  Existing targets are denied before RNG.
    """
    config = _checked_config(config)
    directory_fd = _folder(folder)
    try:
        artifacts = {}
        fit, _ = _split_arrays(config, 0)
        artifacts["fit.npz"] = _npz(directory_fd, "fit.npz", fit)
        del fit
        support, query = _split_arrays(config, 1)
        artifacts["support.npz"] = _npz(directory_fd, "support.npz", support)
        artifacts["query.npz"] = _npz(directory_fd, "query.npz", query)
        manifest = {
            "schema": SCHEMA,
            "config": config,
            "artifacts": artifacts,
            "counts": {
                "fit_groups": config["fit_groups"],
                "evaluation_groups": config["evaluation_groups"],
            },
        }
        payload = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")
        _publish(directory_fd, "fixture.json", lambda stream: stream.write(payload))
        return manifest
    finally:
        os.close(directory_fd)
