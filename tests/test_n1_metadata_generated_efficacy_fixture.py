"""Toy seed913 only: waveform/serialization tests, never optimizer or efficacy."""

import copy
import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis import metadata_trca_prior as native
from cfeg.analysis import n1_metadata_generated_efficacy_fixture as fixture
from cfeg.analysis import task_trca_shape_features as features


@pytest.fixture(scope="module")
def config():
    path = Path(__file__).resolve().parents[1] / "configs/n1_metadata_generated_efficacy_v1.json"
    value = json.loads(path.read_text())
    value.update(seed=913, fit_groups=2, evaluation_groups=2)
    return value


@pytest.fixture(scope="module")
def group(config):
    assert config["seed"] != 20260916
    return fixture.generate_group(config, 0, 0)


def independent_rng(config, split, group, purpose, band=None):
    values = [config["seed"], split, group, purpose]
    if band is not None:
        values.append(band)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(values)))


def test_exact_geometry_streams_complements_and_packets(config, group):
    expected = {
        "support": (3, 12, 5, 8, 256),
        "source": (2, 12, 5, 8, 256),
        "query": (2, 48, 5, 8, 256),
        "packet_coupled": (2, 3, 8),
        "packet_null": (2, 3, 8),
        "bad": (2, 8),
        "null_mask": (8,),
        "phases": (5, 12),
        "source_delta": (12,),
        "query_delta": (4, 12),
    }
    assert set(group) == set(expected)
    for name, shape in expected.items():
        assert group[name].shape == shape
        assert group[name].dtype == (bool if name in ("bad", "null_mask") else np.float64)
        assert np.isfinite(group[name]).all()
    for purpose, actual in ((1, group["bad"][0]), (2, group["null_mask"])):
        positions = independent_rng(config, 0, 0, purpose).choice(8, 4, replace=False)
        expected_mask = np.zeros(8, dtype=bool)
        expected_mask[positions] = True
        np.testing.assert_array_equal(actual, expected_mask)
    np.testing.assert_array_equal(group["bad"][1], ~group["bad"][0])
    np.testing.assert_array_equal(
        group["phases"], independent_rng(config, 0, 0, 3).uniform(0, 2 * np.pi, (5, 12))
    )
    np.testing.assert_array_equal(
        group["source_delta"], independent_rng(config, 0, 0, 5).uniform(-0.01, 0.01, 12)
    )
    np.testing.assert_array_equal(
        group["query_delta"], independent_rng(config, 0, 0, 6).uniform(-0.01, 0.01, (4, 12))
    )
    np.testing.assert_array_equal(
        group["packet_coupled"],
        np.repeat(np.expm1(1 + 2 * group["bad"].astype(float))[:, None], 3, axis=1),
    )
    np.testing.assert_array_equal(group["packet_null"][0], group["packet_null"][1])
    assert not np.array_equal(group["source_delta"], group["query_delta"][0])


def test_fourier_noise_geometry_and_native_sc_formula(group):
    n = 256
    t = np.arange(n)
    phase = group["phases"]
    signal = np.sqrt(2) * np.sin(
        2 * np.pi * np.arange(6, 18)[None, :, None] * t / n + phase[..., None]
    )
    np.testing.assert_allclose(signal.mean(-1), 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose((signal**2).sum(-1), n, rtol=0, atol=1e-11)
    basis = fixture._fourier_basis(n)
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), rtol=0, atol=2e-14)
    recovered = (group["support"] - signal.transpose(1, 0, 2)[None, :, :, None]) / 0.2
    np.testing.assert_allclose(
        recovered, np.repeat(recovered[:, :1], 12, axis=1), rtol=0, atol=2e-15
    )
    for band in range(5):
        noise = recovered[:, 0, band].reshape(24, n)
        np.testing.assert_allclose(noise @ noise.T, n * np.eye(24), rtol=0, atol=1e-11)
        np.testing.assert_allclose(noise @ basis, 0, rtol=0, atol=1e-12)
        for label in range(12):
            s, c = native.trca_matrices(group["support"][:, label, band])
            np.testing.assert_allclose(s, 6 * n * np.ones((8, 8)), rtol=0, atol=1e-8)
            np.testing.assert_allclose(
                c, 3 * n * (np.ones((8, 8)) + 0.04 * np.eye(8)), rtol=0, atol=1e-8
            )
            assert np.linalg.cond(c) == pytest.approx(201, abs=1e-9)


def test_noise_stream_band_and_qr_sign_normalization(config, group):
    basis = fixture._fourier_basis(256)
    for band in (0, 4):
        g = independent_rng(config, 0, 0, 4, band).normal(size=(256, 24))
        g = g - basis @ (basis.T @ g)
        g = g - basis @ (basis.T @ g)
        q, r = np.linalg.qr(g, mode="reduced")
        q *= np.sign(np.diag(r))[None]
        phase = group["phases"][band, 0]
        signal = np.sqrt(2) * np.sin(2 * np.pi * 6 * np.arange(256) / 256 + phase)
        expected = signal[None, None] + 0.2 * 16 * q.T.reshape(3, 8, 256)
        np.testing.assert_array_equal(group["support"][:, 0, band], expected)


def test_source_query_formula_and_independent_block_deltas(group):
    signal = np.sqrt(2) * np.sin(
        2 * np.pi * np.arange(6, 18)[:, None, None] * np.arange(256) / 256
        + group["phases"].T[..., None]
    )
    for member in range(2):
        for label in (0, 11):
            for channel in range(8):
                bad = group["bad"][member, channel]
                target = (label + 1) % 12 if bad else label
                direction = -1 if bad else 1
                source = (1 + direction * group["source_delta"][label]) * signal[target]
                np.testing.assert_array_equal(group["source"][member, label, :, channel], source)
                for block in range(4):
                    expected = (1 + direction * group["query_delta"][block, label]) * signal[target]
                    np.testing.assert_array_equal(
                        group["query"][member, block * 12 + label, :, channel], expected
                    )
    assert all(
        not np.array_equal(group["query"][:, :12], group["query"][:, 12 * b : 12 * (b + 1)])
        for b in (1, 2, 3)
    )


def test_actual_q15_mask_only_and_m_two_feature_contract(config, group):
    frequencies = 250 * np.arange(6, 18) / 256
    original = features.support_q_mask(
        group["support"], np.ones((3, 8), dtype=bool), 0, 0, frequencies
    )
    assert original.shape == (5, 8, 15)
    assert np.isfinite(original).all()
    for regime in ("packet_coupled", "packet_null"):
        for member in range(2):
            packet = group[regime][member]
            q = features.support_q_mask(group["support"], np.isfinite(packet), 0, 0, frequencies)
            np.testing.assert_array_equal(q, original)
            m, available = features.metadata_features(packet)
            assert available.all()
            expected_mask = (
                group["bad"][member] if regime == "packet_coupled" else group["null_mask"]
            )
            np.testing.assert_allclose(
                m[:, 0], 2 * expected_mask.astype(float) - 1, rtol=0, atol=1e-15
            )
            np.testing.assert_array_equal(m[:, 1], np.zeros(8))
    q_mean = np.mean(original[..., :3], axis=1)
    np.testing.assert_allclose(q_mean[:, 0], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(q_mean[:, 1], np.log(0.04 / 1.04), rtol=0, atol=1e-13)
    np.testing.assert_allclose(q_mean[:, 2], 1 - 1 / 1.04, rtol=0, atol=1e-14)


def test_original_constructors_same_q_and_support_statistics_across_regimes(group):
    # Native analytical support fitting only; no fit_pipeline/optimizer/scoring.
    from cfeg.analysis import task_trca_n1_evaluation as evaluation
    from cfeg.analysis import task_trca_n1_learning as learning

    frequencies = 250 * np.arange(6, 18) / 256
    cases, states = [], []
    for regime in ("packet_coupled", "packet_null"):
        packet = group[regime][0]
        cases.append(
            learning.make_task_case(
                11001,
                0,
                0,
                group["support"],
                packet,
                frequencies,
                group["source"][0],
                weights=np.ones(5),
            )
        )
        states.append(
            evaluation.support_state(
                21001,
                0,
                0,
                group["support"],
                packet,
                frequencies,
                np.ones(5),
            )
        )
    for collection in (cases, states):
        for name in ("q", "available", "mask"):
            np.testing.assert_array_equal(
                getattr(collection[0], name), getattr(collection[1], name)
            )
        for name in ("s", "c", "anchors", "weights"):
            np.testing.assert_array_equal(
                getattr(collection[0], name).numpy(), getattr(collection[1], name).numpy()
            )
        np.testing.assert_array_equal(
            collection[0].q,
            features.support_q_mask(
                group["support"],
                np.ones((3, 8), dtype=bool),
                0,
                0,
                frequencies,
            ),
        )
    np.testing.assert_array_equal(cases[0].q, states[0].q)


def test_stream_replay_split_group_and_global_state_isolation(config, group):
    state = np.random.get_state()
    same = fixture.generate_group(config, 0, 0)
    alternate = fixture.generate_group(config, 1, 1)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    for key in group:
        np.testing.assert_array_equal(group[key], same[key])
    assert not np.array_equal(group["phases"], alternate["phases"])
    assert not np.array_equal(group["query_delta"], alternate["query_delta"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("seed", -1),
        ("fit_groups", 1),
        ("fit_groups", 9),
        ("evaluation_groups", 33),
        ("samples", 128),
        ("noise_amplitude", 0.3),
        ("weights", [1] * 4),
        ("weights", [True] * 5),
        ("updates_total", 800),
    ],
)
def test_invalid_config_denied_before_rng(config, monkeypatch, field, value):
    changed = copy.deepcopy(config)
    changed[field] = value
    monkeypatch.setattr(fixture, "_rng", lambda *a, **k: pytest.fail("RNG opened"))
    with pytest.raises(ValueError):
        fixture.generate_group(changed, 0, 0)


@pytest.mark.parametrize(
    "split,group", [(True, 0), (-1, 0), (2, 0), (0, True), (0, -1), (0, 2), (1, 2)]
)
def test_role_bounds_before_rng(config, monkeypatch, split, group):
    monkeypatch.setattr(fixture, "_rng", lambda *a, **k: pytest.fail("RNG opened"))
    with pytest.raises(ValueError):
        fixture.generate_group(config, split, group)


@pytest.fixture(scope="module")
def written(tmp_path_factory, config):
    folder = tmp_path_factory.mktemp("fixture")
    (folder / "primary_start.json").write_text("{}\n")
    (folder / "primary_start.json").chmod(0o400)
    (folder / "events.jsonl").write_text('{"event":"generation_started"}\n')
    manifest = fixture.write_fixture(folder, config)
    return folder, manifest


def test_exclusive_numeric_manifest_layout_and_member_major_ids(written, config):
    folder, manifest = written
    assert manifest == json.loads((folder / "fixture.json").read_text())
    assert manifest["schema"] == fixture.SCHEMA
    assert manifest["config"] == config
    assert manifest["counts"] == {"fit_groups": 2, "evaluation_groups": 2}
    assert set(manifest["artifacts"]) == {"fit.npz", "support.npz", "query.npz"}
    assert (folder / "primary_start.json").read_text() == "{}\n"
    assert (folder / "events.jsonl").read_text() == '{"event":"generation_started"}\n'
    for name, descriptor in manifest["artifacts"].items():
        path = folder / name
        assert stat.S_IMODE(path.stat().st_mode) == 0o400
        assert path.stat().st_nlink == 1
        assert descriptor == {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        with zipfile.ZipFile(path) as archive:
            assert all(row.compress_type == zipfile.ZIP_STORED for row in archive.infolist())
            assert all(
                row.filename.endswith(".npy") and "/" not in row.filename
                for row in archive.infolist()
            )
        with np.load(path, allow_pickle=False) as arrays:
            base = 11000 if name == "fit.npz" else 21000
            np.testing.assert_array_equal(arrays["ids"], np.arange(base + 1, base + 5))
            np.testing.assert_array_equal(arrays["groups"], [0, 1, 0, 1])
            np.testing.assert_array_equal(arrays["members"], [0, 0, 1, 1])
            assert all(arrays[key].dtype.kind in "biuf" for key in arrays.files)
    assert stat.S_IMODE((folder / "fixture.json").stat().st_mode) == 0o400


def test_persisted_roles_shared_support_and_exact_donor_cross_group(written):
    folder, _ = written
    for name in ("fit.npz", "support.npz"):
        with np.load(folder / name, allow_pickle=False) as data:
            assert "query" not in data.files and "query_delta" not in data.files
            assert ("source" in data.files) == (name == "fit.npz")
            assert ("source_delta" in data.files) == (name == "fit.npz")
            for group in range(2):
                np.testing.assert_array_equal(data["support"][group], data["support"][group + 2])
                np.testing.assert_array_equal(data["bad"][group], ~data["bad"][group + 2])
                np.testing.assert_array_equal(
                    data["packet_null"][group], data["packet_null"][group + 2]
                )
            ids = data["ids"].tolist()
            groups = dict(zip(ids, data["groups"].tolist(), strict=True))
            donors = features.donor_map(ids, np.ones((4, 3, 8), dtype=bool), [0] * 4, 0)
            assert all(groups[pid] != groups[donor] for pid, donor in donors.items())
    with np.load(folder / "query.npz", allow_pickle=False) as data:
        assert set(data.files) == {"ids", "groups", "members", "query", "query_delta"}
        assert data["query"].shape == (4, 48, 5, 8, 256)


def test_overwrite_denied_before_rng(written, config, monkeypatch):
    folder, _ = written
    before = (folder / "fixture.json").read_bytes()
    monkeypatch.setattr(fixture, "generate_group", lambda *a: pytest.fail("RNG opened"))
    with pytest.raises(ValueError):
        fixture.write_fixture(folder, config)
    assert (folder / "fixture.json").read_bytes() == before


@pytest.mark.parametrize(
    "kind",
    [
        "parent_symlink",
        "target_symlink",
        "unexpected",
        "journal_symlink",
        "journal_hardlink",
        "journal_directory",
    ],
)
def test_unsafe_folder_denied_before_rng(config, tmp_path, monkeypatch, kind):
    folder = tmp_path / "fixture"
    folder.mkdir()
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("untouched")
    if kind == "parent_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(folder, target_is_directory=True)
        folder = alias
    elif kind == "target_symlink":
        (folder / "fit.npz").symlink_to(sentinel)
    elif kind == "unexpected":
        (folder / "unapproved.json").write_text("{}")
    elif kind == "journal_symlink":
        (folder / "events.jsonl").symlink_to(sentinel)
    elif kind == "journal_hardlink":
        os.link(sentinel, folder / "events.jsonl")
    else:
        (folder / "events.jsonl").mkdir()
    monkeypatch.setattr(fixture, "generate_group", lambda *a: pytest.fail("RNG opened"))
    with pytest.raises((ValueError, OSError)):
        fixture.write_fixture(folder, config)
    assert sentinel.read_text() == "untouched"


def test_object_arrays_never_pickled_and_partial_is_preserved(tmp_path):
    folder = tmp_path / "new"
    fd = fixture._folder(folder)
    try:
        with pytest.raises(ValueError, match="no pickle"):
            fixture._npz(fd, "fit.npz", {"bad": np.array([object()], dtype=object)})
        path = folder / "fit.npz"
        assert path.exists() and stat.S_IMODE(path.stat().st_mode) == 0o400
        with pytest.raises(FileExistsError):
            fixture._npz(fd, "fit.npz", {"good": np.ones(2)})
    finally:
        os.close(fd)


def test_write_error_not_ignored_and_no_replacement(tmp_path):
    folder = tmp_path / "new"
    fd = fixture._folder(folder)

    def broken(stream):
        stream.write(b"partial")
        raise OSError("simulated full disk")

    try:
        with pytest.raises(OSError, match="simulated full disk"):
            fixture._publish(fd, "query.npz", broken)
        assert (folder / "query.npz").read_bytes() == b"partial"
        assert stat.S_IMODE((folder / "query.npz").stat().st_mode) == 0o400
        assert not (folder / "fixture.json").exists()
    finally:
        os.close(fd)
