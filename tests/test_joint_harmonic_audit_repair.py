"""No-fitting generated diagnosis/regression for the single authorized audit repair."""

from __future__ import annotations

import io
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from cfeg.analysis.joint_harmonic import ContextScaler
from cfeg.analysis.joint_harmonic_audit import (
    audit_saved,
    check_normalizer,
    independent_contexts,
    restore_matlab_context_layout,
)
from cfeg.analysis.joint_harmonic_data import extract_participant, prefix_contexts, select_role
from cfeg.analysis.joint_harmonic_runner import file_sha


def test_generated_mat_layout_roundtrip_diagnosis():
    config = json.loads((Path(__file__).resolve().parents[1]
                         / "configs/analysis/joint_harmonic_program_v1.json").read_text())
    rng = np.random.default_rng(2026092301)
    raw = rng.normal(0, 3, config["raw_shape"])
    findings = []
    for order in ("C", "F"):
        extracted = extract_participant(np.array(raw, order=order), config)
        for key in ("q", "q2"):
            original = np.stack([extracted[key] + i / 16 for i in range(3)])
            stream = io.BytesIO()
            np.savez(stream, values=original)
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as stored:
                restored = stored["values"]
            np.testing.assert_array_equal(original, restored)
            differences = []
            for k in (1, 3, 5):
                before = original[0, 0, :k].mean(axis=(0, 1))
                after = restored[0, 0, :k].mean(axis=(0, 1))
                differences.append(float(np.abs(before - after).max()))
            findings.append({"raw_order": order, "feature": key,
                             "shape": original.shape, "strides_before": original.strides,
                             "strides_after": restored.strides, "max_mean_differences": differences})
    print(json.dumps(findings, indent=2))
    assert any(max(item["max_mean_differences"]) > 1e-10 for item in findings)


@pytest.fixture(scope="module")
def matlab_caches():
    config = json.loads((Path(__file__).resolve().parents[1]
                         / "configs/analysis/joint_harmonic_program_v1.json").read_text())
    rng = np.random.default_rng(2026092301)
    raw = np.asfortranarray(rng.normal(0, 3, config["raw_shape"]).astype(np.float32))
    part = extract_participant(raw, config)
    original = {key: np.stack([value + i / 16 for i in range(12)])
                for key, value in part.items()}
    impedance = rng.uniform(0, 100, (12, 2, 5, 8))
    impedance[:, :, :, 0] = np.nan
    impedance[::2, :, 2, 1] = np.nan
    original.update(ids=np.arange(12), orders=np.arange(12) % 2, impedance=impedance)
    stream = io.BytesIO()
    np.savez(stream, **original)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as stored:
        restored = {key: stored[key] for key in stored.files}
    repaired = dict(restored)
    for key in ("q", "q2"):
        repaired[key] = restore_matlab_context_layout(restored[key])
        np.testing.assert_array_equal(original[key], repaired[key])
        assert original[key].strides == repaired[key].strides
    return original, restored, repaired


@pytest.mark.parametrize("arm", ["Q", "Q2", "QM", "SHAM"])
def test_matlab_restoration_keeps_strict_normalizer_check(matlab_caches, arm):
    original, restored, repaired = matlab_caches
    people = [0, 1, 2, 3, 4, 5, 6, 7]
    role = select_role(original, people)
    contexts, donors = prefix_contexts(role, arm, sham_seed=91)
    np.testing.assert_allclose(independent_contexts(repaired, people, arm, donors), contexts,
                               atol=1e-12, rtol=1e-12)
    scaler = ContextScaler.fit(contexts.reshape(-1, 8, 10),
                               subject_ids=np.repeat(people, 6), allowed_fit_ids=people)
    record = {"fit_subject_ids": people, "arm": arm, "sham_donors": donors,
              "normalizer_mean": scaler.mean.tolist(), "normalizer_scale": scaler.scale.tolist()}
    mean_error, scale_error = check_normalizer(repaired, record)
    assert max(mean_error, scale_error) < 1e-12
    # Same values but wrong reduction layout reproduces the original failure.
    with pytest.raises(AssertionError):
        check_normalizer(restored, record)
    bad = dict(record)
    bad["normalizer_mean"] = (scaler.mean + 1e-5).tolist()
    with pytest.raises(AssertionError):
        check_normalizer(repaired, bad)
    bad["normalizer_mean"] = scaler.mean.tolist()
    bad["normalizer_scale"] = (scaler.scale + 1e-5).tolist()
    with pytest.raises(AssertionError):
        check_normalizer(repaired, bad)


def test_repaired_auditor_on_existing_generated_checkpoints(tmp_path, monkeypatch):
    source = os.environ.get("CFEG_AUDIT_GENERATED_FIXTURE")
    if not source:
        pytest.skip("Requires already generated fixture; no fitting is permitted here.")
    output = tmp_path / "saved-generated"
    shutil.copytree(Path(source), output)
    protected = {path.name: file_sha(path) for path in output.iterdir() if path.is_file()}
    from cfeg.analysis import joint_harmonic_data, joint_harmonic_runner

    def forbidden(*args, **kwargs):
        raise AssertionError("Raw reading, production execution and fitting are forbidden.")

    monkeypatch.setattr(joint_harmonic_data, "read_human_cache", forbidden)
    monkeypatch.setattr(joint_harmonic_runner, "train_one", forbidden)
    monkeypatch.setattr(joint_harmonic_runner, "execute", forbidden)
    report = audit_saved(output, generated=True, verification_r1=True)
    assert report["status"] == "PASS_SAVED_FORWARD_ROLES_AND_METRICS"
    assert report["normalizers_checked"] == 48
    assert report["forward_models_checked"] == 24
    assert report["forward_trials_checked"] == 4 * 2 * 12 * 2 * 3 * 60
    assert report["inner_choices_checked"] == 12
    assert report["numpy_argmax_differences"] == report["fits"] == report["raw_reads"] == 0
    assert report["feature_cache_loads"] == 1
    assert all(file_sha(output / name) == sha for name, sha in protected.items())
    with pytest.raises(FileExistsError):
        audit_saved(output, generated=True, verification_r1=True)
    print(json.dumps(report, indent=2))


def test_layout_repair_rejects_wrong_schema():
    with pytest.raises(ValueError):
        restore_matlab_context_layout(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        restore_matlab_context_layout(np.zeros((1, 2, 10, 12, 8, 2), dtype=np.float64))
