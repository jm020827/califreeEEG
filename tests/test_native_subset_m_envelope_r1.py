"""Envelope recovery: real serialized loader/core path; all participant data artificial."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup():
    runner = load("run_native_subset_m_envelope_r1")
    plan = json.loads((ROOT / "configs/analysis/native_subset_m_source39_v1.json").read_text())
    amendment = runner.load_execution(runner.EXECUTION_PATH)
    packets = [
        {
            "subject_id": subject,
            "interface": interface,
            "block_id": block,
            "impedance_kohm": [float(block + channel + p) for channel in range(8)],
            "headband_order": "dry",
            "condition_period": "first" if interface == "dry" else "second",
        }
        for p, subject in enumerate(plan["source_subject_ids"])
        for interface in ("dry", "wet")
        for block in range(10)
    ]
    envelope = {
        "manifest_sha256": plan["source_projection"]["manifest_sha256"],
        "packets": packets,
        "returned_rows": 9360,
        "returned_packets": 780,
        "returned_subject_ids": plan["source_subject_ids"],
        "columns": plan["source_projection"]["columns"],
        **amendment["envelope_provenance"],
    }
    return runner, plan, amendment, envelope


def test_strict_envelope_to_unchanged_core_and_namespace(setup):
    runner, plan, amendment, envelope = setup
    before = copy.deepcopy(envelope)
    core = load("native_subset_m_core")
    content = runner.content_projection(envelope, amendment)
    z, order = core.projection_arrays(content, plan)
    assert z.shape == (39, 2, 10, 8) and not order.any()
    assert envelope == before and len(envelope) == 11 and len(content) == 6
    assert amendment["study_id"] == amendment["randomization_namespace"] == plan["study_id"]
    assert amendment["scientific_changes"] == []
    for path, expected in amendment["source_helpers"].items():
        assert runner.sha(ROOT / path) == expected


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "missing",
        "schema",
        "study_id",
        "plan_sha256",
        "source_commit",
        "start_sha256",
        "packet_extra",
    ],
)
def test_rejects_wrong_envelope_or_packet(setup, damage):
    runner, plan, amendment, envelope = setup
    if damage == "extra":
        envelope["unknown"] = 1
    elif damage == "missing":
        del envelope["source_commit"]
    elif damage == "packet_extra":
        envelope["packets"][0]["query_label"] = 0
    else:
        envelope[damage] = "wrong"
    with pytest.raises(ValueError):
        content = runner.content_projection(envelope, amendment)
        load("native_subset_m_core").projection_arrays(content, plan)


def test_amendment_hash_and_helper_pin_rejection(setup, tmp_path, monkeypatch):
    runner, _, amendment, _ = setup
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match="hash"):
        runner.load_execution(path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.load_helpers(amendment)


def test_guard_uses_new_path_and_denies_old_attempt_and_protected_inputs(setup):
    runner, plan, amendment, _ = setup
    guard = runner.guard_for(amendment, plan)
    guard("open", (amendment["output_root"] + "/start.json", "wb", os.O_CREAT | os.O_WRONLY))
    guard("open", (plan["source_projection"]["path"], "rb", os.O_RDONLY))
    guard("open", (plan["raw_root"] + "/S004.mat", "rb", os.O_RDONLY))
    for path in (
        amendment["previous_attempt"]["path"] + "/start.json",
        plan["raw_root"] + "/S001.mat",
        plan["raw_root"] + "/S005.mat",
        plan["raw_root"] + "/Impedance.mat",
        "/tmp/unknown.npz",
        "/tmp/manifest.json",
    ):
        with pytest.raises(RuntimeError):
            guard("open", (path, "rb", os.O_RDONLY))
    for event in ("socket.connect", "subprocess.Popen"):
        with pytest.raises(RuntimeError):
            guard(event, ())


def prepare_lifecycle(setup, tmp_path, monkeypatch, corrupt=False):
    runner, plan, amendment, envelope = setup
    producer, core = load("run_native_subset_m_source"), load("native_subset_m_core")
    output, input_path = tmp_path / "attempt", tmp_path / "projection.json"
    if corrupt:
        envelope["source_commit"] = "wrong"
    encoded = producer.json_bytes(envelope)
    input_path.write_bytes(encoded)
    fake_sha = hashlib.sha256(encoded).hexdigest()
    monkeypatch.setattr(producer, "PROJECTION_PATH", input_path)
    monkeypatch.setattr(producer, "PROJECTION_SHA256", fake_sha)
    plan["source_projection"].update(path=str(input_path), sha256=fake_sha)
    amendment["output_root"] = str(output)
    provenance = {
        "plan_sha256": runner.SCIENCE_SHA256,
        "execution_plan_sha256": runner.EXECUTION_SHA256,
        "attempt_id": amendment["attempt_id"],
        "source_commit": producer.git(ROOT, "rev-parse", "HEAD"),
        "source_tree": producer.git(ROOT, "rev-parse", "HEAD^{tree}"),
        "upstream_revision": producer.REVISION,
        "python": "3.9.21",
        "dependencies": producer.VERSIONS,
    }

    def preflight(path):
        if output.exists():
            raise FileExistsError("No retry")
        return amendment, plan, provenance, producer, core

    monkeypatch.setattr(runner, "preflight", preflight)
    monkeypatch.setattr(runner.sys, "addaudithook", lambda guard: None)
    monkeypatch.setattr(producer, "import_native", lambda path: dict(plan["upstream"]["pins"]))
    events = []

    def gather(actual_plan, order):
        assert actual_plan is plan and not order.any()
        assert (output / "start.json").exists() and (output / "source-projection.json").exists()
        assert json.loads((output / "source-projection.json").read_text())["projection"] == envelope
        events.append("eeg_boundary")
        cache = {key: np.zeros(spec["shape"]) for key, spec in plan["cache"].items()}
        for i in range(2):
            for w, n in enumerate(plan["sample_counts"]):
                for bi, k in enumerate((3, 5)):
                    for j in range(k):
                        q = cache["q_features"][:, i, w, bi, j]
                        q[..., 0], q[..., 1], q[..., 2] = i, n / 500, k / 5
                        q[..., 3], q[..., 4] = np.arange(5, 10)[None, :, None] / 9, j / 4
                        q[..., 5], q[..., 7] = (np.arange(5, 10)[None, :, None] - j) / 9, i
                        q[..., 13], q[..., 17], q[..., 18] = 1, 1, 1
        records = [
            {
                "subject": p,
                "path": plan["raw_root"] + f"/S{p:03d}.mat",
                "sha256": "0" * 64,
                "stored_dtype": "float64",
                "shape": plan["raw_shape"],
            }
            for p in plan["source_subject_ids"]
        ]
        return cache, records

    monkeypatch.setattr(producer, "gather_participants", gather)
    monkeypatch.setattr(
        producer,
        "check_baseline",
        lambda cache, p: {
            "input_path": p["baseline_reference"]["path"],
            "input_sha256": p["baseline_reference"]["sha256"],
            "a0_r": {"max_abs_error": 0.0, "argmax_exact": True},
            "full_expert_r": {"max_abs_error": 0.0, "argmax_exact": True},
        },
    )
    original_evaluate = core.evaluate_all

    def evaluate(*args):
        frozen = json.loads((output / "fold-freezes.json").read_text())
        assert len(frozen["folds"]) == 3
        events.append("outer")
        return original_evaluate(*args)

    # A spy around the actual core, not a replacement mathematical fixture.
    monkeypatch.setattr(core, "evaluate_all", evaluate)
    return runner, amendment, plan, output, events


def test_bad_original_envelope_stops_before_eeg_and_preserves_start(setup, tmp_path, monkeypatch):
    runner, _, _, output, events = prepare_lifecycle(setup, tmp_path, monkeypatch, corrupt=True)
    with pytest.raises(ValueError, match="provenance"):
        runner.execute(tmp_path / "execution.json")
    assert events == [] and [p.name for p in output.iterdir()] == ["start.json"]
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path / "execution.json")


def test_real_loader_adapter_core_and_independent_auditor(setup, tmp_path, monkeypatch):
    runner, amendment, plan, output, events = prepare_lifecycle(setup, tmp_path, monkeypatch)
    result = runner.execute(tmp_path / "execution.json")
    assert events == ["eeg_boundary", "outer"] and len(result["rows"]) == 4680
    audit = load("audit_native_subset_m_envelope_r1")
    independent = load("audit_native_subset_m_source")
    wrapper = json.loads((output / "source-projection.json").read_text())
    content = audit.content_projection(wrapper["projection"], amendment)
    assert len(wrapper["projection"]) == 11 and len(content) == 6
    z, order = independent.load_projection({**wrapper, "projection": content}, plan)
    with np.load(output / "features.npz", allow_pickle=False) as stored:
        cache = {key: stored[key] for key in stored.files}
    scores = independent.cache_scores(cache, plan)
    assert independent.verify_q(cache, scores, order, plan) < 2e-12
    freeze = json.loads((output / "fold-freezes.json").read_text())
    assert independent.verify_fits(cache, scores, z, plan, freeze) < 2e-10
    rows, diagnostics = independent.replay_evaluation(cache, scores, z, plan, freeze)
    independent.equal(result["rows"], rows, "connected rows")
    independent.equal(result["diagnostics"], diagnostics, "connected diagnostics")
    for name, values in zip(
        ("summary", "contrasts", "attainment"), independent.replay_reporting(rows, plan)
    ):
        independent.equal(result[name], values, "connected " + name)
    for key in ("plan_sha256", "execution_plan_sha256", "attempt_id", "study_id"):
        assert result[key] == freeze[key]
    assert sorted(p.name for p in output.iterdir()) == sorted(runner.ARTIFACTS)
    assert all((p.stat().st_mode & 0o777) == 0o400 for p in output.iterdir())
    # Substitute only artificial input/destination authority. The complete new
    # auditor, committed provenance checks and all numerical replay run for real.
    monkeypatch.setattr(audit, "load_amendment", lambda path: amendment)
    monkeypatch.setattr(audit, "load_helper", lambda repository, value: independent)
    monkeypatch.setattr(independent, "load_plan", lambda path: plan)
    report = audit.audit(output, tmp_path / "execution.json", repository=ROOT)
    assert report["status"] == "PASS" and report["independently_replayed_routers"] == 9
    assert report["projection_envelope_keys"] == 11 and report["scientific_changes"] == []
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path / "execution.json")
