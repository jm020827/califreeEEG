import copy
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_trca_support_geometry.py"
spec = importlib.util.spec_from_file_location("geometry_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def artificial(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    path = native / "S999.npz"
    rng = np.random.default_rng(113)
    x = np.full((2, 10, 12, 5, 8, 16), np.nan)
    x[:, :5] = rng.normal(size=(2, 5, 12, 5, 8, 16))
    with path.open("wb") as stream:
        np.savez(stream, x_16=x, full_16=np.array([np.nan]), a0_16=np.array([np.nan]))
    path.chmod(0o400)
    plan = copy.deepcopy(
        json.loads((REPO / "configs/analysis/trca_support_geometry_v1.json").read_text())
    )
    plan.update(
        source_subject_ids=[999],
        native_directory=str(native),
        sample_counts=[16],
        records_expected=240,
        output_directory=str(tmp_path / "output"),
    )
    manifest = {
        "schema": "cfeg.metadata-prior-source.native-result.v1",
        "status": "COMPLETE",
        "source_subject_ids": [999],
        "sample_counts": [16],
        "plan_sha256": plan["native_plan_sha256"],
        "files": [
            {
                "subject": 999,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": runner.sha(path),
            }
        ],
    }
    manifest_path = native / "result.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_path.chmod(0o400)
    plan["native_manifest_sha256"] = runner.sha(manifest_path)
    # Real source file final rehashes are deliberately nonempty, guarding the old exporter failure.
    provenance = {
        "plan_sha256": "a" * 64,
        "source_commit": "artificial",
        "source_hashes": {str(SCRIPT): runner.sha(SCRIPT)},
    }
    return plan, provenance, path


def test_lifecycle_start_before_input_and_immutable_publication(tmp_path, monkeypatch):
    plan, provenance, archive = artificial(tmp_path)
    digest = runner.sha(archive)
    original = runner.read_json
    output = Path(plan["output_directory"])

    def assert_start(path, expected):
        assert (output / "start.json").exists()
        assert not (output / "result.json").exists()
        return original(path, expected)

    monkeypatch.setattr(runner, "read_json", assert_start)
    result = runner.compute(plan, output, provenance, install_guard=False)
    assert result["status"] == "GEOMETRY_DIAGNOSED"
    assert result["record_count"] == 240 and len(result["band_groups"]) == 20
    assert result["classification_executed"] is False
    assert result["query_values_decoded"] is False
    assert result["numeric_metadata_read"] is False
    assert runner.sha(archive) == digest
    for name in ("start.json", "geometry.npz", "result.json"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o400
    with np.load(output / "geometry.npz") as arrays:
        assert arrays["C"].shape == (240, 8, 8)
        assert arrays["full_unit"].shape == (240, 8)
    with pytest.raises(ValueError, match="Fresh output"):
        runner.compute(plan, output, provenance, install_guard=False)


def test_cold_guarded_lifecycle_real_source_hashes(tmp_path):
    plan, provenance, _ = artificial(tmp_path)
    payload = tmp_path / "artificial.json"
    payload.write_text(json.dumps({"plan": plan, "provenance": provenance}))
    code = """
import importlib.util, io, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("runner", sys.argv[1])
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
d = json.load(open(sys.argv[2]))
r.warm_artificial_runtime()
r.compute(d["plan"], r.Path(d["plan"]["output_directory"]), d["provenance"])
"""
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(REPO / "src"),
    )
    process = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), str(payload)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert (
        json.loads((Path(plan["output_directory"]) / "result.json").read_text())["status"]
        == "GEOMETRY_DIAGNOSED"
    )


@pytest.mark.parametrize("case", ["manifest", "archive", "source_hash", "count"])
def test_failure_is_preserved_not_rescued(tmp_path, case):
    plan, provenance, path = artificial(tmp_path)
    if case == "manifest":
        plan["native_manifest_sha256"] = "0" * 64
    elif case == "archive":
        path.chmod(0o600)
    elif case == "source_hash":
        provenance["source_hashes"][str(SCRIPT)] = "0" * 64
    else:
        plan["records_expected"] += 1
    output = Path(plan["output_directory"])
    with pytest.raises(ValueError):
        runner.compute(plan, output, provenance, install_guard=False)
    assert (output / "start.json").exists()
    assert not (output / "result.json").exists()
    assert json.loads((output / "failure.json").read_text())["status"] == "VALIDITY_FAILURE"


@pytest.mark.parametrize(
    "action", ["M", "query_scores", "raw_mat", "subprocess", "socket", "delete"]
)
def test_runtime_guard_denies_out_of_scope(tmp_path, action):
    guard = runner.guard_for({tmp_path / "S999.npz"}, tmp_path / "output")
    if action in ("M", "query_scores", "raw_mat"):
        event, args = "open", (str(tmp_path / action), "r", os.O_RDONLY)
    else:
        event, args = (
            {"subprocess": "subprocess.Popen", "socket": "socket.connect", "delete": "os.remove"}[
                action
            ],
            (),
        )
    with pytest.raises(ValueError):
        guard(event, args)


def test_publish_exclusive_and_budget(tmp_path):
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError):
        runner.publish(path, {"x": 1}, 1)
    receipt = runner.publish(path, {"x": 1}, 100)
    assert runner.sha(path) == receipt["sha256"]
    with pytest.raises(FileExistsError):
        runner.publish(path, {"x": 2}, 100)


def test_original_operator_and_plan_pins():
    plan = json.loads((REPO / "configs/analysis/trca_support_geometry_v1.json").read_text())
    assert runner.sha(REPO / "configs/analysis/trca_support_geometry_v1.json") == runner.PLAN_SHA
    assert runner.sha(REPO / "src/cfeg/analysis/metadata_trca_prior.py") == plan["operator_sha256"]
    assert plan["records_expected"] == 39 * 2 * 4 * 2 * 12 * 5
    assert plan["gamma"] == 0.1 and plan["support_blocks"] == list(range(5))


def test_executed_module_origin_required(monkeypatch):
    runner.verify_origins()
    monkeypatch.setattr(runner.native_support_prefix, "__file__", "/tmp/unapproved.py")
    with pytest.raises(ValueError, match="origin"):
        runner.verify_origins()
