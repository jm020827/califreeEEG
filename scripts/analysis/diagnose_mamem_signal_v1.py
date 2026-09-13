"""Single pinned development signal diagnostic. No fits, retries or downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "docs/reports/mamem_signal_validity_v1_run"
MAT = Path("/home/whwovy/data/mamem_i_v1_20260913/development_first.mat")
ROLE = MAT.with_name("development_role.json")
MAT_SHA = "57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a"
ROLE_SHA = "dc26f85099ed667208ec7b992430d257d01f2a4aa727b7b98f8a5195f6d9af18"
DEADLINE = "2026-09-13T14:00:00+00:00"
CAP = 128 * 1024
WALL_END = None
UNCHANGED = {
    "src/cfeg/mamem_events_v1.py": "1c6762783fed206201a9c0539de5aa4c94d45564190bb08954ec4b0c242c096a",
    "src/cfeg/mamem_events_v2.py": "e25565ebbf3b0af8bdd469c3847f4114040268dfe2cff19a708d66a13914c666",
    "src/cfeg/mamem_signal_v1.py": "164d21d638fc0b792b58067dc44d34ba41aaa585811ccade91101a6a0c69d2fc",
}
PINS = [
    "scripts/analysis/diagnose_mamem_signal_v1.py",
    "src/cfeg/mamem_signal_diagnostic_v1.py",
    "docs/mamem_signal_validity_v1_contract.md",
    *UNCHANGED,
]
FIXED = {
    "schema": "cfeg.mamem-signal-validity-v1.manifest",
    "deadline_utc": DEADLINE,
    "mat_path": str(MAT),
    "mat_sha256": MAT_SHA,
    "mat_bytes": 137357437,
    "role_sha256": ROLE_SHA,
    "subject": "S001",
    "variables": ["eeg", "DIN_1", "samplingRate"],
    "fits": 0,
    "attempt_budget": 1,
    "wall_seconds": 120,
    "cpu_seconds": 90,
    "address_space_bytes": 2 * 1024**3,
    "output_cap": CAP,
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def remaining():
    seconds = datetime.fromisoformat(DEADLINE).timestamp() - time.time()
    if WALL_END is not None:
        seconds = min(seconds, WALL_END - time.monotonic())
    require(seconds > 0, "deadline")
    return min(seconds, 120)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            remaining()
            digest.update(chunk)
    return digest.hexdigest()


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(path, value):
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    require(len(raw) <= CAP, "output_cap")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_dir(Path(path).parent)


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= CAP,
            "bounded_regular_json")
    return json.loads(path.read_bytes(), parse_constant=lambda _: require(False, "json_nonfinite"))


def validate_manifest(cfg):
    require(set(cfg) == set(FIXED) | {"code_sha256", "created_utc"}, "manifest_keyset")
    require(all(cfg[key] == value for key, value in FIXED.items()), "manifest_scope")
    require(set(cfg["code_sha256"]) == set(PINS), "code_pin_keyset")
    require(all(cfg["code_sha256"][key] == value for key, value in UNCHANGED.items()),
            "unchanged_core_pins")
    require(all(sha(ROOT / key) == value for key, value in cfg["code_sha256"].items()),
            "code_hash")
    return cfg


def manifest():
    remaining()
    return validate_manifest(read_json(RUN / "manifest.json"))


def freeze():
    remaining()
    require(shutil.disk_usage(MAT.parent).free >= 8 * 1024**3, "free_reserve")
    hashes = {name: sha(ROOT / name) for name in PINS}
    cfg = dict(FIXED, code_sha256=hashes, created_utc=now())
    validate_manifest(cfg)
    RUN.mkdir()
    fsync_dir(RUN.parent)
    save(RUN / "manifest.json", cfg)


def validate_input():
    require(str(MAT.resolve()) == str(MAT) and MAT.is_file() and not MAT.is_symlink()
            and MAT.stat().st_size == FIXED["mat_bytes"] and sha(MAT) == MAT_SHA, "MAT_pin")
    require(sha(ROLE) == ROLE_SHA, "role_pin")
    role = read_json(ROLE)
    require(role["subject"] == "S001" and role["subject_role"] == "development_only_all_records"
            and role["mat_path"] == str(MAT) and role["mat_sha256"] == MAT_SHA
            and role["bytes"] == FIXED["mat_bytes"]
            and role["member"] == "EEG-SSVEP-Part1/S001a.mat", "development_role")


def inspect(loader=None, header_loader=None, access=None):
    access = [] if access is None else access
    manifest()
    validate_input()
    access.append("input_and_code_pins_verified")
    if loader is None or header_loader is None:
        from scipy.io import loadmat, whosmat

        loader, header_loader = loader or loadmat, header_loader or whosmat
    headers = header_loader(MAT)
    for name, shape, kind in (("eeg", (257, 117917), "double"),
                              ("DIN_1", (4, 1966), "cell")):
        require([(s, k) for n, s, k in headers if n == name] == [(shape, kind)], "MAT_header")
    rate_header = [(s, k) for n, s, k in headers if n == "samplingRate"]
    numeric = {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32",
               "int64", "uint64"}
    require(len(rate_header) == 1 and rate_header[0][0] == (1, 1)
            and rate_header[0][1] in numeric, "samplingRate_header")
    remaining()
    access.append("three_variables_load_requested")
    loaded = loader(MAT, variable_names=FIXED["variables"], squeeze_me=False,
                    struct_as_record=True, verify_compressed_data_integrity=True)
    require(set(FIXED["variables"]) <= set(loaded)
            <= set(FIXED["variables"]) | {"__header__", "__version__", "__globals__"},
            "loaded_variable_whitelist")
    access.append("three_variables_decoded")
    from cfeg.mamem_events_v1 import _scalar
    from cfeg.mamem_signal_diagnostic_v1 import diagnose, validate_report

    require(float(_scalar(loaded["samplingRate"])) == 250, "samplingRate")
    require(loaded["eeg"].shape == (257, 117917) and loaded["DIN_1"].shape == (4, 1966),
            "loaded_shapes")
    result = diagnose(loaded["eeg"], loaded["DIN_1"], checkpoint=remaining)
    access.append("15_main_windows_diagnosed")
    validate_report(result)
    remaining()
    return dict(result, status="DIAGNOSTIC_COMPLETE_NOT_EFFICACY", mat_sha256=MAT_SHA,
                stored_samplingRate_hz=250, completed_utc=now())


def worker():
    global WALL_END
    WALL_END = time.monotonic() + 120
    for limit, cap in ((resource.RLIMIT_AS, FIXED["address_space_bytes"]),
                       (resource.RLIMIT_CPU, 90), (resource.RLIMIT_FSIZE, CAP)):
        resource.setrlimit(limit, (cap, cap))
    manifest()
    started = read_json(RUN / "started.json")
    require(started["status"] == "STARTED" and started["attempt"] == 1
            and started["parent_pid"] == os.getppid()
            and started["manifest_sha256"] == sha(RUN / "manifest.json")
            and not (RUN / "terminal.json").exists(), "active_parent")
    save(RUN / "worker_claim.json", {"pid": os.getpid(), "attempt": 1, "started_utc": now()})
    access = []
    try:
        result = inspect(access=access)
        remaining()
        save(RUN / "diagnostic.json", dict(result, access_stages=access))
    except Exception as error:  # Preserve partial access, then re-raise.
        save(RUN / "worker_failure.json", {"status": "STOPPED", "access_stages": access,
             "error_type": type(error).__name__, "reason": str(error) if isinstance(error, ValueError)
             else "see_bounded_stderr", "completed_utc": now()})
        raise


def execute():
    manifest()
    require(shutil.disk_usage(MAT.parent).free >= 8 * 1024**3, "free_reserve")
    started = {"status": "STARTED", "attempt": 1, "parent_pid": os.getpid(), "started_utc": now(),
               "manifest_sha256": sha(RUN / "manifest.json")}
    save(RUN / "started.json", started)
    terminal = {"attempts": 1, "fits": 0, "started_utc": started["started_utc"]}
    try:
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", BLIS_NUM_THREADS="1", PYTHONPATH=str(ROOT / "src"))
        with (RUN / "stdout.txt").open("xb") as out, (RUN / "stderr.txt").open("xb") as err:
            timeout = remaining()
            process = subprocess.Popen([sys.executable, __file__, "worker"], stdout=out, stderr=err,
                                       env=env)
            try:
                process.wait(timeout=timeout)
                require(process.returncode == 0, "worker_failed")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            require(out.tell() <= CAP and err.tell() <= CAP, "capture_cap")
        remaining()
        result = read_json(RUN / "diagnostic.json")
        from cfeg.mamem_signal_diagnostic_v1 import validate_report

        require(result["status"] == "DIAGNOSTIC_COMPLETE_NOT_EFFICACY"
                and result["mat_sha256"] == MAT_SHA and result["stored_samplingRate_hz"] == 250,
                "result_binding")
        validate_report(result)
        manifest()
        terminal.update(status="COMPLETE", windows=15, diagnostic_sha256=sha(RUN / "diagnostic.json"))
        remaining()
    except Exception as error:  # noqa: BLE001 -- never retry a failed attempt
        terminal.update(status="STOPPED_NO_RETRY", error_type=type(error).__name__,
                        reason=str(error) if isinstance(error, ValueError) else "see_bounded_log")
    terminal["ended_utc"] = now()
    save(RUN / "terminal.json", terminal)
    print(json.dumps(terminal))
    if terminal["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "execute", "worker"))
    {"freeze": freeze, "execute": execute, "worker": worker}[parser.parse_args().mode]()
