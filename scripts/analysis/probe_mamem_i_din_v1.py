"""One development-record header/DIN probe; no EEG materialization or fitting.

Run as a fresh subprocess. DIN_1 is decoded in full by scipy, but only the
prespecified prefix's timestamp/sample scalar values enter the calculation.
"""
import argparse
import hashlib
import json
import os
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path

CAP = 512 * 1024**2


class Stop(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise Stop(reason)


def scalar(value):
    import numpy as np
    a = np.asarray(value)
    require(a.size == 1 and a.dtype.kind in "iuf", "non_numeric_scalar")
    result = float(a.item())
    require(np.isfinite(result), "nonfinite_scalar")
    return result


def summarize_prefix(din, sample_limit=None):
    import numpy as np
    require(isinstance(din, np.ndarray) and din.ndim == 2 and din.shape[0] >= 4,
            "unsupported_DIN_shape_no_transpose")
    require(din.dtype == object, "expected_author_cell_array")
    n = min(200, din.shape[1])
    timestamps, samples = [], []
    boundary = False
    for i in range(n):
        timestamp = scalar(din[1, i])
        if timestamps:
            interval = timestamp - timestamps[-1]
            require(interval > 0, "nonincreasing_timestamp")
            if interval > 2000:
                boundary = True
                break  # Do not inspect this event's sample or any following values.
        sample = scalar(din[3, i])
        require(sample >= 1 and sample.is_integer(), "invalid_sample_index")
        require(sample_limit is None or sample <= sample_limit, "sample_outside_EEG_header")
        if samples:
            require(sample > samples[-1], "nonincreasing_sample")
        timestamps.append(timestamp)
        samples.append(sample)
    dt, ds = np.diff(timestamps), np.diff(samples)
    residual = dt - 4 * ds
    report = {
        "events_in_first_group_prefix": len(timestamps),
        "timestamp_scalars_inspected": len(timestamps) + int(boundary),
        "sample_scalars_inspected": len(samples),
        "boundary_observed": boundary,
        "group_complete": boundary,
        "censored_without_boundary": not boundary,
        "maximum_events": 200,
        "units": "author loader: row2 milliseconds, row4 one-based 250Hz samples",
        "interpretation": {
            "physical_jitter_verified": False, "independent_M_verified": False,
            "calibration_savings_evaluated": False, "frequency_or_label_inference": False,
            "EEG_values_materialized": False, "fits": 0, "held60": 0,
        },
    }
    if len(dt) == 0:
        report.update(status="INSUFFICIENT_FIRST_PREFIX", interval_summary=None)
        return report
    max_residual = float(np.max(np.abs(residual)))
    report.update(
        status="DIN_PREFIX_SCHEMA_OBSERVED" if max_residual <= 4.001 else "CLOCK_MAPPING_UNRESOLVED",
        interval_summary={
            "n": len(dt), "min_ms": float(dt.min()), "max_ms": float(dt.max()),
            "mean_ms": float(dt.mean()), "std_ms_population": float(dt.std()),
            "timestamp_minus_4sample_max_abs_ms": max_residual,
            "timestamp_minus_4sample_mean_ms": float(residual.mean()),
            "clock_mapping_tolerance_ms": 4.001,
        },
    )
    return report


def decoded_bytes(value):
    """Conservative allocation accounting, not a scan of numeric data values."""
    import numpy as np
    seen, stack, total = set(), [value], 0
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        total += sys.getsizeof(current)
        if isinstance(current, np.ndarray):
            total += current.nbytes  # Conservative double count of owning buffers.
            if current.dtype == object:
                stack.extend(current.flat)
        require(total <= CAP, "decoded_DIN_memory_cap")
    return total


def hash_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path, mode, role, loader=None, header_loader=None):
    from scipy.io import loadmat, whosmat
    loader = loader or loadmat
    header_loader = header_loader or whosmat
    require(path.is_file() and not path.is_symlink(), "regular_MAT_required")
    require(path.stat().st_size <= CAP, "MAT_file_cap")
    require(str(path.resolve()) == role["mat_path"], "role_path_mismatch")
    require(hash_file(path) == role["mat_sha256"], "role_hash_mismatch")
    require(role["subject_role"] == "development_only_all_records", "role_required")
    require(role["member_selection"] == "lexicographic_first_MAT_across_both_archives",
            "selection_required")
    headers = header_loader(path)
    report = {"variables": [{"name": n, "shape": list(s), "class": c} for n, s, c in headers],
              "mat_sha256": role["mat_sha256"], "subject": role["subject"], "mode": mode}
    if mode == "schema":
        return dict(report, status="HEADER_ONLY", EEG_values_materialized=False,
                    DIN_values_materialized=False)
    din_headers = [(s, c) for n, s, c in headers if n == "DIN_1"]
    require(len(din_headers) == 1, "DIN_1_not_unique_or_missing")
    shape, cls = din_headers[0]
    require(cls == "cell" and len(shape) == 2 and shape[0] >= 4, "unsupported_DIN_header")
    require(shape[0] * shape[1] * 8 <= CAP, "DIN_pointer_cap")
    eeg_headers = [(s, c) for n, s, c in headers if n == "eeg"]
    require(len(eeg_headers) == 1 and len(eeg_headers[0][0]) == 2,
            "expected_2D_eeg_header")
    sample_limit = eeg_headers[0][0][1]
    loaded = loader(path, variable_names=["DIN_1"], squeeze_me=False, struct_as_record=True,
                    verify_compressed_data_integrity=True)
    require(set(loaded) <= {"__header__", "__version__", "__globals__", "DIN_1"},
            "unexpected_materialized_variable")
    din = loaded["DIN_1"]
    report["DIN_decoded_allocation_upper_estimate"] = decoded_bytes(din)
    report["DIN_variable_fully_decoded"] = True
    report["numeric_value_scope"] = "first group prefix <=200 events; boundary timestamp only"
    return dict(report, **summarize_prefix(din, sample_limit=sample_limit))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["schema", "din"])
    parser.add_argument("role")
    parser.add_argument("output")
    args = parser.parse_args()
    # Must precede scipy/numpy import; regular EEG arrays are never requested.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    role = json.loads(Path(args.role).read_bytes())
    output = Path(args.output)
    require(not output.exists(), "report_no_overwrite")
    result = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "schema": "cfeg.mamem-i-din-probe.v1", "mode": args.mode,
              "EEG_values_materialized": False, "fits": 0, "held60": 0}
    try:
        result.update(inspect(Path(role["mat_path"]), args.mode, role))
    except Exception as e:
        result.update(status="PROBE_STOPPED", error_type=type(e).__name__,
                      reason=str(e) if isinstance(e, Stop) else "details_not_exported")
    result["ended_utc"] = datetime.now(timezone.utc).isoformat()
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
