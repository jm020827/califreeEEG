"""One-pass event/t metadata inspection of the already verified public pair.

No waveform variables are requested. Memory limits are set by the CLI before
selective MAT decoding; scalar-count guards apply afterwards, explicitly.
"""

import argparse
import hashlib
import json
import resource
import signal
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

FIELDS = ["event", "t"]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def project(value, limits, counter, depth=0):
    require(depth <= limits["nesting_depth"], "depth_limit")
    if isinstance(value, dict):
        require(len(value) <= 64, "field_limit")
        result = {}
        for key, child in value.items():
            require(isinstance(key, str) and len(key) <= 128, "field_name")
            result[key] = project(child, limits, counter, depth + 1)
        return result
    array = np.asarray(value)
    counter[0] += array.size
    require(counter[0] <= limits["postdecode_scalar_elements_total"], "element_limit")
    require(array.dtype.kind in "biufUS", "unsupported_metadata_dtype")
    if array.dtype.kind in "biuf":
        require(np.isfinite(array).all(), "nonfinite_metadata")
    else:
        require(array.nbytes <= 8192, "string_limit")
    return {"shape": list(array.shape), "dtype": str(array.dtype), "values": array.tolist()}


def inspect(config, loader=loadmat):
    require(config["variables"] == FIELDS and len(config["files"]) == 2, "scope")
    limits = config["limits"]
    result = {"schema": "cfeg.mobilebci-event-time-observation.v1", "records": []}
    count = [0]
    calls = 0
    try:
        for item in config["files"]:
            signal.alarm(limits["seconds_per_file"])
            started = time.monotonic()
            path = Path(item["path"])
            before = path.stat()
            require(
                path.resolve(strict=True) == path and before.st_size == item["bytes"], "input_stat"
            )
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(65536), b""):
                    digest.update(chunk)
            require(digest.hexdigest() == item["sha256"], "input_sha256")
            calls += 1
            values = loader(path, variable_names=FIELDS, simplify_cells=True)
            require(
                set(values) - {"__header__", "__version__", "__globals__"} == set(FIELDS),
                "unexpected_fields",
            )
            record = {"path": str(path), "sha256": digest.hexdigest(), "metadata": {}}
            for field in FIELDS:
                record["metadata"][field] = project(values[field], limits, count)
            after = path.stat()
            require(
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                "input_changed",
            )
            record["seconds"] = round(time.monotonic() - started, 6)
            result["records"].append(record)
            signal.alarm(0)
        result["status"] = "COMPLETE_EVENT_TIME_METADATA_ONLY"
    except (ValueError, OSError, MemoryError, TimeoutError, KeyError, TypeError) as error:
        result["status"] = "STOPPED"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)[:256]
    finally:
        signal.alarm(0)
    result.update(
        decode_calls=calls, scalar_elements=count[0], waveform_decodes=0, fits=0, outcomes=0
    )
    result["limitations"] = (
        "Event/t metadata access only; numerical agreement alone is not a verified raw clock/export definition or zero drift."
    )
    rendered = json.dumps(result, indent=2, allow_nan=False)
    require(len(rendered.encode()) <= limits["output_bytes"], "output_limit")
    return rendered, result["status"] == "COMPLETE_EVENT_TIME_METADATA_ONLY"


def selftest():
    from io import BytesIO

    from scipy.io import savemat

    limits = {"nesting_depth": 5, "postdecode_scalar_elements_total": 16}
    assert project({"latency": np.array([1, 2]), "type": 11}, limits, [0])["type"]["values"] == 11
    rejected = 0
    for bad, settings in [
        (np.array([np.nan]), limits),
        (np.array([1 + 2j]), limits),
        (np.ones(17), limits),
        ({"a": {"b": 1}}, {**limits, "nesting_depth": 1}),
        (np.array([object()]), limits),
    ]:
        try:
            project(bad, settings, [0])
        except ValueError:
            rejected += 1
    assert rejected == 5
    buffer = BytesIO()
    savemat(
        buffer,
        {
            "event": {"latency": np.array([1, 2]), "type": np.array([11, 12])},
            "t": np.array([0, 0.01]),
            "raw_x": np.ones((2, 2)),
        },
    )
    buffer.seek(0)
    values = loadmat(buffer, variable_names=FIELDS, simplify_cells=True)
    assert "raw_x" not in values and values["event"]["type"].tolist() == [11, 12]
    assert len(project(values["t"], limits, [0])["values"]) == 2
    print(
        json.dumps({"status": "PASS", "cases": 8, "human_reads": 0, "fixture_numeric_elements": 10})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    else:
        require(args.config is not None, "missing_config")
        config = json.loads(args.config.read_text())
        limit = config["limits"]["address_space_bytes"]
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        def deadline(_signum, _frame):
            raise TimeoutError("event_metadata_deadline")

        signal.signal(signal.SIGALRM, deadline)
        output, ok = inspect(config)
        print(output)
        raise SystemExit(0 if ok else 1)
