"""Bounded MAT cell/struct projection for the same event/t recovery, not waveforms."""

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


class Projector:
    """Limits bound visited containers/leaves after decoding, not MAT allocations."""

    def __init__(self, limits):
        self.limits = limits
        self.visited_units = 0
        self.string_bytes = 0
        self.context = {}

    def charge(self, count):
        self.visited_units += count
        require(self.visited_units <= self.limits["visited_units"], "visited_limit")

    def text(self, value):
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8", errors="strict")
        value = str(value)
        self.string_bytes += len(value.encode("utf-8"))
        require(self.string_bytes <= self.limits["string_bytes"], "string_limit")
        return value

    def project(self, value, path, depth=0):
        self.context = {"path": path, "python_type": type(value).__name__}
        if isinstance(value, np.ndarray):
            self.context.update(dtype=str(value.dtype), shape=list(value.shape))
        require(depth <= self.limits["nesting_depth"], "depth_limit")
        if isinstance(value, dict):
            self.charge(1)
            require(len(value) <= 64, "field_limit")
            result = {}
            for key, child in value.items():
                require(isinstance(key, str) and len(key) <= 128, "field_name")
                result[key] = self.project(child, f"{path}.{key}", depth + 1)
            return {"kind": "struct", "fields": result}
        if isinstance(value, (list, tuple)):
            self.charge(1)
            return {
                "kind": "sequence",
                "shape": [len(value)],
                "items": [
                    self.project(child, f"{path}[{i}]", depth + 1) for i, child in enumerate(value)
                ],
            }
        if isinstance(value, np.ndarray):
            self.charge(1)
            if value.dtype.kind == "O":
                return {
                    "kind": "cell",
                    "shape": list(value.shape),
                    "order": "C",
                    "items": [
                        self.project(child, f"{path}.flat[{i}]", depth + 1)
                        for i, child in enumerate(value.flat)
                    ],
                }
            require(value.dtype.kind in "biufUS", "unsupported_array_dtype")
            self.charge(value.size)
            if value.dtype.kind in "biuf":
                require(np.isfinite(value).all(), "nonfinite_metadata")
                values = value.tolist()
            else:
                # Flattened string storage has an explicit shape/order; bytes become UTF-8 text.
                values = [self.text(child) for child in value.flat]
            return {
                "kind": "array",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "values": values,
            }
        self.charge(1)
        if isinstance(value, (str, bytes, np.str_, np.bytes_)):
            return {"kind": "scalar", "value": self.text(value)}
        if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            require(bool(np.isfinite(value)), "nonfinite_metadata")
            return {
                "kind": "scalar",
                "value": value.item() if isinstance(value, np.generic) else value,
            }
        # Do not coerce arbitrary objects through __array__ or __str__.
        raise ValueError("unsupported_leaf_type")


def render_bounded(result, output_limit):
    try:
        rendered = json.dumps(result, indent=2, allow_nan=False)
        require(len(rendered.encode()) <= output_limit, "output_limit")
        return rendered
    except (ValueError, TypeError, MemoryError):
        result.update(
            status="STOPPED", error="metadata_serialization_or_output_limit", values_omitted=True
        )
        result["records"] = [
            {"path": row["path"], "sha256": row["sha256"]} for row in result["records"]
        ]
        rendered = json.dumps(result, allow_nan=False)
        require(len(rendered.encode()) <= output_limit, "compact_output_limit")
        return rendered


def inspect(config, loader=loadmat):
    require(config["variables"] == FIELDS and len(config["files"]) == 2, "scope")
    limits = config["limits"]
    require(2048 <= limits["output_bytes"] <= 65536, "output_limit_config")
    projector = Projector(limits)
    result = {"schema": "cfeg.mobilebci-event-time-observation.v2", "records": []}
    calls = 0
    checksum_passes = 0
    current_path = None
    try:
        for item in config["files"]:
            signal.alarm(limits["seconds_per_file"])
            started = time.monotonic()
            path = Path(item["path"])
            current_path = str(path)
            projector.context = {"phase": "input_check"}
            before = path.stat()
            require(
                path.resolve(strict=True) == path and before.st_size == item["bytes"], "input_stat"
            )
            digest = hashlib.sha256()
            checksum_passes += 1
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(65536), b""):
                    digest.update(chunk)
            require(digest.hexdigest() == item["sha256"], "input_sha256")
            projector.context = {"phase": "selected_decode"}
            calls += 1
            values = loader(path, variable_names=FIELDS, simplify_cells=True)
            projector.context = {"phase": "selected_keys", "variables": FIELDS}
            require(
                set(values) - {"__header__", "__version__", "__globals__"} == set(FIELDS),
                "unexpected_fields",
            )
            record = {"path": str(path), "sha256": digest.hexdigest(), "metadata": {}}
            for field in FIELDS:
                record["metadata"][field] = projector.project(values[field], field)
            projector.context = {"phase": "post_stat", "variables": FIELDS}
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
    except (
        ValueError,
        OSError,
        MemoryError,
        TimeoutError,
        KeyError,
        TypeError,
        OverflowError,
    ) as error:
        result.update(
            status="STOPPED",
            error_type=type(error).__name__,
            error=str(error)[:256],
            failed_file=current_path,
            error_context=projector.context,
        )
    finally:
        signal.alarm(0)
    result.update(
        decode_calls=calls,
        checksum_passes=checksum_passes,
        projector_visited_units=projector.visited_units,
        projected_string_bytes=projector.string_bytes,
        waveform_decodes=0,
        fits=0,
        outcomes=0,
    )
    result["limitations"] = (
        "Visited units count each container plus primitive scalar leaves, not total MAT allocation. Shapes are scipy-decoded shapes (simplify_cells squeezes MATLAB dimensions). Event metadata correspondence is not proof of raw index origin, physical synchronization or zero clock drift."
    )
    rendered = render_bounded(result, limits["output_bytes"])
    return rendered, result["status"] == "COMPLETE_EVENT_TIME_METADATA_ONLY"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    limit = config["limits"]["address_space_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    def deadline(_signum, _frame):
        raise TimeoutError("event_metadata_deadline")

    signal.signal(signal.SIGALRM, deadline)
    output, ok = inspect(config)
    print(output)
    raise SystemExit(0 if ok else 1)
