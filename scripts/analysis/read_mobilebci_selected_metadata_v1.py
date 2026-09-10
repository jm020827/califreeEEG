"""Read only four explicitly observed small metadata variables from the verified pair."""

import argparse
import hashlib
import json
import signal
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def main(config):
    fields = ["raw_fs", "raw_clab", "preprocess_fs", "preprocess_clab"]
    require(config["variables"] == fields and len(config["files"]) == 2, "unexpected_scope")
    limits = config["limits"]
    result = {"schema": "cfeg.mobilebci-selected-metadata-observation.v1", "records": []}
    total_label_bytes = 0
    for item in config["files"]:
        path = Path(item["path"])
        before = path.stat()
        require(path.resolve(strict=True) == path and before.st_size == item["bytes"], "input_stat")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(65536), b""):
                digest.update(chunk)
        require(digest.hexdigest() == item["sha256"], "input_sha256")
        values = loadmat(path, variable_names=fields, simplify_cells=True)
        record = {"path": str(path)}
        for field in fields:
            value = np.asarray(values[field])
            if field.endswith("_fs"):
                require(value.size == 1, "fs_not_scalar")
                rate = float(value.item())
                require(np.isfinite(rate) and 0 < rate <= 1e6, "fs_invalid")
                record[field] = rate
            else:
                require(value.size <= limits["labels_per_variable"], "labels_limit")
                labels = []
                for label in value.ravel():
                    require(isinstance(label, (str, np.str_)), "non_string_label")
                    size = len(label.encode("utf-8"))
                    require(size <= limits["label_utf8_bytes"], "label_length")
                    total_label_bytes += size
                    require(total_label_bytes <= limits["total_label_bytes"], "total_label_limit")
                    labels.append(str(label))
                record[field] = labels
        after = path.stat()
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "input_changed",
        )
        result["records"].append(record)
    result.update(
        status="COMPLETE_METADATA_ONLY", loadmat_calls=2, total_label_bytes=total_label_bytes
    )
    result["limitations"] = (
        "No event/t/raw_x/preprocess_x values; labels and rates do not establish units or exact clock/trigger alignment."
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    def stop(_signum, _frame):
        raise TimeoutError("metadata_deadline")

    signal.signal(signal.SIGALRM, stop)
    signal.alarm(config["limits"]["total_seconds"])
    try:
        main(config)
    finally:
        signal.alarm(0)
