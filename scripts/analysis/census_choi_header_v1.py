"""Enumerate names/types at root and cnt without reading dataset/attribute values."""

import argparse
import json
import signal
import stat
import time
from pathlib import Path

import h5py


class CensusStop(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise CensusStop(reason)


def names(keys, limit, bytes_limit):
    result = []
    for key in keys:
        require(len(result) < limit, "name_count_limit")
        require(len(key.encode("utf-8")) <= bytes_limit, "name_length_limit")
        result.append(key)
    return sorted(result)


def object_metadata(obj, limits):
    result = {
        "type": "group" if isinstance(obj, h5py.Group) else "dataset",
        "attribute_names": names(
            obj.attrs.keys(),
            limits["attribute_names_per_object_max"],
            limits["object_names_bytes_max"],
        ),
    }
    if isinstance(obj, h5py.Dataset):
        result.update(shape=None if obj.shape is None else list(obj.shape), dtype=str(obj.dtype))
    return result


def enumerate_group(group, limits):
    result = object_metadata(group, limits)
    children = {}
    for key in names(group.keys(), limits["keys_per_group_max"], limits["object_names_bytes_max"]):
        link = group.get(key, getlink=True)
        if isinstance(link, h5py.HardLink):
            children[key] = {"link": "hard", **object_metadata(group[key], limits)}
        else:
            # Even the target strings of soft/external links are not needed here.
            children[key] = {"link": type(link).__name__, "opened": False}
    result["children"] = children
    result["immediate_keys_complete"] = True
    return result


def identity(path):
    info = path.stat()
    return {
        "bytes": info.st_size,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
    }


def run(config):
    result = {
        "schema": "cfeg.choi-header-census-observation.v1",
        "status": "STARTED",
        "hdf5_opens": 0,
        "dataset_value_reads": 0,
        "attribute_value_reads": 0,
        "reference_dereferences": 0,
    }
    started = time.monotonic()
    path = Path(config["input_path"])
    limits = config["budget"]

    def deadline(_signal, _frame):
        raise CensusStop("wall_deadline")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, limits["run_wall_seconds_max"])
    try:
        require(
            path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),
            "input_not_exact_regular_file",
        )
        before = identity(path)
        require(before["bytes"] == config["expected_file_bytes"], "input_size_changed")
        require(config["groups_to_enumerate"] == ["/", "/cnt"], "unsupported_scope")
        result["path"], result["stat_before"] = str(path), before
        result["hdf5_opens"] += 1
        with h5py.File(path, "r") as source:
            result["groups"] = {"/": enumerate_group(source, limits)}
            require(isinstance(source.get("cnt", getlink=True), h5py.HardLink), "cnt_not_hard_link")
            require(isinstance(source["cnt"], h5py.Group), "cnt_not_group")
            result["groups"]["/cnt"] = enumerate_group(source["cnt"], limits)
        result["stat_after"] = identity(path)
        require(result["stat_after"] == before, "input_stat_changed")
        result["status"] = "COMPLETE_NAMES_ONLY_CENSUS"
    except (CensusStop, OSError) as error:
        result = {
            "schema": result["schema"],
            "status": "STOPPED",
            "hdf5_opens": result["hdf5_opens"],
            "failure_type": type(error).__name__,
            "failure_reason": str(error) if isinstance(error, CensusStop) else "file_or_hdf5_error",
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result["elapsed_seconds"] = round(time.monotonic() - started, 6)
    result["limitations"] = (
        "Only root/cnt immediate keys and direct-child metadata; no recursion/reference dereference/attribute values/user block/full-file checksum. Value-read zeros describe this reviewed code path, not HDF5 internal physical IO instrumentation."
    )
    rendered = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    if len(rendered.encode("utf-8")) > limits["output_bytes_max"]:
        rendered = (
            json.dumps(
                {
                    "schema": result["schema"],
                    "status": "STOPPED",
                    "hdf5_opens": result["hdf5_opens"],
                    "failure_reason": "output_limit",
                }
            )
            + "\n"
        )
    require(len(rendered.encode("utf-8")) <= limits["output_bytes_max"], "output_limit_too_small")
    print(rendered, end="")


def selftest():
    limits = {
        "keys_per_group_max": 128,
        "attribute_names_per_object_max": 64,
        "object_names_bytes_max": 256,
    }
    with h5py.File("generated-census-only.h5", "w", driver="core", backing_store=False) as source:
        cnt = source.create_group("cnt")
        cnt.create_dataset("x", shape=(2, 3), dtype="f8")
        cnt.attrs["history"] = "generated; must not be read"
        source["soft"] = h5py.SoftLink("/cnt/x")
        source["external"] = h5py.ExternalLink("never-open.h5", "/x")
        original_dataset_get = h5py.Dataset.__getitem__
        original_attribute_get = h5py.AttributeManager.__getitem__

        def deny_values(*_args, **_kwargs):
            raise AssertionError("value_read")

        h5py.Dataset.__getitem__ = deny_values
        h5py.AttributeManager.__getitem__ = deny_values
        try:
            root = enumerate_group(source, limits)
            child = enumerate_group(cnt, limits)
        finally:
            h5py.Dataset.__getitem__ = original_dataset_get
            h5py.AttributeManager.__getitem__ = original_attribute_get
        assert root["children"]["soft"]["opened"] is False
        assert root["children"]["external"]["opened"] is False
        assert child["attribute_names"] == ["history"]
        assert child["children"]["x"]["shape"] == [2, 3]
    print(
        json.dumps(
            {
                "status": "PASS",
                "cases": [
                    "metadata_without_value_getters",
                    "soft_link_not_followed",
                    "external_link_not_followed",
                    "attribute_names_not_values",
                ],
                "fixture_elements": 6,
                "human_file_opens": 0,
                "persisted_fixture_files": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    else:
        run(json.loads(args.config.read_text()))
