"""Independent saved-correlation audit; never imports producer, toolbox or raw loaders."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import t

PLAN_SHA256 = "d36886d4dfd80961d1d8b7ceac685d468ca12211ff6149ecbf993cc0b4502462"
KEYS = ("stage", "view", "interface", "n_samples", "method", "k")
IDS = (
    4,
    6,
    8,
    11,
    14,
    21,
    22,
    25,
    28,
    29,
    30,
    31,
    32,
    33,
    37,
    41,
    42,
    43,
    44,
    46,
    54,
    55,
    56,
    61,
    63,
    65,
    67,
    73,
    74,
    77,
    80,
    82,
    83,
    84,
    89,
    92,
    97,
    100,
    102,
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    radius = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {
        "mean": mean,
        "ci95_low": mean - radius,
        "ci95_high": mean + radius,
        "n_participants": len(values),
    }


def validate_cache(cache, plan):
    if set(cache) != {"a0_r", "etrca_lobo_r", "etrca_chrono_r"}:
        raise ValueError("Unexpected cache keys")
    for name, array in cache.items():
        if array.shape != tuple(plan["cache"][name]["shape"]) or array.dtype != np.float64:
            raise ValueError(f"Cache shape/dtype mismatch: {name}")
        if not np.isfinite(array).all() or np.max(np.abs(array)) > 1 + 1e-10:
            raise ValueError(f"Invalid correlations: {name}")


def accuracy(correlations, weights, rotate=0):
    """Same documented native reduction order, separately implemented traversal."""
    flat = correlations.reshape(-1, 5, 12)
    weights = np.asarray(weights).reshape(1, 5)
    labels = np.tile(np.arange(12), len(flat) // 12)
    predictions = [int(np.argmax(weights @ np.roll(r, rotate, axis=1))) for r in flat]
    return float(np.mean(np.asarray(predictions) == labels))


def reconstruct(cache, plan):
    validate_cache(cache, plan)
    rows = []

    def emit(participant, stage, view, interface, n_samples, method, k, corr):
        base = "A0_author" if method == "A0_author" else "ETRCA"
        weights = plan["methods"]["weights"][base][interface]
        if method == "ETRCA_rotated":
            ba = float(np.mean([accuracy(corr, weights, shift) for shift in range(1, 12)]))
        else:
            ba = accuracy(corr, weights)
        rows.append(
            {
                "participant": participant,
                "stage": stage,
                "view": view,
                "interface": interface,
                "n_samples": n_samples,
                "method": method,
                "k": k,
                "ba": ba,
                "query_count": int(np.prod(corr.shape[:-2])),
                "label_count": 12 * k,
            }
        )

    for p, participant in enumerate(IDS):
        for i, interface in enumerate(plan["interfaces"]):
            for view, block_slice in [("all10", slice(None)), ("last5", slice(5, 10))]:
                for method, k, corr in [
                    ("A0_author", 0, cache["a0_r"][p, i, 3, block_slice]),
                    ("ETRCA", 9, cache["etrca_lobo_r"][p, i, block_slice]),
                    ("ETRCA_rotated", 9, cache["etrca_lobo_r"][p, i, block_slice]),
                ]:
                    emit(participant, "native_lobo_2s", view, interface, 500, method, k, corr)
            for w, n_samples in enumerate(plan["sample_counts"]):
                stage = "bridge_chronological_2s" if n_samples == 500 else "bridge_short_windows"
                emit(
                    participant,
                    stage,
                    "last5",
                    interface,
                    n_samples,
                    "A0_author",
                    0,
                    cache["a0_r"][p, i, w, 5:10],
                )
                for b, k in enumerate((3, 5)):
                    for method in ("ETRCA", "ETRCA_rotated"):
                        emit(
                            participant,
                            stage,
                            "last5",
                            interface,
                            n_samples,
                            method,
                            k,
                            cache["etrca_chrono_r"][p, i, w, b],
                        )
    rows.sort(key=lambda row: tuple(row[key] for key in ("participant", *KEYS)))
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in KEYS)].append(row["ba"])
    summary = []
    for key, values in sorted(groups.items()):
        stats = interval(values)
        summary.append({**dict(zip(KEYS, key)), "mean_ba": stats.pop("mean"), **stats})
    return rows, summary


def diagnostics(rows):
    lookup = {tuple(row[k] for k in ("participant", *KEYS)): row["ba"] for row in rows}

    def values(stage, view, interface, n_samples, method, k):
        return np.array([lookup[(p, stage, view, interface, n_samples, method, k)] for p in IDS])

    def chrono(interface, n_samples, method, k):
        stage = "bridge_chronological_2s" if n_samples == 500 else "bridge_short_windows"
        return values(stage, "last5", interface, n_samples, method, k)

    contrasts, harms, attainment = [], [], []

    def contrast(name, interface, n_samples, delta):
        contrasts.append(
            {"contrast": name, "interface": interface, "n_samples": n_samples, **interval(delta)}
        )

    for interface in ("dry", "wet"):
        for view in ("all10", "last5"):
            contrast(
                f"native_ETRCA9_minus_A0_{view}",
                interface,
                500,
                values("native_lobo_2s", view, interface, 500, "ETRCA", 9)
                - values("native_lobo_2s", view, interface, 500, "A0_author", 0),
            )
        native_last = values("native_lobo_2s", "last5", interface, 500, "ETRCA", 9)
        for k in (3, 5):
            contrast(
                f"chrono_ETRCA{k}_minus_native9_last5",
                interface,
                500,
                chrono(interface, 500, "ETRCA", k) - native_last,
            )
        for n_samples in (125, 188, 250, 500):
            anchor = chrono(interface, n_samples, "A0_author", 0)
            adapted = {k: chrono(interface, n_samples, "ETRCA", k) for k in (3, 5)}
            for k in (3, 5):
                delta = adapted[k] - anchor
                contrast(f"chrono_ETRCA{k}_minus_A0", interface, n_samples, delta)
                harms.extend(
                    {
                        "participant": p,
                        "interface": interface,
                        "n_samples": n_samples,
                        "k": k,
                        "delta_ba": float(d),
                    }
                    for p, d in zip(IDS, delta)
                )
            contrast("chrono_ETRCA5_minus_ETRCA3", interface, n_samples, adapted[5] - adapted[3])
            for p_idx, p in enumerate(IDS):
                first = next(
                    (
                        k
                        for k, ba in [
                            (0, anchor[p_idx]),
                            (3, adapted[3][p_idx]),
                            (5, adapted[5][p_idx]),
                        ]
                        if ba >= 0.8
                    ),
                    None,
                )
                attainment.append(
                    {
                        "participant": p,
                        "interface": interface,
                        "n_samples": n_samples,
                        "first_observed_k": first,
                        "censored_above_5": first is None,
                        "label_count": None if first is None else 12 * first,
                        "retained_seconds": None if first is None else 12 * first * n_samples / 250,
                        "history_seconds": None if first is None else 12 * first * 0.14,
                    }
                )
            if n_samples != 500:
                for method, k in [("A0_author", 0), ("ETRCA", 3), ("ETRCA", 5)]:
                    contrast(
                        f"{method}{k}_short_minus_2s",
                        interface,
                        n_samples,
                        chrono(interface, n_samples, method, k) - chrono(interface, 500, method, k),
                    )
    overall_harm = []
    for k in (3, 5):
        # All eight chronological cells have 60 queries. Use integer correct
        # counts for signs so cancelling floating fractions remain true ties.
        net_correct = [
            sum(round(60 * r["delta_ba"]) for r in harms if r["k"] == k and r["participant"] == p)
            for p in IDS
        ]
        deltas = [
            float(np.mean([r["delta_ba"] for r in harms if r["k"] == k and r["participant"] == p]))
            for p in IDS
        ]
        overall_harm.append(
            {
                "k": k,
                "cell_weighting": "all8 equal, including2s",
                "help": sum(d > 0 for d in net_correct),
                "harm": sum(d < 0 for d in net_correct),
                "tie": sum(d == 0 for d in net_correct),
                **interval(deltas),
            }
        )
    return {
        "contrasts": contrasts,
        "harm": harms,
        "overall_harm": overall_harm,
        "attainment": attainment,
    }


def compare_records(actual, expected, label):
    if len(actual) != len(expected):
        raise ValueError(f"{label}: record count differs")
    for i, (left, right) in enumerate(zip(actual, expected)):
        if set(left) != set(right):
            raise ValueError(f"{label}[{i}]: keys differ")
        for key, value in right.items():
            if isinstance(value, float):
                if (
                    not isinstance(left[key], (int, float))
                    or not np.isfinite(left[key])
                    or abs(left[key] - value) > 1e-12
                ):
                    raise ValueError(f"{label}[{i}].{key}: numerical mismatch")
            elif left[key] != value:
                raise ValueError(f"{label}[{i}].{key}: mismatch")


def audit(plan_path, output):
    if digest(plan_path) != PLAN_SHA256:
        raise ValueError("Not the frozen plan")
    plan = json.loads(plan_path.read_text())
    expected_files = set(plan["execution"]["artifacts"])
    if output.is_symlink() or {p.name for p in output.iterdir()} != expected_files:
        raise ValueError("Unexpected output inventory")
    for name in expected_files:
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Invalid output artifact")
        if path.stat().st_mode & 0o222 or path.stat().st_nlink != 1:
            raise ValueError("Output artifact must be read-only and singly linked")
    result = json.loads((output / "result.json").read_text())
    start = json.loads((output / "start.json").read_text())
    if (
        result["schema"] != plan["reporting"]["schema"]
        or result["status"] != "COMPATIBILITY_ASSESSMENT_COMPLETE"
    ):
        raise ValueError("Invalid result schema/status")
    if start["schema"] != "cfeg.author-etrca-source.start.v1":
        raise ValueError("Invalid start schema")
    for key in (
        "study_id",
        "plan_sha256",
        "source_commit",
        "source_tree",
        "upstream_revision",
        "started_at",
    ):
        if start[key] != result[key]:
            raise ValueError(f"Start binding mismatch: {key}")
    if result["plan_sha256"] != PLAN_SHA256 or result["study_id"] != plan["study_id"]:
        raise ValueError("Plan binding mismatch")
    if result["upstream_revision"] != plan["upstream"]["revision"]:
        raise ValueError("Upstream binding mismatch")
    for name, expected in plan["upstream"]["pins"].items():
        if start["imported_source_hashes"].get(name) != expected:
            raise ValueError("Imported source binding mismatch")
    if start["python"] != "3.9.21" or start["dependencies"] != {
        "numpy": "1.23.4",
        "scipy": "1.13.0",
        "joblib": "1.4.2",
        "scikit-learn": "1.3.0",
        "mat73": "0.63",
        "h5py": "3.11.0",
        "threadpoolctl": "3.5.0",
    }:
        raise ValueError("Runtime binding mismatch")
    for key in ("source_commit", "source_tree"):
        if not re.fullmatch(r"[0-9a-f]{40}", result[key]):
            raise ValueError("Invalid git identity")
    if start["source_subject_ids"] != list(IDS) or any(
        start[k] for k in ("metadata_access", "held_access", "retired_access")
    ):
        raise ValueError("Invalid input authority")
    if start["raw_root"] != plan["raw_root"]:
        raise ValueError("Raw root mismatch")
    if start["output_root"] != plan["execution"]["output_root"]:
        raise ValueError("Output root mismatch")
    began = datetime.fromisoformat(start["started_at"].replace("Z", "+00:00"))
    ended = datetime.fromisoformat(result["completed_at"].replace("Z", "+00:00"))
    if began.tzinfo is None or ended.tzinfo is None or ended < began:
        raise ValueError("Invalid completion chronology")
    raw_files = result["raw_files"]
    if [r["subject"] for r in raw_files] != list(IDS):
        raise ValueError("Raw provenance does not cover exact39")
    for record in raw_files:
        expected_path = str(Path(plan["raw_root"]) / f"S{record['subject']:03d}.mat")
        if record["path"] != expected_path or record["shape"] != plan["raw_shape"]:
            raise ValueError("Raw path/shape mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) or not record["stored_dtype"]:
            raise ValueError("Invalid raw hash/dtype record")
    if digest(output / "correlations.npz") != result["cache_sha256"]:
        raise ValueError("Cache hash mismatch")
    with np.load(output / "correlations.npz", allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    rows, summary = reconstruct(cache, plan)
    if len(rows) != 2028 or len(summary) != 52:
        raise ValueError("Audit dimension error")
    compare_records(result["rows"], rows, "rows")
    compare_records(result["summary"], summary, "summary")
    return {
        "status": "AUDIT_PASS",
        "rows": len(rows),
        "summary_count": len(summary),
        "result_sha256": digest(output / "result.json"),
        "summary": summary,
        **diagnostics(rows),
        "scope": "saved-correlation arithmetic, not raw EEG replay or efficacy",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    report = audit(args.plan, args.output)
    if not args.details:
        report = {
            k: report[k] for k in ("status", "rows", "summary_count", "result_sha256", "scope")
        }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
