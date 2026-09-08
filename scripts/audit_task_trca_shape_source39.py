"""Cold saved-artifact audit; no raw EEG/M, training or producer imports."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_shape_artifact_audit as audit
from cfeg.analysis.task_trca_shape_audit import summarize

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


def regular(path):
    info = path.stat(follow_symlinks=False)
    audit.require(
        path.resolve() == path and stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
        "regular unaliased audit input",
    )


def load_json(path):
    regular(path)
    return json.loads(path.read_bytes())


def load_arrays(path):
    regular(path)
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def run(output):
    output = output.absolute()
    audit.require(
        output.resolve() == output and not (output / "failure.json").exists(),
        "Failure receipt has precedence",
    )
    audit.require(not (output / "cold_audit.json").exists(), "Cold audit is append-only once")
    start = time.perf_counter()
    result = load_json(output / "result.json")
    initial = load_json(output / "start.json")
    manifest = initial["manifest"]
    audit.require(
        result["status"] == "COMPLETE" and result["manifest_sha256"] == initial["manifest_sha256"],
        "completion provenance",
    )
    audit.require(
        manifest["source_ids"] == list(IDS) and manifest["held60_authorized"] is False,
        "fixed source scope",
    )
    for path, digest in manifest["code_pins"].items():
        audit.require(audit.sha(ROOT / path) == digest, "audit code pin")
    artifacts = result["artifacts"]
    for value in artifacts.values():
        path = Path(value["path"])
        audit.require(path.parent == output, "artifact path escape")
        regular(path)
        audit.require(
            path.stat().st_size == value["bytes"] and audit.sha(path) == value["sha256"],
            "artifact byte binding",
        )
    freeze = load_json(output / "globalfreeze.json")
    audit.require(
        freeze["status"] == "ALL_MODELS_FROZEN"
        and freeze["query_access_count"] == 0
        and freeze["manifest_sha256"] == result["manifest_sha256"],
        "global freeze provenance",
    )
    access = [json.loads(line) for line in (output / "access.jsonl").read_text().splitlines()]
    counts = Counter()
    for row in access:
        fold = row["outer_fold"]
        audit.require(type(fold) is int and fold in (0, 1, 2), "access outer fold")
        eval_ids = IDS[fold::3]
        role = "evaluation" if row["participant_id"] in eval_ids else "fit"
        audit.require(
            row["participant_id"] in IDS and row["role"] == role, "access participant roles"
        )
        kind, blocks = row["kind"], row["blocks"]
        if kind in ("query", "a0", "full_k3", "full_k5"):
            audit.require(
                role == "evaluation"
                and blocks == [6, 7, 8, 9]
                and row["freeze_sha256"] == artifacts["freeze"]["sha256"],
                "query freeze rights",
            )
        elif kind == "source_supervision":
            audit.require(role == "fit" and blocks == [5], "source block5 rights")
        else:
            audit.require(
                kind in ("support", "metadata_support") and blocks in ([0, 1, 2], [0, 1, 2, 3, 4]),
                "support-only rights",
            )
        counts[(role, kind)] += 1
    expected = {
        ("fit", "source_supervision"): 624,
        ("fit", "support"): 1248,
        ("fit", "metadata_support"): 312,
        ("evaluation", "support"): 624,
        ("evaluation", "metadata_support"): 156,
        **{("evaluation", k): 312 for k in ("query", "a0", "full_k3", "full_k5")},
    }
    audit.require(dict(counts) == expected, "exact complete access event totals")
    aggregate = load_arrays(output / "scores.npz")
    np.testing.assert_array_equal(aggregate["weights"], manifest["native_weights"]["ETRCA"])
    np.testing.assert_array_equal(aggregate["a0_weights"], manifest["native_weights"]["A0_author"])
    training_receipts, evaluation_receipts = [], []
    for fold in range(3):
        evaluation_ids = IDS[fold::3]
        fit_ids = tuple(p for p in IDS if p not in evaluation_ids)
        model = load_json(output / f"model{fold}.json")
        audit.require(
            model["source_artifact"] == artifacts[f"source{fold}"]
            and model["manifest_sha256"] == result["manifest_sha256"],
            "model source binding",
        )
        source = load_arrays(output / f"source{fold}.npz")
        np.testing.assert_array_equal(source["weights"], aggregate["weights"][source["keys"][:, 1]])
        training_receipts.append(audit.audit_training(source, model, fit_ids, evaluation_ids))
        del source
        parts = [load_arrays(output / f"evaluation_S{pid:03d}.npz") for pid in evaluation_ids]
        data = {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}
        np.testing.assert_array_equal(data["weights"], aggregate["weights"][data["keys"][:, 1]])
        audit.require(
            sorted(set(data["keys"][:, 0].tolist())) == list(evaluation_ids),
            "exact outer evaluation participant set",
        )
        evaluation_receipts.append(audit.audit_evaluation(data, model["pipeline"]))
        donors = audit.donor_positions(data, range(len(data["keys"])))
        for j, (pid, interface, n, k) in enumerate(data["keys"]):
            p, ni, ki = IDS.index(int(pid)), audit.SAMPLES.index(int(n)), (3, 5).index(int(k))
            np.testing.assert_array_equal(
                aggregate["scores"][p, interface, ni, ki], data["scores"][j]
            )
            a0 = np.einsum(
                "nbc,b->nc", data["a0_correlations"][j], aggregate["a0_weights"][interface]
            )
            np.testing.assert_array_equal(aggregate["a0"][p, interface, ni], a0)
            r_change = np.max(np.abs(data["r"][j, 3] - data["r"][j, 1]))
            w_change = np.max(np.abs(data["filters"][j, 4] - data["filters"][j, 2]))
            audit.require(
                aggregate["r_changes"][p, interface, ni, ki] == r_change
                and aggregate["filter_changes"][p, interface, ni, ki] == w_change,
                "actuation aggregate binding",
            )
            if k == 3:
                for name, value in (
                    ("coverage_m", data["m"][j]),
                    ("coverage_donor_m", data["m"][donors[j]]),
                    ("coverage_available", data["available"][j]),
                ):
                    np.testing.assert_array_equal(aggregate[name][p, interface], value)
        np.testing.assert_array_equal(
            aggregate["qm_coefficients"][fold], model["pipeline"]["residuals"]["QM"]["coefficients"]
        )
        print(
            json.dumps(
                {
                    "event": "cold_fold_audit_pass",
                    "fold": fold,
                    "seconds": time.perf_counter() - start,
                }
            ),
            flush=True,
        )
        del parts, data
    summary = summarize(
        aggregate["scores"],
        aggregate["a0"],
        {
            "m": aggregate["coverage_m"],
            "donor_m": aggregate["coverage_donor_m"],
            "available": aggregate["coverage_available"],
        },
        {
            "band_weights": aggregate["weights"],
            "a0_band_weights": aggregate["a0_weights"],
            "qm_coefficients": aggregate["qm_coefficients"],
            "r_max_abs_qm_minus_q": aggregate["r_changes"],
            "filter_max_abs_qm_minus_q": aggregate["filter_changes"],
        },
    )
    audit.require(summary == result["summary"], "saved endpoint/terminal exact reconstruction")
    return {
        "status": "COLD_INDEPENDENT_AUDIT_PASS",
        "terminal": summary["terminal"],
        "result_sha256": audit.sha(output / "result.json"),
        "manifest_sha256": result["manifest_sha256"],
        "source_audits": training_receipts,
        "evaluation_audits": evaluation_receipts,
        "exact_access_counts": {f"{a}:{b}": v for (a, b), v in counts.items()},
        "elapsed_seconds": time.perf_counter() - start,
        "held60_access": False,
        "raw_input_access": False,
        "limitations": [
            "not an independent replay of raw preprocessing/support Q EEG formulas",
            "not a second full Adam implementation",
            "development evidence, not independent confirmation",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        receipt = run(args.output)
    except Exception as exc:
        failure = args.output / "cold_audit_failure.json"
        with failure.open("x") as stream:
            json.dump(
                {"status": "VALIDITY_FAILURE", "error_type": type(exc).__name__, "error": str(exc)},
                stream,
                indent=2,
            )
        raise
    with (args.output / "cold_audit.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, indent=2), flush=True)
