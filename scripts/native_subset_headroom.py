"""One frozen saved-prediction ceiling diagnostic; no learning or metadata I/O."""

from __future__ import annotations

import codecs
import hashlib
import io
import json
import os
import signal
import stat
import subprocess
import sys
import time
import zipfile  # noqa: F401 -- eager NPZ dependency before the exact-path runtime guard
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import scipy
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "configs/analysis/native_subset_headroom_source39_cold_r1.json"
PLAN_SHA = "e7affb66540ed68703ac8aaea53c24841f32df944feaa4e698f073863b4e69a0"
CELL = ("participant", "interface", "n_samples", "k")
ROW = (*CELL, "method")
METRICS = (
    "full_ba",
    "q_ba",
    "qm_ba",
    "oracle_ba",
    "potential_over_q",
    "potential_over_qm",
    "disagreement_fraction",
    "unrecoverable_fraction",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def integer_array(value, shape, upper):
    x = np.asarray(value)
    require(
        not any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat),
        "Boolean is not an integer index",
    )
    require(x.shape == shape and x.dtype.kind in "iu", "Integer shape/type")
    require(np.all((x >= 0) & (x < upper)), "Integer range")
    return x


def cell_metrics(predictions, truth, q_actions, qm_actions):
    pred = np.asarray(predictions)
    require(pred.ndim == 2 and min(pred.shape) > 0, "Candidate/query dimensions")
    c, n = pred.shape
    pred = integer_array(pred, (c, n), 12)
    truth = integer_array(truth, (n,), 12)
    q = integer_array(q_actions, (n,), c)
    qm = integer_array(qm_actions, (n,), c)
    columns = np.arange(n)
    qc, qmc = pred[q, columns] == truth, pred[qm, columns] == truth
    correct = pred == truth
    oracle = correct.any(axis=0)
    disagreement = (pred != pred[0]).any(axis=0)
    counts = {
        "full_correct": int(correct[0].sum()),
        "q_correct": int(qc.sum()),
        "qm_correct": int(qmc.sum()),
        "oracle_correct": int(oracle.sum()),
        "all_wrong": int((~oracle).sum()),
        "disagreement": int(disagreement.sum()),
        "disagreement_any_correct": int((disagreement & oracle).sum()),
        "disagreement_all_wrong": int((disagreement & ~oracle).sum()),
        "recoverable_q": int((oracle & ~qc).sum()),
        "recoverable_qm": int((oracle & ~qmc).sum()),
        "recoverable_full": int((oracle & ~correct[0]).sum()),
        "qm_q_selector_changes": int((q != qm).sum()),
        "qm_q_prediction_changes": int((pred[q, columns] != pred[qm, columns]).sum()),
        "qm_repairs_q": int((qmc & ~qc).sum()),
        "qm_damages_q": int((qc & ~qmc).sum()),
    }
    require(counts["oracle_correct"] + counts["all_wrong"] == n, "Oracle partition")
    require(counts["recoverable_full"] <= counts["disagreement"], "Disagreement bound")
    require(
        counts["qm_correct"] - counts["q_correct"]
        == counts["qm_repairs_q"] - counts["qm_damages_q"],
        "Repair identity",
    )
    return {
        "query_count": n,
        "candidate_count": c,
        **counts,
        **{m + "_ba": counts[m + "_correct"] / n for m in ("full", "q", "qm", "oracle")},
        "potential_over_q": counts["recoverable_q"] / n,
        "potential_over_qm": counts["recoverable_qm"] / n,
        "disagreement_fraction": counts["disagreement"] / n,
        "unrecoverable_fraction": counts["all_wrong"] / n,
    }


def indexed(rows, fields):
    result = {tuple(row[f] for f in fields): row for row in rows}
    require(len(result) == len(rows), "Duplicated saved keys")
    return result


def interval(values):
    x = np.asarray(values, dtype=float)
    require(x.shape == (39,) and np.isfinite(x).all(), "39 participant means required")
    mean = float(x.mean())
    half = float(t.ppf(0.975, 38) * x.std(ddof=1) / np.sqrt(39))
    return {"mean": mean, "ci95": [mean - half, mean + half], "n_participants": 39}


def first80(counts):
    return next((k for k in (0, 3, 5) if counts[k] >= 48), None)


def evaluate(cache, saved, science, core, auditor):
    """Replay ALL fixed comparators before computing any new oracle aggregate."""
    core.validate_cache(cache, science)
    scores = core.native_scores(cache, science)
    independent = auditor.cache_scores(cache, science)
    maximum = 0.0
    for name, values in scores.items():
        delta = float(np.max(np.abs(values - independent[name])))
        require(delta <= 1e-12, "Independent native scores differ")
        require(np.array_equal(values.argmax(-1), independent[name].argmax(-1)), "Argmax drift")
        maximum = max(maximum, delta)
    del independent
    rows, diags = indexed(saved["rows"], ROW), indexed(saved["diagnostics"], CELL)
    require(len(rows) == 4680 and len(diags) == 624, "Saved grid size")
    ids, interfaces, windows = (
        science["source_subject_ids"],
        science["interfaces"],
        science["sample_counts"],
    )
    expected_cells = {
        (p, i, n, k) for p in ids for i in interfaces for n in windows for k in (3, 5)
    }
    require(set(diags) == expected_cells, "Saved cell inventory")
    expected_rows = {
        (p, i, n, k, m)
        for p in ids
        for i in interfaces
        for n in windows
        for k, methods in (
            (0, ("A0_author",)),
            (3, science["controls"]["methods"][1:]),
            (5, science["controls"]["methods"][1:]),
        )
        for m in methods
    }
    require(set(rows) == expected_rows, "Saved row inventory")
    truth, pending, a0_counts = np.tile(np.arange(12), 5), [], {}

    def reconcile(key, count):
        row = rows[key]
        value = row["ba"]
        require(type(value) in (int, float) and np.isfinite(value), "Saved BA type")
        require(abs(value - count / 60) <= 1e-12, "Saved BA replay mismatch")
        require(row["query_count"] == 60 and row["label_count"] == 12 * key[3], "Saved cost")

    for p, subject in enumerate(ids):
        for i, interface in enumerate(interfaces):
            for w, n in enumerate(windows):
                a0 = scores["a0_r"][p, i, w].reshape(60, 12).argmax(-1)
                count = int((a0 == truth).sum())
                a0_counts[(subject, interface, n)] = count
                reconcile((subject, interface, n, 0, "A0_author"), count)
                for bi, k in enumerate((3, 5)):
                    key = (subject, interface, n, k)
                    pred = (
                        scores["expert_r"][p, i, w, bi, : k + 1].reshape(k + 1, 60, 12).argmax(-1)
                    )
                    q, qm = [
                        integer_array(diags[key]["actions"][m], (60,), k + 1) for m in ("Q", "QM")
                    ]
                    for method, selected in (
                        ("FULL", pred[0]),
                        ("Q", pred[q, np.arange(60)]),
                        ("QM", pred[qm, np.arange(60)]),
                    ):
                        reconcile((*key, method), int((selected == truth).sum()))
                    pending.append((key, pred, q, qm))

    # No oracle calls above this boundary; an invalid late cell cannot leak an early ceiling.
    cells = [
        dict(zip(CELL, key), **cell_metrics(pred, truth, q, qm)) for key, pred, q, qm in pending
    ]
    by_cell = indexed(cells, CELL)
    groups = [("all8", None, None)] + [(f"{i}_{n}", i, n) for i in interfaces for n in windows]
    summary, costs = [], []
    for group, interface, n in groups:
        subset = [
            r
            for r in cells
            if interface is None or (r["interface"], r["n_samples"]) == (interface, n)
        ]
        for k in (3, 5):
            selected = [r for r in subset if r["k"] == k]
            summary.append(
                {
                    "group": group,
                    "k": k,
                    "cell_count": len(selected),
                    "metrics": {
                        m: interval(
                            [
                                np.mean([r[m] for r in selected if r["participant"] == p])
                                for p in ids
                            ]
                        )
                        for m in METRICS
                    },
                }
            )
        differences = [
            np.mean(
                [
                    r["oracle_ba"] - by_cell[(p, r["interface"], r["n_samples"], 5)]["q_ba"]
                    for r in subset
                    if r["participant"] == p and r["k"] == 3
                ]
            )
            for p in ids
        ]
        costs.append(
            {
                "group": group,
                "oracle3_minus_fixed_q5": interval(differences),
                "labels_k3": 36,
                "labels_k5": 60,
                "descriptive_ni_reference": -1 / 60,
            }
        )
    attainment = []
    for key, a0_count in a0_counts.items():
        for method, field in (
            ("FULL", "full_correct"),
            ("Q", "q_correct"),
            ("QM", "qm_correct"),
            ("ORACLE_HINDSIGHT", "oracle_correct"),
        ):
            counts = {0: a0_count, **{k: by_cell[(*key, k)][field] for k in (3, 5)}}
            first = first80(counts)
            attainment.append(
                {
                    **dict(zip(CELL[:3], key)),
                    "method": method,
                    "first_observed_k": first,
                    "unreached_tested_grid": first is None,
                    "label_count": None if first is None else 12 * first,
                }
            )
    return {
        "comparator_replay_verified": True,
        "score_max_abs_difference": maximum,
        "replayed_ba_rows": 2184,
        "cells": cells,
        "summary": summary,
        "cost": costs,
        "attainment": attainment,
    }


def artifact_bytes(path, limit):
    require(path.absolute() == path.resolve(), "Canonical artifact path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o400
            and info.st_nlink == 1
            and info.st_size <= limit,
            "Artifact type/mode/size",
        )
        raw = stream.read(limit + 1)
    require(len(raw) == info.st_size and len(raw) <= limit, "Artifact size drift")
    return raw


def publish(path, payload, budget):
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    require(
        sum(p.stat().st_size for p in path.parent.iterdir()) + len(raw) <= budget, "Output budget"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, "wb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o400
            and info.st_nlink == 1,
            "Published artifact mode/type/link count",
        )
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return digest(raw)


def guard_for(plan):
    inputs = {Path(v["path"]) for v in plan["inputs"].values()}
    output = Path(plan["output_root"])
    outputs = {output / name for name in plan["artifacts"]}

    def guard(event, args):
        if event == "open" and not isinstance(args[0], int):
            path = Path(os.fsdecode(args[0]))
            require(path.is_absolute() and path == path.resolve(), "Noncanonical runtime open")
            flags = args[2]
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            if writing:
                require(
                    path in outputs and flags & os.O_EXCL and flags & os.O_NOFOLLOW,
                    "Undeclared output or nonexclusive write",
                )
            else:
                require(path in inputs or path == output, "Undeclared input: " + str(path))
        elif event in ("os.listdir", "os.scandir", "os.mkdir"):
            require(Path(args[0]) == output, "Undeclared directory access")
        elif event.startswith(("subprocess.", "socket.")) or event in (
            "os.system",
            "os.remove",
            "os.rename",
            "os.rmdir",
            "os.link",
            "os.symlink",
            "os.chmod",
            "os.truncate",
            "os.utime",
            "os.chown",
            "os.setxattr",
            "os.removexattr",
        ):
            raise ValueError("Forbidden runtime operation: " + event)

    return guard


def validate_recovery(plan, previous):
    require(
        set(plan) == set(previous) | {"attempt_id", "prior_plan", "prior_attempt"},
        "Recovery fields",
    )
    for key in previous:
        if key not in ("authority", "output_root"):
            require(plan[key] == previous[key], "Recovery changed scientific contract: " + key)
    require(set(plan["authority"]) == set(previous["authority"]), "Recovery authority fields")
    for key in previous["authority"]:
        if key != "basis":
            require(
                plan["authority"][key] == previous["authority"][key], "Recovery authority drift"
            )
    require(plan["output_root"] != previous["output_root"], "Failed attempt must remain untouched")
    require(
        plan["attempt_id"]
        == Path(plan["output_root"]).name.replace("headroom-", "native-subset-headroom-", 1),
        "Explicit new attempt identity",
    )


def preflight():
    require(len(sys.argv) == 1, "No CLI overrides")
    raw = (ROOT / PLAN_PATH).read_bytes()
    require(digest(raw) == PLAN_SHA, "Frozen plan hash")
    plan = json.loads(raw)
    require(Path(sys.executable).absolute() == ROOT / ".venv/bin/python", "Existing main venv")
    for actual, key in (
        (sys.version.split()[0], "python"),
        (np.__version__, "numpy"),
        (scipy.__version__, "scipy"),
    ):
        require(actual == plan["runtime"][key], "Runtime version: " + key)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        require(os.environ.get(key) == "1", "Single-thread environment")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args])

    require(git("branch", "--show-current").strip() == b"main", "Main branch only")
    require(not git("status", "--porcelain").strip(), "Clean source required")
    commit, tree = [git("rev-parse", spec).decode().strip() for spec in ("HEAD", "HEAD^{tree}")]
    sources = {
        PLAN_PATH: PLAN_SHA,
        plan["prior_plan"]["path"]: plan["prior_plan"]["sha256"],
        plan["science"]["path"]: plan["science"]["sha256"],
        **plan["helpers"],
        "scripts/native_subset_headroom.py": None,
    }
    modules, loaded, hashes = {}, {}, {}
    for relative, pin in sources.items():
        content = (ROOT / relative).read_bytes()
        require(content == git("show", commit + ":" + relative), "Uncommitted source bytes")
        hashes[relative] = digest(content)
        require(pin is None or hashes[relative] == pin, "Pinned source drift")
        loaded[relative] = content
    validate_recovery(plan, json.loads(loaded[plan["prior_plan"]["path"]]))
    for relative in plan["helpers"]:
        module = ModuleType(Path(relative).stem)
        module.__file__ = str(ROOT / relative)
        # Execute only exact SHA-pinned, committed local helper definitions.
        exec(compile(loaded[relative], module.__file__, "exec"), module.__dict__)  # noqa: S102
        modules[module.__name__] = module
    t.ppf(0.975, 38)  # Resolve lazy numerical imports before the exact-path runtime guard.
    output = Path(plan["output_root"])
    require(output == output.resolve() and not output.exists(), "Fresh canonical output only")
    return plan, json.loads(loaded[plan["science"]["path"]]), modules, commit, tree, hashes


def execute():
    began = time.monotonic()
    plan, science, modules, commit, tree, hashes = preflight()
    codecs.lookup("cp437")  # ZIP member-name decoding, no human data required.
    common = {
        "study_id": plan["study_id"],
        "attempt_id": plan["attempt_id"],
        "prior_attempt": plan["prior_attempt"],
        "source_commit": commit,
        "source_tree": tree,
        "execution_plan_sha256": PLAN_SHA,
        "source_hashes": hashes,
        "scientific_plan_sha256": plan["science"]["sha256"],
        "inputs": plan["inputs"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "authority": plan["authority"],
        "interpretation": plan["interpretation"],
        "runtime": plan["runtime"],
    }

    def timeout(*_):
        raise TimeoutError("Diagnostic runtime ceiling; no retry")

    signal.signal(signal.SIGALRM, timeout)
    remaining = plan["runtime"]["ceiling_seconds"] - (time.monotonic() - began)
    require(remaining > 0, "Runtime exhausted before start")
    signal.setitimer(signal.ITIMER_REAL, remaining)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    # Avoid exception rendering opening undeclared source files under the guard.
    sys.excepthook = lambda kind, value, tb: print(
        kind.__name__ + ": " + str(value), file=sys.stderr
    )
    sys.dont_write_bytecode = True
    sys.addaudithook(guard_for(plan))
    output = Path(plan["output_root"])
    output.mkdir(mode=0o700)
    budget = plan["runtime"]["output_budget_bytes"]
    start_sha = publish(
        output / "start.json", {"schema": "cfeg.headroom.start.v1", **common}, budget
    )
    print("Start published before two pinned input reads", flush=True)
    raw_inputs = {}
    for name, spec in plan["inputs"].items():
        raw = artifact_bytes(Path(spec["path"]), plan["runtime"]["input_budget_bytes"])
        require(digest(raw) == spec["sha256"], "Input hash: " + name)
        raw_inputs[name] = raw
    require(
        sum(map(len, raw_inputs.values())) <= plan["runtime"]["input_budget_bytes"], "Input budget"
    )
    saved = json.loads(raw_inputs["result"])
    for key in ("study_id", "source_commit"):
        require(saved[key] == plan["saved_result"][key], "Saved result provenance: " + key)
    require(saved["scientific_plan_sha256"] == plan["science"]["sha256"], "Saved science")
    require(saved["status"] == "DEVELOPMENT_ASSESSMENT_COMPLETE", "Completed parent required")
    with np.load(io.BytesIO(raw_inputs["cache"]), allow_pickle=False) as archive:
        require(len(archive.files) == 3 and set(archive.files) == set(science["cache"]), "NPZ keys")
        cache = {name: archive[name] for name in archive.files}
    del raw_inputs
    result = evaluate(
        cache,
        saved,
        science,
        modules["native_subset_m_core"],
        modules["audit_native_subset_m_source"],
    )
    for key, spec in (
        ("cells", "cell_rows"),
        ("summary", "group_summaries"),
        ("cost", "cost_rows"),
        ("attainment", "attainment_rows"),
    ):
        require(len(result[key]) == plan["reporting"][spec], "Complete reporting: " + key)
    for spec in plan["inputs"].values():
        require(
            digest(artifact_bytes(Path(spec["path"]), plan["runtime"]["input_budget_bytes"]))
            == spec["sha256"],
            "Input changed during diagnostic",
        )
    elapsed = time.monotonic() - began
    require(elapsed <= plan["runtime"]["ceiling_seconds"], "Elapsed ceiling")
    result.update(
        {
            **common,
            "schema": "cfeg.headroom.result.v1",
            "status": "DIAGNOSTIC_COMPLETE",
            "start_sha256": start_sha,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
        }
    )
    result_sha = publish(output / "result.json", result, budget)
    signal.alarm(0)
    print(
        json.dumps(
            {
                "status": result["status"],
                "sha256": result_sha,
                "cells": len(result["cells"]),
                "elapsed_seconds": elapsed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    execute()
