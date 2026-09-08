"""Fixed artificial learner/CPU-CUDA engineering check; no human-data inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_shape_learning as learning
from cfeg.analysis import task_trca_shape_operator as operator


def fixture():
    """One predeclared seed/task. No seed/accuracy screen or study files."""
    rng = np.random.default_rng(20260908)
    prototype = rng.normal(size=(12, 5, 8, 31))
    support = prototype[None] + rng.normal(size=(3, 12, 5, 8, 31))
    source = prototype + rng.normal(size=prototype.shape)
    packet = rng.uniform(0, 50, (3, 8))
    return support, source, packet


def device_probe(case):
    logits = torch.linspace(-0.2, 0.2, 40, dtype=torch.float64, device=case.s.device)
    logits = logits.reshape(5, 8).clone().requires_grad_(True)
    r = operator.shape_prior(logits)
    w = operator.bounded_filters(case.s, case.c, case.anchors, r[:, None, :])
    scores, _ = operator.score_gram(w.transpose(-1, -2), case.statistics, case.weights)
    loss = learning._ce(scores, case)
    (gradient,) = torch.autograd.grad(loss, logits)
    return {
        "R": r.detach().cpu().numpy(),
        "filter": w.detach().cpu().numpy(),
        "score": scores.detach().cpu().numpy(),
        "gradient": gradient.cpu().numpy(),
    }


def run(*, cuda=False):
    torch.set_num_threads(1)
    support, source, packet = fixture()
    inputs_sha = hashlib.sha256(
        b"".join(v.astype("<f8").tobytes() for v in (support, source, packet))
    ).hexdigest()
    arguments = (10001, 0, 0, support, packet, np.linspace(9, 14.5, 12), source)
    case = learning.make_task_case(*arguments, weights=np.ones(5))
    start = time.perf_counter()
    cpu = device_probe(case)
    cpu_time = time.perf_counter() - start
    start = time.perf_counter()
    fit = learning.fit_pipeline((case,), 0.001)
    fit_time = time.perf_counter() - start
    if not fit.q.final_loss < fit.q.initial_loss:
        raise AssertionError("Fixed artificial Q optimizer did not reduce its objective")
    q = learning.predict(fit, (case,), "Q")[case.key]
    missing = learning.predict(fit, (case,), "MISSING")[case.key]
    np.testing.assert_array_equal(q, missing)
    native_scores, native_corr = operator.native_zero_scores(support, source, np.ones(5))
    original = learning.native.fit_trca(support, np.ones((5, 8)), 0)
    expected, expected_corr = learning.native.score_trca(original, source, np.ones(5))
    np.testing.assert_array_equal(native_scores, expected)
    np.testing.assert_array_equal(native_corr, expected_corr)
    gpu = {"requested": cuda, "executed": False}
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit CUDA engineering check requires available CUDA")
        torch.cuda.reset_peak_memory_stats()
        gpu_case = learning.make_task_case(*arguments, device="cuda", weights=np.ones(5))
        torch.cuda.synchronize()
        start = time.perf_counter()
        actual = device_probe(gpu_case)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        errors = {}
        for name in cpu:
            atol, rtol = (1e-7, 1e-5) if name == "gradient" else (1e-9, 0)
            np.testing.assert_allclose(actual[name], cpu[name], atol=atol, rtol=rtol)
            errors[name] = float(np.max(np.abs(actual[name] - cpu[name])))
        np.testing.assert_array_equal(actual["score"].argmax(-1), cpu["score"].argmax(-1))
        gpu.update(
            executed=True,
            name=torch.cuda.get_device_name(),
            seconds=elapsed,
            max_abs_errors=errors,
            predictions_match=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    return {
        "status": "ENGINEERING_CHECKS_PASSED",
        "scope": "fixed artificial arrays only",
        "seed": 20260908,
        "input_sha256": inputs_sha,
        "human_data_access": False,
        "held60_access": False,
        "metadata_efficacy_established": False,
        "calibration_savings_established": False,
        "native_zero_project_exact": True,
        "missing_exact_q": True,
        "cpu_probe_seconds": cpu_time,
        "cpu_four_head_fit_seconds": fit_time,
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "Q_initial_objective": fit.q.initial_loss,
        "Q_final_objective": fit.q.final_loss,
        "head_steps": {
            "Q": len(fit.q.trace),
            **{k: len(v.trace) for k, v in fit.residuals.items()},
        },
        "cuda": gpu,
        "fit": fit.record(),
        "limitations": [
            "not a human outcome",
            "no actual nested39 run",
            "no human archive reader or executable manifest",
            "no independent persisted-artifact auditor yet",
            "CUDA probe timing includes first-call overhead, not a speed benchmark",
        ],
    }


def run_nested():
    """Execute the full fit graph once on six generated people, one condition.

    Same predeclared seed, no feature/seed/optimizer selection beyond the frozen
    three-lambda inner loop. This is not the 39-person/8-condition study.
    """
    torch.set_num_threads(1)
    rng = np.random.default_rng(20260908)
    cases = []
    digest = hashlib.sha256()
    for pid in range(20001, 20007):
        prototype = rng.normal(size=(12, 5, 8, 17))
        support = prototype[None] + rng.normal(size=(3, 12, 5, 8, 17))
        source = prototype + rng.normal(size=prototype.shape)
        packet = rng.uniform(0, 50, (3, 8))
        for array in (support, source, packet):
            digest.update(array.astype("<f8").tobytes())
        cases.append(
            learning.make_task_case(
                pid,
                0,
                0,
                support,
                packet,
                np.linspace(9, 14.5, 12),
                source,
                weights=np.ones(5),
            )
        )
    start = time.perf_counter()
    model, selection = learning.nested_fit(tuple(cases), outer_evaluation_ids=(30001, 30002))
    seconds = time.perf_counter() - start
    for row in selection["inner"]:
        assert not set(row["fit_ids"]) & set(row["validation_ids"])
        assert len(row["fit_ids"]) == 4 and len(row["validation_ids"]) == 2
        assert set(row["pipeline"]["q_scaler"]["fit_ids"]) == set(row["fit_ids"])
        assert all(d["donor_id"] in row["fit_ids"] for d in row["pipeline"]["donors"])
        assert row["pipeline"]["Q"]["steps"] == 200
        assert all(v["steps"] == 200 for v in row["pipeline"]["residuals"].values())
    assert learning.choose_lambda(selection["inner"]) == selection["selected_lambda"]
    return {
        "status": "ARTIFICIAL_NESTED_GRAPH_PASS",
        "seed": 20260908,
        "input_sha256": digest.hexdigest(),
        "artificial_participants": 6,
        "conditions_per_participant": 1,
        "samples": 17,
        "k": 3,
        "actual_pipeline_fits": 10,
        "actual_head_fits": 40,
        "actual_optimizer_steps": 8000,
        "cpu_seconds": seconds,
        "selected_lambda": selection["selected_lambda"],
        "human_data_access": False,
        "held60_access": False,
        "efficacy_established": False,
        "selection": selection,
        "fit": model.record(),
        "limitations": [
            "artificial one-condition six-person engineering graph only",
            "not source39 or 8-condition study",
            "no outer-evaluation query values",
            "not an independent artifact auditor or efficacy experiment",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--nested", action="store_true", help="Full small artificial nested graph")
    parser.add_argument("--output", type=Path, help="Exclusive new engineering receipt file")
    args = parser.parse_args()
    if args.cuda and args.nested:
        parser.error("Nested graph is CPU-only; run CUDA parity separately")
    if args.output is not None and args.output.exists():
        raise FileExistsError("Engineering receipts cannot be overwritten")
    try:
        result = run_nested() if args.nested else run(cuda=args.cuda)
    except Exception as error:
        if args.output is not None:
            with args.output.open("x") as stream:
                json.dump(
                    {
                        "status": "ENGINEERING_FAILURE",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "seed": 20260908,
                        "human_data_access": False,
                    },
                    stream,
                    indent=2,
                    allow_nan=False,
                )
        raise
    if args.output is not None:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        summary = {key: value for key, value in result.items() if key not in ("fit", "selection")}
        summary["full_receipt"] = str(args.output.resolve())
        summary["full_receipt_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        print(json.dumps(summary, indent=2, allow_nan=False))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
