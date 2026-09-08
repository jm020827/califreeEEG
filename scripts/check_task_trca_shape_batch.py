"""Artificial batch/scalar/CUDA parity and workload sizing before human access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_shape_learning as learning
from cfeg.analysis.task_trca_shape_batch import TaskBatch


def move(case, device):
    values = {
        name: getattr(case, name).to(device) for name in ("s", "c", "anchors", "weights", "labels")
    }
    stats = case.statistics
    values["statistics"] = replace(
        stats,
        **{f.name: getattr(stats, f.name).to(device) for f in fields(stats) if f.name != "samples"},
    )
    return replace(case, **values)


def cases():
    rng = np.random.default_rng(20260909)
    result = []
    for pid in (11001, 11002):
        packet5 = rng.uniform(0, 30, (5, 8))
        packet5[:, 2] = np.nan
        for n in (17, 23):
            proto = rng.normal(size=(12, 5, 8, n)) + 0.7
            x = proto[None] + rng.normal(size=(5, 12, 5, 8, n))
            y = proto + rng.normal(size=proto.shape)
            for k in (3, 5):
                result.append(
                    learning.make_task_case(
                        pid,
                        0,
                        0,
                        x[:k],
                        packet5[:k],
                        np.linspace(9, 14.5, 12),
                        y,
                        weights=np.array([1.25, 0.67, 0.50, 0.43, 0.38]),
                    )
                )
    return tuple(result)


def probe(rows):
    batch = TaskBatch(rows)
    logits = (
        torch.linspace(-0.2, 0.2, len(rows) * 40, dtype=torch.float64, device=rows[0].s.device)
        .reshape(-1, 5, 8)
        .requires_grad_()
    )
    # One warmup then three complete forward/backward calls, no scientific fit.
    values = []
    for iteration in range(4):
        if rows[0].s.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        scores = batch.scores(logits)
        scaled = scores / batch.weights.sum(-1)[:, None, None] / 0.1
        loss = torch.nn.functional.cross_entropy(scaled.flatten(0, 1), batch.labels.flatten())
        (grad,) = torch.autograd.grad(loss, logits)
        if rows[0].s.is_cuda:
            torch.cuda.synchronize()
        if iteration:
            values.append(time.perf_counter() - start)
    return scores.detach().cpu().numpy(), grad.cpu().numpy(), values


def run():
    torch.set_num_threads(1)
    rows = cases()
    receipt = {"seed": 20260909, "human_data_access": False, "held60_access": False, "fits": {}}
    fits = {}
    for name, device, backend in (
        ("cpu_scalar", "cpu", "scalar"),
        ("cpu_batch", "cpu", "batch"),
        ("cuda_batch", "cuda", "batch"),
    ):
        selected = tuple(move(c, device) for c in rows)
        start = time.perf_counter()
        fits[name] = learning.fit_pipeline(selected, 0.001, backend=backend)
        receipt["fits"][name] = {
            "seconds": time.perf_counter() - start,
            "record": fits[name].record(),
        }
        print(json.dumps({"event": name, "seconds": receipt["fits"][name]["seconds"]}), flush=True)
    reference = fits["cpu_scalar"]
    errors = {}
    for name in ("cpu_batch", "cuda_batch"):
        errors[name] = {}
        for head in ("Q", *learning.RESIDUAL_ARMS):
            a = reference.q if head == "Q" else reference.residuals[head]
            b = fits[name].q if head == "Q" else fits[name].residuals[head]
            np.testing.assert_allclose(a.coefficients, b.coefficients, atol=1e-7, rtol=1e-5)
            np.testing.assert_allclose(a.final_loss, b.final_loss, atol=1e-9, rtol=0)
            errors[name][head] = float(np.max(np.abs(a.coefficients - b.coefficients)))
        for arm in ("Q", *learning.RESIDUAL_ARMS):
            a, b = learning.predict(reference, rows, arm), learning.predict(fits[name], rows, arm)
            for key in a:
                np.testing.assert_allclose(a[key], b[key], atol=1e-8, rtol=0)
                np.testing.assert_array_equal(a[key].argmax(-1), b[key].argmax(-1))
    receipt["coefficient_errors"] = errors
    receipt["prediction_parity"] = True
    # 416 cases equals max outer26*16 computational batch shape. Artificial IDs
    # and only four conditions are used here; this is not a scientific grid.
    big = tuple(replace(c, participant_id=pid) for pid in range(12001, 12105) for c in rows[:4])
    outputs = {}
    receipt["workload_probe"] = {}
    for device in ("cpu", "cuda"):
        selected = tuple(move(c, device) for c in big)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        scores, grad, times = probe(selected)
        outputs[device] = (scores, grad)
        receipt["workload_probe"][device] = {"cases": 416, "seconds": times}
        if device == "cuda":
            receipt["workload_probe"][device].update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                name=torch.cuda.get_device_name(),
            )
        print(
            json.dumps({"event": "workload_probe", "device": device, "seconds": times}), flush=True
        )
    for a, b in zip(outputs["cpu"], outputs["cuda"]):
        np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-5)
    np.testing.assert_array_equal(outputs["cpu"][0].argmax(-1), outputs["cuda"][0].argmax(-1))
    receipt["status"] = "ARTIFICIAL_BATCH_PARITY_PASS"
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        receipt = run()
    except Exception as exc:
        receipt = {"status": "ENGINEERING_FAILURE", "error": str(exc), "human_data_access": False}
        with args.output.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
        raise
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        ),
        flush=True,
    )
