"""Fixed generated-array check; optional one PINNED, already exposed failure point.

No arbitrary input path, human archive reader, training replay, or query adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy import linalg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis.metadata_trca_prior import trca_matrices
from cfeg.analysis.task_trca_shape_operator import bounded_filters, shape_prior
from cfeg.analysis.task_trca_shape_signfree import (
    bounded_projectors,
    score_temporal_gram,
    temporal_statistics,
)

POINT = Path("/home/whwovy/task-trca-shape-anchor-failure-diagnostic-v1.json")
POINT_SHA = "86eda34fc11f9e20a3874ef0e952a2aedc8b60e7f908bdd290b84ccf84b1561b"
SEED = 20260908
D = torch.float64


def scipy_projectors(s, c, r):
    """Independent generalized eigh; no producer whitening or normalization helper."""
    r = np.broadcast_to(r, s.shape[:-1])
    result = []
    vectors = []
    for ss, cc, rr in zip(s.reshape(-1, 8, 8), c.reshape(-1, 8, 8), r.reshape(-1, 8)):
        tau = 0.1 * linalg.eigvalsh(cc)[0] / (16 / 9)
        _, vv = linalg.eigh(ss, cc + tau * np.diag(rr))
        v = vv[:, -1]
        v = v / np.sqrt(v @ cc @ v)
        vectors.append(v)
        result.append(np.outer(v, v))
    return np.array(result).reshape(s.shape), np.array(vectors).reshape(s.shape[:-1])


def numpy_scores(vectors, templates, query, weights):
    """Independent literal projection/centering/flattening, no Gram helper."""
    ux = np.einsum("bki,nbit->nbkt", vectors, query)
    ut = np.einsum("bki,cbit->cbkt", vectors, templates)
    ux = (ux - ux.mean(-1, keepdims=True)).reshape(query.shape[0], len(weights), -1)
    ut = (ut - ut.mean(-1, keepdims=True)).reshape(templates.shape[0], len(weights), -1)
    ux /= np.linalg.norm(ux, axis=-1, keepdims=True)
    ut /= np.linalg.norm(ut, axis=-1, keepdims=True)
    corr = np.einsum("nbt,cbt->nbc", ux, ut)
    return np.einsum("nbc,b->nc", corr, weights), corr


def fixture():
    rng = np.random.default_rng(SEED)
    prototype = rng.normal(size=(3, 2, 8, 31))
    support = prototype[None] + rng.normal(size=(3, 3, 2, 8, 31))
    query = prototype + rng.normal(size=prototype.shape)
    support += rng.normal(size=(3, 3, 2, 8, 1)) * 2
    query += rng.normal(size=(3, 2, 8, 1)) * 2
    s, c = np.empty((2, 3, 8, 8)), np.empty((2, 3, 8, 8))
    for b in range(2):
        for cls in range(3):
            s[b, cls], c[b, cls] = trca_matrices(support[:, cls, b])
    features = rng.normal(size=(2, 8, 2))
    return s, c, support.mean(0), query, np.array([1.4, 0.7]), features


def fit_generated(arrays, device):
    s, c, t, x, weights, features = [torch.tensor(v, dtype=D, device=device) for v in arrays]
    stats = temporal_statistics(t, x)
    labels = torch.arange(3, device=device)
    q = torch.zeros((2, 8), dtype=D, device=device, requires_grad=True)
    a = np.log(2) / 2

    def predict(z):
        return score_temporal_gram(
            bounded_projectors(s, c, shape_prior(z)[:, None, :]), stats, weights
        )[0]

    def optimize(coef, logits):
        optimizer = torch.optim.Adam([coef], lr=0.01)
        trace = []
        for _ in range(200):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(predict(logits()) / 0.1, labels)
            loss = loss + 0.001 * coef.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite artificial loss")
            trace.append(float(loss.detach().cpu()))
            loss.backward()
            if coef.grad is None or not torch.isfinite(coef.grad).all():
                raise ValueError("nonfinite or absent artificial gradient")
            optimizer.step()
        final = torch.nn.functional.cross_entropy(predict(logits()) / 0.1, labels)
        final = final + 0.001 * coef.square().mean()
        if not float(final.detach()) < trace[0]:
            raise AssertionError("fixed generated optimizer did not reduce its own objective")
        return {
            "steps": 200,
            "initial": trace[0],
            "final": float(final.detach()),
            "trace": trace,
            "coef": coef.detach().cpu().tolist(),
        }

    initial = predict(torch.zeros_like(q)).detach()
    qfit = optimize(q, lambda: 0.8 * a * q.tanh())
    frozen_q = (0.8 * a * q.tanh()).detach().clone()
    residual = torch.zeros(3, dtype=D, device=device, requires_grad=True)

    def combined():
        value = residual[0] + torch.einsum("bif,f->bi", features, residual[1:])
        return frozen_q + 0.2 * a * value.tanh()

    rfit = optimize(residual, combined)
    z = combined().detach().clone().requires_grad_()
    r = shape_prior(z)
    f = bounded_projectors(s, c, r[:, None, :])
    scores, corr = score_temporal_gram(f, stats, weights)
    ce = torch.nn.functional.cross_entropy(scores / 0.1, labels)
    gradient = torch.autograd.grad(ce, z)[0]

    def numpy(v):
        return v.detach().cpu().numpy()

    ref_f, ref_v = scipy_projectors(arrays[0], arrays[1], numpy(r)[:, None, :])
    ref_score, ref_corr = numpy_scores(ref_v, arrays[2], arrays[3], arrays[4])
    errors = {}
    for name, value, ref in (
        ("projector", f, ref_f),
        ("score", scores, ref_score),
        ("correlation", corr, ref_corr),
    ):
        np.testing.assert_allclose(numpy(value), ref, atol=1e-10, rtol=0)
        errors[name] = float(np.max(np.abs(numpy(value) - ref)))
    return {
        "BASE_LOGITS": qfit,
        "RANDOM_RESIDUAL": rfit,
        "projectors": numpy(f).tolist(),
        "scores": numpy(scores).tolist(),
        "gradient": numpy(gradient).tolist(),
        "independent_scipy_literal_max_abs_error": errors,
        "score_change_from_initial": float((scores.detach() - initial).abs().max().cpu()),
        "argmax": numpy(scores.argmax(-1)).tolist(),
    }


def check_point(*, cuda):
    # One open, hash-before-decode; bounded bytes and no caller-supplied file path.
    descriptor = os.open(POINT, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        identity = os.fstat(stream.fileno())
        if not stat.S_ISREG(identity.st_mode) or identity.st_size > 1_048_576:
            raise ValueError("saved failure point must be a bounded regular file")
        raw = stream.read(1_048_577)
    if len(raw) > 1_048_576 or hashlib.sha256(raw).hexdigest() != POINT_SHA:
        raise ValueError("saved failure point does not match the pinned artifact")
    point = json.loads(raw)["point"]
    s, c, r, anchor = [
        torch.tensor(point[name], dtype=D) for name in ("S", "C", "r", "native_c_normalized_anchor")
    ]
    try:
        bounded_filters(s, c, anchor, r)
    except ValueError as exc:
        if "anchor" not in str(exc):
            raise
        old_error = str(exc)
    else:
        raise AssertionError("old pinned anchor failure unexpectedly disappeared")
    f = bounded_projectors(s, c, r)
    expected, _ = scipy_projectors(s.numpy(), c.numpy(), r.numpy())
    np.testing.assert_allclose(f.numpy(), expected, atol=1e-10, rtol=0)
    log_r = r.log()
    z = (log_r - (log_r.max() + log_r.min()) / 2).requires_grad_()
    torch.testing.assert_close(shape_prior(z), r, atol=2e-15, rtol=0)
    weight = torch.arange(64, dtype=D).reshape(8, 8) / 64
    fp = bounded_projectors(s, c, shape_prior(z))
    gradient = torch.autograd.grad((fp * weight).sum(), z)[0]
    if not torch.autograd.gradcheck(
        lambda zz: bounded_projectors(s, c, shape_prior(zz)),
        (z,),
        atol=1e-5,
        rtol=1e-3,
    ):
        raise AssertionError("saved-point first-order gradcheck failed")
    result = {
        "scope": "KNOWN_EXPOSED_FAILURE_POINT_NOT_INDEPENDENT_SYNTHETIC_DATA",
        "input_path": str(POINT),
        "input_sha256": POINT_SHA,
        "case_key": point["case_key"],
        "band_zero_based": point["band_zero_based"],
        "class_zero_based": point["class_zero_based"],
        "old_error": old_error,
        "old_cpu_cosine": point["cpu_abs_c_cosine"],
        "new_c_trace": float((f * c).sum()),
        "new_scipy_max_abs_error": float(np.max(np.abs(f.numpy() - expected))),
        "new_projector": f.tolist(),
        "new_r_logit_gradient": gradient.tolist(),
        "first_order_gradcheck": True,
        "cuda": {"executed": False},
        "training_updates": 0,
        "query_score_computed": False,
    }
    if cuda:
        zg = z.detach().cuda().requires_grad_()
        fg = bounded_projectors(s.cuda(), c.cuda(), shape_prior(zg))
        gg = torch.autograd.grad((fg * weight.cuda()).sum(), zg)[0]
        np.testing.assert_allclose(fg.detach().cpu(), fp.detach(), atol=1e-10, rtol=0)
        np.testing.assert_allclose(gg.cpu(), gradient, atol=1e-7, rtol=1e-5)
        result["cuda"] = {
            "executed": True,
            "max_projector_error": float((fg.detach().cpu() - fp.detach()).abs().max()),
            "max_gradient_error": float((gg.cpu() - gradient).abs().max()),
        }
    return result


def run(*, cuda=False, saved_failure_point=False):
    torch.set_num_threads(1)
    arrays = fixture()
    start = time.perf_counter()
    cpu = fit_generated(arrays, "cpu")
    cpu_seconds = time.perf_counter() - start
    gpu = {"executed": False}
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("explicit CUDA probe requires CUDA")
        torch.cuda.set_per_process_memory_fraction(0.04)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_fit = fit_generated(arrays, "cuda")
        torch.cuda.synchronize()
        errors = {}
        for name in ("projectors", "scores", "gradient"):
            a, b = np.array(cpu[name]), np.array(gpu_fit[name])
            np.testing.assert_allclose(a, b, atol=1e-7 if name == "gradient" else 1e-9, rtol=0)
            errors[name] = float(np.max(np.abs(a - b)))
        for head in ("BASE_LOGITS", "RANDOM_RESIDUAL"):
            a, b = np.array(cpu[head]["coef"]), np.array(gpu_fit[head]["coef"])
            np.testing.assert_allclose(a, b, atol=1e-7, rtol=0)
            errors[head + "_coef"] = float(np.max(np.abs(a - b)))
        if cpu["argmax"] != gpu_fit["argmax"]:
            raise AssertionError("generated CPU/CUDA argmax differs")
        gpu = {
            "executed": True,
            "device": torch.cuda.get_device_name(),
            "seconds": time.perf_counter() - start,
            "max_abs_errors": errors,
            "argmax_equal": True,
            "fit": gpu_fit,
        }
    point = check_point(cuda=cuda) if saved_failure_point else {"executed": False}
    if cuda:
        gpu["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        gpu["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if gpu["peak_allocated_bytes"] > 1_073_741_824:
            raise RuntimeError("fixed engineering GPU allocation budget exceeded")
    paths = [
        "src/cfeg/analysis/task_trca_shape_operator.py",
        "src/cfeg/analysis/task_trca_shape_signfree.py",
        "src/cfeg/analysis/metadata_trca_prior.py",
        "scripts/check_task_trca_shape_signfree.py",
    ]
    return {
        "status": "SIGNFREE_ENGINEERING_PASS",
        "seed": SEED,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "code_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "generated_inputs_sha256": hashlib.sha256(
            b"".join(a.tobytes() for a in arrays)
        ).hexdigest(),
        "generated_shape": {
            "support_blocks": 3,
            "classes": 3,
            "bands": 2,
            "channels": 8,
            "samples": 31,
            "source_trials": 3,
        },
        "cpu_seconds": cpu_seconds,
        "cpu": cpu,
        "cuda": gpu,
        "saved_failure_point": point,
        "human_training_updates": 0,
        "new_human_archive_access": False,
        "held60_access": False,
        "query_outcomes_access": False,
        "known_source_derived_point_access": saved_failure_point,
        "metadata_efficacy_established": False,
        "calibration_savings_established": False,
        "native_score_equivalence_for_arbitrary_means": False,
        "limitations": [
            "new score semantics, no human execution adapter",
            "two artificial heads, not complete source39 pipeline/nested selection",
            "base is 16 free band/channel logits, NOT the shared Q15 feature learner",
            "residual inputs are generated random features, NOT acquisition metadata",
            "toy CE uses scores/.1, NOT frozen source39 scores/sum(weights)/.1",
            "this easy generated case has near-zero initial CE; reduction is not utility evidence",
            "known failure point is regression evidence, not a fresh validation sample",
            "simple top eigenspace/positive temporal variance still required",
            "first-order derivatives only; no timing speed claim",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--saved-failure-point", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve before work: pre-existing receipts are never overwritten, including failures.
    with args.output.open("x") as output:
        try:
            result = run(cuda=args.cuda, saved_failure_point=args.saved_failure_point)
        except Exception as exc:
            json.dump(
                {
                    "status": "ENGINEERING_FAILURE",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                output,
                indent=2,
            )
            output.write("\n")
            output.flush()
            args.output.chmod(0o400)
            raise
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    args.output.chmod(0o400)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "cpu_seconds": result["cpu_seconds"],
                "cuda_executed": result["cuda"]["executed"],
            }
        )
    )
