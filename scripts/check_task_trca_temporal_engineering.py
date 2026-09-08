"""Fixed generated-array actual Q15/metadata learner integration. No human inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_temporal_evaluation as ev
from cfeg.analysis import task_trca_temporal_learning as learn
from cfeg.analysis.task_trca_temporal_batch import TaskBatch

SEED = 20260909
SOURCE_IDS = tuple(range(21001, 21007))
EVAL_IDS = (31001, 31002)
WEIGHTS = np.array([1.25, 0.67, 0.50, 0.43, 0.38])
FREQUENCIES = np.linspace(9, 14.5, 12)
CODE = (
    "src/cfeg/analysis/task_trca_temporal_learning.py",
    "src/cfeg/analysis/task_trca_temporal_batch.py",
    "src/cfeg/analysis/task_trca_temporal_evaluation.py",
    "src/cfeg/analysis/task_trca_temporal_audit.py",
    "src/cfeg/analysis/task_trca_shape_signfree.py",
    "src/cfeg/analysis/task_trca_shape_features.py",
    "src/cfeg/analysis/task_trca_shape_operator.py",
    "src/cfeg/analysis/metadata_trca_prior.py",
    "src/cfeg/analysis/metadata_prior_source.py",
    "src/cfeg/analysis/task_trca_shape_audit.py",
    "src/cfeg/analysis/task_trca_shape_artifact_audit.py",
    "scripts/check_task_trca_temporal_engineering.py",
    "scripts/audit_task_trca_temporal_engineering.py",
    "configs/analysis/task_trca_temporal_v1_design.json",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o400)


def fixture():
    """One seed, all fixed generated participants; no difficulty/accuracy screen."""
    rng = np.random.default_rng(SEED)
    bundles = {}
    for pid in (*SOURCE_IDS, *EVAL_IDS):
        prototype = rng.normal(size=(12, 5, 8, 23))
        support = prototype[None] + rng.normal(size=(5, 12, 5, 8, 23))
        support += rng.normal(size=(5, 12, 5, 8, 1))
        supervised = prototype + rng.normal(size=prototype.shape)
        supervised += rng.normal(size=(12, 5, 8, 1))
        packet = rng.uniform(0, 40, (5, 8))
        bundles[pid] = (support, supervised, packet)
    return bundles


def make_case(pid, bundle, samples=17, k=3, *, device="cpu"):
    x, y, packet = bundle
    return learn.make_task_case(
        pid,
        0,
        0,
        x[:k, ..., :samples],
        packet[:k],
        FREQUENCIES,
        y[..., :samples],
        weights=WEIGHTS,
        device=device,
    )


def loss_gradient(cases, backend):
    z = (
        torch.linspace(-0.17, 0.17, len(cases) * 40, dtype=torch.float64, device=cases[0].s.device)
        .reshape(len(cases), 5, 8)
        .requires_grad_()
    )
    if backend == "batch":
        loss = TaskBatch(cases).loss(z)
    else:
        loss = torch.stack(
            [learn._ce(learn._prior_scores(c, z[j]), c) for j, c in enumerate(cases)]
        ).mean()
    gradient = torch.autograd.grad(loss, z)[0]
    return float(loss.detach().cpu()), gradient.detach().cpu().numpy()


def parity(bundles, *, cuda):
    specifications = [(pid, n, k) for pid in SOURCE_IDS[:2] for n in (17, 23) for k in (3, 5)]
    cpu_cases = tuple(make_case(pid, bundles[pid], n, k) for pid, n, k in specifications)
    scalar_loss, scalar_grad = loss_gradient(cpu_cases, "scalar")
    batch_loss, batch_grad = loss_gradient(cpu_cases, "batch")
    np.testing.assert_allclose(batch_loss, scalar_loss, atol=1e-12, rtol=0)
    np.testing.assert_allclose(batch_grad, scalar_grad, atol=1e-10, rtol=0)
    start = time.perf_counter()
    cpu = learn.fit_pipeline(cpu_cases, 0.001, backend="batch")
    cpu_seconds = time.perf_counter() - start
    cpu_predictions = {
        arm: np.stack(list(learn.predict(cpu, cpu_cases, arm).values()))
        for arm in ("Q", *learn.RESIDUAL_ARMS)
    }
    result = {
        "case_keys": [list(c.key) for c in cpu_cases],
        "actual_head_count": 4,
        "steps_per_head": 200,
        "cpu_seconds": cpu_seconds,
        "cpu_pipeline": cpu.record(),
        "scalar_batch_loss_error": abs(batch_loss - scalar_loss),
        "scalar_batch_gradient_error": float(np.max(np.abs(batch_grad - scalar_grad))),
        "cpu_scores": {arm: values.tolist() for arm, values in cpu_predictions.items()},
        "cuda": {"executed": False},
    }
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit CUDA engineering request requires CUDA")
        torch.cuda.set_per_process_memory_fraction(0.04)
        torch.cuda.reset_peak_memory_stats()
        gpu_cases = tuple(
            make_case(pid, bundles[pid], n, k, device="cuda") for pid, n, k in specifications
        )
        gpu_loss, gpu_grad = loss_gradient(gpu_cases, "batch")
        np.testing.assert_allclose(gpu_loss, batch_loss, atol=1e-10, rtol=0)
        np.testing.assert_allclose(gpu_grad, batch_grad, atol=1e-8, rtol=1e-5)
        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu = learn.fit_pipeline(gpu_cases, 0.001, backend="batch")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        heads = {
            "Q": (cpu.q, gpu.q),
            **{a: (cpu.residuals[a], gpu.residuals[a]) for a in learn.RESIDUAL_ARMS},
        }
        errors = {}
        for arm, (left, right) in heads.items():
            np.testing.assert_allclose(left.coefficients, right.coefficients, atol=1e-7, rtol=0)
            errors[arm] = float(np.max(np.abs(left.coefficients - right.coefficients)))
        predictions, score_errors = {}, {}
        for arm in cpu_predictions:
            predictions[arm] = np.stack(list(learn.predict(gpu, gpu_cases, arm).values()))
            np.testing.assert_allclose(predictions[arm], cpu_predictions[arm], atol=1e-9, rtol=0)
            np.testing.assert_array_equal(
                predictions[arm].argmax(-1), cpu_predictions[arm].argmax(-1)
            )
            score_errors[arm] = float(np.max(np.abs(predictions[arm] - cpu_predictions[arm])))
        result["cuda"] = {
            "executed": True,
            "device": torch.cuda.get_device_name(),
            "seconds": elapsed,
            "pipeline": gpu.record(),
            "scores": {arm: values.tolist() for arm, values in predictions.items()},
            "loss_error": abs(gpu_loss - batch_loss),
            "gradient_error": float(np.max(np.abs(gpu_grad - batch_grad))),
            "coefficient_errors": errors,
            "score_errors": score_errors,
            "argmax_equal": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        if result["cuda"]["peak_allocated_bytes"] > 1_073_741_824:
            raise RuntimeError("Fixed1GiB GPU engineering budget exceeded")
    return result


def run(output_root, *, cuda):
    torch.set_num_threads(1)
    if not output_root.is_dir() or any(output_root.iterdir()):
        raise ValueError("An existing empty dedicated output directory is required")
    start = {
        "kind": "GENERATED_TEMPORAL_ENGINEERING",
        "schema": ev.SCHEMA,
        "score_schema": ev.SCORE_SCHEMA,
        "seed": SEED,
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "source_ids": list(SOURCE_IDS),
        "evaluation_ids": list(EVAL_IDS),
        "human_data_access": False,
        "held60_access": False,
    }
    write_json(output_root / "start.json", start)
    try:
        bundles = fixture()
        generated_sha = hashlib.sha256(
            b"".join(a.tobytes() for values in bundles.values() for a in values)
        ).hexdigest()
        checked = parity(bundles, cuda=cuda)
        write_json(output_root / "parity.json", checked)
        cases = tuple(make_case(pid, bundles[pid]) for pid in SOURCE_IDS)
        packed = ev.pack_cases(cases)
        with (output_root / "source.npz").open("xb") as stream:
            np.savez_compressed(stream, **packed)
        (output_root / "source.npz").chmod(0o400)
        began = time.perf_counter()

        def progress(event):
            print(json.dumps(event), flush=True)

        pipeline, selection = learn.nested_fit(
            cases, outer_evaluation_ids=EVAL_IDS, backend="batch", progress=progress
        )
        nested_seconds = time.perf_counter() - began
        model = {
            "schema": ev.SCHEMA,
            "score_schema": ev.SCORE_SCHEMA,
            "pipeline": pipeline.record(),
            "selection": selection,
        }
        write_json(output_root / "model.json", model)
        restored = ev.pipeline_from_record(
            json.loads((output_root / "model.json").read_text())["pipeline"]
        )
        if restored.record() != pipeline.record():
            raise ValueError("Pipeline JSON roundtrip changed the trained model")
        states = tuple(
            ev.support_state(
                pid, 0, 0, bundles[pid][0][:3, ..., :17], bundles[pid][2][:3], FREQUENCIES, WEIGHTS
            )
            for pid in EVAL_IDS
        )
        partition = ev.EvaluationPartition(states, EVAL_IDS, ((0, 17, 3),))
        payload = {
            "schema": np.array(ev.SCHEMA),
            "score_schema": np.array(ev.SCORE_SCHEMA),
            "keys": np.array([s.key for s in states]),
            "orders": np.zeros(2, dtype=np.int64),
            "arms": np.array(ev.ARMS),
        }
        outputs, queries = [], []
        for state in states:
            query = np.tile(bundles[state.participant_id][1][..., :17], (4, 1, 1, 1))
            queries.append(query)
            outputs.append(ev.evaluate(restored, state, query, partition=partition))
        for name in ("q", "m", "available", "packet", "s", "c", "weights"):
            payload[name] = np.stack(
                [
                    getattr(s, name).numpy()
                    if isinstance(getattr(s, name), torch.Tensor)
                    else getattr(s, name)
                    for s in states
                ]
            )
        payload["query"] = np.stack(queries)
        payload["templates"] = np.stack([s.model.templates for s in states])
        for name in ("scores", "r", "projectors", "native_filters", "native_full_correlations"):
            payload[name] = np.stack([v[name] for v in outputs])
        for group in ("statistics", "native_statistics"):
            for name in outputs[0][group]:
                payload[group + "_" + name] = np.stack([v[group][name] for v in outputs])
        with (output_root / "evaluation.npz").open("xb") as stream:
            np.savez_compressed(stream, **payload)
        (output_root / "evaluation.npz").chmod(0o400)
        filenames = ("start.json", "parity.json", "source.npz", "model.json", "evaluation.npz")
        size = sum((output_root / name).stat().st_size for name in filenames)
        if size > 2 * 1024**3:
            raise RuntimeError("Fixed generated output2GiB budget exceeded")
        result = {
            "status": "GENERATED_INTEGRATION_COMPLETE_COLD_AUDIT_PENDING",
            "kind": start["kind"],
            "schema": ev.SCHEMA,
            "score_schema": ev.SCORE_SCHEMA,
            "generated_input_sha256": generated_sha,
            "artifacts": {
                name: {
                    "sha256": sha(output_root / name),
                    "bytes": (output_root / name).stat().st_size,
                }
                for name in filenames
            },
            "nested_seconds": nested_seconds,
            "nested_pipeline_fits": 10,
            "nested_head_fits": 40,
            "nested_optimizer_updates": 8000,
            "selected_lambda": selection["selected_lambda"],
            "parity_updates_cpu": 800,
            "parity_updates_cuda": 800 if cuda else 0,
            "evaluation_conditions": [[0, 17, 3]],
            "evaluation_actuation": [v["actuation_QM_minus_Q"] for v in outputs],
            "native_full_project_exact": True,
            "new_human_data_access": False,
            "held60_access": False,
            "metadata_effect": "NOT_EVALUATED",
            "calibration_savings": "NOT_EVALUATED",
            "output_bytes_before_receipt": size,
            "limitations": [
                "fixed generated arrays, not human evidence or DGP efficacy",
                "one-condition6person nested graph, not full39person8condition run",
                "no actual human reader/execution manifest or completed-path access audit",
                "independent audit begins after recorded Q/S/C/statistics, not full raw/Adam replay",
                "synthetic lambda cannot be transferred into a human experiment",
            ],
        }
        write_json(output_root / "receipt.json", result)
        return result
    except Exception as exc:
        write_json(
            output_root / "failure.json",
            {
                "status": "GENERATED_INTEGRATION_FAILURE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "human_data_access": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    result = run(args.output_root.resolve(), cuda=args.cuda)
    print(json.dumps({k: result[k] for k in ("status", "nested_seconds", "selected_lambda")}))
