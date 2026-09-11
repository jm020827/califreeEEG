"""One bounded, generated-only, known-optimum router diagnostic; no human IO."""

import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import torch

from cfeg.analysis.source_expert_borrowing import Router, mix_probabilities

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reports/router_saturation_numerical_v1"
CONTRACT = ROOT / "docs/router_saturation_numerical_v1_contract.md"
BASE = ROOT / "src/cfeg/analysis/source_expert_borrowing.py"
BASE_SHA = "3c080cafa2bb9c943aab32084a5ea502c42e0ebc819c9f1cb243528043c143bd"
LIMIT = 4 * 1024**2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if len(data.encode()) > LIMIT:
        raise RuntimeError("report_size_limit")
    with path.open("x") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def fixture(kind, arm, representation):
    contexts = [1, 1, -1, -1] if kind == "CONTEXT" else [1, 1]
    self_p, source_p = [], []
    for m in contexts:
        a, b = ([0.9, 0.8], [0.4, 0.3]) if kind == "BOUNDARY" else ([0.9, 0.2], [0.4, 0.8])
        if kind == "CONTEXT" and m == -1:
            a, b = b, a
        self_p.append(a)
        source_p.append(b)
    correct = torch.tensor([[a] + [b] * 12 for a, b in zip(self_p, source_p)], dtype=torch.float64)
    probabilities = torch.stack((correct, 1 - correct), -1)
    probabilities[:, :, 1] = probabilities[:, :, 1].flip(-1)
    labels = torch.tensor([[0, 1]] * len(contexts), dtype=torch.long)
    copies = 1 if representation == "ONE" else 59
    inputs = torch.zeros((len(contexts), 13, copies + (arm == "QM")), dtype=torch.float64)
    inputs[:, 0, :copies] = 1
    if arm == "QM":
        inputs[:, 0, -1] = torch.tensor(contexts, dtype=torch.float64)
    scale = torch.ones(inputs.shape[-1], dtype=torch.float64)
    if representation == "MEAN59":
        scale[:copies] = 1 / copies
    optimum = [
        1.0
        if kind == "BOUNDARY"
        else 0.5
        if kind == "CONTEXT" and arm == "Q"
        else 4 / 15
        if m == 1
        else 11 / 15
        for m in contexts
    ]
    oracle = -sum(
        math.log(s * a[n] + (1 - s) * b[n])
        for s, a, b in zip(optimum, self_p, source_p)
        for n in range(2)
    ) / (len(contexts) * 2)
    fixture_bytes = sum(
        t.numel() * t.element_size() for t in [correct, probabilities, labels, inputs, scale]
    )
    if fixture_bytes > 2 * 1024**2:
        raise RuntimeError("fixture_size_limit")
    return inputs, scale, probabilities, labels, optimum, oracle, fixture_bytes


def loss_of(probabilities, labels):
    return -probabilities.gather(-1, labels[..., None]).squeeze(-1).log().mean()


def run_fit(kind, arm, representation, journal, counter):
    x, scale, p, y, optimum, oracle, size = fixture(kind, arm, representation)
    router = Router(x)
    optimizer = torch.optim.AdamW(
        router.layer.parameters(), lr=0.05, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    def forward(inputs):
        logits = router.layer(((inputs - router.mean) / router.std) * scale).squeeze(-1)
        gates = logits.softmax(-1)
        return logits, gates, mix_probabilities(p, gates)

    fit_id = f"{kind}_{arm}_{representation}"
    losses, first, final = [], None, None
    for step in range(101):
        optimizer.zero_grad(set_to_none=True)
        logits, gates, mixed = forward(x)
        logits.retain_grad()
        loss = loss_of(mixed, y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite_loss")
        # On these balanced two-class fixtures macro/class-balanced NLL equals plain NLL.
        macro = torch.stack([-mixed[..., c][y == c].log().mean() for c in range(2)]).mean()
        if not bool(torch.allclose(loss, macro, atol=1e-14, rtol=0)):
            raise RuntimeError("balanced_loss_mismatch")
        if representation != "MEAN59" and not bool(
            torch.allclose(gates, router.weights(x), atol=1e-14, rtol=0)
        ):
            raise RuntimeError("production_router_mismatch")
        loss.backward()
        params = list(router.layer.parameters())
        if not all(bool(torch.isfinite(v.grad).all()) for v in params):
            raise RuntimeError("nonfinite_gradient")
        state = {
            "fit": fit_id,
            "step": step,
            "loss": float(loss.detach()),
            "self": gates[:, 0].detach().tolist(),
            "source_mass": gates[:, 1:].sum(-1).detach().tolist(),
            "exact_zero_source_weights": int((gates[:, 1:] == 0).sum()),
            "gap": (logits[:, 0] - logits[:, 1]).detach().tolist(),
            "gap_derivative_holding_sources_fixed": logits.grad[:, 0].tolist(),
            "gradient_norm": math.sqrt(sum(float(v.grad.square().sum()) for v in params)),
            "parameter_norm": math.sqrt(sum(float(v.detach().square().sum()) for v in params)),
        }
        journal.write(json.dumps(state, allow_nan=False) + "\n")
        journal.flush()
        if journal.tell() > LIMIT - 128 * 1024:
            raise RuntimeError("journal_size_limit")
        losses.append(state["loss"])
        if first is None:
            first = state
        final = state
        if step < 100:
            if counter["updates"] >= 1200:
                raise RuntimeError("update_limit")
            optimizer.step()
            counter["updates"] += 1
    with torch.no_grad():
        _, gates, mixed = forward(x)
        flipped = x.clone()
        if arm == "QM":
            flipped[:, 0, -1] *= -1
        _, changed_gates, changed_p = forward(flipped)
        gate_delta = float((gates - changed_gates).abs().max())
        margin_delta = float(
            ((mixed[..., 0] - mixed[..., 1]) - (changed_p[..., 0] - changed_p[..., 1])).abs().max()
        )
    gate_error = max(abs(s - t) for s, t in zip(final["self"], optimum))
    nll_gap = final["loss"] - oracle
    if nll_gap < -1e-12:
        raise RuntimeError("oracle_violated")
    counter["fits"] += 1
    return {
        "fit": fit_id,
        "fixture_bytes": size,
        "states": 101,
        "updates": 100,
        "oracle_self": optimum,
        "oracle_nll": oracle,
        "initial": first,
        "final": final,
        "minimum_recorded_nll": min(losses),
        "final_minus_initial_nll": losses[-1] - losses[0],
        "final_nll_gap": nll_gap,
        "max_gate_error": gate_error,
        "engineering_converged": nll_gap <= 0.001 and gate_error <= 0.02,
        "m_flip_gate_delta": gate_delta,
        "m_flip_margin_delta": margin_delta,
        "m_actuation": arm == "QM" and gate_delta > 1e-12 and margin_delta > 1e-12,
        "model": router.state(),
        "post_standardization_scale": scale.tolist(),
    }


def main():
    if sha(BASE) != BASE_SHA:
        raise RuntimeError("frozen_router_hash_mismatch")
    # Existing directory/start prevents resuming or silently replacing any attempt.
    OUT.mkdir(exist_ok=False)
    write_new(
        OUT / "start.json",
        {
            "schema": "cfeg.router-saturation-numerical.v1",
            "attempt": 1,
            "contract_sha256": sha(CONTRACT),
            "script_sha256": sha(Path(__file__)),
            "router_sha256": sha(BASE),
            "torch": torch.__version__,
            "human_data_access": False,
            "budget_fits": 12,
            "budget_updates": 1200,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    counter = {"fits": 0, "updates": 0}
    began = time.monotonic()

    def deadline(_signum, _frame):
        raise TimeoutError("numeric_wall_budget_30s")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(30)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        # Direct derivative check; no optimization or fixture selection.
        derivative = lambda s: -0.5 * (0.5 / (0.4 + 0.5 * s) - 0.6 / (0.8 - 0.6 * s))
        if not (derivative(4 / 15 - 0.01) < 0 < derivative(4 / 15 + 0.01)):
            raise RuntimeError("oracle_derivative_check")
        results = []
        with (OUT / "trajectory.jsonl").open("x") as journal:
            for kind in ["INTERIOR", "BOUNDARY", "CONTEXT"]:
                for arm in ["Q", "QM"] if kind == "CONTEXT" else ["Q"]:
                    for representation in ["ONE", "SUM59", "MEAN59"]:
                        results.append(run_fit(kind, arm, representation, journal, counter))
            journal.flush()
            os.fsync(journal.fileno())
        if counter != {"fits": 12, "updates": 1200}:
            raise RuntimeError("incomplete_budget")
        report = {
            "schema": "cfeg.router-saturation-numerical.v1",
            "status": "COMPLETED_DIAGNOSTIC_NOT_HUMAN_EFFICACY",
            **counter,
            "numerical_seconds_before_report": time.monotonic() - began,
            "trajectory_sha256": sha(OUT / "trajectory.jsonl"),
            "results": results,
            "human_fits": 0,
            "human_data_reads": 0,
        }
        write_new(OUT / "result.json", report)
        if sum(v.stat().st_size for v in OUT.iterdir()) > LIMIT:
            raise RuntimeError("total_output_limit")
        print(json.dumps({k: v for k, v in report.items() if k != "results"}))
        for r in results:
            print(
                r["fit"],
                "final_self",
                r["final"]["self"],
                "nll_gap",
                r["final_nll_gap"],
                "converged",
                r["engineering_converged"],
            )
    except Exception as exc:
        write_new(
            OUT / "stopped.json",
            {
                "status": "STOPPED_NO_RETRY",
                "error": str(exc),
                **counter,
                "seconds": time.monotonic() - began,
            },
        )
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
