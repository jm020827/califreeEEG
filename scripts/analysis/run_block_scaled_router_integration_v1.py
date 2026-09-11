"""Single generated EEG/statistics integration attempt, no human files or requests."""

import hashlib
import itertools
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from cfeg.analysis.block_scaled_router import BlockRouter, actuation, train
from cfeg.analysis.mobilebci_features import query_statistics, references
from cfeg.analysis.mobilebci_prior_learning import require
from cfeg.analysis.mobilebci_source_borrowing import (
    ARMS,
    KS,
    SPEEDS,
    Dataset,
    balanced_probability_loss,
)
from cfeg.analysis.source_expert_borrowing import mix_probabilities

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/reports/block_scaled_router_integration_v1"
POOL = [f"b{i:02d}" for i in range(4)]
VARIANTS = ["ORIGINAL", "SCALE", "SAFE"]
PINS = {
    "src/cfeg/analysis/mobilebci_source_borrowing.py": "4c634ea313f6cb6df3ebe95f08a01937229e44a4b348443e9d6eb95f03083ca5",
    "src/cfeg/analysis/source_expert_borrowing.py": "3c080cafa2bb9c943aab32084a5ea502c42e0ebc819c9f1cb243528043c143bd",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def generated_data():
    cells = list(itertools.product([-1, 1], repeat=3))
    people = [(s, (-1 if i < 2 else 1,) * 2 + (1,), 0) for i, s in enumerate(POOL)]
    people += [
        (f"{prefix}{i:02d}", cell, split)
        for prefix, split in [("t", 0), ("v", 1)]
        for i, cell in enumerate(cells)
    ]
    runs = [{"subject": s, "speed": v} for s, _, _ in people for v in SPEEDS]
    a = {
        "support_cov": np.empty((60, 4, 3, 9, 9)),
        "support_cross": np.empty((60, 4, 3, 9, 6)),
        "q": np.empty((60, 4, 54)),
        "q2": np.empty((60, 4, 2)),
        "m": np.empty((60, 4, 2)),
        "common": np.empty((60, 4, 5)),
    }
    refs = references(128, 128)
    worlds = {w: [] for w in ["POSITIVE", "NULL"]}
    for pi, (subject, (h, m, r), split) in enumerate(people):
        bank = subject.startswith("b")
        for vi, speed in enumerate(SPEEDS):
            i = 3 * pi + vi
            for c in range(3):
                seed = 9000 + 100 * vi + 10 * c + (pi if bank else 20 + (r + 1) + 4 * split)
                rng = np.random.default_rng(seed)
                trials = 0.12 * rng.normal(size=(5, 9, 128))
                if bank:
                    trials[:, (0 if m == -1 else 3) + c] += refs[c, 0]
                else:
                    trials[:, c] += refs[c, 0] / np.sqrt(2)
                    trials[:, 3 + c] += refs[c, 0] / np.sqrt(2)
                trials -= trials.mean(-1, keepdims=True)
                for ki, k in enumerate(KS):
                    a["support_cov"][i, ki, c] = np.einsum(
                        "nct,ndt->cd", trials[:k], trials[:k]
                    ) / (k * 128)
                    a["support_cross"][i, ki, c] = np.einsum("nct,ht->ch", trials[:k], refs[c]) / (
                        k * 128
                    )
            for ki, k in enumerate(KS):
                j = np.arange(54) + 1
                # No h,m,person ID, or query-derived values enter recipient Q/common.
                a["q"][i, ki] = 1 + 0.05 * np.sin(j * (0.13 + 0.01 * r)) + 0.01 * vi + 0.02 * ki
                a["q2"][i, ki] = [r, 0.5 * r]
                a["m"][i, ki] = [m, 0.5 * m]
                a["common"][i, ki] = [float(speed), vi, k, 3 * k + 1, 9 * (3 * k + 1)]
            for world, world_rows in worlds.items():
                group = m if world == "POSITIVE" else h
                rng = np.random.default_rng(700 + 20 * vi + 5 * (r + 1) + (group + 1) + 100 * split)
                x = (0.16 + 0.02 * vi) * rng.normal(size=(6, 9, 128))
                for n, c in enumerate([0, 0, 1, 1, 2, 2]):
                    active, inactive = (0, 3) if group == -1 else (3, 0)
                    phase = 0.3 * split + 0.07 * (n % 2)
                    x[n, active + c] += np.cos(phase) * refs[c, 0] + np.sin(phase) * refs[c, 1]
                    wrong = (c + 1) % 3
                    x[n, inactive + wrong] += 1.8 * (
                        np.cos(phase) * refs[wrong, 0] + np.sin(phase) * refs[wrong, 1]
                    )
                require(x.nbytes <= 1024**2, "waveform_budget")
                x -= x.mean(-1, keepdims=True)
                world_rows.append(query_statistics(x, refs))
    data = {}
    for world, rows in worlds.items():
        arrays = dict(a)
        for key in ["query_cov", "query_cross", "reference_gram"]:
            arrays[key] = np.stack([r[key] for r in rows])
        arrays["query_labels"] = np.tile([0, 0, 1, 1, 2, 2], (60, 1))
        require(sum(v.nbytes for v in arrays.values()) <= 4 * 1024**2, "array_budget")
        data[world] = Dataset(arrays, runs, samples=128)
    return data, cells


def metric(batch, model, arm):
    with torch.no_grad():
        gate = model.weights(batch["inputs"][arm])
        p = mix_probabilities(batch["probabilities"], gate)
        loss = float(balanced_probability_loss(p, batch["labels"]))
        accuracy = float((p.argmax(-1) == batch["labels"]).double().mean())
        restore = BlockRouter.from_state(model.state())
        require(
            torch.allclose(
                p,
                mix_probabilities(batch["probabilities"], restore.weights(batch["inputs"][arm])),
                atol=1e-12,
                rtol=0,
            ),
            "state_roundtrip",
        )
        order = torch.arange(12, -1, -1)
        perm = mix_probabilities(
            batch["probabilities"][:, order], model.weights(batch["inputs"][arm][:, order])
        )
        require(torch.allclose(p, perm, atol=1e-12, rtol=0), "joint_permutation")
    return {
        "nll": loss,
        "accuracy": accuracy,
        "self_mass_mean": float(gate[:, 0].mean()),
        "source_mass_min": float(gate[:, 1:].sum(-1).min()),
        "predictions": p.argmax(-1).tolist(),
    }


def main():
    for path, expected in PINS.items():
        require(sha(ROOT / path) == expected, "frozen_code_pin:" + path)
    OUT.mkdir(exist_ok=False)
    write(
        OUT / "start.json",
        {
            "attempt": 1,
            "base": "da11bf3",
            "old_pins": PINS,
            "runner_sha": sha(Path(__file__)),
            "model_sha": sha(ROOT / "src/cfeg/analysis/block_scaled_router.py"),
            "contract_sha": sha(ROOT / "docs/block_scaled_router_integration_v1_contract.md"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    counter = {
        "fits": 0,
        "proposals": 0,
        "proposals_attempted": 0,
        "trial_checks": 0,
        "trial_loss_evaluations": 0,
        "ridge_solves": 0,
        "ridge_attempted": 0,
    }
    started = time.monotonic()

    def timeout(_s, _f):
        raise TimeoutError("numeric_budget120s")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        torch.set_num_threads(1)
        datasets, cells = generated_data()

        def solve(done):
            if done:
                counter["ridge_solves"] += 1
            else:
                counter["ridge_attempted"] += 1
            require(counter["ridge_attempted"] <= 720, "ridge_budget")

        datasets["POSITIVE"].fit_experts(solve)
        datasets["NULL"].weights = datasets["POSITIVE"].weights.detach().clone()
        require(counter["ridge_solves"] == 720, "ridge_count")
        results, audit, roles = {}, {}, {}
        with (OUT / "trajectory.jsonl").open("x") as journal:
            for world, dataset in datasets.items():
                training = dataset.indices([f"t{i:02d}" for i in range(8)])
                evaluation = dataset.indices([f"v{i:02d}" for i in range(8)])
                batch = dataset.batch(training, POOL, 0)
                # Paired cells differ only M, or only h: routing Q cannot see either.
                for i, (h, m, r) in enumerate(cells):
                    j = cells.index((h, -m, r))
                    jh = cells.index((-h, m, r))
                    for vi in range(3):
                        torch.testing.assert_close(
                            batch["inputs"]["Q"][3 * i + vi],
                            batch["inputs"]["Q"][3 * j + vi],
                            rtol=0,
                            atol=0,
                        )
                        torch.testing.assert_close(
                            batch["inputs"]["Q"][3 * i + vi],
                            batch["inputs"]["Q"][3 * jh + vi],
                            rtol=0,
                            atol=0,
                        )
                        if world == "NULL":
                            torch.testing.assert_close(
                                batch["probabilities"][3 * i + vi],
                                batch["probabilities"][3 * j + vi],
                                rtol=0,
                                atol=0,
                            )
                for arm in ARMS:
                    audit[f"{world}_train_{arm}"] = batch["inputs"][arm].numpy()
                audit[f"{world}_train_p"] = batch["probabilities"].numpy()
                audit[f"{world}_train_y"] = batch["labels"].numpy()
                heads, records = {}, {}
                for variant in VARIANTS:
                    for arm in ARMS:
                        name = f"{variant}_{arm}"

                        def step(row, world=world, name=name):
                            counter["proposals"] += 1
                            journal.write(json.dumps({"world": world, "fit": name, **row}) + "\n")
                            journal.flush()

                        def attempt():
                            counter["proposals_attempted"] += 1
                            require(counter["proposals_attempted"] <= 2400, "proposal_budget")

                        def trial(kind):
                            key = (
                                "trial_checks" if kind == "candidate" else "trial_loss_evaluations"
                            )
                            counter[key] += 1
                            require(counter[key] <= 6400, "trial_budget")

                        model, record = train(
                            batch["inputs"][arm],
                            batch["probabilities"],
                            batch["labels"],
                            variant,
                            100,
                            step,
                            attempt,
                            trial,
                        )
                        record.pop("trajectory")
                        heads[name] = model
                        records[name] = {"training": record, "model": model.state()}
                        counter["fits"] += 1
                        journal.write(
                            json.dumps(
                                {
                                    "world": world,
                                    "fit": name,
                                    "event": "fit_complete",
                                    **records[name],
                                }
                            )
                            + "\n"
                        )
                        journal.flush()
                require(
                    all(not p.requires_grad for m in heads.values() for p in m.layer.parameters()),
                    "heads_not_frozen",
                )
                journal.write(
                    json.dumps({"world": world, "event": "heads_frozen", "counters": dict(counter)})
                    + "\n"
                )
                journal.flush()
                eb = dataset.batch(evaluation, POOL, 0)
                roles[world] = {"training": batch["records"], "evaluation": eb["records"]}
                for phase, phase_batch in [("training", batch), ("evaluation", eb)]:
                    for record in phase_batch["records"]:
                        require(
                            record["subject"].startswith("t" if phase == "training" else "v"),
                            "recipient_phase",
                        )
                        require(
                            all(
                                dataset.subjects[ix] in POOL
                                for ix in record["source_runs"] + record["donor_runs"]
                            ),
                            "bank_role_all",
                        )
                    for i, (h, m, r) in enumerate(cells):
                        j = cells.index((h, -m, r))
                        jh = cells.index((-h, m, r))
                        for vi in range(3):
                            for other in [j, jh]:
                                torch.testing.assert_close(
                                    phase_batch["inputs"]["Q"][3 * i + vi],
                                    phase_batch["inputs"]["Q"][3 * other + vi],
                                    rtol=0,
                                    atol=0,
                                )
                            if world == "NULL":
                                torch.testing.assert_close(
                                    phase_batch["probabilities"][3 * i + vi],
                                    phase_batch["probabilities"][3 * j + vi],
                                    rtol=0,
                                    atol=0,
                                )
                for arm in ARMS:
                    audit[f"{world}_eval_{arm}"] = eb["inputs"][arm].numpy()
                audit[f"{world}_eval_p"] = eb["probabilities"].numpy()
                audit[f"{world}_eval_y"] = eb["labels"].numpy()
                for variant in VARIANTS:
                    for arm in ARMS:
                        name = f"{variant}_{arm}"
                        records[name]["evaluation"] = metric(eb, heads[name], arm)
                    records[f"{variant}_QM"]["m_actuation"] = actuation(
                        eb["inputs"]["QM"],
                        eb["inputs"]["SHAM"],
                        eb["probabilities"],
                        heads[f"{variant}_QM"],
                    )
                # Feature access boundaries checked without any further fit.
                ix = evaluation[0]
                before, _, roles = dataset.features(ix, POOL, 0)
                reverse, _, _ = dataset.features(ix, list(reversed(POOL)), 0)
                require(
                    all(
                        s.startswith("b")
                        for s in [dataset.subjects[i] for i in roles["source_runs"]]
                    ),
                    "bank_role",
                )
                label = dataset.arrays["query_labels"][ix].copy()
                dataset.arrays["query_labels"][ix] = (label + 1) % 3
                changed, _, _ = dataset.features(ix, POOL, 0)
                changed_batch = dataset.batch([ix], POOL, 0)
                dataset.arrays["query_labels"][ix] = label
                torch.testing.assert_close(
                    changed_batch["probabilities"], eb["probabilities"][:1], rtol=0, atol=0
                )
                for arm in ARMS:
                    torch.testing.assert_close(before[arm], reverse[arm], rtol=0, atol=0)
                    torch.testing.assert_close(before[arm], changed[arm], rtol=0, atol=0)
                cov = dataset.arrays["query_cov"][ix].copy()
                dataset.arrays["query_cov"][ix] *= 2
                changed, _, _ = dataset.features(ix, POOL, 0)
                dataset.arrays["query_cov"][ix] = cov
                for arm in ARMS:
                    torch.testing.assert_close(before[arm], changed[arm], rtol=0, atol=0)
                torch.testing.assert_close(
                    mix_probabilities(
                        eb["probabilities"],
                        heads["SAFE_Q"].weights(eb["inputs"]["Q"]),
                        no_transfer=True,
                    ),
                    eb["probabilities"][:, 0],
                    rtol=0,
                    atol=0,
                )
                results[world] = records
            journal.flush()
            os.fsync(journal.fileno())
        positive, negative = results["POSITIVE"], results["NULL"]
        qm = positive["SAFE_QM"]
        gates = {
            "positive_nll": all(
                positive[f"SAFE_{a}"]["evaluation"]["nll"] - qm["evaluation"]["nll"] >= 0.02
                for a in ["Q", "Q2"]
            ),
            "positive_accuracy": all(
                qm["evaluation"]["accuracy"] - positive[f"SAFE_{a}"]["evaluation"]["accuracy"]
                >= 0.10
                for a in ["Q", "Q2"]
            ),
            "m_actuation": all(v > 1e-4 for v in qm["m_actuation"].values()),
            "null_nll": negative["SAFE_Q"]["evaluation"]["nll"]
            - negative["SAFE_QM"]["evaluation"]["nll"]
            <= 0.01,
            "null_accuracy": negative["SAFE_QM"]["evaluation"]["accuracy"]
            - negative["SAFE_Q"]["evaluation"]["accuracy"]
            <= 0.02,
        }
        require(counter["fits"] == 24 and counter["proposals"] == 2400, "incomplete_count")
        with (OUT / "audit_arrays.npz").open("xb") as f:
            np.savez_compressed(f, **audit)
        report = {
            "status": "GATE_PASS" if all(gates.values()) else "GATE_FAIL_NO_RETRY",
            "gates": gates,
            "counters": counter,
            "results": results,
            "roles": roles,
            "numeric_seconds_before_report": time.monotonic() - started,
            "journal_sha": sha(OUT / "trajectory.jsonl"),
            "arrays_sha": sha(OUT / "audit_arrays.npz"),
            "human_fits": 0,
            "human_numeric_reads": 0,
        }
        expected_bytes = len((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
        require(
            sum(p.stat().st_size for p in OUT.iterdir()) + expected_bytes <= 16 * 1024**2,
            "output_budget",
        )
        write(OUT / "result.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "results"}))
        for world, rows in results.items():
            for name, row in rows.items():
                print(world, name, row["evaluation"]["nll"], row["evaluation"]["accuracy"])
    except Exception as exc:
        write(
            OUT / "stopped.json",
            {
                "status": "STOPPED_NO_RETRY",
                "error": str(exc),
                "counters": counter,
                "seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
