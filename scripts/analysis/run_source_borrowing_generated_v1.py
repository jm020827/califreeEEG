"""Single constructive capacity test; no human files, caches, or outcomes."""

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch

from cfeg.analysis.mobilebci_features import query_statistics, references, support_statistics
from cfeg.analysis.mobilebci_reference_ridge import fit_reference_ridge
from cfeg.analysis.source_expert_borrowing import (
    expert_scores,
    fit_router,
    metadata_differences,
    mix_probabilities,
    pair_features,
    validate_source_subjects,
)


def fixture():
    rng = np.random.default_rng(20260911)
    y = references(128, 128)
    source0 = rng.normal(0, 0.03, size=(3, 5, 6, 128))
    target = np.empty((3, 1, 6, 128))
    for c in range(3):
        wave = y[c, 0] + 0.35 * y[c, 2]
        source0[c, :, c] += wave
        block = rng.normal(0, 0.03, size=(3, 128))
        block[c] += wave
        target[c, 0] = np.concatenate((block, block), axis=0)
    sources = np.stack((source0, source0[:, :, [3, 4, 5, 0, 1, 2]]))
    contexts = np.tile([-1.0, -1.0, 1.0, 1.0], 4)
    sham = np.tile([-1.0, 1.0, -1.0, 1.0], 4)
    labels = np.tile([0, 1, 2, 0, 1, 2], (16, 1))
    queries = rng.normal(0, 0.015, size=(16, 6, 6, 128))
    for episode in range(16):
        group = 0 if contexts[episode] < 0 else 1
        for n, c in enumerate(labels[episode]):
            decoy = (c + 1) % 3
            queries[episode, n, 3 * group + c] += y[c, 0] + 0.35 * y[c, 2]
            queries[episode, n, 3 * (1 - group) + decoy] += 1.15 * (
                y[decoy, 0] + 0.35 * y[decoy, 2]
            )
    arrays = {
        "reference": y,
        "source_support": sources,
        "target_support": target,
        "queries": queries,
        "labels": labels,
        "contexts": contexts,
        "sham": sham,
    }
    if (
        sum(a.size for a in arrays.values()) > 200000
        or sum(a.nbytes for a in arrays.values()) > 2 * 1024 * 1024
    ):
        raise ValueError("fixture_budget")
    return arrays


def run(config):
    paths = {key: Path(config[key]) for key in ["result_path", "fixture_path", "journal_path"]}
    if any(path.exists() for path in paths.values()):
        raise ValueError("no_restart")
    state = {
        "schema": "cfeg.source-borrowing-generated.v1",
        "status": "RUNNING",
        "router_fits_attempted": 0,
        "router_fits_completed": 0,
        "optimizer_steps": 0,
        "ridge_class_solves": 0,
        "ridge_class_solves_attempted": 0,
        "human_inputs": 0,
        "human_fits": 0,
        "models": {},
        "arms": {},
        "seed": 20260911,
    }
    started = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        pinned_paths = {
            "module_sha256": Path("src/cfeg/analysis/source_expert_borrowing.py"),
            "driver_sha256": Path(__file__),
            "test_sha256": Path("scripts/analysis/test_source_expert_borrowing_v1.py"),
            "unit_receipt_sha256": Path(config["unit_receipt_path"]),
        }
        for key, path in pinned_paths.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != config[key]:
                raise ValueError("frozen_artifact_mismatch:" + key)
        receipt = json.loads(pinned_paths["unit_receipt_sha256"].read_text())
        if not (
            receipt["failures"] == receipt["errors"] == 0
            and receipt["tests"] == 9
            and receipt["suite_calls"] <= 3
            and receipt["module_sha256"] == config["module_sha256"]
            and receipt["test_sha256"] == config["test_sha256"]
        ):
            raise ValueError("unit_receipt_invalid")
        state["frozen_artifacts"] = {key: config[key] for key in pinned_paths}
        state["unit_receipt_pass"] = True
        arrays = fixture()
        with paths["fixture_path"].open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        state["fixture_sha256"] = hashlib.sha256(paths["fixture_path"].read_bytes()).hexdigest()
        state["fixture_numeric_elements"] = sum(a.size for a in arrays.values())
        state["fixture_numeric_bytes"] = sum(a.nbytes for a in arrays.values())
        y = torch.tensor(arrays["reference"])
        gram = y @ y.transpose(-1, -2)
        supports = [arrays["target_support"], *arrays["source_support"]]
        weights, covariances, crosses = [], [], []
        for support in supports:
            class_weights = []
            for c in range(3):
                state["ridge_class_solves_attempted"] += 1
                class_weights.append(
                    fit_reference_ridge(
                        torch.tensor(support[c]), y[c], torch.eye(6, dtype=torch.float64), 0.1
                    )
                )
                state["ridge_class_solves"] += 1
            weights.append(torch.stack(class_weights))
            k = support.shape[1]
            stats, _ = support_statistics(
                support.reshape(3 * k, 6, 128),
                np.repeat(np.arange(3), k),
                np.zeros((3 * k, 3, 128)),
                arrays["reference"],
            )
            covariances.append(torch.tensor(stats["support_cov"]))
            crosses.append(torch.tensor(stats["support_cross"]))
        bank = torch.stack(weights)
        bank_cov = torch.stack(covariances)
        q = [query_statistics(x, arrays["reference"]) for x in arrays["queries"]]
        query_cov = torch.tensor(np.stack([row["query_cov"] for row in q]))
        query_cross = torch.tensor(np.stack([row["query_cross"] for row in q]))
        scores, energy_floors = expert_scores(bank, query_cov, query_cross, gram)
        probabilities = torch.softmax(10 * scores, -1)
        base, q2 = pair_features(bank, covariances[0], crosses[0], bank_cov, gram, 128)
        base = base[None].repeat(16, 1, 1)
        q2 = q2[None].repeat(16, 1, 1)
        source_m = torch.tensor([[-1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        target_m = torch.tensor(np.stack((arrays["contexts"], np.ones(16)), axis=-1))
        sham_m = torch.tensor(np.stack((arrays["sham"], np.ones(16)), axis=-1))
        m = metadata_differences(target_m, source_m)
        sm = metadata_differences(sham_m, source_m)
        inputs = {
            "Q": base,
            "Q2": torch.cat((base, q2), -1),
            "QM": torch.cat((base, m), -1),
            "SHAM": torch.cat((base, sm), -1),
        }
        labels = torch.tensor(arrays["labels"], dtype=torch.long)
        train, evaluation = slice(0, 8), slice(8, 16)
        validate_source_subjects(
            ["source0", "source1"], ["source0", "source1"], [f"episode{i}" for i in range(16)]
        )
        routers = {}

        def accuracy(p):
            return float((p.argmax(-1) == labels[evaluation]).to(torch.float64).mean())

        def count_step():
            state["optimizer_steps"] += 1

        with paths["journal_path"].open("x") as journal:
            for arm in ["Q", "Q2", "QM", "SHAM"]:
                state["current_arm"] = arm
                state["router_fits_attempted"] += 1
                router, diagnostics = fit_router(
                    inputs[arm][train],
                    probabilities[train],
                    labels[train],
                    steps=100,
                    on_step=count_step,
                )
                state["router_fits_completed"] += 1
                routers[arm] = router
                gate = router.weights(inputs[arm][evaluation])
                p = mix_probabilities(probabilities[evaluation], gate)
                state["models"][arm] = router.state()
                state["arms"][arm] = {
                    "accuracy": accuracy(p),
                    "gate": gate.tolist(),
                    "predictions": p.argmax(-1).tolist(),
                    "probabilities": p.tolist(),
                    "training": diagnostics,
                }
                journal.write(
                    json.dumps({"arm": arm, **state["arms"][arm]}, allow_nan=False) + "\n"
                )
                journal.flush()
                os.fsync(journal.fileno())
        qm = routers["QM"]
        true_gate = qm.weights(inputs["QM"][evaluation])
        sham_gate = qm.weights(inputs["SHAM"][evaluation])
        true_p = mix_probabilities(probabilities[evaluation], true_gate)
        wrong_m_p = mix_probabilities(probabilities[evaluation], sham_gate)
        uniform_p = mix_probabilities(probabilities[evaluation], torch.full_like(true_gate, 1 / 3))
        wrong_bank = bank.clone()
        wrong_bank[1:] = wrong_bank[1:, [1, 2, 0]]
        wrong_scores, _ = expert_scores(
            wrong_bank, query_cov[evaluation], query_cross[evaluation], gram
        )
        wrong_class_p = mix_probabilities(torch.softmax(10 * wrong_scores, -1), true_gate)
        # Orthogonal basis rotation is representational, not physical onset correction.
        theta = 0.37
        rotation = torch.tensor(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=torch.float64
        )
        rotation = torch.block_diag(rotation, rotation, rotation)
        rotated_scores, _ = expert_scores(
            bank @ rotation, query_cov[evaluation], query_cross[evaluation], gram
        )
        order = [0, 2, 1]
        reordered = mix_probabilities(
            probabilities[evaluation][:, order], qm.weights(inputs["QM"][evaluation][:, order])
        )
        changed_margin = (
            ((true_p[..., 1:] - true_p[..., :1]) - (wrong_m_p[..., 1:] - wrong_m_p[..., :1]))
            .abs()
            .max()
        )
        contingency = [
            [
                int(((arrays["contexts"][s] == c) & (arrays["sham"][s] == mval)).sum())
                for mval in [-1, 1]
            ]
            for s in [train, evaluation]
            for c in [-1, 1]
        ]
        checks = {
            f"qm_beats_{arm.lower()}_by_10pp": state["arms"]["QM"]["accuracy"]
            - state["arms"][arm]["accuracy"]
            >= 0.1
            for arm in ["Q", "Q2", "SHAM"]
        }
        checks.update(
            pinned_unit_receipt_pass=state["unit_receipt_pass"],
            qm_beats_uniform_by_10pp=accuracy(true_p) - accuracy(uniform_p) >= 0.1,
            true_m_beats_mismatch_by_10pp=accuracy(true_p) - accuracy(wrong_m_p) >= 0.1,
            correct_source_classes_matter=accuracy(true_p) - accuracy(wrong_class_p) >= 0.1,
            k1_metadata_changes_gate=float((true_gate - sham_gate).abs().max()) > 1e-5,
            k1_metadata_changes_class_margin=float(changed_margin) > 1e-5,
            source_order_invariance=float((reordered - true_p).abs().max()) <= 1e-9,
            reference_rotation_invariance=float((rotated_scores - scores[evaluation]).abs().max())
            <= 1e-9,
            exact_no_transfer=torch.equal(
                mix_probabilities(probabilities[evaluation], true_gate, no_transfer=True),
                probabilities[evaluation, 0],
            ),
            independent_binary_sham=contingency == [[2, 2]] * 4,
            finite_probabilities=bool(torch.isfinite(true_p).all()),
        )
        state.update(
            status="GENERATED_MECHANISM_PASS"
            if all(checks.values())
            else "GENERATED_MECHANISM_FAIL",
            checks={key: bool(value) for key, value in checks.items()},
            diagnostics={
                "uniform_accuracy": accuracy(uniform_p),
                "mismatched_m_accuracy": accuracy(wrong_m_p),
                "wrong_source_class_accuracy": accuracy(wrong_class_p),
                "target_only_accuracy": accuracy(probabilities[evaluation, 0]),
                "m_only_gate_max_change": float((true_gate - sham_gate).abs().max()),
                "m_only_margin_max_change": float(changed_margin),
                "source_order_max_error": float((reordered - true_p).abs().max()),
                "rotation_max_error": float((rotated_scores - scores[evaluation]).abs().max()),
                "energy_floor_count": energy_floors,
                "sham_contingency_rows": contingency,
            },
            evaluation_labels=labels[evaluation].tolist(),
            bank_weights=bank.tolist(),
            support_pair_base=base[0].tolist(),
            all_expert_probabilities=probabilities[evaluation].tolist(),
        )
    except (
        ValueError,
        RuntimeError,
        OSError,
        MemoryError,
        IndexError,
        KeyError,
        TypeError,
    ) as error:
        state.update(
            status="STOPPED_NO_RETRY", error_type=type(error).__name__, error=str(error)[:512]
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    state["seconds"] = round(time.monotonic() - started, 6)
    state["interpretation"] = (
        "Deliberately M-informative generated mechanism check only; no human benefit, calibration saving, or independent cohort result."
    )
    rendered = json.dumps(state, indent=2, allow_nan=False)
    existing_bytes = sum(
        paths[key].stat().st_size for key in ["fixture_path", "journal_path"] if paths[key].exists()
    )
    if existing_bytes + len(rendered.encode()) + 1 > 8 * 1024 * 1024:
        raise ValueError("report_budget")
    with paths["result_path"].open("x") as stream:
        stream.write(rendered + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": state["status"],
                "arms": {key: row["accuracy"] for key, row in state["arms"].items()},
                "diagnostics": state.get("diagnostics"),
                "checks": state.get("checks"),
                "seconds": state["seconds"],
                "error": state.get("error"),
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    torch.set_num_threads(1)
    torch.manual_seed(20260911)
    torch.use_deterministic_algorithms(True)

    def deadline(_signal, _frame):
        raise TimeoutError("generated_deadline")

    signal.signal(signal.SIGALRM, deadline)
    run(config)
