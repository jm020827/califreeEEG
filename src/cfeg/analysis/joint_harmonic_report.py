"""Fixed descriptive summaries; no training, selection or data reader."""

from __future__ import annotations

import numpy as np

from cfeg.analysis.joint_harmonic import ARMS, observed_threshold_cost, participant_splits


def summarize(learned: np.ndarray, common: np.ndarray, baselines: np.ndarray,
              choices: dict, config: dict) -> dict:
    n, classes = len(config["source_subject_ids"]), len(config["frequencies"])
    expected = (4, 2, n, 2, 3, 5 * classes, classes)
    if learned.shape != expected or common.shape != (n, 2, 5 * classes, classes):
        raise ValueError("Incomplete or unexpected learned/common score shape.")
    if baselines.shape != (2, n, 2, 3, 5 * classes, classes):
        raise ValueError("Incomplete baseline score shape.")
    if not all(np.isfinite(a).all() for a in (learned, common, baselines)):
        raise ValueError("Cannot summarize nonfinite scores.")
    labels = np.tile(np.arange(classes), 5)
    acc = (learned.argmax(-1) == labels).mean(-1)  # arm,seed,person,interface,k
    cca = (common.argmax(-1) == labels).mean(-1)
    base_acc = (baselines.argmax(-1) == labels).mean(-1)
    low_by_seed = acc[..., :2].mean(axis=(-2, -1))
    low = low_by_seed.mean(axis=1)
    rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
    draws = rng.integers(n, size=(config["evaluation"]["bootstrap_draws"], n))

    def contrast(values: np.ndarray) -> dict:
        intervals = np.quantile(values[draws].mean(axis=1) * 100, [0.025, 0.975])
        return {"mean_pp": float(100 * values.mean()),
                "descriptive_paired_bootstrap_95_pp": intervals.tolist(),
                "participant_values_pp": (100 * values).tolist()}

    contrasts = {f"QM_minus_{arm}": contrast(low[2] - low[i])
                 for i, arm in enumerate(ARMS) if arm != "QM"}
    policies = np.empty((4, n), int)
    fallback = np.empty((4, n), bool)
    ids = config["source_subject_ids"]
    for split in participant_splits(ids):
        positions = [ids.index(s) for s in split["outer_query"]]
        for ai, arm in enumerate(ARMS):
            policy = choices[str(split["fold"])][arm]["policy"]
            policies[ai, positions] = policy["k"]
            fallback[ai, positions] = policy["fallback"]
    policy_acc = np.empty((4, 2, n, 2), float)
    for ai in range(4):
        for i in range(n):
            k = policies[ai, i]
            policy_acc[ai, :, i] = cca[i] if k == 0 else acc[ai, :, i, :, (1, 3, 5).index(k)]
    # Each seed has its own threshold curve; never threshold an ensemble of accuracies.
    zero = np.broadcast_to(cca[None, None, ..., None], (4, 2, n, 2, 1))
    all_curve = np.concatenate((zero, acc), axis=-1)
    costs = observed_threshold_cost(all_curve, config["evaluation"]["target_ba"])
    qcost, mcost = costs[0], costs[2]
    jointly = np.isfinite(qcost) & np.isfinite(mcost)
    threshold = {
        "unit": "seed x participant x interface (repeated, not independent people)",
        "jointly_attained": int(jointly.sum()),
        "qm_only_attained": int((~np.isfinite(qcost) & np.isfinite(mcost)).sum()),
        "q_only_attained": int((np.isfinite(qcost) & ~np.isfinite(mcost)).sum()),
        "neither_attained": int((~np.isfinite(qcost) & ~np.isfinite(mcost)).sum()),
        "paired_mean_q_trials": float(qcost[jointly].mean()) if jointly.any() else None,
        "paired_mean_qm_trials": float(mcost[jointly].mean()) if jointly.any() else None,
        "paired_mean_trial_saving": float((qcost - mcost)[jointly].mean()) if jointly.any() else None,
        "not_an_online_stopping_policy": True,
    }
    arms = {}
    for ai, arm in enumerate(ARMS):
        arms[arm] = {
            "low_k_mean_ba": float(low[ai].mean()),
            "low_k_seed_ba": low_by_seed[ai].mean(axis=1).tolist(),
            "ba_by_k": acc[ai].mean(axis=(0, 1, 2)).tolist(),
            "ba_by_interface_k": acc[ai].mean(axis=(0, 1)).tolist(),
            "participant_interface_k_seed_mean": acc[ai].mean(axis=0).tolist(),
            "policy_mean_ba": float(policy_acc[ai].mean()),
            "policy_mean_acquired_trials": float((classes * policies[ai]).mean()),
            "policy_mean_used_eeg_seconds": float((config["n_samples"] / config["sfreq"]
                                                   * classes * policies[ai]).mean()),
            "policy_ks_by_participant": policies[ai].tolist(),
            "policy_fallback_participants": int(fallback[ai].sum()),
            "policy_target_attained_fraction": float((policy_acc[ai] >= 0.8).mean()),
            "observed_threshold_attained_fraction": float(np.isfinite(costs[ai]).mean()),
        }
    q_trials = arms["Q"]["policy_mean_acquired_trials"]
    m_trials = arms["QM"]["policy_mean_acquired_trials"]
    saving = (q_trials - m_trials) / q_trials if q_trials > 0 else None
    seed_deltas = (low_by_seed[2] - low_by_seed[0]).mean(axis=1) * 100
    harm = float(((low[2] - low[0]) * 100 < config["promotion"]["harm_threshold_pp"]).mean())
    policy_delta = (policy_acc[2] - policy_acc[0]).mean() * 100
    p = config["promotion"]
    gates = {
        "low_q_gain": contrasts["QM_minus_Q"]["mean_pp"] >= p["qm_minus_q_pp"],
        "beats_q2": contrasts["QM_minus_Q2"]["mean_pp"] > p["qm_minus_q2_pp_strict"],
        "beats_sham": contrasts["QM_minus_SHAM"]["mean_pp"] >= p["qm_minus_sham_pp"],
        "both_seeds_positive": bool((seed_deltas > 0).all()),
        "policy_target": arms["QM"]["policy_mean_ba"] >= p["policy_ba"],
        "policy_noninferiority": policy_delta >= p["policy_noninferiority_pp"],
        "policy_trial_reduction": saving is not None and saving >= p["acquired_trial_reduction"],
        "harm_fraction": harm <= p["maximum_harmed_fraction"],
    }
    return {
        "schema": "cfeg.joint-harmonic-result.v1",
        "scientific_screen_passed": all(gates.values()), "gates": {k: bool(v) for k, v in gates.items()},
        "arms": arms, "low_k_contrasts": contrasts,
        "qm_minus_q_by_seed_pp": seed_deltas.tolist(), "harmed_participant_fraction": harm,
        "policy_qm_minus_q_pp": float(policy_delta), "policy_trial_reduction_fraction": saving,
        "observed_threshold_comparison": threshold,
        "baseline": {"common_cca_ba": float(cca.mean()),
                     "common_cca_interface_ba": cca.mean(axis=0).tolist(),
                     "b0_ba_by_k": base_acc[0].mean(axis=(0, 1)).tolist(),
                     "direct_m_ba_by_k": base_acc[1].mean(axis=(0, 1)).tolist(),
                     "q_minus_b0_low_k": contrast(low[0] - base_acc[0, ..., :2].mean(axis=(1, 2))),
                     "direct_m_minus_b0_low_k": contrast((base_acc[1, ..., :2]
                                                          - base_acc[0, ..., :2]).mean(axis=(1, 2)))},
        "limits_of_inference": [
            "Repeatedly used development source39, not independent confirmation",
            "Descriptive participant bootstrap, not confirmatory significance",
            "Calibration trial cost is a complete-block offline acquisition-prefix replay",
            "No new online acquisition or online stopping experiment was performed",
            "Used EEG seconds omit cue/rest/setup and impedance-measurement time",
            "Overall ready-time saving UNKNOWN",
            "Fixed CCA and prototype baselines are not a strongest-method reproduction",
        ],
    }
