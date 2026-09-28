"""Predeclared descriptive source39 endpoints, not confirmatory inference."""

from __future__ import annotations

import numpy as np


def summarize(rows: list[dict], config: dict, *, generated: bool) -> dict:
    ids = sorted({r["target"] for r in rows})

    def mean(arm, ks, *, person=None, seed=None):
        selected = [r for r in rows if r["arm"] == arm and r["k"] in ks
                    and (person is None or r["target"] == person)
                    and (seed is None or r["seed"] == seed)]
        if not selected:
            raise ValueError("Missing endpoint cell; do not silently change the evaluation grid.")
        if len({r["trials"] for r in selected}) != 1:
            raise ValueError("Unequal query counts violate this complete-block protocol.")
        return sum(r["correct"] for r in selected) / (len(selected) * selected[0]["trials"])

    primary = config["evaluation"]["primary_k"]
    contrasts = {}
    for comparator in ("Q", "Q2", "SHAM"):
        delta = np.array([mean("QM", primary, person=s) - mean(comparator, primary, person=s)
                          for s in ids])
        rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
        draws = rng.choice(delta, size=(config["evaluation"]["bootstrap_draws"], len(ids))).mean(1)
        contrasts[comparator] = {"mean_pp": (mean("QM", primary) - mean(comparator, primary)) * 100,
                                 "descriptive_ci95_pp": (np.quantile(draws, [0.025, 0.975]) * 100).tolist(),
                                 "participant_delta_pp": dict(zip(map(str, ids), (delta * 100).tolist()))}
    curves = {arm: {str(k): mean(arm, [k]) for k in config["budgets"]}
              for arm in [*config["arms"], "TRCA"]}
    cca = mean("CCA", [0])
    seeds = {str(seed): (mean("QM", primary, seed=seed) - mean("Q", primary, seed=seed)) * 100
             for seed in config["training"]["seeds"]}
    harm = float(np.mean(np.array(list(contrasts["Q"]["participant_delta_pp"].values())) < -5))
    qm3, q5 = curves["QM"]["3"], curves["Q"]["5"]
    gates = {"qm_q_2pp": contrasts["Q"]["mean_pp"] >= 2,
             "qm_q2_positive": contrasts["Q2"]["mean_pp"] > 0,
             "qm_sham_1pp": contrasts["SHAM"]["mean_pp"] >= 1,
             "each_seed_positive": all(v > 0 for v in seeds.values()),
             "qm3_at_least_80pct": qm3 >= 0.8,
             "qm3_noninferior_q5": qm3 - q5 >= -0.01,
             "qm3_noninferior_cca": qm3 - cca >= -0.01,
             "harm_fraction_at_most_20pct": harm <= 0.2}
    attainment = []
    for person in ids:
        for interface in sorted({r["interface"] for r in rows if r["target"] == person}):
            for seed in config["training"]["seeds"]:
                for arm in config["arms"]:
                    group = [r for r in rows if r["target"] == person and r["interface"] == interface
                             and r["seed"] == seed and r["arm"] == arm]
                    passing = [r["k"] * len(config["frequencies"]) for r in group if r["accuracy"] >= .8]
                    attainment.append({"target": person, "interface": interface, "seed": seed,
                                       "arm": arm, "first_grid_80pct_trials": min(passing) if passing else None})
    return {"generated": generated, "status": "PRODUCER_COMPLETE_AUDIT_PENDING",
            "curves": curves, "cca_accuracy": cca, "primary_qm_minus": contrasts,
            "primary_qm_minus_q_by_seed_pp": seeds, "harm_fraction": harm,
            "baseline_u_minus_trca_primary_pp": (mean("U", primary) - mean("TRCA", primary)) * 100,
            "fixed_cost_contrast": {"qm_trials": 36, "q_trials": 60,
                                    "qm3_accuracy": qm3, "q5_accuracy": q5,
                                    "nominal_trial_reduction": 0.4,
                                    "performance_criteria_met": qm3 >= .8 and qm3 - q5 >= -.01
                                    and qm3 - cca >= -.01,
                                    "validated_metadata_calibration_reduction": False},
            "promotion_gates": gates, "producer_all_gates_pass": all(gates.values()),
            "attainment_descriptive_only": attainment,
            "rows": rows,
            "limits_of_inference": ["Repeatedly reused source39 development participants.",
                                    "Independent paper-based protocol adaptation, not author reproduction.",
                                    "Bootstrap is descriptive; independent confirmation is pending.",
                                    "Electrode setup/measurement/rest ready-time is UNKNOWN.",
                                    "Fixed prefix contrast, not an online adaptive stopping policy."]}
