"""Frozen-parent replay followed by the single known-zero inference revision.

The evaluator has no filesystem, fitting, or raw-data access. Authenticated
helper definitions and already verified inputs are injected by the new I/O
owner. No old entrypoint or helper global is replaced.
"""

import math

import numpy as np

TABLES = ("rows", "summary", "contrasts", "diagnostics", "attainment")
MODES = ("Q", "QM", "SHAM_REFIT", "M_STALE")
KEYS = ("participant", "interface", "n_samples", "k")


def compare_parent(actual, expected, path="parent"):
    """Require the entire prior scientific payload, not selected headline rows."""
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or set(actual) != set(expected):
            raise ValueError("Parent replay mapping mismatch: " + path)
        for key in actual:
            compare_parent(actual[key], expected[key], path + "." + key)
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError("Parent replay list mismatch: " + path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_parent(left, right, path + "[" + str(index) + "]")
    elif isinstance(actual, float):
        if (
            type(expected) not in (int, float)
            or not math.isfinite(actual)
            or not math.isfinite(expected)
            or abs(actual - expected) > 1e-12
        ):
            raise ValueError("Parent replay numeric mismatch: " + path)
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError("Parent replay value mismatch: " + path)


def revision_metrics(pred, gains, old_action, new_action, core, rule):
    """Report structural changes; truth is used only for the final metrics."""
    count = pred.shape[1]
    column = np.arange(count)
    original_prefallback = core.choose(gains)
    projected = rule.project_gains(gains, pred)
    corrected_prefallback = core.choose(projected)
    before, after = pred[old_action, column], pred[new_action, column]
    truth = np.tile(np.arange(12), 5)
    old_right, new_right = before == truth, after == truth
    alias = (old_action != 0) & (before == pred[0])
    return {
        "query_count": count,
        "old_full_selected": int(np.sum(old_action == 0)),
        "old_alias_selected": int(np.sum(alias)),
        "old_nonalias_selected": int(np.sum((old_action != 0) & ~alias)),
        "projected_gain_entries_changed": int(np.sum(projected != gains)),
        "prefallback_action_changes": int(np.sum(original_prefallback != corrected_prefallback)),
        "prefallback_prediction_changes": int(
            np.sum(pred[original_prefallback, column] != pred[corrected_prefallback, column])
        ),
        "action_changes": int(np.sum(old_action != new_action)),
        "prediction_changes": int(np.sum(before != after)),
        "repaired_previous": int(np.sum(new_right & ~old_right)),
        "damaged_previous": int(np.sum(old_right & ~new_right)),
        "old_ba": float(np.mean(old_right)),
        "new_ba": float(np.mean(new_right)),
    }


def evaluate(cache, content_projection, freezes, parent_result, science, core, rule):
    # No revised gain projection or routing is permitted before this whole-parent
    # replay succeeds. The old fit definitions are never called by this evaluator.
    replay = core.evaluate_all(cache, content_projection, freezes, science)
    for name in (*TABLES, "evidence_scope"):
        if name not in parent_result:
            raise ValueError("Parent replay missing field: " + name)
        compare_parent(replay[name], parent_result[name], name)

    old_diagnostics = {tuple(row[key] for key in KEYS): row for row in replay["diagnostics"]}
    if len(old_diagnostics) != 624:
        raise ValueError("Parent diagnostic cell grid differs")
    z, _ = core.projection_arrays(content_projection, science)
    scores = core.native_scores(cache, science)
    truth = np.tile(np.arange(12), 5)
    query_columns = np.arange(60)
    rows, diagnostics, revisions = [], [], []

    for p, participant in enumerate(science["source_subject_ids"]):
        fold = p % 3
        frozen = freezes["folds"][fold]
        routers, scaler = frozen["routers"], frozen["metadata_scaler"]
        for i, interface in enumerate(science["interfaces"]):
            for w, n_samples in enumerate(science["sample_counts"]):
                cell = {"participant": participant, "interface": interface, "n_samples": n_samples}

                def emit(
                    method, k, prediction=None, ba=None, interventions=1, available=True, cell=cell
                ):
                    rows.append(
                        {
                            **cell,
                            "method": method,
                            "k": k,
                            "ba": float(np.mean(prediction == truth) if ba is None else ba),
                            "query_count": 60,
                            "label_count": 12 * k,
                            "intervention_count": interventions,
                            "control_available": bool(available),
                        }
                    )

                a0 = scores["a0_r"][p, i, w].reshape(60, 12).argmax(-1)
                emit("A0_author", 0, a0)
                for bi, k in enumerate(science["budgets"]):
                    pred = (
                        scores["expert_r"][p, i, w, bi, : k + 1].reshape(k + 1, 60, 12).argmax(-1)
                    )
                    gains, inputs, fallbacks = {}, {}, {}
                    for mode in MODES:
                        x, fallback = core.inputs_for_cell(
                            cache, z, science, scaler, p, i, w, bi, mode, fold
                        )
                        gains[mode] = core.predict_gain(
                            x, routers["QM" if mode == "M_STALE" else mode]
                        )
                        inputs[mode], fallbacks[mode] = x, fallback
                    maps = core.support_maps(z[p, i, :k])
                    interventions = maps or [tuple(range(k))]
                    shuffle_inputs, shuffle_gains, shuffle_masks = [], [], []
                    for mapping in interventions:
                        x, fallback = core.inputs_for_cell(
                            cache, z, science, scaler, p, i, w, bi, "QM", fold, mapping
                        )
                        shuffle_inputs.append(x)
                        shuffle_gains.append(core.predict_gain(x, routers["QM"]))
                        shuffle_masks.append(fallback)
                    family = rule.route_family(
                        pred,
                        gains,
                        {mode: fallbacks[mode] for mode in MODES[1:]},
                        shuffle_gains,
                        shuffle_masks,
                    )
                    actions = family["actions"]
                    decisions = {
                        mode: pred[action, query_columns] for mode, action in actions.items()
                    }
                    decisions["FULL"] = pred[0]
                    previous = old_diagnostics[(participant, interface, n_samples, k)]
                    previous_actions = {
                        mode: np.asarray(previous["actions"][mode], dtype=int) for mode in MODES
                    }
                    previous_actions["M_MISSING"] = previous_actions["Q"]
                    method_revisions = {
                        mode: revision_metrics(
                            pred,
                            gains["Q" if mode == "M_MISSING" else mode],
                            previous_actions[mode],
                            actions[mode],
                            core,
                            rule,
                        )
                        for mode in (*MODES, "M_MISSING")
                    }
                    for mode in ("FULL", *MODES):
                        available = bool(maps) if mode == "SHAM_REFIT" else True
                        emit(
                            mode,
                            k,
                            decisions[mode],
                            interventions=int(available),
                            available=available,
                        )
                    emit("M_MISSING", k, decisions["M_MISSING"], interventions=0)
                    shuffle_records, shuffle_revisions = [], []
                    for index, (mapping, action, x, gain) in enumerate(
                        zip(interventions, family["shuffle_actions"], shuffle_inputs, shuffle_gains)
                    ):
                        old_record = previous["shuffle_interventions"][index]
                        if old_record["mapping"] != list(mapping):
                            raise ValueError("Original shuffle map/order changed")
                        prediction = pred[action, query_columns]
                        shuffle_records.append(
                            {
                                "mapping": list(mapping),
                                "control_available": bool(maps),
                                "packet_changes": core.packet_changes(z[p, i, :k], mapping),
                                "feature_changes": core.feature_changes(inputs["QM"], x),
                                "selector_changes": int(np.sum(action != actions["QM"])),
                                "prediction_changes": int(np.sum(prediction != decisions["QM"])),
                                "ba": float(np.mean(prediction == truth)),
                                "actions": action.tolist(),
                            }
                        )
                        shuffle_revisions.append(
                            {
                                "mapping": list(mapping),
                                **revision_metrics(
                                    pred,
                                    gain,
                                    np.asarray(old_record["actions"], dtype=int),
                                    action,
                                    core,
                                    rule,
                                ),
                            }
                        )
                    emit(
                        "M_SHUFFLE",
                        k,
                        ba=np.mean([row["ba"] for row in shuffle_records]),
                        interventions=len(maps),
                        available=bool(maps),
                    )
                    qright, mright = decisions["Q"] == truth, decisions["QM"] == truth
                    diagnostics.append(
                        {
                            **cell,
                            "k": k,
                            "expert_disagreement_queries": int(
                                np.sum(np.any(pred[1:] != pred[0], axis=0))
                            ),
                            "qm_vs_q_selector_changes": int(np.sum(actions["QM"] != actions["Q"])),
                            "qm_vs_q_prediction_changes": int(
                                np.sum(decisions["QM"] != decisions["Q"])
                            ),
                            "qm_repaired_q": int(np.sum(mright & ~qright)),
                            "qm_damaged_q": int(np.sum(qright & ~mright)),
                            "qm_missing_fallback_queries": int(fallbacks["QM"].sum()),
                            "admissible_shuffle_maps": len(maps),
                            "mean_shuffled_packet_changes": float(
                                np.mean([row["packet_changes"] for row in shuffle_records])
                            ),
                            "mean_shuffle_feature_changes": float(
                                np.mean([row["feature_changes"] for row in shuffle_records])
                            ),
                            "mean_shuffle_selector_changes": float(
                                np.mean([row["selector_changes"] for row in shuffle_records])
                            ),
                            "mean_shuffle_prediction_changes": float(
                                np.mean([row["prediction_changes"] for row in shuffle_records])
                            ),
                            "sham_feature_changes": core.feature_changes(
                                inputs["QM"], inputs["SHAM_REFIT"]
                            ),
                            "stale_feature_changes": core.feature_changes(
                                inputs["QM"], inputs["M_STALE"]
                            ),
                            "feature_change_tolerance": 1e-12,
                            "shuffle_interventions": shuffle_records,
                            "actions": {mode: actions[mode].tolist() for mode in MODES},
                        }
                    )
                    revisions.append(
                        {**cell, "k": k, "methods": method_revisions, "shuffle": shuffle_revisions}
                    )

    rows.sort(key=lambda row: tuple(row[key] for key in ("participant", *core.KEYS)))
    summary, contrasts, attainment = core.reporting(rows, science)
    result = {
        "parent_replay_verified": True,
        "rows": rows,
        "summary": summary,
        "contrasts": contrasts,
        "diagnostics": diagnostics,
        "attainment": attainment,
        "revision_diagnostics": revisions,
        "evidence_scope": replay["evidence_scope"],
    }
    for name, expected in (
        ("rows", 4680),
        ("summary", 120),
        ("contrasts", 153),
        ("diagnostics", 624),
        ("attainment", 1248),
        ("revision_diagnostics", 624),
    ):
        if len(result[name]) != expected:
            raise ValueError("Incomplete revised reporting grid: " + name)
    return result


def main():
    from native_subset_known_zero_io import execute

    execute(evaluate)


if __name__ == "__main__":
    main()
