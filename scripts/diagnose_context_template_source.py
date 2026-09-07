#!/usr/bin/env python3
"""Post-hoc, read-only diagnosis of the closed source39 cache; never retune a gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from audit_context_template_source import qweights, sha, softmax, template_scores


def diagnose(root: Path, plan: dict) -> dict:
    with np.load(root / "features.npz", allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    folds = json.loads((root / "fold-freezes.json").read_text())["folds"]
    candidates = {c["id"]: c for c in plan["q"]["candidates"]}
    truth = cache["query_labels"]
    people = len(cache["subject_ids"])
    rows = []
    for l, cell in enumerate(cache["cell_ids"]):
        a0 = np.stack(
            [
                softmax(cache["a0_scores"][p, l], folds[p % 3]["a0_temperature"]["temperature"])
                for p in range(people)
            ]
        )
        uniform_mix = 0.5 * a0 + 0.5 / plan["n_classes"]
        for k in (1, 3, 5):
            scores, supports = [], []
            for p in range(people):
                fold = folds[p % 3]
                selected = fold["q_selection"]["candidate_id"]
                score = template_scores(cache, p, l, qweights(cache, p, l, candidates[selected], k))
                scores.append(score)
                supports.append(
                    softmax(score, fold["support_temperatures"][selected][str(k)]["temperature"])
                )
            scores, supports = np.stack(scores), np.stack(supports)
            fused = 0.5 * a0 + 0.5 * supports
            accuracy = (scores.argmax(-1) == truth).mean(-1)
            rows.append(
                {
                    "cell_id": str(cell),
                    "budget": k,
                    "standalone_template_mean_BA": float(accuracy.mean()),
                    "standalone_template_participant_BA": accuracy.tolist(),
                    "support_posterior_mean_max": float(supports.max(-1).mean()),
                    "support_posterior_mean_class_range": float(np.ptp(supports, axis=-1).mean()),
                    "fused_vs_A0_prediction_flips": float(
                        (fused.argmax(-1) != a0.argmax(-1)).mean()
                    ),
                    "fused_vs_uniform_mix_max_absolute_probability_difference": float(
                        np.abs(fused - uniform_mix).max()
                    ),
                    "fused_mean_NLL": float(
                        -np.log(
                            np.maximum(
                                np.take_along_axis(
                                    fused,
                                    np.broadcast_to(truth[None, :, None], (people, len(truth), 1)),
                                    -1,
                                ),
                                1e-300,
                            )
                        ).mean()
                    ),
                    "A0_mean_NLL": float(
                        -np.log(
                            np.maximum(
                                np.take_along_axis(
                                    a0,
                                    np.broadcast_to(truth[None, :, None], (people, len(truth), 1)),
                                    -1,
                                ),
                                1e-300,
                            )
                        ).mean()
                    ),
                    "uniform_mix_mean_NLL": float(
                        -np.log(
                            np.maximum(
                                np.take_along_axis(
                                    uniform_mix,
                                    np.broadcast_to(truth[None, :, None], (people, len(truth), 1)),
                                    -1,
                                ),
                                1e-300,
                            )
                        ).mean()
                    ),
                }
            )
    return {
        "schema": "cfeg.context-template-source.posthoc-diagnostic.v1",
        "evidence_role": "posthoc_saved_feature_diagnostic_not_new_efficacy_test_or_retuning",
        "result_sha256": sha(root / "result.json"),
        "features_sha256": sha(root / "features.npz"),
        "fold_freezes_sha256": sha(root / "fold-freezes.json"),
        "human_raw_access": False,
        "changes_to_frozen_study": False,
        "interpretation": "uniform mixture uses no support; temperature preserves standalone class ranking; no grid or condition is selected from this diagnostic",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Existing closed artifact directory, read-only"
    )
    args = parser.parse_args()
    print(json.dumps(diagnose(args.output, json.loads(args.plan.read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
