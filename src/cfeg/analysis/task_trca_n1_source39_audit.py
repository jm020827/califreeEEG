"""New experiment endpoints; unchanged independent N1 math and temporal statistics.

No producer, learner, operator, reader or optimizer imports. Algorithm schema is
explicitly reused, never relabeled as the new experiment's execution authority.
"""

from __future__ import annotations

import numpy as np

from cfeg.analysis import task_trca_n1_audit as independent
from cfeg.analysis import task_trca_shape_audit as legacy
from cfeg.analysis import task_trca_temporal_audit as endpoint

STUDY_ID = "task-trca-n1-source39-v1"
SCHEMA = independent.SCHEMA
SCORE_SCHEMA = independent.SCORE_SCHEMA
SUMMARY_SCHEMA = "cfeg.task_trca_n1_source39.independent_summary.v1"
ARMS = independent.ARMS


def summarize(scores, a0, coverage, actuation):
    """Every frozen metadata/calibration/cost/harm branch, with transparent reuse."""
    result = endpoint.summarize(scores, a0, coverage, actuation)
    return {
        **result,
        "schema": SUMMARY_SCHEMA,
        "study_id": STUDY_ID,
        "algorithm_schema": SCHEMA,
        "endpoint_reuse": "task_trca_temporal_audit.summarize; unchanged decision",
    }


def learning_path_coverage(models, *, profile):
    """Generated actuation is a prerequisite; human zero learning is descriptive."""
    independent._require(profile in ("generated", "human"), "fixed coverage profile")
    if profile == "generated":
        return independent.learning_path_coverage(models)
    independent._require(
        isinstance(models, (list, tuple)) and len(models) == 3, "three outer coverage models"
    )
    families = {
        name: {"max_gradient_norm": 0.0, "max_abs_coefficient": 0.0, "head_records": 0}
        for name in ("Q", "Q2", "QM", "SHAM_REFIT")
    }
    pipelines = []
    for model in models:
        independent._schema(model["selection"])
        rows = model["selection"]["inner"]
        independent._require(isinstance(rows, list) and len(rows) == 9, "nine inner pipelines")
        pipelines.extend(row["pipeline"] for row in rows)
        pipelines.append(model["pipeline"])
    for pipeline in pipelines:
        independent._schema(pipeline)
        independent._require(
            set(pipeline["residuals"]) == {"Q2", "QM", "SHAM_REFIT"}, "exact residual heads"
        )
        for name, head in (("Q", pipeline["Q"]), *pipeline["residuals"].items()):
            coefficient = legacy._array(
                head["coefficients"], "coverage coefficients", (16 if name == "Q" else 3,)
            )
            trace = head["trace"]
            independent._require(
                type(head["steps"]) is int
                and head["steps"] == 200
                and isinstance(trace, list)
                and len(trace) == 200
                and all(type(row["step"]) is int for row in trace)
                and [row["step"] for row in trace] == list(range(1, 201)),
                "individual complete200-step coverage trace",
            )
            norms = legacy._array([row["gradient_norm"] for row in trace], "gradient norms", (200,))
            independent._require(np.all(norms >= 0), "nonnegative gradient norms")
            family = families[name]
            family["max_gradient_norm"] = max(family["max_gradient_norm"], float(norms.max()))
            family["max_abs_coefficient"] = max(
                family["max_abs_coefficient"], float(np.abs(coefficient).max())
            )
            family["head_records"] += 1
    for family in families.values():
        family["exercised"] = (
            family["head_records"] == 30
            and family["max_gradient_norm"] > 1e-12
            and family["max_abs_coefficient"] > 1e-12
        )
    return {
        "pipelines": 30,
        "heads_total": 120,
        "registered_updates": 24000,
        "families": families,
        "exercised": all(family["exercised"] for family in families.values()),
        "descriptive_only": True,
        "metadata_effect": "DETERMINED_BY_FROZEN_ENDPOINT_NOT_GRADIENT_COVERAGE",
        "scope": "Recorded gradient/coefficient actuation; not independent Adam replay or efficacy",
    }
