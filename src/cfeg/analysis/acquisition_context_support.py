"""Bind generated/declaration-based support context to counted collection events.

No data reader or model fit is provided. Caller-supplied identity/timing evidence
is not independently verified by consistency checks on these pure records.
"""

from dataclasses import dataclass

from cfeg.analysis.calibration_collection_cost import (
    CollectedTrial,
    CollectionCost,
    balanced_prefix_cost,
)
from cfeg.data.acquisition_context_boundary import ContextFeatures, require_context_pairing


@dataclass(frozen=True)
class BoundSupportContext:
    cost: CollectionCost
    trials: tuple[CollectedTrial, ...]
    contexts: tuple[ContextFeatures, ...]
    required_precollection_seconds: float


def bind_selected_contexts(
    trials: tuple[CollectedTrial, ...],
    cost: CollectionCost,
    contexts: dict[str, ContextFeatures],
) -> BoundSupportContext:
    """Require exact selected-trial membership and matching run/clock/rate/onset.

    Recompute the cost from the supplied collection schedule, not a claimed count.
    This is observed k-shot availability, not reaching an accuracy target.
    All selected trials must have context in this engineering contract; a missing
    sensor fallback for a future learner must be designed separately.
    """
    if type(cost) is not CollectionCost:
        raise ValueError("CollectionCost required")
    CollectionCost.__post_init__(cost)
    if cost.clock_id is None:
        raise ValueError("known collection clock required for context binding")
    reproduced = balanced_prefix_cost(
        trials,
        run=cost.run,
        classes=cost.classes,
        k=cost.k,
        sfreq=cost.sfreq,
        collection_start_sample=cost.collection_start_sample,
        setup_seconds=cost.setup_seconds,
        extra_baseline_seconds=cost.extra_baseline_seconds,
        clock_id=cost.clock_id,
    )
    if reproduced != cost:
        raise ValueError("cost receipt does not match the supplied schedule")
    if not cost.attained or cost.selected_trial_ids is None:
        raise ValueError("requested k-shot support was not collected")
    if type(contexts) is not dict or set(contexts) != set(cost.selected_trial_ids):
        raise ValueError("context IDs must match exactly the selected support")
    lookup = {trial.trial_id: trial for trial in trials}
    selected = tuple(lookup[identifier] for identifier in cost.selected_trial_ids)
    ordered = tuple(contexts[trial.trial_id] for trial in selected)
    for trial, features in zip(selected, ordered, strict=True):
        require_context_pairing(
            features,
            run=trial.run,
            clock_id=cost.clock_id,
            sfreq=cost.sfreq,
            cutoff_sample=trial.onset_sample,
        )
    earliest = min(
        (feature.dependency.start for feature in ordered), default=cost.collection_start_sample
    )
    lead = max(0, cost.collection_start_sample - earliest) / cost.sfreq
    if cost.extra_baseline_seconds is not None and cost.extra_baseline_seconds < lead:
        raise ValueError("extra baseline omits required precollection context time")
    # If extra baseline is unknown, cost.total_seconds already remains unknown.
    return BoundSupportContext(cost, selected, ordered, lead)
