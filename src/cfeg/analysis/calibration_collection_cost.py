"""Pure observed-schedule accounting, not adaptive stopping or data authority.

Labels are nonnegative Python integers; ``classes`` enumerates the exact task
labels, not a class count. Every supplied trial is a labeled task trial. The
whole schedule is validated without sorting; no files or waveform values are
read. Run identities and sample times are caller-provided provenance claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cfeg.data.acquisition_context_contract import RunKey, require_int


def _run(value: object) -> RunKey:
    if type(value) is not RunKey:
        raise ValueError("run must be a RunKey")
    # Frozen dataclasses are not a security boundary; recheck caller fields.
    RunKey(
        *(
            getattr(value, name, None)
            for name in ("dataset", "participant", "day", "band", "session")
        )
    )
    return value


def _trial_id(value: object) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError("trial_id must be a nonempty, unpadded Python string")
    return value


def _classes(value: object) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) < 2:
        raise ValueError("classes must be a tuple of at least two distinct labels")
    for label in value:
        require_int(label, "class label")
    if len(set(value)) != len(value):
        raise ValueError("classes must not contain duplicate labels")
    return value


def _seconds(value: object, name: str, *, optional: bool = True) -> float | None:
    if value is None and optional:
        return None
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be finite nonnegative Python int/float")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside finite float seconds") from exc
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite nonnegative seconds")
    return converted


def _ids(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple")
    for identifier in value:
        _trial_id(identifier)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate trial IDs")
    return value


def _clock(value: object) -> str | None:
    if value is not None and (
        type(value) is not str or not value.strip() or value != value.strip()
    ):
        raise ValueError("clock_id must be None or a nonempty, unpadded Python string")
    return value


@dataclass(frozen=True)
class CollectedTrial:
    """One labeled task interval [onset_sample, end_sample) on one run grid."""

    run: RunKey
    trial_id: str
    label: int
    onset_sample: int
    end_sample: int

    def __post_init__(self) -> None:
        _run(self.run)
        _trial_id(self.trial_id)
        require_int(self.label, "label")
        require_int(self.onset_sample, "onset_sample")
        require_int(self.end_sample, "end_sample", minimum=self.onset_sample + 1)


@dataclass(frozen=True)
class CollectionCost:
    """Selected labels versus all task trials collected before attainment.

    ``selected_labels`` and ``collected_labels`` are integer label COUNTS, not
    class-label sequences. Missing attainment preserves the observed collected
    prefix and its elapsed recording time, but target selection and total time
    to attainment remain None. Setup and extra baseline are separate charges;
    unknown is never replaced by zero. Extra baseline must be outside the elapsed
    recording interval: this pure type cannot detect caller double-counting.
    """

    run: RunKey
    classes: tuple[int, ...]
    k: int
    sfreq: int
    collection_start_sample: int
    clock_id: str | None
    selected_trial_ids: tuple[str, ...] | None
    selected_labels: int | None
    collected_trial_ids: tuple[str, ...]
    collected_trials: int
    collected_labels: int
    elapsed_recording_seconds: float
    setup_seconds: float | None
    extra_baseline_seconds: float | None
    total_seconds: float | None
    attained: bool

    def __post_init__(self) -> None:
        _run(self.run)
        _classes(self.classes)
        require_int(self.k, "k")
        require_int(self.sfreq, "sfreq", minimum=1)
        require_int(self.collection_start_sample, "collection_start_sample")
        _clock(self.clock_id)
        if type(self.attained) is not bool:
            raise ValueError("attained must be a Python bool")
        collected = _ids(self.collected_trial_ids, "collected_trial_ids")
        require_int(self.collected_trials, "collected_trials")
        require_int(self.collected_labels, "collected_labels")
        if self.collected_trials != len(collected) or self.collected_labels != len(collected):
            raise ValueError("all collected task trials must be counted and labeled")
        elapsed = _seconds(self.elapsed_recording_seconds, "elapsed", optional=False)
        if (not collected and elapsed != 0) or (collected and elapsed == 0):
            raise ValueError("elapsed recording time must agree with collected trials")
        setup = _seconds(self.setup_seconds, "setup_seconds")
        baseline = _seconds(self.extra_baseline_seconds, "extra_baseline_seconds")
        total = _seconds(self.total_seconds, "total_seconds")
        if self.attained:
            selected = _ids(self.selected_trial_ids, "selected_trial_ids")
            require_int(self.selected_labels, "selected_labels")
            if self.selected_labels != len(selected) or len(selected) != self.k * len(self.classes):
                raise ValueError("attained selection must contain k labels per class")
            selected_set = set(selected)
            if tuple(value for value in collected if value in selected_set) != selected:
                raise ValueError("selected trials must be a chronological collected subsequence")
            expected = None if setup is None or baseline is None else elapsed + setup + baseline
            if expected is not None and not math.isfinite(expected):
                raise ValueError("total seconds overflow")
            if total != expected:
                raise ValueError("total_seconds must preserve missing or sum all known charges")
        elif (
            self.selected_trial_ids is not None
            or self.selected_labels is not None
            or total is not None
        ):
            raise ValueError("unattained target selection and total_seconds must be None")
        if self.k == 0 and (not self.attained or collected or elapsed != 0):
            raise ValueError("k0 anchor must attain without collecting task trials")


def balanced_prefix_cost(
    trials: tuple[CollectedTrial, ...] | list[CollectedTrial],
    *,
    run: RunKey,
    classes: tuple[int, ...],
    k: int,
    sfreq: int,
    collection_start_sample: int,
    setup_seconds: float | None = None,
    extra_baseline_seconds: float | None = None,
    clock_id: str | None = None,
) -> CollectionCost:
    """Count the earliest observed prefix that contains k trials of each class.

    Input must already be chronological, single-run, nonoverlapping and duplicate
    free; even trials after attainment are validated. Adjacent half-open intervals
    are allowed. Selection is each class's first k trials in chronological order,
    while collection charges every intervening task trial. Elapsed time includes
    rests/gaps after collection_start_sample through the final collected trial's
    END. No outcome, accuracy, or prospective stopping policy is evaluated.

    All sample indices and sfreq belong to the same caller-declared run grid.
    Optional clock_id is retained; omission does not establish clock equivalence.
    An integration join must require explicit matching clock_id and sfreq. Neither
    clock truth nor evidence provenance can be established from these declarations.
    """
    _run(run)
    labels = _classes(classes)
    require_int(k, "k")
    require_int(sfreq, "sfreq", minimum=1)
    require_int(collection_start_sample, "collection_start_sample")
    _clock(clock_id)
    setup = _seconds(setup_seconds, "setup_seconds")
    baseline = _seconds(extra_baseline_seconds, "extra_baseline_seconds")
    if type(trials) not in (tuple, list):
        raise ValueError("trials must be a tuple/list of CollectedTrial records")
    records = tuple(trials)
    seen: set[str] = set()
    previous: CollectedTrial | None = None
    for trial in records:
        if type(trial) is not CollectedTrial:
            raise ValueError("each trial must be a CollectedTrial")
        # Reconstruct for field validation in case frozen records were forged.
        CollectedTrial(
            *(
                getattr(trial, name, None)
                for name in ("run", "trial_id", "label", "onset_sample", "end_sample")
            )
        )
        if trial.run != run:
            raise ValueError("mixed run identity is forbidden")
        if trial.label not in labels:
            raise ValueError("trial label is outside the exact task classes")
        if trial.trial_id in seen:
            raise ValueError("duplicate trial ID")
        seen.add(trial.trial_id)
        if trial.onset_sample < collection_start_sample:
            raise ValueError("collection starts after a supplied trial onset")
        if previous is not None and trial.onset_sample < previous.end_sample:
            raise ValueError("trial chronology/overlap is invalid; inputs are never sorted")
        previous = trial

    counts = dict.fromkeys(labels, 0)
    selected: list[str] = []
    prefix = 0
    attained = k == 0
    if k:
        for prefix, trial in enumerate(records, start=1):
            if counts[trial.label] < k:
                selected.append(trial.trial_id)
                counts[trial.label] += 1
            if all(count >= k for count in counts.values()):
                attained = True
                break
    collected = records[:prefix]
    try:
        elapsed = (collected[-1].end_sample - collection_start_sample) / sfreq if collected else 0.0
    except OverflowError as exc:
        raise ValueError("elapsed recording seconds overflow") from exc
    elapsed = _seconds(elapsed, "elapsed recording seconds", optional=False)
    total = (
        None if not attained or setup is None or baseline is None else elapsed + setup + baseline
    )
    _seconds(total, "total_seconds")
    return CollectionCost(
        run=run,
        classes=labels,
        k=k,
        sfreq=sfreq,
        collection_start_sample=collection_start_sample,
        clock_id=clock_id,
        selected_trial_ids=tuple(selected) if attained else None,
        selected_labels=len(selected) if attained else None,
        collected_trial_ids=tuple(trial.trial_id for trial in collected),
        collected_trials=len(collected),
        collected_labels=len(collected),
        elapsed_recording_seconds=elapsed,
        setup_seconds=setup,
        extra_baseline_seconds=baseline,
        total_seconds=total,
        attained=attained,
    )
