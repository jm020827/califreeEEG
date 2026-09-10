# Schema validation uniformly reports ValueError, including malformed field types.
# ruff: noqa: TRY004
"""Finite, interval-only context extraction with explicit temporal dependencies.

There is no file reader here. Timing evidence is a caller-supplied declaration,
not verification of a device/export or authority to read a human recording.
All dependency bounds use one explicitly identified, aligned sample grid.
"""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from cfeg.data.acquisition_context_contract import (
    RunKey,
    SampleSpan,
    require_int,
    validate_run,
    validate_span,
)


def seconds_to_samples(seconds: str, sfreq: int) -> int:
    """Convert an exact nonnegative decimal/rational string; never round."""
    require_int(sfreq, "sfreq", minimum=1)
    if not isinstance(seconds, str):
        raise ValueError("seconds must be an exact string, not a binary float")
    try:
        count = Fraction(seconds) * sfreq
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid exact time") from exc
    if count < 0 or count.denominator != 1:
        raise ValueError("time must be nonnegative and on the sample grid")
    return count.numerator


def marker_1based_to_onset(marker: int) -> int:
    return require_int(marker, "marker", minimum=1) - 1


@dataclass(frozen=True)
class TimingEvidence:
    run: RunKey
    clock_id: str
    sfreq: int
    export_history_samples: int | None
    export_future_samples: int | None
    marker_anchor: str | None
    evidence_id: str | None

    def validate(self) -> None:
        validate_run(self.run)
        if (
            type(self.clock_id) is not str
            or not self.clock_id.strip()
            or self.clock_id != self.clock_id.strip()
        ):
            raise ValueError("clock_id is required")
        require_int(self.sfreq, "sfreq", minimum=1)
        require_int(self.export_history_samples, "known export history")
        require_int(self.export_future_samples, "known export future")
        if self.marker_anchor != "trial_onset":
            raise ValueError("verified trial_onset marker anchor required")
        if (
            type(self.evidence_id) is not str
            or not self.evidence_id.strip()
            or self.evidence_id != self.evidence_id.strip()
        ):
            raise ValueError("timing evidence_id is required")


def _covered(span: SampleSpan, allowed: tuple[SampleSpan, ...]) -> bool:
    """Check a union, not only its min/max; adjacent intervals may join."""
    cursor = span.start
    for interval in sorted(allowed, key=lambda value: value.start):
        if interval.stop <= cursor:
            continue
        if interval.start > cursor:
            return False
        cursor = max(cursor, interval.stop)
        if cursor >= span.stop:
            return True
    return False


@dataclass(frozen=True)
class ContextWindowPlan:
    timing: TimingEvidence
    run: RunKey
    output: SampleSpan
    cutoff_sample: int
    allowed: tuple[SampleSpan, ...]
    taps: tuple[float, ...]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def exported_read(self) -> SampleSpan:
        return SampleSpan(self.output.start - len(self.taps) + 1, self.output.stop)

    @property
    def dependency(self) -> SampleSpan:
        # validate() has already rejected None before these bounds are used.
        history = self.timing.export_history_samples
        future = self.timing.export_future_samples
        if history is None or future is None:
            raise ValueError("unknown export dependency")
        read = self.exported_read
        return SampleSpan(read.start - history, read.stop + future)

    def validate(self) -> None:
        if type(self.timing) is not TimingEvidence:
            raise ValueError("TimingEvidence required")
        TimingEvidence.validate(self.timing)
        validate_run(self.run)
        if self.run != self.timing.run:
            raise ValueError("run pairing mismatch")
        validate_span(self.output)
        require_int(self.cutoff_sample, "cutoff_sample")
        if type(self.allowed) is not tuple or not self.allowed:
            raise ValueError("nonempty immutable allowed spans required")
        for span in self.allowed:
            validate_span(span)
        if type(self.taps) is not tuple or not 1 <= len(self.taps) <= 4096:
            raise ValueError("immutable FIR taps of length 1..4096 required")
        if any(type(tap) not in (int, float) or not np.isfinite(tap) for tap in self.taps):
            raise ValueError("finite real FIR taps required")
        dependency = self.dependency
        if dependency.stop > self.cutoff_sample:
            raise ValueError("dependency includes samples at or after cutoff")
        if not _covered(dependency, self.allowed):
            raise ValueError("dependency includes unauthorized history or a gap")
        if self.exported_read.length > 200000:
            raise ValueError("request too large for bounded context extractor")


def plan_context_window(
    *,
    timing: TimingEvidence,
    run: RunKey,
    output: SampleSpan,
    cutoff_sample: int,
    allowed: tuple[SampleSpan, ...],
    taps: tuple[float, ...] = (1.0,),
) -> ContextWindowPlan:
    return ContextWindowPlan(timing, run, output, cutoff_sample, allowed, taps)


@dataclass(frozen=True)
class SignalSlice:
    run: RunKey
    clock_id: str
    sfreq: int
    span: SampleSpan
    values: np.ndarray


@dataclass(frozen=True)
class ContextFeatures:
    run: RunKey
    clock_id: str
    sfreq: int
    cutoff_sample: int
    output: SampleSpan
    exported_read: SampleSpan
    dependency: SampleSpan
    evidence_id: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    mean_absolute_difference: tuple[float, ...]


def require_context_pairing(
    features: ContextFeatures,
    *,
    run: RunKey,
    clock_id: str,
    sfreq: int,
    cutoff_sample: int,
) -> None:
    """Bind an extracted receipt to a trial's declared onset and clock.

    The trial/clock must come from verified provenance at a real reader boundary;
    self-consistent caller declarations are not a device verification mechanism.
    """
    if type(features) is not ContextFeatures:
        raise ValueError("ContextFeatures required")
    validate_run(run)
    validate_run(features.run)
    require_int(sfreq, "sfreq", minimum=1)
    require_int(features.sfreq, "feature sfreq", minimum=1)
    require_int(cutoff_sample, "cutoff_sample")
    require_int(features.cutoff_sample, "feature cutoff")
    if type(clock_id) is not str or not clock_id.strip() or clock_id != clock_id.strip():
        raise ValueError("known clock_id required to pair context")
    if (
        features.run != run
        or features.clock_id != clock_id
        or features.sfreq != sfreq
        or features.cutoff_sample != cutoff_sample
    ):
        raise ValueError("context/trial identity, clock, rate or onset mismatch")
    for span in (features.output, features.exported_read, features.dependency):
        validate_span(span)
    if (
        not features.dependency.contains(features.exported_read)
        or not features.exported_read.contains(features.output)
        or features.dependency.stop > cutoff_sample
    ):
        raise ValueError("invalid context receipt dependency")
    if type(features.evidence_id) is not str or not features.evidence_id.strip():
        raise ValueError("context timing provenance required")
    columns = (features.mean, features.std, features.mean_absolute_difference)
    if any(type(values) is not tuple or not values for values in columns):
        raise ValueError("immutable nonempty feature vectors required")
    if len({len(values) for values in columns}) != 1:
        raise ValueError("inconsistent context channel counts")
    for values in columns:
        if any(type(value) not in (int, float) or not np.isfinite(value) for value in values):
            raise ValueError("nonfinite context receipt")
    if any(value < 0 for values in columns[1:] for value in values):
        raise ValueError("negative variation feature")


def extract_context(
    plan: ContextWindowPlan,
    fetch: Callable[[RunKey, SampleSpan], SignalSlice],
) -> ContextFeatures:
    """Read only the planned interval and reset finite FIR history per call.

    The callback must enforce its own physical IO/provenance boundaries. This
    helper validates the request/returned slice, not an arbitrary callback's IO.
    Coefficient taps[j] multiplies the exported sample j steps in the past.
    """
    if type(plan) is not ContextWindowPlan:
        raise ValueError("ContextWindowPlan required")
    ContextWindowPlan.validate(plan)  # Reject all plan/timing failures before invoking fetch.
    read = plan.exported_read
    packet = fetch(plan.run, read)
    if type(packet) is not SignalSlice:
        raise ValueError("SignalSlice required")
    validate_run(packet.run)
    validate_span(packet.span)
    if packet.run != plan.run or packet.clock_id != plan.timing.clock_id:
        raise ValueError("returned slice identity/clock mismatch")
    if type(packet.sfreq) is not int or packet.sfreq != plan.timing.sfreq:
        raise ValueError("returned slice sampling grid mismatch")
    if packet.span != read:
        raise ValueError("returned interval differs from the authorized request")
    x = packet.values
    if type(x) is not np.ndarray or x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("plain channel-by-sample ndarray required")
    if x.shape[1] != read.length or x.size > 200000:
        raise ValueError("wrong or excessive returned array shape")
    if x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise ValueError("finite real numeric values required in allowed slice")
    x = np.asarray(x, dtype=np.float64)
    history = len(plan.taps) - 1
    # Finite support needs neither padding nor state from an earlier call.
    filtered = np.zeros((x.shape[0], plan.output.length), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        try:
            for lag, coefficient in enumerate(plan.taps):
                start = history - lag
                filtered += coefficient * x[:, start : start + plan.output.length]
            means = filtered.mean(axis=1)
            stds = filtered.std(axis=1)
            differences = (
                np.abs(np.diff(filtered, axis=1)).mean(axis=1)
                if plan.output.length > 1
                else np.zeros(x.shape[0])
            )
        except FloatingPointError as exc:
            raise ValueError("nonfinite context arithmetic") from exc
    if not all(np.isfinite(values).all() for values in (means, stds, differences)):
        raise ValueError("nonfinite context features")
    return ContextFeatures(
        plan.run,
        plan.timing.clock_id,
        plan.timing.sfreq,
        plan.cutoff_sample,
        plan.output,
        read,
        plan.dependency,
        str(plan.timing.evidence_id),
        tuple(float(value) for value in means),
        tuple(float(value) for value in stds),
        tuple(float(value) for value in differences),
    )
