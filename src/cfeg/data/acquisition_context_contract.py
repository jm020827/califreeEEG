"""Pure acquisition identity/time types; these do not grant data access."""

from dataclasses import dataclass


def require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int >= {minimum}")
    return value


@dataclass(frozen=True)
class RunKey:
    dataset: str
    participant: str
    day: str
    band: str
    session: str

    def __post_init__(self) -> None:
        for name in ("dataset", "participant", "day", "band", "session"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a nonempty, unpadded string")


@dataclass(frozen=True)
class SampleSpan:
    start: int
    stop: int

    def __post_init__(self) -> None:
        require_int(self.start, "start")
        require_int(self.stop, "stop", minimum=self.start + 1)

    @property
    def length(self) -> int:
        return self.stop - self.start

    def contains(self, other: "SampleSpan") -> bool:
        return self.start <= other.start and other.stop <= self.stop


def validate_run(run: RunKey) -> None:
    """Recheck even a frozen instance; frozen does not guarantee field integrity."""
    if type(run) is not RunKey:
        raise ValueError("exact RunKey required")
    RunKey.__post_init__(run)


def validate_span(span: SampleSpan) -> None:
    if type(span) is not SampleSpan:
        raise ValueError("exact SampleSpan required")
    SampleSpan.__post_init__(span)
