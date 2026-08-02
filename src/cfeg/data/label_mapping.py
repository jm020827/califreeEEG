from __future__ import annotations

import json
from pathlib import Path


SSVEP_40_FREQUENCIES = tuple(round(8.0 + 0.2 * i, 1) for i in range(40))


def frequency_to_label(
    frequency_hz: float,
    canonical_frequencies: list[float] | tuple[float, ...],
    *,
    tolerance_hz: float = 1e-4,
) -> int:
    """Return the canonical class id for a stimulus frequency."""
    distances = [abs(float(frequency_hz) - float(value)) for value in canonical_frequencies]
    if not distances or min(distances) > tolerance_hz:
        raise ValueError(
            f"Frequency {frequency_hz:g} Hz is absent from the canonical class space "
            f"(tolerance={tolerance_hz:g} Hz)."
        )
    return int(min(range(len(distances)), key=distances.__getitem__))


def remap_source_label(
    source_label: int,
    source_frequencies: list[float],
    canonical_frequencies: list[float],
) -> tuple[int, float]:
    if source_label < 0 or source_label >= len(source_frequencies):
        raise ValueError(
            f"Source class {source_label} is outside 0..{len(source_frequencies) - 1}."
        )
    frequency = float(source_frequencies[source_label])
    return frequency_to_label(frequency, canonical_frequencies), frequency


def write_class_map(
    freqs: list[float] | tuple[float, ...], out_dir: str | Path
) -> dict[str, dict[str, float | int]]:
    class_map = {
        str(i): {"label": i, "stimulus_frequency_hz": float(freq)}
        for i, freq in enumerate(freqs)
    }
    path = Path(out_dir) / "class_map.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)
    return class_map
