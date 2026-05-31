from __future__ import annotations

import json
from pathlib import Path


def write_class_map(freqs: list[float], out_dir: str | Path) -> dict[str, dict[str, float | int]]:
    class_map = {
        str(i): {"label": i, "stimulus_frequency_hz": float(freq)}
        for i, freq in enumerate(freqs)
    }
    path = Path(out_dir) / "class_map.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)
    return class_map

