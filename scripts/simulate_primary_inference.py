#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.analysis.inference_simulation import run_target_free_inference_simulation
from cfeg.utils.checkpoint import save_json
from cfeg.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the target-outcome-free confirmatory inference stress simulation."
    )
    parser.add_argument(
        "--config",
        default="configs/analysis/wearable_target_free_simulation.yaml",
    )
    parser.add_argument(
        "--out",
        default="outputs/governance/wearable_target_free_simulation.json",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    result = run_target_free_inference_simulation(load_config(config_path))
    result["simulation_plan"] = str(config_path)
    result["simulation_plan_sha256"] = _sha256_file(config_path)
    output = Path(args.out).expanduser()
    save_json(output, result)
    print(json.dumps({"status": result["status"], "receipt": str(output)}, sort_keys=True))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
