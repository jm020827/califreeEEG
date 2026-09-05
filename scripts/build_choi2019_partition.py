#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from _bootstrap import add_src_to_path

add_src_to_path()

from cfeg.data.choi2019_partition import (
    construct_partition,
    require_safe_regular_file,
    verify_governance_contract,
    write_partition_bundle_atomically,
)
from cfeg.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="/home/whwovy/eeg-data/processed/choi2019_v1")
    parser.add_argument("--data-config", default="configs/data/choi2019.yaml")
    parser.add_argument(
        "--anchor-config", default="configs/baselines/fbcca_choi2019_bandwise_v1.yaml"
    )
    parser.add_argument(
        "--governance-config",
        default="configs/data/choi2019_processed_partition_v1.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "/home/whwovy/eeg-data/processed/choi2019_v1/partitions/fbcca_choi2019_bandwise_v1"
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Verify the frozen governance config and atomically write the partition bundle.",
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    data_argument = Path(args.data_config)
    data_path = require_safe_regular_file(
        data_argument if data_argument.is_absolute() else repository / data_argument,
        "Choi data config",
    )
    anchor_argument = Path(args.anchor_config)
    anchor_path = require_safe_regular_file(
        anchor_argument if anchor_argument.is_absolute() else repository / anchor_argument,
        "Choi anchor config",
    )
    data_cfg = load_config(data_path, strict_env=False)
    anchor_cfg = load_config(anchor_path, strict_env=False)
    plan = construct_partition(Path(args.processed_dir), data_cfg, anchor_cfg)
    if not args.write:
        print(json.dumps(plan.summary, indent=2, sort_keys=True))
        return

    governance_argument = Path(args.governance_config)
    governance_path = (
        governance_argument.absolute()
        if governance_argument.is_absolute()
        else (repository / governance_argument).absolute()
    )
    governance_path = require_safe_regular_file(governance_path, "Choi governance config")
    with governance_path.open("r", encoding="utf-8") as handle:
        governance_cfg = yaml.safe_load(handle)
    receipt = verify_governance_contract(Path(args.processed_dir), repository, governance_cfg, plan)
    try:
        governance_label = governance_path.relative_to(repository).as_posix()
    except ValueError:
        governance_label = str(governance_path)
    receipt = write_partition_bundle_atomically(
        Path(args.out_dir),
        plan,
        receipt,
        governance_path,
        governance_config_label=governance_label,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
