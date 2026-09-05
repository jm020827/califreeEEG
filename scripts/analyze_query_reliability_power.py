from __future__ import annotations

import argparse
import json

from cfeg.analysis.query_reliability_power import (
    build_query_reliability_sensitivity_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Outcome-free sensitivity audit for query-reliability-spatial-v1."
    )
    parser.add_argument("--draws", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    result = build_query_reliability_sensitivity_audit(
        draws=args.draws,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
