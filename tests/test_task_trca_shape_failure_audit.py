"""Only generated access events; no actual failure files are opened."""

import importlib.util
from pathlib import Path

import pytest


def test_exact_access_multiset_rejects_global_count_preserving_corruption():
    spec = importlib.util.spec_from_file_location(
        "task_failure_audit",
        Path(__file__).resolve().parents[1] / "scripts/audit_task_trca_shape_failure.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = [11001, 11002, 11003]
    records = []
    for fold in (0, 1):
        for rank, pid in enumerate(ids):
            if rank % 3 == fold:
                continue
            for i in (0, 1):
                for k in (3, 5):
                    records.append(
                        {
                            "outer_fold": fold,
                            "participant_id": pid,
                            "role": "fit",
                            "kind": "metadata_support",
                            "interface": i,
                            "blocks": list(range(k)),
                        }
                    )
                for n in module.audit.SAMPLES:
                    records.append(
                        {
                            "outer_fold": fold,
                            "participant_id": pid,
                            "role": "fit",
                            "kind": "source_supervision",
                            "interface": i,
                            "samples": n,
                            "blocks": [5],
                        }
                    )
                    for k in (3, 5):
                        records.append(
                            {
                                "outer_fold": fold,
                                "participant_id": pid,
                                "role": "fit",
                                "kind": "support",
                                "interface": i,
                                "samples": n,
                                "blocks": list(range(k)),
                            }
                        )
    module.check_access(records, ids)
    for field, value in (("participant_id", 99999), ("interface", 9), ("blocks", [6, 7, 8, 9])):
        bad = [r.copy() for r in records]
        bad[0][field] = value
        with pytest.raises(ValueError, match="multiset"):
            module.check_access(bad, ids)
