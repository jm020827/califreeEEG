"""Synthetic callback/role checks; no data readers or study artifacts."""

import numpy as np
import pytest

from cfeg.analysis.task_trca_shape_inputs import (
    EngineeringBlocks,
    RolePartition,
    reject_design_execution,
)


def test_roles_deny_eval_supervision_before_callback_and_queries_always():
    calls = []

    def eeg(pid, interface, samples, blocks):
        calls.append(("eeg", pid, blocks))
        assert pid != 3 or blocks == (0, 1, 2)
        return np.zeros((len(blocks), 12, 5, 8, samples))

    def m(pid, interface, samples, blocks):
        calls.append(("m", pid, blocks))
        assert blocks == tuple(range(len(blocks)))
        return np.zeros((len(blocks), 8))

    blocks = EngineeringBlocks(RolePartition((1,), (2,), (3,)), eeg, m)
    blocks.support(3, 0, 17, 3)
    blocks.supervision(1, 0, 17)
    blocks.supervision(2, 0, 17)
    before = calls.copy()
    with pytest.raises(PermissionError, match="before callback"):
        blocks.supervision(3, 0, 17)
    with pytest.raises(PermissionError):
        blocks.query(3, 0, 17)
    with pytest.raises(PermissionError):
        blocks.support(99, 0, 17, 3)
    assert calls == before
    assert all(entry[0] != "m" or entry[2] == (0, 1, 2) for entry in calls)


@pytest.mark.parametrize("roles", [((1,), (1,), ()), ((1, 1), (), ()), ((True,), (), ())])
def test_bad_role_partitions(roles):
    with pytest.raises(ValueError):
        RolePartition(*roles)


@pytest.mark.parametrize("k", [1, 2, 4, 6, 3.0, True])
def test_prefix_only(k):
    def poison(*args):
        raise AssertionError("Denied input must not be decoded")

    blocks = EngineeringBlocks(RolePartition((1,)), poison, poison)
    with pytest.raises(ValueError):
        blocks.support(1, 0, 17, k)


@pytest.mark.parametrize("status", ["DESIGN_ONLY", "READY", "EXECUTE", None])
def test_no_human_manifest_enabled(status):
    with pytest.raises(PermissionError):
        reject_design_execution({"status": status, "human_execution_authorized": True})
