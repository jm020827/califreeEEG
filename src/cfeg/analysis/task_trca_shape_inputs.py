"""In-memory engineering role adapter. No filesystem decoder or human-run authority."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RolePartition:
    fit_ids: tuple[int, ...]
    validation_ids: tuple[int, ...] = ()
    evaluation_ids: tuple[int, ...] = ()

    def __post_init__(self):
        groups = (self.fit_ids, self.validation_ids, self.evaluation_ids)
        for ids in groups:
            if not isinstance(ids, tuple) or any(type(i) is not int for i in ids):
                raise ValueError("Role IDs must be tuples of integers")
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate role IDs")
        if any(set(groups[a]) & set(groups[b]) for a in range(3) for b in range(a)):
            raise ValueError("Participant roles must be disjoint")

    def role(self, participant_id):
        for role, ids in (
            ("fit", self.fit_ids),
            ("validation", self.validation_ids),
            ("evaluation", self.evaluation_ids),
        ):
            if type(participant_id) is int and participant_id in ids:
                return role
        raise PermissionError("Participant is outside this role partition")


class EngineeringBlocks:
    """Callback boundary for synthetic fixtures, not a sandbox or raw archive reader.

    Callbacks receive (participant_id, interface, N, blocks). Numeric metadata is
    requested separately and only for the support prefix. Final queries are
    deliberately unavailable in this engineering-only adapter.
    """

    def __init__(self, partition: RolePartition, eeg: Callable, metadata: Callable):
        if not isinstance(partition, RolePartition):
            raise TypeError("Expected explicit role partition")
        self.partition, self._eeg, self._metadata = partition, eeg, metadata
        self.access_log = []

    def _context(self, participant_id, interface, samples):
        role = self.partition.role(participant_id)
        if type(interface) is not int or interface not in (0, 1):
            raise ValueError("Invalid interface")
        if type(samples) is not int or samples < 11:
            raise ValueError("Invalid sample count")
        return role

    def support(self, participant_id, interface, samples, k):
        role = self._context(participant_id, interface, samples)
        if type(k) is not int or k not in (3, 5):
            raise ValueError("Only support budgets3/5 are allowed")
        blocks = tuple(range(k))
        self.access_log.append((role, participant_id, "support", blocks))
        x = np.array(self._eeg(participant_id, interface, samples, blocks), copy=True)
        m = np.array(self._metadata(participant_id, interface, samples, blocks), copy=True)
        if x.shape != (k, 12, 5, 8, samples) or m.shape != (k, 8):
            raise ValueError("Callback returned non-prefix geometry")
        return x, m

    def supervision(self, participant_id, interface, samples):
        role = self._context(participant_id, interface, samples)
        if role == "evaluation":
            raise PermissionError("Evaluation participant block5 is denied before callback")
        self.access_log.append((role, participant_id, "source_supervision", (5,)))
        x = np.array(self._eeg(participant_id, interface, samples, (5,)), copy=True)
        if x.shape != (1, 12, 5, 8, samples):
            raise ValueError("Callback returned non-block5 geometry")
        return x[0]

    def query(self, *args, **kwargs):
        raise PermissionError("Final query access requires a future human-run reader/contract")


def reject_design_execution(config):
    """No human executable manifest is supported by this implementation stage."""
    if config.get("status") == "DESIGN_ONLY":
        raise PermissionError("DESIGN_ONLY is not an executable experiment plan")
    raise PermissionError("Human execution manifests are not implemented or authorized here")
