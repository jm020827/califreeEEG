"""Array-only participant roles and a complete-freeze query capability gate.

This module is not a human-data loader or an execution authorization. Every
returned support/source packet is a copy; held query arrays are outside this API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfeg.analysis.dan_teacher import SourceChannelScaler, logged_impedance, teacher_block_weights


def fold_roles(ids: tuple[int, ...], fold: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(set(ids)) != len(ids) or tuple(sorted(ids)) != ids or fold not in (0, 1, 2):
        raise ValueError("Sorted unique participants and one of three folds required.")
    source = tuple(person for index, person in enumerate(ids) if index % 3 != fold)[:5]
    target = tuple(person for index, person in enumerate(ids) if index % 3 == fold)
    if len(source) != 5 or not target or set(source) & set(target):
        raise ValueError("Insufficient disjoint source pool.")
    return source, target


def sham_mapping(targets: tuple[int, ...], orders: dict[int, int], seed: int) -> dict[int, int]:
    """One stable derangement per target role/order; no k/interface dependence."""
    if not targets or len(set(targets)) != len(targets) or set(orders) != set(targets):
        raise ValueError("Order packet must contain exactly the target role.")
    if not set(orders.values()) <= {0, 1}:
        raise ValueError("Unknown acquisition order.")
    generator = np.random.default_rng(seed)
    donors = {}
    for order in (0, 1):
        people = sorted(person for person in targets if orders[person] == order)
        if not people:
            continue
        if len(people) < 2:
            raise ValueError("SHAM stratum needs at least two people; no identity fallback.")
        sequence = generator.permutation(people).tolist()
        donors.update(zip(sequence, sequence[1:] + sequence[:1]))
    return donors


@dataclass(frozen=True)
class SupportArrays:
    ids: tuple[int, ...]
    orders: np.ndarray  # [person],0=dry first,1=wet first
    bands: np.ndarray  # [person,interface,band,block0:6,class,channel,time]
    q: np.ndarray  # [person,interface,band,block0:5,channel]
    q2: np.ndarray
    impedance: np.ndarray  # [person,interface,block0:5,channel]

    def validate(self) -> None:
        n = len(self.ids)
        if tuple(sorted(set(self.ids))) != self.ids or self.orders.shape != (n,):
            raise ValueError("Participant schema differs.")
        if not np.isin(self.orders, [0, 1]).all():
            raise ValueError("Invalid order code.")
        if (self.bands.ndim != 7 or self.bands.shape[:4] != (n, 2, 3, 6)
                or min(self.bands.shape[4:]) < 2 or not np.isfinite(self.bands).all()):
            raise ValueError("Expected source-support bands without late query blocks.")
        channels = self.bands.shape[-2]
        if (self.q.shape != (n, 2, 3, 5, channels) or self.q2.shape != self.q.shape
                or not np.isfinite(self.q).all() or not np.isfinite(self.q2).all()
                or self.impedance.shape != (n, 2, 5, channels)):
            raise ValueError("Quality/metadata support-only geometry differs.")
        logged_impedance(self.impedance.reshape(-1, channels))


class FoldSupport:
    def __init__(self, data: SupportArrays, fold: int, *, sham_seed: int):
        data.validate()
        self.data = data
        self.source_ids, self.target_ids = fold_roles(data.ids, fold)
        self.fit_ids = self.source_ids[:4]
        self.index = {person: index for index, person in enumerate(data.ids)}
        orders = {person: int(data.orders[self.index[person]]) for person in self.target_ids}
        self.donors = sham_mapping(self.target_ids, orders, sham_seed + fold)
        self.scalers = {}
        rows = [person for person in self.fit_ids for _ in range(5)]
        fit_indices = [self.index[person] for person in self.fit_ids]
        for interface in range(2):
            m, flags = logged_impedance(data.impedance[fit_indices, interface].reshape(
                -1, data.bands.shape[-2]))
            self.scalers[interface, "M"] = SourceChannelScaler.fit(
                m, flags, row_subject_ids=rows, allowed_fit_ids=self.fit_ids)
            for band in range(3):
                for name in ("q", "q2"):
                    x = getattr(data, name)[fit_indices, interface, band].reshape(-1, m.shape[-1])
                    self.scalers[interface, band, name] = SourceChannelScaler.fit(
                        x, np.ones_like(x, dtype=bool), row_subject_ids=rows,
                        allowed_fit_ids=self.fit_ids)

    def packet(self, target: int, interface: int, band: int, k: int, arm: str) -> tuple:
        if (target not in self.target_ids or interface not in (0, 1) or band not in (0, 1, 2)
                or k not in (2, 3, 5)):
            raise ValueError("Role/interface/band/calibration prefix not permitted.")
        row = self.index[target]
        q = self.data.q[row, interface, band, :k].copy()
        q = self.scalers[interface, band, "q"].transform(q, np.ones_like(q, dtype=bool))
        auxiliary, flags = None, None
        if arm == "Q2":
            raw = self.data.q2[row, interface, band, :k].copy()
            flags = np.ones_like(raw, dtype=bool)
            auxiliary = self.scalers[interface, band, "q2"].transform(raw, flags)
        elif arm in ("QM", "SHAM"):
            donor = self.index[self.donors[target]] if arm == "SHAM" else row
            raw, flags = logged_impedance(self.data.impedance[donor, interface, :k].copy())
            auxiliary = self.scalers[interface, "M"].transform(raw, flags)
        weights = teacher_block_weights(q, arm, auxiliary=auxiliary, observed=flags)
        source = self.data.bands[[self.index[person] for person in self.source_ids],
                                 interface, band].copy()
        support = self.data.bands[row, interface, band, :k].copy()
        return source, support, weights


class FreezeGate:
    """Require all expected model artifacts before one final query capability.

    The runner must hash durable artifacts itself; a digest string alone is not
    independent proof of saving. This gate only checks completeness/order/uniqueness.
    """
    def __init__(self, expected_cells: set[tuple]):
        if not expected_cells:
            raise ValueError("An explicit nonempty full expected cell set is required.")
        self.expected = frozenset(expected_cells)
        self.registered = {}
        self.sealed = False
        self.revealed = False

    def register(self, cell: tuple, artifact_sha256: str) -> None:
        if self.sealed or cell not in self.expected or cell in self.registered:
            raise RuntimeError("Unknown/duplicate cell or training after freeze.")
        if len(artifact_sha256) != 64 or any(c not in "0123456789abcdef" for c in artifact_sha256):
            raise ValueError("Durable artifact SHA256 required.")
        self.registered[cell] = artifact_sha256

    def seal(self) -> None:
        if self.sealed or set(self.registered) != self.expected:
            raise RuntimeError("Cannot freeze until every expected cell is registered exactly once.")
        self.sealed = True

    def reveal_once(self) -> None:
        if not self.sealed or self.revealed:
            raise RuntimeError("Exactly one query batch after complete model freeze is permitted.")
        self.revealed = True
