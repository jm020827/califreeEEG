"""Support-only task-shape features and partition-local preprocessing.

This module performs no I/O.  In particular, ``support_q_mask`` and
``donor_map`` accept a boolean missingness pattern, never numeric metadata.
The caller owns the role-bound selection of the first three or five blocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from cfeg.analysis.metadata_prior_source import m_features

Q_DIMENSIONS = 15
M_DIMENSIONS = 2
Q2_INDICES = (1, 2)
SCALER_SD_MIN = 1e-12


def _real_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    return np.asarray(array, dtype=np.float64)


def _boolean_array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "b" or array.shape != shape:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return array


def _binary_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be the integer 0 or 1")  # noqa: TRY004 -- validation API
    if value not in (0, 1):
        raise ValueError(f"{name} must be the integer 0 or 1")
    return int(value)


def _ids(values: Sequence[int], name: str, *, unique: bool) -> tuple[int, ...]:
    try:
        items = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of positive integer IDs") from error
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0
        for value in items
    ):
        raise ValueError(f"{name} must contain positive integer IDs")
    result = tuple(int(value) for value in items)
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def support_q_mask(support, mask, interface: int, order: int, frequencies) -> np.ndarray:
    """Return the frozen 15 Q features without accepting numerical impedance.

    ``support`` is ``[k, 12, 5, 8, N]`` for k=3 or 5 and ``mask`` is a
    strict boolean ``[k, 8]`` array.  The mask describes only those k blocks.
    Equations match the previous Q reference, including its native 250 Hz
    sinusoidal basis and repeat-correlation convention; no proxy is computed.
    """
    x = _real_array(support, "support")
    if (
        x.ndim != 5
        or x.shape[0] not in (3, 5)
        or x.shape[1:4] != (12, 5, 8)
        or x.shape[-1] < 2
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid native support geometry or nonfinite support")
    observed = _boolean_array(mask, (len(x), 8), "mask")
    interface = _binary_integer(interface, "interface")
    order = _binary_integer(order, "order")
    frequency = _real_array(frequencies, "frequencies")
    if frequency.shape != (12,) or not np.isfinite(frequency).all() or np.any(frequency <= 0):
        raise ValueError("frequencies must contain 12 finite positive values")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            power = np.mean(x**2, axis=(0, 1, 4))
            global_power = np.maximum(power.mean(-1, keepdims=True), 1e-12)
            off = np.zeros((5, 8))
            time = np.arange(x.shape[-1]) / 250
            for label, frequency_value in enumerate(frequency):
                reference = np.stack(
                    [
                        fn(2 * np.pi * frequency_value * harmonic * time)
                        for harmonic in range(1, 6)
                        for fn in (np.sin, np.cos)
                    ],
                    axis=1,
                )
                reference -= reference.mean(0)
                basis = np.linalg.qr(reference)[0]
                residual = x[:, label] - (x[:, label] @ basis) @ basis.T
                off += np.mean(residual**2, axis=(0, 3)) / 12
            centered = x - x.mean(-1, keepdims=True)
            correlations = []
            for left_index in range(len(x)):
                for right_index in range(left_index + 1, len(x)):
                    left, right = centered[left_index], centered[right_index]
                    norm = np.sqrt(np.sum(left**2, -1) * np.sum(right**2, -1))
                    if np.any(norm <= 1e-24):
                        raise ValueError("Zero centered support variance")
                    correlations.append(np.clip(np.sum(left * right, -1) / norm, -1, 1))
            result = np.stack(
                (
                    np.log(np.maximum(power / global_power, 1e-12)),
                    np.log(np.clip(off / np.maximum(power, 1e-12), 1e-6, 1)),
                    1 - np.mean(correlations, axis=(0, 1)),
                    np.full((5, 8), np.log(len(x))),
                    np.broadcast_to(observed.mean(0), (5, 8)),
                    np.full((5, 8), order),
                    np.full((5, 8), interface != order),
                ),
                axis=-1,
            )
            result = np.concatenate((result, np.broadcast_to(np.eye(8), (5, 8, 8))), axis=-1)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Invalid support feature arithmetic") from error
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite support features")
    return result


def metadata_features(packet) -> tuple[np.ndarray, np.ndarray]:
    """Return centered log1p mean/population SD and channel availability.

    Only explicit 3/5-block prefixes are accepted.  Zero is observed metadata;
    only NaN is missing.  This separate numeric-M API cannot affect Q features.
    """
    raw = _real_array(packet, "packet")
    if raw.ndim != 2 or raw.shape not in ((3, 8), (5, 8)):
        raise ValueError("packet must be a 3- or 5-block, 8-channel support prefix")
    features, available = m_features(raw)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite metadata features")
    return features, available


@dataclass(frozen=True, eq=False)
class FeatureScaler:
    """Immutable fit-partition statistics; transform never estimates statistics."""

    mean: np.ndarray
    scale: np.ndarray
    fit_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        mean = _real_array(self.mean, "mean")
        scale = _real_array(self.scale, "scale")
        if (
            mean.ndim != 1
            or not len(mean)
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("Scaler mean/scale must be finite feature vectors with positive scale")
        fit_ids = tuple(sorted(_ids(self.fit_ids, "fit_ids", unique=True)))
        if not fit_ids:
            raise ValueError("fit_ids must not be empty")
        # A bytes-backed copy cannot be made writable with setflags(write=True).
        object.__setattr__(self, "mean", np.frombuffer(mean.tobytes(), dtype=np.float64))
        object.__setattr__(self, "scale", np.frombuffer(scale.tobytes(), dtype=np.float64))
        object.__setattr__(self, "fit_ids", fit_ids)

    def transform(self, values, availability=None) -> np.ndarray:
        """Standardize available rows and set unavailable rows exactly to zero."""
        x = _real_array(values, "values")
        if x.ndim < 1 or x.shape[-1] != len(self.mean):
            raise ValueError("values have an incompatible feature dimension")
        observed = (
            np.ones(x.shape[:-1], dtype=bool)
            if availability is None
            else _boolean_array(availability, x.shape[:-1], "availability")
        )
        chosen = x.reshape(-1, x.shape[-1])[observed.reshape(-1)]
        if not np.isfinite(chosen).all():
            raise ValueError("Available transform rows must be finite")
        result = np.zeros(x.shape, dtype=np.float64)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result.reshape(-1, x.shape[-1])[observed.reshape(-1)] = (
                    chosen - self.mean
                ) / self.scale
        except FloatingPointError as error:
            raise ValueError("Nonfinite standardized features") from error
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized features")
        return result


def fit_scaler(values, participant_ids, fit_ids, availability=None) -> FeatureScaler:
    """Fit per-feature population statistics on selected participants only.

    Values have shape ``[participant-or-case, ..., feature]``.  Participant IDs
    correspond to the first axis and may repeat when cases are stacked; every
    row of a participant is included or excluded together.  Availability must
    match ``values.shape[:-1]`` exactly.  Nonfinite excluded/unavailable rows
    are deliberately not inspected for finiteness and do not affect the fit.
    """
    x = _real_array(values, "values")
    if x.ndim < 2 or x.shape[-1] < 1:
        raise ValueError("values must have a participant axis and a nonempty feature axis")
    ids = _ids(participant_ids, "participant_ids", unique=False)
    requested = tuple(sorted(_ids(fit_ids, "fit_ids", unique=True)))
    if len(ids) != x.shape[0] or not requested or not set(requested).issubset(ids):
        raise ValueError("fit_ids must be nonempty and present on the participant axis")
    observed = (
        np.ones(x.shape[:-1], dtype=bool)
        if availability is None
        else _boolean_array(availability, x.shape[:-1], "availability")
    )
    selected = np.array([participant in requested for participant in ids], dtype=bool)
    fit_rows = x[selected]
    chosen = fit_rows[observed[selected]]
    if not np.isfinite(chosen).all():
        raise ValueError("Available fitting rows must be finite")
    if not len(chosen):
        mean = np.zeros(x.shape[-1], dtype=np.float64)
        scale = np.ones(x.shape[-1], dtype=np.float64)
    else:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                mean = chosen.mean(axis=0)
                scale = chosen.std(axis=0, ddof=0)
        except FloatingPointError as error:
            raise ValueError("Invalid scaler fitting arithmetic") from error
        scale = np.where(scale < SCALER_SD_MIN, 1.0, scale)
    return FeatureScaler(mean, scale, requested)


def donor_map(ids, masks, orders, interface: int) -> dict[int, int]:
    """Sorted-ID cyclic donors within this partition's exact prefix strata.

    The provided IDs are the complete permitted donor partition.  Strata use
    interface, order, and every bit of the k-by-8 boolean pattern, not merely
    channel-level availability or its count.  Singletons map to themselves.
    Numeric metadata values are not accepted and cannot select donors.
    """
    participants = _ids(ids, "ids", unique=True)
    pattern = np.asarray(masks)
    if (
        pattern.ndim != 3
        or pattern.shape[0] != len(participants)
        or pattern.shape[1] not in (3, 5)
        or pattern.shape[2] != 8
    ):
        raise ValueError("masks must have shape [people, k=3 or 5, 8]")
    pattern = _boolean_array(pattern, pattern.shape, "masks")
    interface = _binary_integer(interface, "interface")
    order_values = np.asarray(orders)
    if order_values.shape != (len(participants),) or order_values.dtype.kind not in "iu":
        raise ValueError("orders must be an integer vector matching ids")
    if np.any((order_values != 0) & (order_values != 1)):
        raise ValueError("orders must contain only 0 or 1")
    groups: dict[tuple[int, int, bytes], list[int]] = {}
    for index, participant in enumerate(participants):
        # Truth values, not implementation-specific bool storage bytes, define
        # the missingness pattern (a true byte need not equal 1).
        key = (interface, int(order_values[index]), np.packbits(pattern[index]).tobytes())
        groups.setdefault(key, []).append(participant)
    result: dict[int, int] = {}
    for group in groups.values():
        group.sort()
        for index, participant in enumerate(group):
            result[participant] = group[(index + 1) % len(group)]
    return dict(sorted(result.items()))
