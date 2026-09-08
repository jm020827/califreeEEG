"""Frozen C2 pair features and weighted fit-only preprocessing, without I/O.

The caller supplies exactly the permitted k=3/5 support prefix. Numeric
impedance has a separate API and cannot enter the mask-only EEG Q features.
Pairs use their original chronological coordinates, not arbitrary relabeling.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from cfeg.analysis.task_trca_shape_features import (
    FeatureScaler,
    _binary_integer,
    _boolean_array,
    _ids,
    _real_array,
    donor_map,
)

Q_DIMENSIONS = 15
M_DIMENSIONS = 2
Q2_INDICES = (2, 4)
SCALER_SD_MIN = 1e-12

__all__ = [
    "M_DIMENSIONS",
    "Q2_INDICES",
    "Q_DIMENSIONS",
    "FeatureScaler",
    "donor_map",
    "fit_weighted_scaler",
    "metadata_features",
    "pair_coordinates",
    "pair_cross_products",
    "stale_packet",
    "support_q_mask",
]


def pair_coordinates(k) -> np.ndarray:
    """Return immutable lexicographic [n,2] pairs for the actual k prefix."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k not in (3, 5):
        raise ValueError("Pair prefix k must be integer 3 or 5")
    pairs = np.array([(i, j) for i in range(k) for j in range(i + 1, k)], dtype=np.int64)
    return np.frombuffer(pairs.tobytes(), dtype=np.int64).reshape(-1, 2)


def _support(support) -> np.ndarray:
    x = _real_array(support, "support")
    if (
        x.ndim != 5
        or x.shape[0] not in (3, 5)
        or x.shape[1:4] != (12, 5, 8)
        or x.shape[-1] < 2
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid native support geometry or nonfinite support")
    return x


def support_q_mask(support, mask, interface, order, frequencies) -> np.ndarray:
    """Return float64[5,n,15] in the exact frozen C2 feature order.

    The harmonic reference is temporally centered before QR, while the signal
    used for residual power is NOT centered. Repeat correlations alone center
    each trial in time. Channel summaries make Q channel-permutation invariant.
    """
    x = _support(support)
    observed = _boolean_array(mask, (len(x), 8), "mask")
    interface = _binary_integer(interface, "interface")
    order = _binary_integer(order, "order")
    frequency = _real_array(frequencies, "frequencies")
    if frequency.shape != (12,) or not np.isfinite(frequency).all() or np.any(frequency <= 0):
        raise ValueError("frequencies must contain 12 finite positive values")
    pairs = pair_coordinates(len(x))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            power = np.mean(x**2, axis=(1, 4))  # [block,band,channel]
            global_power = np.maximum(power.mean(axis=(0, 2)), 1e-12)
            log_power = np.log(np.maximum(power / global_power[None, :, None], 1e-12))
            off = np.zeros_like(power)
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
                off += np.mean(residual**2, axis=-1) / 12
            log_off_fraction = np.log(np.clip(off / np.maximum(power, 1e-12), 1e-6, 1))
            centered = x - x.mean(-1, keepdims=True)
            result = np.empty((5, len(pairs), Q_DIMENSIONS), dtype=np.float64)
            for index, (i, j) in enumerate(pairs):
                left, right = centered[i], centered[j]
                norm = np.sqrt(np.sum(left**2, -1) * np.sum(right**2, -1))
                if np.any(norm <= 1e-24):
                    raise ValueError("Zero centered support variance")
                disagreement = np.mean(1 - np.clip(np.sum(left * right, -1) / norm, -1, 1), axis=0)
                result[:, index, :6] = np.stack(
                    (
                        ((log_power[i] + log_power[j]) / 2).mean(-1),
                        np.sqrt(np.mean((log_power[i] - log_power[j]) ** 2, axis=-1)),
                        ((log_off_fraction[i] + log_off_fraction[j]) / 2).mean(-1),
                        np.sqrt(np.mean((log_off_fraction[i] - log_off_fraction[j]) ** 2, axis=-1)),
                        disagreement.mean(-1),
                        disagreement.std(-1, ddof=0),
                    ),
                    axis=-1,
                )
                result[:, index, 6:] = (
                    np.log(len(x)),
                    interface,
                    order,
                    float(interface != order),
                    (i + j) / (2 * (len(x) - 1)),
                    (j - i) / (len(x) - 1),
                    np.mean(observed[i] & observed[j]),
                    np.mean(observed[i] ^ observed[j]),
                    abs(int(observed[i].sum()) - int(observed[j].sum())) / 8,
                )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Invalid pair support feature arithmetic") from error
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite pair support features")
    return result


def pair_cross_products(support) -> np.ndarray:
    """Return uncentered X_i X_j.T + X_j X_i.T as [5,n,12,8,8]."""
    x = _support(support)
    pairs = pair_coordinates(len(x))
    try:
        with np.errstate(over="raise", invalid="raise"):
            left, right = x[pairs[:, 0]], x[pairs[:, 1]]
            cross = left @ right.swapaxes(-1, -2) + right @ left.swapaxes(-1, -2)
    except FloatingPointError as error:
        raise ValueError("Invalid pair cross-product arithmetic") from error
    if not np.isfinite(cross).all():
        raise ValueError("Nonfinite pair cross products")
    return cross.transpose(2, 0, 1, 3, 4).copy()


def _packet(packet) -> np.ndarray:
    raw = _real_array(packet, "packet")
    if raw.shape not in ((3, 8), (5, 8)):
        raise ValueError("packet must be a 3- or 5-block, 8-channel support prefix")
    if np.any(np.isinf(raw)) or np.any(raw < 0):
        raise ValueError("Impedance must be nonnegative finite or NaN missing")
    return raw


def metadata_features(packet) -> tuple[np.ndarray, np.ndarray]:
    """Return pair M2 and common-channel availability from exactly this prefix.

    Per-channel centering uses all observed blocks in the current k prefix;
    pair summaries then use only channels observed in both of its two blocks.
    Zero is observed. Missing pairs have two zero features and false availability.
    """
    raw = _packet(packet)
    observed = np.isfinite(raw)
    pairs = pair_coordinates(len(raw))
    result = np.zeros((len(pairs), M_DIMENSIONS), dtype=np.float64)
    available = np.zeros(len(pairs), dtype=bool)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            z = np.log1p(np.where(observed, raw, 0.0))
            count = observed.sum(0)
            mean = np.divide(z.sum(0), count, out=np.zeros(8), where=count > 0)
            centered = z - mean
            for index, (i, j) in enumerate(pairs):
                common = observed[i] & observed[j]
                available[index] = common.any()
                if available[index]:
                    result[index] = (
                        np.mean((centered[i, common] + centered[j, common]) / 2),
                        np.sqrt(np.mean((z[i, common] - z[j, common]) ** 2)),
                    )
    except FloatingPointError as error:
        raise ValueError("Invalid pair metadata feature arithmetic") from error
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite pair metadata features")
    return result, available


def stale_packet(packet) -> np.ndarray:
    """Repeat block0 values only at originally observed cells with a block0 value."""
    raw = _packet(packet)
    keep = np.isfinite(raw) & np.isfinite(raw[0])[None, :]
    return np.where(keep, raw[0][None, :], np.nan)


def fit_weighted_scaler(
    values, participant_ids, fit_ids, masses, availability=None
) -> FeatureScaler:
    """Fit globally normalized weighted population statistics on flattened rows.

    Callers assign pair masses 1/n and deduplicate M units by pid/interface/k,
    checking equal packets across bands/windows before calling. This function
    does not infer units or reweight each person's available pairs separately.
    Every row must be a real pair with positive mass; padding rows are forbidden.
    Non-fit and unavailable numeric rows cannot affect these moments.
    """
    x = _real_array(values, "values")
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("values must be flattened [rows, nonempty features]")
    ids = _ids(participant_ids, "participant_ids", unique=False)
    requested = tuple(sorted(_ids(fit_ids, "fit_ids", unique=True)))
    if len(ids) != len(x) or not requested or not set(requested).issubset(ids):
        raise ValueError("fit_ids must be nonempty and present on the row participant axis")
    mass = _real_array(masses, "masses")
    if mass.shape != (len(x),) or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("masses must be a finite strictly positive vector matching real pair rows")
    observed = (
        np.ones(len(x), dtype=bool)
        if availability is None
        else _boolean_array(availability, (len(x),), "availability")
    )
    selected = np.array([pid in requested for pid in ids]) & observed
    chosen = x[selected]
    if not np.isfinite(chosen).all():
        raise ValueError("Available fitting rows must be finite")
    if not len(chosen):
        mean, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
    else:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                normalized_mass = mass[selected] / mass[selected].sum()
                mean = np.sum(chosen * normalized_mass[:, None], axis=0)
                scale = np.sqrt(np.sum((chosen - mean) ** 2 * normalized_mass[:, None], axis=0))
        except FloatingPointError as error:
            raise ValueError("Invalid weighted scaler arithmetic") from error
        scale = np.where(scale < SCALER_SD_MIN, 1.0, scale)
    return FeatureScaler(mean, scale, requested)
