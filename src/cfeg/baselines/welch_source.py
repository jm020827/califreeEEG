"""Reader-free source-only Welch/SVM core, NOT an author-baseline reproduction.

Caller owns filtering, channel/time selection, units, and trustworthy participant
IDs. No human-data runner or acquisition-M interface is provided. See
docs/welch_source_baseline_v1_contract.md for conventions and remaining gaps.
scikit-learn is an optional dependency, required only for source fitting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


@dataclass(frozen=True)
class WelchSpec:
    """All spectral choices explicit; no implicit MATLAB/SciPy defaults."""

    fs: float
    samples: int
    nperseg: int
    noverlap: int
    nfft: int

    def __post_init__(self):
        _require(type(self.fs) in (int, float) and np.isfinite(self.fs) and self.fs > 0,
                 "invalid_fs")
        _require(all(type(v) is int for v in
                     (self.samples, self.nperseg, self.noverlap, self.nfft)), "integer_spec")
        _require(2 <= self.nperseg <= self.samples and self.nfft >= self.nperseg,
                 "segment_fft_length")
        _require(0 <= self.noverlap < self.nperseg, "overlap_range")


def welch_power(windows, spec: WelchSpec) -> np.ndarray:
    """Per-trial mean removal, symmetric Hamming, mean one-sided PSD density.

    Only complete segments enter the FFT; all samples enter the trial mean.
    No per-segment detrend, filtering, log, gain normalization or statistics
    shared across trials. Units: input**2/Hz.
    Extreme inputs that overflow are rejected rather than silently rescaled.
    """
    _require(isinstance(spec, WelchSpec), "spec_required")
    x = np.asarray(windows)
    _require(x.ndim == 2 and x.shape[0] > 0 and x.shape[1] == spec.samples
             and x.dtype.kind in "fiu", "windows_shape_dtype")
    x = np.asarray(x, dtype=np.float64)
    _require(bool(np.isfinite(x).all()), "windows_finite")
    # Test raw equality before summation: mean(0.1 repeated) need not equal 0.1.
    _require(bool(np.any(x != x[:, :1], axis=1).all()), "constant_trial")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            centered = x - x.mean(axis=1, keepdims=True)
            _require(bool(np.any(centered != 0, axis=1).all()), "constant_trial")
            _, power = welch(
                centered, fs=spec.fs, window=np.hamming(spec.nperseg),
                nperseg=spec.nperseg, noverlap=spec.noverlap, nfft=spec.nfft,
                detrend=False, return_onesided=True, scaling="density",
                axis=1, average="mean",
            )
    except FloatingPointError as exc:
        raise ValueError("spectral_overflow") from exc
    _require(bool(np.isfinite(power).all() and (power >= 0).all()
                  and np.any(power > 0, axis=1).all()), "invalid_power")
    return power


def _ids(values, n: int | None = None) -> tuple[str, ...]:
    _require(isinstance(values, (list, tuple, np.ndarray)), "ids_sequence")
    if isinstance(values, np.ndarray):
        _require(values.ndim == 1, "ids_dimension")
    ids = tuple(values)
    _require(bool(ids) and (n is None or len(ids) == n), "ids_length")
    _require(all(isinstance(v, str) and v and v == v.strip() for v in ids),
             "ids_nonempty_strings")
    return ids


@dataclass(frozen=True)
class SourceWelchSVM:
    """Immutable exported linear heads; query prediction does not call a fitter.

    Margins are NOT calibrated probabilities. Role checks validate declared IDs,
    not provenance of waveform bytes; a future acquisition runner must bind them.
    """

    spec: WelchSpec
    classes: tuple[int, ...]
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]

    def __post_init__(self):
        _require(isinstance(self.spec, WelchSpec), "spec_required")
        _require(isinstance(self.classes, (list, tuple)) and len(self.classes) >= 2
                 and all(type(c) is int and 0 <= c <= 1_000_000 for c in self.classes),
                 "classes_integer")
        classes = tuple(self.classes)
        _require(classes == tuple(sorted(set(classes))), "classes_unique_sorted")
        source, target = _ids(self.source_ids), _ids(self.target_ids)
        _require(len(source) >= 2 and len(set(source)) == len(source)
                 and len(set(target)) == len(target) and set(source).isdisjoint(target),
                 "model_roles")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "source_ids", tuple(sorted(source)))
        object.__setattr__(self, "target_ids", tuple(sorted(target)))
        dimension = self.spec.nfft // 2 + 1
        for name, shape in (("mean", (dimension,)), ("scale", (dimension,)),
                            ("coefficients", (len(classes), dimension)),
                            ("intercepts", (len(classes),))):
            value = np.asarray(getattr(self, name))
            _require(value.shape == shape and value.dtype.kind in "fiu", "model_state_shape")
            value = np.asarray(value, dtype=np.float64)
            _require(bool(np.isfinite(value).all()), "model_state_finite")
            if name == "scale":
                _require(bool((value > 0).all()), "model_scale_positive")
            frozen = (tuple(tuple(float(v) for v in row) for row in value)
                      if value.ndim == 2 else tuple(float(v) for v in value))
            object.__setattr__(self, name, frozen)

    def decision_function(self, windows, participant_ids) -> np.ndarray:
        ids = _ids(participant_ids)
        _require(set(ids).isdisjoint(self.source_ids) and set(ids) <= set(self.target_ids),
                 "query_role")
        features = welch_power(windows, self.spec)
        _require(features.shape[0] == len(ids), "ids_length")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                normalized = (features - np.asarray(self.mean)) / np.asarray(self.scale)
                result = normalized @ np.asarray(self.coefficients).T + self.intercepts
        except FloatingPointError as exc:
            raise ValueError("prediction_overflow") from exc
        _require(bool(np.isfinite(result).all()), "prediction_finite")
        return result

    def predict(self, windows, participant_ids) -> np.ndarray:
        scores = self.decision_function(windows, participant_ids)
        return np.asarray(self.classes)[np.argmax(scores, axis=1)]


def fit_source(windows, labels, participant_ids, *, target_ids, classes,
               spec: WelchSpec) -> SourceWelchSVM:
    """Fit scaling and one binary linear SVC per class using only source arrays.

    Requires at least two source people per declared class. Target arrays are not
    arguments. Source standardization and margin OVR are deliberate conventions,
    not faithful reproduction of the author's probability-based LIBSVMFast.
    """
    y = np.asarray(labels)
    _require(y.ndim == 1 and y.size > 0 and y.dtype.kind in "iu", "labels_integer")
    _require(isinstance(classes, (list, tuple)) and len(classes) >= 2
             and all(type(c) is int and 0 <= c <= 1_000_000 for c in classes),
             "classes_integer")
    expected = tuple(sorted(classes))
    _require(len(set(expected)) == len(expected) and set(y.tolist()) == set(expected),
             "class_coverage")
    source = _ids(participant_ids, len(y))
    target = _ids(target_ids)
    _require(len(set(target)) == len(target), "target_ids_unique")
    _require(set(source).isdisjoint(target), "source_target_overlap")
    _require(len(set(source)) >= 2, "source_people")
    _require(all(len({source[i] for i in np.flatnonzero(y == c)}) >= 2 for c in expected),
             "class_people_coverage")
    features = welch_power(windows, spec)
    _require(features.shape[0] == y.size, "labels_length")

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    heads, offsets = [], []
    try:
        with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise"):
            warnings.simplefilter("error", ConvergenceWarning)
            scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            z = scaler.fit_transform(features)
            _require(bool(np.isfinite(z).all() and np.isfinite(scaler.mean_).all()
                          and np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all()),
                     "scaling_finite")
            for c in expected:
                binary = np.where(y == c, 1, -1)
                model = SVC(C=1.0, kernel="linear", probability=False, tol=1e-6,
                            max_iter=100_000, class_weight=None, shrinking=False,
                            cache_size=64)
                model.fit(z, binary)
                _require(model.fit_status_ == 0, "svm_not_converged")
                _require(np.array_equal(model.classes_, [-1, 1]), "binary_class_orientation")
                heads.append(tuple(float(v) for v in model.coef_[0]))
                offsets.append(float(model.intercept_[0]))
    except (ConvergenceWarning, FloatingPointError) as exc:
        raise ValueError("source_fit_numeric_failure") from exc
    _require(bool(np.isfinite(heads).all() and np.isfinite(offsets).all()), "heads_finite")
    return SourceWelchSVM(
        spec, expected, tuple(sorted(set(source))), tuple(sorted(target)),
        tuple(float(v) for v in scaler.mean_), tuple(float(v) for v in scaler.scale_),
        tuple(heads), tuple(offsets),
    )
