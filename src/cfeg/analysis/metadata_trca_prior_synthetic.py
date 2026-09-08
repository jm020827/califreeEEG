"""One bounded artificial mechanism screen; no data-loading or human-data API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfeg.analysis import metadata_trca_prior as op

SCENARIOS = ("informative", "independent", "q_sufficient", "phase_only")
ARMS = ("FULL", "ISO", "Q", "Q2", "QM", "SHAM_REFIT", "PERMUTED", "STALE", "MISSING")
SEED = 26090817
BUDGETS = (3, 5)
GAMMA, RIDGE, BOUND = 0.1, 0.1, 0.2


@dataclass(frozen=True)
class ArtificialParticipant:
    eeg: np.ndarray  # block,class,band,channel,time; not a human-data container
    impedance: np.ndarray  # block,band,channel
    oracle_q: np.ndarray  # explicitly privileged synthetic null control


def generate_participant(seed: int, scenario: str, participant: int) -> ArtificialParticipant:
    """No hidden file, model, dataset, global RNG or external metadata inputs."""
    if scenario not in SCENARIOS or participant < 0 or seed < 0:
        raise ValueError("Invalid artificial generator identity")
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, SCENARIOS.index(scenario), participant])
    )
    risk = rng.normal(size=8)
    m_risk = rng.normal(size=8) if scenario == "independent" else risk
    gain = np.exp(rng.normal(0, 0.2, size=8))
    sigma = np.full(8, 0.7) if scenario == "phase_only" else 0.7 * np.exp(0.6 * risk)
    impedance = 20 * np.exp(m_risk[None, None, :] + 0.1 * rng.normal(size=(8, 1, 8)))
    eeg = np.empty((8, 12, 1, 8, 125))
    time = np.arange(125) / 250
    for block in range(8):
        for label in range(12):
            phase = (
                rng.normal(size=8) * 0.5 * np.exp(0.5 * risk)
                if scenario == "phase_only"
                else np.zeros(8)
            )
            angle = 2 * np.pi * (9 + 0.5 * label) * time[None, :] + phase[:, None]
            signal = gain[:, None] * (np.sin(angle) + 0.3 * np.cos(2 * angle))
            signal += sigma[:, None] * rng.normal(size=(8, 125))
            eeg[block, label, 0] = signal - signal.mean(axis=-1, keepdims=True)
    oracle = (2 * np.log(sigma) if scenario == "q_sufficient" else np.zeros(8))[None, :]
    return ArtificialParticipant(eeg, impedance, oracle)


def metadata_features(packet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Class-blind support M means and variability; NaN alone denotes missing."""
    raw = np.asarray(packet)
    if raw.ndim != 3 or min(raw.shape) < 1 or np.iscomplexobj(raw):
        raise ValueError("Expected support,band,channel metadata")
    raw = np.asarray(raw, dtype=np.float64)
    if np.isinf(raw).any() or np.any(raw[np.isfinite(raw)] <= 0):
        raise ValueError("Observed metadata must be finite positive; NaN is missing")
    valid = np.isfinite(raw)
    count = valid.sum(axis=0)
    logm = np.log(np.where(valid, raw, 1.0))
    mean = logm.sum(axis=0) / np.maximum(count, 1)
    available = count > 0
    center = (mean * available).sum(axis=-1, keepdims=True) / np.maximum(
        available.sum(axis=-1, keepdims=True), 1
    )
    variance = np.sum(np.where(valid, (logm - mean[None]) ** 2, 0), axis=0)
    std = np.sqrt(variance / np.maximum(count, 1))
    features = np.stack([mean - center, std], axis=-1)
    return np.where(available[..., None], features, 0), available


def support_features(support: np.ndarray, packet: np.ndarray, oracle_q: np.ndarray) -> np.ndarray:
    """Inputs are support only; no future-repeat/query argument exists."""
    x = np.asarray(support, dtype=np.float64)
    if x.ndim != 5 or x.shape[0] not in BUDGETS or not np.isfinite(x).all():
        raise ValueError("Invalid support feature input")
    if x.shape[1:] != (12, 1, 8, 125):
        raise ValueError("Only the frozen artificial geometry is implemented")
    metadata_features(packet)
    if packet.shape != (x.shape[0], 1, 8):
        raise ValueError("Support M shape mismatch")
    oracle = np.asarray(oracle_q, dtype=float)
    if oracle.shape != (1, 8) or not np.isfinite(oracle).all():
        raise ValueError("Invalid artificial oracle control")
    power = np.mean(x**2, axis=(0, 1, 4))
    global_power = np.maximum(power.mean(axis=-1, keepdims=True), 1e-12)
    off_power = np.zeros((1, 8))
    time = np.arange(125) / 250
    for label in range(12):
        freq = 9 + 0.5 * label
        reference = np.stack(
            [
                fn(2 * np.pi * freq * harmonic * time)
                for harmonic in (1, 2)
                for fn in (np.sin, np.cos)
            ],
            axis=1,
        )
        reference -= reference.mean(axis=0, keepdims=True)
        basis = np.linalg.qr(reference)[0]
        trial = x[:, label]
        residual = trial - (trial @ basis) @ basis.T
        off_power += np.mean(residual**2, axis=(0, 3)) / 12
    correlations = []
    centered = x - x.mean(axis=-1, keepdims=True)
    for left in range(len(x)):
        for right in range(left + 1, len(x)):
            a, b = centered[left], centered[right]
            denominator = np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))
            if np.any(denominator <= 1e-24):
                raise ValueError("Zero support channel variance")
            correlations.append(np.clip(np.sum(a * b, axis=-1) / denominator, -1, 1))
    disagreement = 1 - np.mean(correlations, axis=(0, 1))
    return np.stack(
        [
            np.log(np.maximum(power / global_power, 1e-12)),
            np.log(np.clip(off_power / np.maximum(power, 1e-12), 1e-6, 1)),
            disagreement,
            np.full_like(power, np.log(len(x))),
            np.isfinite(packet).mean(axis=0),
            oracle,
        ],
        axis=-1,
    )


def proxy_target(support: np.ndarray, future: np.ndarray) -> np.ndarray:
    x, later = np.asarray(support), np.asarray(future)
    if x.ndim != 5 or later.shape != x.shape[1:]:
        raise ValueError("Invalid future-repeat proxy geometry")
    if not np.isfinite(x).all() or not np.isfinite(later).all():
        raise ValueError("Nonfinite proxy input")
    scale = np.maximum(np.mean(x * x, axis=(0, 1, 3, 4)), 1e-12)[:, None]
    error = np.mean((later - x.mean(axis=0)) ** 2, axis=(0, 3))
    return np.log(np.maximum(error / scale, 1e-12))


@dataclass(frozen=True)
class LinearModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale) @ self.coefficient + self.intercept

    def record(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "intercept": self.intercept,
        }


def fit_linear(features: np.ndarray, target: np.ndarray, *, intercept: bool) -> LinearModel:
    x, y = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) == 0:
        raise ValueError("Invalid regression shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite regression input")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-12, 1, scale)
    z = (x - mean) / scale
    bias = float(y.mean()) if intercept else 0.0
    coefficient = np.linalg.solve(
        z.T @ z / len(x) + RIDGE * np.eye(x.shape[1]), z.T @ (y - bias) / len(x)
    )
    return LinearModel(mean.copy(), scale.copy(), coefficient, bias)


def derangement(indices: np.ndarray, seed: int) -> dict[int, int]:
    ids = np.asarray(indices)
    if ids.ndim != 1 or len(ids) < 2 or len(np.unique(ids)) != len(ids):
        raise ValueError("Derangement needs distinct participant IDs")
    order = np.random.default_rng(seed).permutation(ids)
    return {int(a): int(b) for a, b in zip(order, np.roll(order, 1))}


def _features(data: list[ArtificialParticipant], k: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([support_features(p.eeg[:k], p.impedance[:k], p.oracle_q) for p in data]),
        np.stack([metadata_features(p.impedance[:k])[0] for p in data]),
    )


def fit_fold(
    data: list[ArtificialParticipant], fit_ids: np.ndarray, scenario_index: int, fold: int
):
    """Receive fit participants only; evaluation arrays cannot enter this function."""
    if len(data) != len(fit_ids) or len(set(map(int, fit_ids))) != len(data):
        raise ValueError("Invalid fit participant membership")
    if np.any(np.asarray(fit_ids) % 3 == fold):
        raise ValueError("Evaluation participant in fitting set")
    index_map = derangement(fit_ids, SEED + 1000 * scenario_index + 10 * fold + 1)
    local = {int(pid): i for i, pid in enumerate(fit_ids)}
    donor = [local[index_map[int(pid)]] for pid in fit_ids]
    qrows, mrows, shamrows, targets = [], [], [], []
    for k in BUDGETS:
        q, m = _features(data, k)
        qrows.append(q.reshape(-1, 6))
        mrows.append(m.reshape(-1, 2))
        shamrows.append(m[donor].reshape(-1, 2))
        targets.append(np.stack([proxy_target(p.eeg[:k], p.eeg[5]) for p in data]).ravel())
    qx, mx, sx, y = map(np.concatenate, (qrows, mrows, shamrows, targets))
    qmodel = fit_linear(qx, y, intercept=True)
    residual = y - qmodel.predict(qx)
    models = {
        "Q": qmodel,
        "Q2": fit_linear(qx, residual, intercept=True),
        "QM": fit_linear(mx, residual, intercept=False),
        "SHAM_REFIT": fit_linear(sx, residual, intercept=False),
    }
    receipt = {
        "fit_ids": list(map(int, fit_ids)),
        "rows": len(y),
        "derangement": index_map,
        "models": {k: v.record() for k, v in models.items()},
    }
    return models, receipt


def priors_for_participant(p: ArtificialParticipant, donor: ArtificialParticipant, k: int, models):
    support, packet = p.eeg[:k], p.impedance[:k]
    q = support_features(support, packet, p.oracle_q)
    mf, available = metadata_features(packet)
    perm, perm_available = metadata_features(donor.impedance[:k])
    stale, stale_available = metadata_features(np.repeat(packet[:1], k, axis=0))
    base_prediction = models["Q"].predict(q)
    base_prior = op.trace_normalize(np.exp(np.clip(base_prediction, -3, 3)))
    ones = np.ones_like(base_prior)
    priors = {"FULL": ones, "ISO": ones, "Q": base_prior}
    proxy = {"FULL": None, "ISO": None, "Q": base_prediction}
    residuals = {
        "Q2": (models["Q2"].predict(q), np.ones_like(available)),
        "QM": (models["QM"].predict(mf), available),
        "SHAM_REFIT": (models["SHAM_REFIT"].predict(perm), available & perm_available),
        "PERMUTED": (models["QM"].predict(perm), available & perm_available),
        "STALE": (models["QM"].predict(stale), stale_available),
        "MISSING": (np.zeros_like(base_prediction), np.zeros_like(available)),
    }
    for name, (delta, mask) in residuals.items():
        clipped = np.where(mask, np.clip(delta, -BOUND, BOUND), 0)
        priors[name] = op.residual_prior(base_prior, delta, mask, bound=BOUND)
        proxy[name] = base_prediction + clipped
    return priors, proxy


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    summary, contrasts = [], []
    for scenario in SCENARIOS:
        for k in BUDGETS:
            selected = [r for r in rows if r["scenario"] == scenario and r["k"] == k]
            q_rows = {r["participant"]: r for r in selected if r["arm"] == "Q"}
            for arm in ARMS:
                group = [r for r in selected if r["arm"] == arm]
                differences = np.array(
                    [r["accuracy"] - q_rows[r["participant"]]["accuracy"] for r in group]
                )
                changes = sum(
                    int(
                        np.count_nonzero(
                            np.asarray(r["predictions"]) != q_rows[r["participant"]]["predictions"]
                        )
                    )
                    for r in group
                )
                summary.append(
                    {
                        "scenario": scenario,
                        "k": k,
                        "arm": arm,
                        "accuracy": float(np.mean([r["accuracy"] for r in group])),
                        "help_vs_Q": int((differences > 0).sum()),
                        "tie_vs_Q": int((differences == 0).sum()),
                        "harm_vs_Q": int((differences < 0).sum()),
                        "prediction_changes_vs_Q": changes,
                        "proxy_mse": None
                        if arm in ("FULL", "ISO")
                        else float(np.mean([r["proxy_mse"] for r in group])),
                    }
                )
            for comparator in ("Q", "Q2", "SHAM_REFIT"):
                baseline = {r["participant"]: r for r in selected if r["arm"] == comparator}
                actual = {r["participant"]: r for r in selected if r["arm"] == "QM"}
                diff = np.array([actual[i]["accuracy"] - baseline[i]["accuracy"] for i in baseline])
                contrasts.append(
                    {
                        "scenario": scenario,
                        "k": k,
                        "comparator": comparator,
                        "accuracy_delta": float(diff.mean()),
                        "proxy_mse_delta": float(
                            np.mean(
                                [
                                    actual[i]["proxy_mse"] - baseline[i]["proxy_mse"]
                                    for i in baseline
                                ]
                            )
                        ),
                        "help": int((diff > 0).sum()),
                        "tie": int((diff == 0).sum()),
                        "harm": int((diff < 0).sum()),
                    }
                )
    lookup = {(r["scenario"], r["k"], r["arm"]): r for r in summary}
    info = lambda arm: lookup["informative", 3, arm]
    proxy_ok = info("QM")["proxy_mse"] < info("Q")["proxy_mse"]
    accuracy_ok = all(
        info("QM")["accuracy"] > info(arm)["accuracy"] for arm in ("Q", "Q2", "SHAM_REFIT")
    )
    null_ok = all(
        abs(lookup[s, 3, "QM"]["accuracy"] - lookup[s, 3, "Q"]["accuracy"]) <= 1 / 60
        for s in ("independent", "q_sufficient")
    )
    screen = {
        "informative_proxy": proxy_ok,
        "informative_accuracy": accuracy_ok,
        "null_accuracy_bound": null_ok,
        "status": "SYNTHETIC_SCREEN_SUPPORTED"
        if proxy_ok and accuracy_ok and null_ok
        else "SYNTHETIC_CANDIDATE_NOT_ESTABLISHED",
        "human_promotion": False,
    }
    return summary, contrasts, screen


def run_suite() -> dict:
    """The one declared artificial suite, called only after contract/code freeze."""
    rows, fits = [], []
    for si, scenario in enumerate(SCENARIOS):
        data = [generate_participant(SEED, scenario, pid) for pid in range(24)]
        for fold in range(3):
            fit_ids = np.array([i for i in range(24) if i % 3 != fold])
            eval_ids = np.array([i for i in range(24) if i % 3 == fold])
            models, receipt = fit_fold([data[i] for i in fit_ids], fit_ids, si, fold)
            perm = derangement(eval_ids, SEED + 1000 * si + 10 * fold + 2)
            fits.append(
                {
                    "scenario": scenario,
                    "fold": fold,
                    "eval_ids": eval_ids.tolist(),
                    "eval_derangement": perm,
                    **receipt,
                }
            )
            for pid in eval_ids:
                p = data[pid]
                for k in BUDGETS:
                    prior, proxy = priors_for_participant(p, data[perm[int(pid)]], k, models)
                    target = proxy_target(p.eeg[:k], p.eeg[5])  # evaluation only, after freeze
                    query = p.eeg[6:8].reshape(24, 1, 8, 125)
                    labels = np.tile(np.arange(12), 2)
                    predictions = {}
                    for arm in ARMS:
                        model = op.fit_trca(p.eeg[:k], prior[arm], 0.0 if arm == "FULL" else GAMMA)
                        scores, _ = op.score_trca(model, query, np.ones(1))
                        predicted = scores.argmax(axis=-1)
                        predictions[arm] = predicted
                        diagnostics = model.diagnostics
                        if arm == "MISSING" and (
                            not np.array_equal(prior[arm], prior["Q"])
                            or not np.array_equal(predicted, predictions["Q"])
                        ):
                            raise AssertionError("Numeric-denial M must reproduce Q exactly")
                        rows.append(
                            {
                                "scenario": scenario,
                                "fold": fold,
                                "participant": int(pid),
                                "k": k,
                                "labels": 12 * k,
                                "arm": arm,
                                "accuracy": float(np.mean(predicted == labels)),
                                "proxy_mse": None
                                if proxy[arm] is None
                                else float(np.mean((proxy[arm] - target) ** 2)),
                                "predictions": predicted.tolist(),
                                "prior": prior[arm].tolist(),
                                "trace_error": float(np.max(np.abs(prior[arm].sum(-1) - 8))),
                                "min_denominator_eigenvalue": float(
                                    np.min(diagnostics["denominator_min_eigenvalues"])
                                ),
                                "min_eigen_gap": float(np.min(diagnostics["top_eigenvalue_gaps"])),
                                "max_denominator_norm_error": float(
                                    np.max(diagnostics["denominator_norm_errors"])
                                ),
                                "penalty_trace": diagnostics["penalty_trace"],
                            }
                        )
    summary, contrasts, screen = summarize(rows)
    attainment = []
    for scenario in SCENARIOS:
        for pid in range(24):
            for arm in ARMS:
                by_k = {
                    r["k"]: r["accuracy"]
                    for r in rows
                    if r["scenario"] == scenario and r["participant"] == pid and r["arm"] == arm
                }
                first = next((k for k in BUDGETS if by_k[k] >= 0.8), None)
                attainment.append(
                    {
                        "scenario": scenario,
                        "participant": pid,
                        "arm": arm,
                        "first_observed_k": first,
                        "labels": None if first is None else 12 * first,
                    }
                )
    return {
        "study_id": "metadata-trca-prior-synthetic-v1",
        "seed": SEED,
        "scope": "artificial engineering; no actual calibration savings",
        "screen": screen,
        "summary": summary,
        "contrasts": contrasts,
        "rows": rows,
        "fits": fits,
        "attainment": attainment,
        "human_data_access": False,
        "k0_measured": False,
        "calibration_savings_established": False,
    }
