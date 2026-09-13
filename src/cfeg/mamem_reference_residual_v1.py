"""DIN-only development structure screen; no EEG, learner, or I/O."""

import numpy as np

from cfeg.mamem_events_v1 import _scalar
from cfeg.mamem_events_v2 import parse_main_trials
from cfeg.mamem_reference_probe_v1 import sample_clock_frequencies

SUBJECTS = tuple(f"S{p:03d}" for p in range(2, 12))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def extract(din, total_samples):
    """Retain only the 15 main-window sample sequences, never descriptor cells."""
    parsed = parse_main_trials(din, total_samples, include_metadata=False)
    samples = np.array([int(_scalar(cell)) for cell in din[3]], dtype=np.int64)
    records = []
    counts = [0] * 5
    for row in parsed:
        label = row["label"]
        inside = (samples - 1 >= row["start0"]) & (samples - 1 < row["end0"])
        selected = samples[inside]
        require(len(selected) == row["event_count"], "event_count")
        records.append({k: row[k] for k in
                        ("group_index", "label", "start0", "end0", "trial_end0")}
                       | {"repeat": counts[label], "event_samples": selected.tolist()})
        counts[label] += 1
    # Pure validation of three class-ordered five-frequency arrays; no bank/scorer call.
    for repeat in range(3):
        ordered = [next(r for r in records if r["label"] == c and r["repeat"] == repeat)
                   for c in range(5)]
        values = sample_clock_frequencies([r["event_samples"] for r in ordered])
        for row, frequency in zip(ordered, values):
            row["frequency_hz"] = float(frequency)
    return records


def common_logs(values, participant):
    """Never use either run of the held participant in either common bank."""
    others = np.arange(10) != participant
    return np.log(values[others]).mean(axis=(0, 3))  # (run, class)


def summarize(values):
    values = np.asarray(values)
    require(values.shape == (10, 2, 5, 3) and values.dtype.kind in "fiu", "frequency_shape")
    values = values.astype(np.float64)
    require(bool(np.isfinite(values).all() and (values > 0).all()
                 and (values < 62.5).all() and (np.diff(values, axis=2) > 0).all()),
            "frequency_range_order")
    repeatability = {}
    for run_index, run in enumerate(("a", "b")):
        frequencies = values[:, run_index]
        within = float(np.var(frequencies, axis=2, ddof=1).mean())
        between = float(np.var(frequencies.mean(axis=2), axis=0, ddof=1).mean())
        mean_noise = within / 3
        excess = float(np.sqrt(max(between - mean_noise, 0)))
        repeatability[run] = {"within_variance_hz2": within,
                             "between_mean_variance_hz2": between,
                             "repeat_mean_noise_proxy_hz2": mean_noise,
                             "excess_rms_hz": excess,
                             "eligible": bool(between >= 2 * mean_noise and excess >= 0.025)}
    people, residuals = [], []
    for p, subject in enumerate(SUBJECTS):
        means = common_logs(values, p)
        residual = np.log(values[p, 0, :, 0]) - means[0]
        target = np.exp(np.log(values[p, 1]).mean(axis=1))
        common = np.exp(means[1])
        direct = np.exp(means[1] + residual)
        errors = {"common": (common - target)**2, "direct": (direct - target)**2}
        mse = {name: float(error.mean()) for name, error in errors.items()}
        people.append({"subject": subject, "common_a_hz": np.exp(means[0]).tolist(),
                       "common_b_hz": common.tolist(), "target_b_hz": target.tolist(),
                       "direct_b_hz": direct.tolist(), "a_first_residual_log": residual.tolist(),
                       "b_mean_residual_log": (np.log(target) - means[1]).tolist(),
                       "squared_error_hz2": {name: a.tolist() for name, a in errors.items()},
                       "mse_hz2": mse, "strict_improvement": mse["common"]-mse["direct"] > 1e-12})
        residuals.append(residual)
    mse = {name: float(np.mean([p["mse_hz2"][name] for p in people]))
           for name in ("common", "direct")}
    floor = mse["common"] <= 1e-24
    gain = 0.0 if floor else 1 - mse["direct"] / mse["common"]
    improved = sum(p["strict_improvement"] for p in people)
    eligible = (all(r["eligible"] for r in repeatability.values()) and not floor
                and gain >= 0.1 and improved >= 7)
    residuals = np.array(residuals)
    denominator = float(np.sum(residuals**2))
    factor = float(np.sum(5 * residuals.mean(axis=1)**2))
    factor_summary = {"log_all_ones_energy_fraction": factor / denominator
                      if denominator > 1e-24 else 0.0,
                      "zero_residual_energy": denominator <= 1e-24,
                      "orthogonal_log_rms": float(np.sqrt(np.mean(
                          (residuals - residuals.mean(axis=1, keepdims=True))**2))),
                      "promotion_allowed": False}
    return {"repeatability": repeatability, "participants": people, "mse_hz2": mse,
            "common_error_floor": floor, "gain_fraction": gain, "improved_participants": improved,
            "factor_descriptive": factor_summary,
            "decision": "ELIGIBLE_CLASSWISE_DIRECT_TRANSFER_ONLY" if eligible
            else "RETIRE_FULL_RESIDUAL_ROUTE"}
