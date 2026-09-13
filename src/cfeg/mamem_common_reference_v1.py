"""Fixed common-reference development comparison; no readers or trained model."""

import numpy as np

from cfeg.mamem_reference_probe_v1 import scores

SUBJECTS = tuple(f"S{p:03d}" for p in range(2, 12))
ARMS = ("NOMINAL", "COMMON")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def common_frequencies(source_b):
    values = np.asarray(source_b)
    require(values.shape == (10, 5, 3) and values.dtype.kind in "fiu", "source_shape_dtype")
    values = values.astype(np.float64)
    require(bool(np.isfinite(values).all() and (values > 0).all() and (values < 62.5).all()
                 and (np.diff(values, axis=1) > 0).all()), "source_frequency_range")
    return {subject: np.exp(np.log(values[np.arange(10) != p]).mean(axis=(0, 2))).tolist()
            for p, subject in enumerate(SUBJECTS)}


def decode_windows(eeg, windows, banks):
    require(len(windows) == 15 and set(banks) == set(ARMS), "decode_scope")
    decoded = []
    for window in windows:
        require(set(window) == {"group_index", "start0", "end0"}, "label_free_windows")
        query = eeg[125, window["start0"]:window["end0"]]
        arms = {}
        for arm in ARMS:
            result = scores(query, banks[arm])  # No label, group index, or subject to scorer.
            centered = query / np.max(np.abs(query))
            centered = centered - centered.mean()
            normalized = centered / np.linalg.norm(centered)
            projection = banks[arm].transpose(0, 2, 1) @ normalized
            require(float(np.max(np.abs(np.sum(projection**2, axis=1)-result))) <= 1e-12,
                    "projection_score")
            arms[arm] = {"projection": projection.tolist(), "scores": result.tolist(),
                         "prediction": int(np.argmax(result))}
        decoded.append(dict(window, arms=arms))
    return decoded


def evaluate(scored, labels):
    require([p["subject"] for p in scored] == list(SUBJECTS) and set(labels) == set(SUBJECTS),
            "evaluation_subjects")
    participants = []
    paired = dict.fromkeys(("both_correct", "nominal_only", "common_only", "both_wrong"), 0)
    changes = dict.fromkeys(("common_better", "nominal_better", "tied"), 0)
    for person in scored:
        subject = person["subject"]
        require(len(person["queries"]) == len(labels[subject]) == 15, "evaluation_count")
        require(all(type(label) is int and 0 <= label < 5 for label in labels[subject])
                and all(labels[subject].count(c) == 3 for c in range(5)), "evaluation_labels")
        queries = []
        for row, label in zip(person["queries"], labels[subject]):
            arms = {arm: dict(row["arms"][arm], correct=row["arms"][arm]["prediction"] == label)
                    for arm in ARMS}
            queries.append(dict(row, label=label, arms=arms))
            a, b = arms["NOMINAL"]["correct"], arms["COMMON"]["correct"]
            key = ("both_correct" if a and b else "nominal_only" if a else "common_only" if b
                   else "both_wrong")
            paired[key] += 1
        summary = {}
        for arm in ARMS:
            counts = [sum(r["arms"][arm]["correct"] for r in queries if r["label"] == c)
                      for c in range(5)]
            summary[arm] = {"correct": sum(counts), "total": 15, "per_class_correct": counts,
                            "ready": sum(counts) >= 12 and min(counts) >= 1}
        delta = summary["COMMON"]["correct"] - summary["NOMINAL"]["correct"]
        changes["common_better" if delta > 0 else "nominal_better" if delta < 0 else "tied"] += 1
        participants.append({"subject": subject, "queries": queries, "summary": summary})
    total = {}
    for arm in ARMS:
        counts = [sum(p["summary"][arm]["per_class_correct"][c] for p in participants)
                  for c in range(5)]
        ready = sum(p["summary"][arm]["ready"] for p in participants)
        total[arm] = {"correct": sum(counts), "total": 150, "accuracy": sum(counts)/150,
                      "per_class_correct": counts, "subjects_ready": ready,
                      "ready": sum(counts) >= 120 and ready >= 7}
    a, b = total["NOMINAL"]["ready"], total["COMMON"]["ready"]
    decision = ("BOTH_ZERO_SUPPORT_READY_DEV" if a and b else "COMMON_ONLY_READY_DEV" if b
                else "NOMINAL_ONLY_READY_DEV" if a else "NEITHER_BASELINE_READY_STOP")
    total.update(paired=paired, subject_changes=changes,
                 common_advantage_pp=100*(total["COMMON"]["correct"]-total["NOMINAL"]["correct"])/150,
                 decision=decision)
    return participants, total
