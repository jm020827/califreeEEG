"""Cold independent audit of PINNED generated temporal integration artifacts only.

No producer imports, raw human paths or training calls. Evidence starts at saved
source Q/S/C, and at saved generated evaluation templates/query/native filters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_temporal_audit as independent

FILES = ("start.json", "parity.json", "source.npz", "model.json", "evaluation.npz")


def require(ok, message):
    if not ok:
        raise ValueError("COLD_GENERATED_AUDIT_FAILURE: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal(filters, templates, query, weights, *, temporal):
    x = np.einsum("bik,nbit->nbkt", filters, query)
    t = np.einsum("bik,cbit->cbkt", filters, templates)
    if temporal:
        x, t = x - x.mean(-1, keepdims=True), t - t.mean(-1, keepdims=True)
    x, t = x.reshape(len(query), 5, -1), t.reshape(12, 5, -1)
    if not temporal:
        x, t = x - x.mean(-1, keepdims=True), t - t.mean(-1, keepdims=True)
    require(np.all(np.linalg.norm(x, axis=-1) > 0), "literal query variance")
    require(np.all(np.linalg.norm(t, axis=-1) > 0), "literal template variance")
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    corr = np.clip(np.einsum("nbt,cbt->nbc", x, t), -1, 1)
    return np.einsum("nbc,b->nc", corr, weights), corr


def check_evaluation(data, pipeline, expected_ids):
    require(str(data["schema"]) == independent.SCHEMA, "evaluation schema")
    require(str(data["score_schema"]) == independent.SCORE_SCHEMA, "evaluation score schema")
    require(data["arms"].tolist() == list(independent.ARMS), "exact ten-arm ordering")
    require(data["keys"].tolist() == [[pid, 0, 17, 3] for pid in expected_ids], "frozen eval grid")
    require(not set(expected_ids) & set(pipeline["fit_ids"]), "evaluation participant overlap")
    require(data["orders"].tolist() == [0] * len(expected_ids), "generated order")
    donor_ids = independent.legacy.independent_donors(
        expected_ids, np.isfinite(data["packet"]), data["orders"].tolist(), 0
    )
    lookup = {pid: j for j, pid in enumerate(expected_ids)}
    errors = {
        "prior": 0.0,
        "projector": 0.0,
        "score": 0.0,
        "native": 0.0,
        "centered_full": 0.0,
        "statistics": 0.0,
    }
    comparisons = 0
    integer_count_comparisons = 0

    def compare(name, a, b):
        nonlocal comparisons
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=0)
        errors[name] = max(errors[name], float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
        comparisons += 1

    def compare_scores(name, a, b):
        nonlocal integer_count_comparisons
        compare(name, a, b)
        pa, pb = np.asarray(a).argmax(-1), np.asarray(b).argmax(-1)
        np.testing.assert_array_equal(pa, pb)
        truth = np.tile(np.arange(12), 4)
        require(
            int(np.count_nonzero(pa == truth)) == int(np.count_nonzero(pb == truth)),
            "independent generated integer correct counts",
        )
        integer_count_comparisons += 1

    for j, pid in enumerate(expected_ids):
        packet, weights = data["packet"][j], data["weights"][j]
        m, available = independent.legacy.independent_metadata(packet)
        np.testing.assert_allclose(m, data["m"][j], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(available, data["available"][j])
        qmask = np.broadcast_to(np.isfinite(packet).mean(0), (5, 8))
        np.testing.assert_array_equal(data["q"][j, ..., 4], qmask)
        templates, query = data["templates"][j], data["query"][j]
        require(
            templates.shape == (12, 5, 8, 17) and query.shape == (48, 5, 8, 17),
            "generated raw shapes",
        )
        require(
            np.isfinite(templates).all() and np.isfinite(query).all(), "generated raw finiteness"
        )
        t = templates / np.max(np.abs(templates), axis=(-2, -1), keepdims=True)
        x = query / np.max(np.abs(query), axis=(-2, -1), keepdims=True)
        mt, mx = t.mean(-1), x.mean(-1)
        tc, xc = t - mt[..., None], x - mx[..., None]
        expected_stats = {
            "query_gram": xc @ xc.swapaxes(-1, -2),
            "template_gram": tc @ tc.swapaxes(-1, -2),
            "cross_gram": np.einsum("nbit,cbjt->ncbij", xc, tc),
        }
        for name, value in expected_stats.items():
            compare("statistics", value, data["statistics_" + name][j])
            compare("statistics", value, data["native_statistics_" + name][j])
        compare("statistics", mx, data["native_statistics_query_mean"][j])
        compare("statistics", mt, data["native_statistics_template_mean"][j])
        require(int(data["statistics_samples"][j]) == 17, "temporal sample count")
        require(int(data["native_statistics_samples"][j]) == 17, "native sample count")
        stats = {name: data["statistics_" + name][j] for name in independent.STATS}
        stats["samples"] = 17
        for centered in (False, True):
            score, corr = literal(
                data["native_filters"][j], templates, query, weights, temporal=centered
            )
            compare_scores(
                "centered_full" if centered else "native", score, data["scores"][j, int(centered)]
            )
            if not centered:
                compare("native", corr, data["native_full_correlations"][j])
        donor_m = data["m"][lookup[donor_ids[pid]]]
        stale_m, stale_available = independent.legacy.independent_metadata(
            np.repeat(packet[:1], 3, 0)
        )
        for a, arm in enumerate(independent.ARMS[2:]):
            r = independent.independent_prior(
                data["q"][j],
                m,
                available,
                pipeline,
                arm,
                donor_m=donor_m,
                stale_m=stale_m,
                stale_available=stale_available,
            )
            compare("prior", r, data["r"][j, a])
            f = independent.independent_projectors(data["s"][j], data["c"][j], r)
            compare("projector", f, data["projectors"][j, a])
            score, _ = independent.independent_scores(f, stats, weights)
            compare_scores("score", score, data["scores"][j, a + 2])
        np.testing.assert_array_equal(data["scores"][j, 3], data["scores"][j, 9])
    return {
        "status": "GENERATED_EVALUATION_AUDIT_PASS",
        "max_abs_errors": errors,
        "array_comparisons": comparisons,
        "integer_count_comparisons": integer_count_comparisons,
        "participants": len(expected_ids),
        "scope": "generated templates/query/Grams, both FULL scorers, M/donors/R/F/scores; no native support-fitting/raw human audit",
    }


def audit(root, expected_receipt_sha):
    start_time = time.perf_counter()
    require(
        not (root / "failure.json").exists(), "producer failure takes precedence over completion"
    )
    require(sha(root / "receipt.json") == expected_receipt_sha, "receipt hash")
    receipt = json.loads((root / "receipt.json").read_text())
    require(receipt["kind"] == "GENERATED_TEMPORAL_ENGINEERING", "generated-only kind")
    require(
        receipt["status"] == "GENERATED_INTEGRATION_COMPLETE_COLD_AUDIT_PENDING", "completion state"
    )
    require(set(receipt["artifacts"]) == set(FILES), "exact generated artifact basenames")
    for name in FILES:
        require(not (root / name).is_symlink(), "artifact symlink")
        require(sha(root / name) == receipt["artifacts"][name]["sha256"], "artifact hash: " + name)
        require(
            (root / name).stat().st_size == receipt["artifacts"][name]["bytes"], "artifact size"
        )
    start = json.loads((root / "start.json").read_text())
    require(start["kind"] == receipt["kind"] and start["seed"] == 20260909, "start kind/seed")
    require(
        start["source_ids"] == list(range(21001, 21007))
        and start["evaluation_ids"] == [31001, 31002],
        "fixed generated IDs",
    )
    require(
        start["human_data_access"] is False and start["held60_access"] is False, "generated scope"
    )
    for name, digest in start["code_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "code path scope")
        require(sha(ROOT / name) == digest, "runtime code changed: " + name)
    model = json.loads((root / "model.json").read_text())
    with np.load(root / "source.npz", allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}
    training = independent.audit_training(
        source, model, start["source_ids"], start["evaluation_ids"]
    )
    with np.load(root / "evaluation.npz", allow_pickle=False) as archive:
        evaluation = check_evaluation(
            {n: archive[n] for n in archive.files}, model["pipeline"], start["evaluation_ids"]
        )
    parity = json.loads((root / "parity.json").read_text())
    require(parity["actual_head_count"] == 4 and parity["steps_per_head"] == 200, "four full heads")
    cuda = {"executed": parity["cuda"]["executed"]}
    if cuda["executed"]:
        require(
            set(parity["cpu_scores"])
            == set(parity["cuda"]["scores"])
            == {"Q", "Q2", "QM", "SHAM_REFIT"},
            "exact four parity prediction arms",
        )
        cuda["score_errors"] = {}
        for arm in ("Q", "Q2", "QM", "SHAM_REFIT"):
            a, b = np.array(parity["cpu_scores"][arm]), np.array(parity["cuda"]["scores"][arm])
            np.testing.assert_allclose(a, b, atol=1e-9, rtol=0)
            np.testing.assert_array_equal(a.argmax(-1), b.argmax(-1))
            cuda["score_errors"][arm] = float(np.max(np.abs(a - b)))
            left = (
                parity["cpu_pipeline"]["Q"]
                if arm == "Q"
                else parity["cpu_pipeline"]["residuals"][arm]
            )
            right = (
                parity["cuda"]["pipeline"]["Q"]
                if arm == "Q"
                else parity["cuda"]["pipeline"]["residuals"][arm]
            )
            require(
                left["steps"] == right["steps"] == len(left["trace"]) == len(right["trace"]) == 200,
                "CPU/CUDA head trace count",
            )
            np.testing.assert_allclose(
                left["coefficients"], right["coefficients"], atol=1e-7, rtol=0
            )
    return {
        "status": "GENERATED_COLD_AUDIT_PASS",
        "receipt_sha256": expected_receipt_sha,
        "training": training,
        "evaluation": evaluation,
        "cuda_stored_array_check": cuda,
        "seconds": time.perf_counter() - start_time,
        "limitations": [
            "saved Q/S/C source evidence, no independent EEG-Q15/nativefit/Adam replay",
            "not an actual human access/completion provenance audit",
            "no metadata efficacy or calibration conclusion",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    output = args.root / "cold_audit.json"
    with output.open("x") as stream:
        try:
            result = audit(args.root.resolve(), args.receipt_sha256)
        except Exception as exc:
            json.dump(
                {"status": "GENERATED_COLD_AUDIT_FAILURE", "error": str(exc)}, stream, indent=2
            )
            stream.write("\n")
            stream.flush()
            output.chmod(0o400)
            raise
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    output.chmod(0o400)
    print(json.dumps(result))
