"""One failure-only replay of saved source statistics; no query or candidate retry.

Fixed target outer1/inner0/lambda.0001 from the preserved failure trace. No
parameter changes, source file rewrite, complete model publication or outcome
evaluation. Stop at the same native-anchor assertion and independently inspect
only that point with SciPy. This adds diagnostics, not a new efficacy attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cfeg.analysis import task_trca_shape_learning as learn
from cfeg.analysis import task_trca_shape_operator as op

SOURCE = Path("/home/whwovy/task-trca-shape-source39-v1-attempt1/source1.npz")
SOURCE_SHA = "217c37501edfe97ab49567e6341d844a5c4ee942e5463f046c34bafdacc92440"
FAILURE = SOURCE.parent / "failure.json"
FAILURE_SHA = "78e540eb45e93378219127c626437a05acecac46633ae8e1fa37924a079f37ae"
OUTPUT = Path("/home/whwovy/task-trca-shape-anchor-failure-diagnostic-v1.json")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def point_diagnostic(s, c, anchors, r, keys):
    """Inspect exactly the failed R point; no sweep or rescue."""
    with torch.no_grad():
        c = (c + c.transpose(-1, -2)) / 2
        s = (s + s.transpose(-1, -2)) / 2
        minimum = torch.linalg.eigvalsh(c)[..., 0]
        tau = 0.1 * minimum / (16 / 9)
        anchor_norm = torch.einsum("...i,...ij,...j->...", anchors, c, anchors).sqrt()
        anchor = anchors / anchor_norm.unsqueeze(-1)
        b = c + torch.diag_embed(tau.unsqueeze(-1) * r)
        lower = torch.linalg.cholesky(b)
        left = torch.linalg.solve_triangular(lower, s, upper=False)
        h = torch.linalg.solve_triangular(lower, left.transpose(-1, -2), upper=False).transpose(
            -1, -2
        )
        h = (h + h.transpose(-1, -2)) / 2
        values, vectors = torch.linalg.eigh(h)
        top = vectors[..., -1]
        w = torch.linalg.solve_triangular(
            lower.transpose(-1, -2), top.unsqueeze(-1), upper=True
        ).squeeze(-1)
        w = w / torch.einsum("...i,...ij,...j->...", w, c, w).sqrt().unsqueeze(-1)
        cosine = torch.einsum("...i,...ij,...j->...", w, c, anchor).abs()
        minimum_index = int(torch.argmin(cosine).item())
        position, band, label = np.unravel_index(minimum_index, cosine.shape)
        indices = (position, band, label)
        ss, cc, aa, bb = [v[indices].cpu().numpy() for v in (s, c, anchor, b)]
        cpu_values, cpu_vectors = la.eigh(ss, bb)
        cpu_w = cpu_vectors[:, -1]
        cpu_w /= np.sqrt(cpu_w @ cc @ cpu_w)
        cpu_cosine = abs(float(cpu_w @ cc @ aa))
        point = {
            "case_key": keys[position],
            "band_zero_based": int(band),
            "class_zero_based": int(label),
            "gpu_min_abs_c_cosine": float(cosine[indices]),
            "cpu_abs_c_cosine": cpu_cosine,
            "all_cosines_finite": bool(torch.isfinite(cosine).all()),
            "frozen_anchor_min": 1e-6,
            "gpu_roots": values[indices].cpu().tolist(),
            "cpu_roots": cpu_values.tolist(),
            "c_minimum": float(minimum[indices]),
            "tau": float(tau[indices]),
            "r": r[position, band, 0].cpu().tolist(),
            "S": ss.tolist(),
            "C": cc.tolist(),
            "native_c_normalized_anchor": aa.tolist(),
            "independent_cpu_confirms_threshold_failure": cpu_cosine <= 1e-6,
        }
        return point


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if digest(SOURCE) != SOURCE_SHA or digest(FAILURE) != FAILURE_SHA:
        raise ValueError("Exact preserved failure/source pins required")
    failure = json.loads(FAILURE.read_bytes())
    if failure["state"]["query_access_count"] != 0 or failure["state"]["outer_fold"] != 1:
        raise ValueError("Wrong failure boundary")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.16)
    with np.load(SOURCE, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    ids = sorted(set(data["keys"][:, 0].tolist()))
    validation_ids = ids[::3]
    positions = [j for j, key in enumerate(data["keys"]) if key[0] not in validation_ids]
    cases = []

    def tensor(value, dtype=torch.float64):
        return torch.tensor(np.array(value, copy=True), dtype=dtype, device="cuda")

    for j in positions:
        pid, interface, samples, k = map(int, data["keys"][j])
        packet = data["packet5"][j, :k]
        stats = op.GramStatistics(
            **{
                f.name: tensor(data[f.name][j])
                for f in fields(op.GramStatistics)
                if f.name != "samples"
            },
            samples=samples,
        )
        cases.append(
            learn.TaskCase(
                pid,
                interface,
                int(data["orders"][j]),
                k,
                samples,
                "fit",
                data["q"][j],
                data["m"][j],
                data["available"][j],
                np.isfinite(packet),
                packet,
                tensor(data["s"][j]),
                tensor(data["c"][j]),
                tensor(data["anchors"][j]),
                stats,
                tensor(data["labels"][j], torch.long),
                tensor(data["weights"][j]),
            )
        )
    original_head, original_filters = learn._fit_head, op.bounded_filters
    state = {"head_index": -1, "head": None, "objective_call": 0, "completed_heads": []}
    evidence = {}

    def tracked_head(dimensions, loss_function, regularization, device):
        state["head_index"] += 1
        state["head"] = ("Q", "Q2", "QM", "SHAM_REFIT")[state["head_index"]]
        state["objective_call"] = 0

        def tracked_loss(coefficient):
            state["objective_call"] += 1
            state["coefficient_at_call"] = coefficient.detach().cpu().tolist()
            return loss_function(coefficient)

        result = original_head(dimensions, tracked_loss, regularization, device)
        state["completed_heads"].append(state["head"])
        return result

    def tracked_filters(s, c, anchors, r, **kwargs):
        try:
            return original_filters(s, c, anchors, r, **kwargs)
        except ValueError as exc:
            if str(exc) == "native anchor is nearly C-orthogonal to the leading direction":
                evidence.update(point_diagnostic(s, c, anchors, r, [list(c.key) for c in cases]))
            raise

    learn._fit_head, op.bounded_filters = tracked_head, tracked_filters
    started = time.perf_counter()
    try:
        learn.fit_pipeline(tuple(cases), 0.0001, backend="batch")
        status, error = "FAILURE_NOT_REPRODUCED_NO_PROMOTION", None
    except ValueError as exc:
        status, error = (
            "FAILURE_POINT_REPRODUCED" if evidence else "DIAGNOSTIC_INCONCLUSIVE",
            str(exc),
        )
    finally:
        learn._fit_head, op.bounded_filters = original_head, original_filters
    if digest(SOURCE) != SOURCE_SHA or digest(FAILURE) != FAILURE_SHA:
        raise ValueError("Preserved artifact changed")
    return {
        "status": status,
        "error": error,
        "seconds": time.perf_counter() - started,
        "state": state,
        "point": evidence,
        "source_sha256": SOURCE_SHA,
        "failure_sha256": FAILURE_SHA,
        "scope": "one fixed failure-only source-statistic replay; no completed model published, no query/outcome/new candidate",
        "candidate_terminal_unchanged": "VALIDITY_FAILURE",
        "held60_access": False,
        "raw_input_access": False,
        "no_threshold_or_scientific_setting_changes": True,
    }


if __name__ == "__main__":

    def alarm(*unused):
        raise TimeoutError("Bounded failure-only replay limit120s")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(120)
    receipt = run()
    signal.alarm(0)
    with OUTPUT.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({k: v for k, v in receipt.items() if k != "point"}, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt["point"].items()
                if k not in ("S", "C", "native_c_normalized_anchor")
            },
            indent=2,
        )
    )
