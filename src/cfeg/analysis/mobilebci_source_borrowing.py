"""Role-aware source-bank routing from previously extracted sufficient statistics."""

import hashlib

import numpy as np
import torch

from cfeg.analysis.mobilebci_prior_learning import balanced_accuracy, require, tensor
from cfeg.analysis.source_expert_borrowing import (
    Router,
    expert_scores,
    mix_probabilities,
    support_compatibility,
    validate_source_subjects,
)

ARMS = ("Q", "Q2", "QM", "SHAM")
KS = (1, 2, 3, 5)
SPEEDS = ("0.0", "0.8", "1.6")


def folds(subjects):
    subjects = sorted(subjects)
    outer_folds = {s: i % 3 for i, s in enumerate(subjects)}
    plans = []
    for outer in range(3):
        source = [s for s in subjects if outer_folds[s] != outer]
        for inner in range(2):
            plans.append(
                {
                    "phase": "inner",
                    "outer": outer,
                    "inner": inner,
                    "source": [s for i, s in enumerate(source) if i % 2 != inner],
                    "evaluation": [s for i, s in enumerate(source) if i % 2 == inner],
                }
            )
        plans.append(
            {
                "phase": "outer",
                "outer": outer,
                "source": source,
                "evaluation": [s for s in subjects if outer_folds[s] == outer],
            }
        )
    return outer_folds, sorted(plans, key=lambda p: p["phase"])


def select_sources(recipient, source_pool):
    eligible = sorted(set(source_pool) - {recipient})
    require(len(eligible) >= 4, "fewer_than_four_source_people")
    selected = sorted(
        eligible,
        key=lambda s: (
            hashlib.sha256(f"mobilebci-source-borrowing-v1|{recipient}|{s}".encode()).hexdigest(),
            s,
        ),
    )[:4]
    validate_source_subjects(selected, source_pool, [recipient])
    return sorted(selected)


def balanced_probability_loss(probabilities, labels):
    require(probabilities.shape == (*labels.shape, 3), "loss_shapes")
    counts = torch.stack([(labels == c).sum(-1) for c in range(3)], -1)
    require(bool((counts > 0).all()), "query_class_missing")
    selected = probabilities.gather(-1, labels[..., None]).squeeze(-1).clamp_min(1e-12)
    return (-selected.log() / (3 * counts.gather(1, labels))).sum(-1).mean()


def train_router(inputs, probabilities, labels, steps=100, on_step=None):
    router = Router(inputs)
    optimizer = torch.optim.AdamW(router.layer.parameters(), lr=0.05, weight_decay=0.01)
    first_loss, max_gradient = None, 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_probability_loss(
            mix_probabilities(probabilities, router.weights(inputs)), labels
        )
        require(bool(torch.isfinite(loss)), "nonfinite_loss")
        if first_loss is None:
            first_loss = float(loss.detach())
        loss.backward()
        require(
            all(bool(torch.isfinite(p.grad).all()) for p in router.layer.parameters()),
            "nonfinite_gradient",
        )
        max_gradient = max(
            max_gradient, max(float(p.grad.abs().max()) for p in router.layer.parameters())
        )
        optimizer.step()
        if on_step:
            on_step()
    for p in router.layer.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        final = balanced_probability_loss(
            mix_probabilities(probabilities, router.weights(inputs)), labels
        )
    return router, {
        "steps": steps,
        "initial_loss": first_loss,
        "final_loss": float(final),
        "max_abs_gradient": max_gradient,
        "std_floors": int((router.std == 1e-12).sum()),
    }


class Dataset:
    def __init__(self, arrays, runs, samples=500):
        self.arrays, self.runs, self.samples = arrays, runs, samples
        self.subjects = [r["subject"] for r in runs]
        self.speeds = [r["speed"] for r in runs]
        self.lookup = {(s, v): i for i, (s, v) in enumerate(zip(self.subjects, self.speeds))}
        require(len(self.lookup) == len(runs), "duplicate_run")
        require(
            all((s, v) in self.lookup for s in set(self.subjects) for v in SPEEDS), "missing_speed"
        )
        n, qn = len(runs), arrays["query_labels"].shape[1]
        shapes = {
            "support_cov": (n, 4, 3, 9, 9),
            "support_cross": (n, 4, 3, 9, 6),
            "q": (n, 4, 54),
            "q2": (n, 4, 2),
            "m": (n, 4, 2),
            "common": (n, 4, 5),
            "query_cov": (n, qn, 9, 9),
            "query_cross": (n, qn, 3, 9, 6),
            "reference_gram": (n, 3, 6, 6),
            "query_labels": (n, qn),
        }
        require(set(arrays) == set(shapes), "cache_keys")
        for name, shape in shapes.items():
            require(
                arrays[name].shape == shape and bool(np.isfinite(arrays[name]).all()),
                "cache_shape_or_finite:" + name,
            )
        require(np.issubdtype(arrays["query_labels"].dtype, np.integer), "label_dtype")
        require(all(set(y) == {0, 1, 2} for y in arrays["query_labels"]), "label_classes")
        self.weights = None

    def indices(self, subjects):
        return [i for i, s in enumerate(self.subjects) if s in subjects]

    def fit_experts(self, on_solve=None):
        a = self.arrays
        rows = []
        for i in range(len(self.runs)):
            budgets = []
            for ki in range(4):
                classes = []
                for c in range(3):
                    if on_solve:
                        on_solve(False)
                    cov, xy = (
                        tensor(a["support_cov"][i, ki, c]),
                        tensor(a["support_cross"][i, ki, c]),
                    )
                    scale = cov.diagonal().mean().clamp_min(1e-12)
                    w = torch.linalg.solve(
                        cov + 0.1 * scale * torch.eye(9, dtype=torch.float64), xy
                    )
                    require(bool(torch.isfinite(w).all()), "nonfinite_expert")
                    classes.append(w)
                    if on_solve:
                        on_solve(True)
                budgets.append(torch.stack(classes))
            rows.append(torch.stack(budgets))
        self.weights = torch.stack(rows)

    def bank(self, index, source_pool):
        people = select_sources(self.subjects[index], source_pool)
        source = [self.lookup[s, v] for s in people for v in SPEEDS]
        donor = [self.lookup[people[(j + 1) % 4], v] for j in range(4) for v in SPEEDS]
        require(
            all(
                self.subjects[i] != self.subjects[d] and self.speeds[i] == self.speeds[d]
                for i, d in zip(source, donor)
            ),
            "sham_identity_or_speed",
        )
        return source, donor

    def features(self, index, source_pool, ki):
        require(self.weights is not None, "experts_not_fit")
        a = self.arrays
        source, donor = self.bank(index, source_pool)
        w = torch.cat((self.weights[index, ki][None], self.weights[source, 3]))
        target_cov, target_cross = (
            tensor(a["support_cov"][index, ki]),
            tensor(a["support_cross"][index, ki]),
        )
        cov = torch.cat((target_cov[None], tensor(a["support_cov"][source, 3])))
        compat = support_compatibility(
            w, target_cov, target_cross, tensor(a["reference_gram"][index]), self.samples
        )
        norm = cov / cov.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None].clamp_min(1e-12)
        distance = (norm - norm[:1]).square().sum((-1, -2)).mean(-1)
        self_flag = torch.zeros(13, dtype=torch.float64)
        self_flag[0] = 1
        base5 = torch.stack(
            (
                self_flag,
                compat.mean(-1),
                compat.std(-1, correction=0),
                compat.min(-1).values,
                distance,
            ),
            -1,
        )
        common = np.concatenate((a["q"], a["common"]), -1)
        target = tensor(common[index, ki])
        delta = torch.cat(
            (torch.zeros((1, 59), dtype=torch.float64), (target - tensor(common[source, 3])).abs())
        )
        base = torch.cat((base5, delta, self_flag[:, None] * target), -1)

        def difference(key, rows):
            return torch.cat(
                (
                    torch.zeros((1, 2), dtype=torch.float64),
                    (tensor(a[key][index, ki]) - tensor(a[key][rows, 3])).abs(),
                )
            )

        q2, m, sham = difference("q2", source), difference("m", source), difference("m", donor)
        for si in range(3):
            ix = list(range(1 + si, 13, 3))
            require(
                sorted(map(tuple, m[ix].tolist())) == sorted(map(tuple, sham[ix].tolist())),
                "sham_vector_multiset",
            )
        change = float((m - sham).abs().max())
        require(change > 0, "sham_aux_unchanged")
        inputs = {
            "Q": base,
            "Q2": torch.cat((base, q2), -1),
            "QM": torch.cat((base, m), -1),
            "SHAM": torch.cat((base, sham), -1),
        }
        record = {
            "run": index,
            "subject": self.subjects[index],
            "speed": self.speeds[index],
            "source_runs": source,
            "donor_runs": donor,
            "sham_aux_max_change": change,
        }
        return inputs, w, record

    def batch(self, recipients, source_pool, ki):
        rows, records, ps, floors = {arm: [] for arm in ARMS}, [], [], 0
        for i in recipients:
            inputs, bank, record = self.features(i, source_pool, ki)
            for arm in ARMS:
                rows[arm].append(inputs[arm])
            score, count = expert_scores(
                bank,
                tensor(self.arrays["query_cov"][i : i + 1]),
                tensor(self.arrays["query_cross"][i : i + 1]),
                tensor(self.arrays["reference_gram"][i]),
            )
            ps.append(torch.softmax(10 * score[0], -1))
            records.append(record)
            floors += count
        return {
            "inputs": {arm: torch.stack(values) for arm, values in rows.items()},
            "probabilities": torch.stack(ps),
            "labels": torch.as_tensor(self.arrays["query_labels"][recipients], dtype=torch.long),
            "records": records,
            "energy_floors": floors,
        }


def predict(batch, router, arm):
    p = mix_probabilities(batch["probabilities"], router.weights(batch["inputs"][arm]))
    return p


def actuation(batch, router):
    true, sham = batch["inputs"]["QM"], batch["inputs"]["SHAM"]
    logits = [router.layer((x - router.mean) / router.std).squeeze(-1) for x in [true, sham]]
    centered = [x - x.mean(-1, keepdim=True) for x in logits]
    gates = [router.weights(x) for x in [true, sham]]
    ps = [mix_probabilities(batch["probabilities"], g) for g in gates]
    margins = [p[..., 1:] - p[..., :1] for p in ps]
    return {
        "aux_max_change": float((true - sham).abs().max()),
        "centered_logit_max_change": float((centered[0] - centered[1]).abs().max()),
        "gate_max_change": float((gates[0] - gates[1]).abs().max()),
        "class_margin_max_change": float((margins[0] - margins[1]).abs().max()),
    }


def outcome_rows(batch, probabilities):
    pred, labels = probabilities.argmax(-1).numpy(), batch["labels"].numpy()
    accuracy = balanced_accuracy(pred, labels)
    return [
        {
            "subject": r["subject"],
            "speed": r["speed"],
            "run": r["run"],
            "balanced_accuracy": float(accuracy[j]),
            "predictions": pred[j].tolist(),
            "labels": labels[j].tolist(),
        }
        for j, r in enumerate(batch["records"])
    ]


def choose_policy(inner_rows):
    choices = {}
    for outer in range(3):
        means, policies = {}, {}
        for arm in ARMS:
            means[arm] = {
                str(k): float(
                    np.mean(
                        [
                            r["balanced_accuracy"]
                            for r in inner_rows
                            if r["outer"] == outer and r["arm"] == arm and r["k"] == k
                        ]
                    )
                )
                for k in KS
            }
            eligible = [k for k in KS if means[arm][str(k)] >= 0.8]
            policies[arm] = {
                "k": min(eligible) if eligible else 5,
                "source_target_unmet": not eligible,
            }
        choices[str(outer)] = {"inner_oof_means": means, "policy": policies}
    return choices
