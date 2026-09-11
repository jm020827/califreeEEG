"""Source-only block scaling and safeguarded AdamW proposals; no data IO."""

import torch

from cfeg.analysis.mobilebci_prior_learning import require
from cfeg.analysis.mobilebci_source_borrowing import balanced_probability_loss
from cfeg.analysis.source_expert_borrowing import Router, mix_probabilities


class BlockRouter(Router):
    def __init__(self, inputs, variant="SAFE"):
        require(variant in {"ORIGINAL", "SCALE", "SAFE"}, "variant")
        require(inputs.ndim == 3 and inputs.shape[1] == 13, "expert_shape")
        require(inputs.shape[-1] in {123, 125}, "feature_shape")
        super().__init__(inputs)
        self.variant = variant
        self.scale = torch.ones_like(self.mean)
        if variant != "ORIGINAL":
            for start, end in [(0, 5), (5, 64), (64, 123), (123, len(self.mean))]:
                if end > start:
                    self.scale[start:end] = 1 / (end - start)

    def logits(self, inputs):
        require(inputs.ndim == 3 and inputs.shape[1:] == (13, len(self.mean)), "input_shape")
        require(bool(torch.isfinite(inputs).all()), "nonfinite_input")
        return self.layer(((inputs - self.mean) / self.std) * self.scale).squeeze(-1)

    def weights(self, inputs):
        return self.logits(inputs).softmax(-1)

    def state(self):
        return {**super().state(), "variant": self.variant, "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, state):
        d = len(state["mean"])
        model = cls(torch.zeros((1, 13, d), dtype=torch.float64), state["variant"])
        for key in ["mean", "std", "scale"]:
            setattr(model, key, torch.tensor(state[key], dtype=torch.float64))
        require(bool((model.std > 0).all()), "invalid_saved_std")
        with torch.no_grad():
            model.layer.weight.copy_(torch.tensor(state["weight"], dtype=torch.float64))
            model.layer.bias.copy_(torch.tensor(state["bias"], dtype=torch.float64))
        model.layer.requires_grad_(False)
        return model


def centered(logits):
    return logits - logits.mean(-1, keepdim=True)


def accept_proposal(layer, old, proposal, evaluate, old_loss, old_logits, on_trial=None):
    """Parameter-only line search. Optimizer state is deliberately not rolled back."""
    parameters = list(layer.parameters())
    trials = []
    for exponent in range(8):
        if on_trial:
            on_trial("candidate")
        factor = 2.0**-exponent
        with torch.no_grad():
            for p, a, b in zip(parameters, old, proposal):
                p.copy_(a + factor * (b - a))
            evaluated = all(bool(torch.isfinite(p).all()) for p in parameters)
            if evaluated:
                if on_trial:
                    on_trial("loss")
                loss, logits = evaluate()
                finite = bool(torch.isfinite(loss) and torch.isfinite(logits).all())
            else:
                finite = False
            delta = float((logits - old_logits).abs().max()) if finite else None
            value = float(loss) if finite else None
            accepted = finite and value <= old_loss + 1e-12 and delta <= 1.0
        trials.append(
            {
                "factor": factor,
                "loss": value,
                "centered_delta": delta,
                "accepted": bool(accepted),
                "loss_evaluated": evaluated,
            }
        )
        if accepted:
            return {"factor": factor, "accepted": True, "trials": trials}
    with torch.no_grad():
        for p, a in zip(parameters, old):
            p.copy_(a)
    return {"factor": 0.0, "accepted": False, "trials": trials}


def train(
    inputs, probabilities, labels, variant, steps=100, on_step=None, on_attempt=None, on_trial=None
):
    require(1 <= steps <= 100, "step_budget")
    require(probabilities.shape[:2] == inputs.shape[:2], "batch_shape")
    model = BlockRouter(inputs, variant)
    optimizer = torch.optim.AdamW(
        model.layer.parameters(), lr=0.05, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    def evaluate():
        logits = model.logits(inputs)
        if not bool(torch.isfinite(logits).all()):
            return torch.tensor(float("inf"), dtype=torch.float64), centered(logits)
        p = mix_probabilities(probabilities, logits.softmax(-1))
        return balanced_probability_loss(p, labels), centered(logits)

    trajectory = []
    trial_evaluations = 0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, logits = evaluate()
        require(bool(torch.isfinite(loss)), "nonfinite_loss")
        before = float(loss.detach())
        old_logits = logits.detach().clone()
        old = [p.detach().clone() for p in model.layer.parameters()]
        loss.backward()
        require(
            all(bool(torch.isfinite(p.grad).all()) for p in model.layer.parameters()),
            "nonfinite_gradient",
        )
        if on_attempt:
            on_attempt()
        optimizer.step()
        proposal = [p.detach().clone() for p in model.layer.parameters()]
        receipt = {"factor": 1.0, "accepted": True, "trials": []}
        if variant == "SAFE":
            receipt = accept_proposal(
                model.layer, old, proposal, evaluate, before, old_logits, on_trial
            )
        trial_evaluations += len(receipt["trials"])
        with torch.no_grad():
            after_loss, after_logits = evaluate()
            after = float(after_loss)
            change = float((after_logits - old_logits).abs().max())
            parameter_change = max(
                float((p - a).abs().max()) for p, a in zip(model.layer.parameters(), old)
            )
        require(bool(torch.isfinite(after_loss)), "nonfinite_final_loss")
        if variant == "SAFE":
            require(after <= before + 1e-12 and change <= 1 + 1e-12, "safeguard_invariant")
        row = {
            "proposal": step + 1,
            "before": before,
            "after": after,
            "centered_delta": change,
            "parameter_delta": parameter_change,
            **receipt,
        }
        trajectory.append(row)
        if on_step:
            on_step(row)
    model.layer.requires_grad_(False)
    return model, {
        "proposals": steps,
        "accepted": sum(r["accepted"] for r in trajectory),
        "rejected": sum(not r["accepted"] for r in trajectory),
        "parameter_noops": sum(r["parameter_delta"] == 0 for r in trajectory),
        "trial_checks": trial_evaluations,
        "trial_loss_evaluations": sum(t["loss_evaluated"] for r in trajectory for t in r["trials"]),
        "ordinary_loss_evaluations": 2 * steps,
        "moment_steps": [int(optimizer.state[p]["step"]) for p in model.layer.parameters()],
        "initial_loss": trajectory[0]["before"],
        "final_loss": trajectory[-1]["after"],
        "max_loss_increase": max(r["after"] - r["before"] for r in trajectory),
        "max_centered_change": max(r["centered_delta"] for r in trajectory),
        "trajectory": trajectory,
    }


def actuation(inputs, sham_inputs, probabilities, model):
    with torch.no_grad():
        gates = [model.weights(x) for x in [inputs, sham_inputs]]
        ps = [mix_probabilities(probabilities, g) for g in gates]
        return {
            "gate_delta": float((gates[0] - gates[1]).abs().max()),
            "margin_delta": float(
                ((ps[0][..., 1:] - ps[0][..., :1]) - (ps[1][..., 1:] - ps[1][..., :1])).abs().max()
            ),
        }
