from types import SimpleNamespace

import torch
import torch.nn.functional as F

from cfeg.train_loop import _branch_regularization


def test_branch_regularization_combines_auxiliary_ce_and_spectral_anchor():
    learned = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    spectral = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    combined = torch.tensor([[1.5, 0.5], [0.5, 1.5]], requires_grad=True)
    labels = torch.tensor([0, 1])
    output = SimpleNamespace(
        logits=combined,
        aux={"learned_logits": learned, "spectral_logits": spectral},
    )
    cfg = {"lambda_learned_aux": 0.25, "lambda_spectral_anchor": 0.5}

    value = _branch_regularization(output, labels, cfg)
    expected = 0.25 * F.cross_entropy(learned, labels) + 0.5 * F.kl_div(
        F.log_softmax(combined, dim=-1),
        F.softmax(spectral, dim=-1),
        reduction="batchmean",
    )
    assert torch.allclose(value, expected)
