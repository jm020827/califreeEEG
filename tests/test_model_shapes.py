from __future__ import annotations

import pytest
import torch

from cfeg.constants import CATEGORICAL_VOCABS
from cfeg.models.adapters import BottleneckAdapter
from cfeg.models.condition_encoder import ConditionEncoder
from cfeg.models.full_model import ConditionedEEGDecoder
from cfeg.utils.checkpoint import load_checkpoint_model_state


def test_full_model_forward_shape():
    cfg = {
        "model": {
            "n_classes": 4,
            "c_max": 64,
            "t_len": 400,
            "d_model": 32,
            "patch_size": 20,
            "depth": 1,
            "n_heads": 4,
            "backbone": {"name": "tiny_transformer"},
            "condition_encoder": {"enabled": True, "n_prompt_tokens": 2},
            "adapter": {"enabled": True, "bottleneck_dim": 8},
            "latent": {"enabled": True, "z_dim": 4},
        }
    }
    model = ConditionedEEGDecoder(cfg)
    assert model.conditioning_architecture == "prompt_adapter_v1"
    assert isinstance(model.condition_encoder, ConditionEncoder)
    assert isinstance(model.adapter, BottleneckAdapter)
    assert model.physical_conditioner is None
    cond = {
        "continuous": torch.zeros(2, 5),
        "continuous_missing": torch.ones(2, 5, dtype=torch.bool),
        "channel_ids": torch.arange(64).unsqueeze(0).repeat(2, 1),
        "channel_mask": torch.ones(2, 64, dtype=torch.bool),
        "sfreq_processed_float": torch.full((2,), 200.0),
    }
    for name in CATEGORICAL_VOCABS:
        cond[name] = torch.zeros(2, dtype=torch.long)
    out = model(torch.randn(2, 64, 400), cond)
    assert out.logits.shape == (2, 4)
    assert out.logits_zero.shape == (2, 4)


def test_trainable_only_checkpoint_rejects_missing_trainable_parameter():
    model = torch.nn.Linear(3, 2)
    checkpoint = {
        "save_trainable_only": True,
        "model_state": {"bias": model.bias.detach().clone()},
    }

    with pytest.raises(ValueError, match="missing_trainable"):
        load_checkpoint_model_state(model, checkpoint)


def test_trainable_only_checkpoint_allows_exact_frozen_omission():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    trainable = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }

    load_checkpoint_model_state(model, {"save_trainable_only": True, "model_state": trainable})


def test_trainable_only_checkpoint_materializes_lazy_projection_before_preflight():
    class LazyProjectionModel(torch.nn.Module):
        materialize_checkpoint_modules = ConditionedEEGDecoder.materialize_checkpoint_modules

        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Module()
            self.backbone.output_proj = None
            self.frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    model = LazyProjectionModel()
    checkpoint = {
        "save_trainable_only": True,
        "model_state": {
            "backbone.output_proj.weight": torch.randn(2, 3),
            "backbone.output_proj.bias": torch.randn(2),
        },
    }

    load_checkpoint_model_state(model, checkpoint)

    assert isinstance(model.backbone.output_proj, torch.nn.Linear)
