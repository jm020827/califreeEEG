"""Artificial orthogonal/parallel points only, never execute the real replay."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("anchor_channel, expected", [(0, 0.0), (7, 1.0)])
def test_independent_point_diagnostic(anchor_channel, expected):
    spec = importlib.util.spec_from_file_location(
        "task_failure_diagnostic",
        Path(__file__).resolve().parents[1] / "scripts/diagnose_task_trca_shape_failure.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = torch.eye(8, dtype=torch.float64).expand(1, 5, 12, 8, 8)
    s = torch.diag(torch.arange(1, 9, dtype=torch.float64)).expand_as(c)
    anchor = torch.zeros((1, 5, 12, 8), dtype=torch.float64)
    anchor[..., anchor_channel] = 1
    r = torch.ones((1, 5, 1, 8), dtype=torch.float64)
    result = module.point_diagnostic(s, c, anchor, r, [[11001, 0, 125, 3]])
    assert result["all_cosines_finite"]
    assert result["cpu_abs_c_cosine"] == pytest.approx(expected)
    assert result["gpu_min_abs_c_cosine"] == pytest.approx(expected)
    assert result["independent_cpu_confirms_threshold_failure"] == (expected == 0)
