from __future__ import annotations

import pytest

from cfeg.utils.config import resolve_interpolations


def test_env_interpolation_strict_missing():
    with pytest.raises(RuntimeError, match="Environment variable EEG_DATA_ROOT is not set"):
        resolve_interpolations({"x": "${env:EEG_DATA_ROOT}"}, strict_env=True)

