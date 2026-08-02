from __future__ import annotations

import numpy as np

from cfeg.data.prepare_wearable import _impedance_for, _wearable_data


def test_wearable_axis_detection_accepts_matlab_and_reversed_storage():
    expected = np.zeros((8, 710, 2, 10, 12), dtype=np.float32)
    assert _wearable_data({"data": expected}).shape == expected.shape
    reversed_data = np.moveaxis(expected, range(5), tuple(reversed(range(5))))
    assert _wearable_data({"data": reversed_data}).shape == expected.shape


def test_wearable_impedance_selects_block_electrode_subject():
    impedance = np.arange(8 * 10 * 2 * 102, dtype=np.float32).reshape(8, 10, 2, 102)
    selected = _impedance_for(impedance, subject=4, electrode=1, block=3)
    np.testing.assert_array_equal(selected, impedance[:, 3, 1, 4])
