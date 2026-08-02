from __future__ import annotations

import pytest

from cfeg.data.label_mapping import SSVEP_40_FREQUENCIES, frequency_to_label, remap_source_label


def test_wang_and_beta_source_orders_share_canonical_labels():
    wang = [round(base + offset, 1) for offset in (0.0, 0.2, 0.4, 0.6, 0.8) for base in range(8, 16)]
    beta = [round(8.6 + 0.2 * i, 1) for i in range(37)] + [8.0, 8.2, 8.4]
    for source in (wang, beta):
        for source_label, frequency in enumerate(source):
            label, returned_frequency = remap_source_label(
                source_label, source, list(SSVEP_40_FREQUENCIES)
            )
            assert returned_frequency == frequency
            assert SSVEP_40_FREQUENCIES[label] == frequency


def test_frequency_mapping_rejects_noncanonical_frequency():
    with pytest.raises(ValueError, match="absent"):
        frequency_to_label(9.25, SSVEP_40_FREQUENCIES)
