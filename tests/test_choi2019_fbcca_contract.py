from pathlib import Path

import numpy as np
import pytest
import yaml

from cfeg.baselines.fbcca import make_reference_signals, resolve_filterbank_parameters

_ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return yaml.safe_load((_ROOT / relative).read_text(encoding="utf-8"))


def _runtime_filterbank(contract: dict, band: str) -> dict:
    common = contract["common_parameters"]
    selected = contract["band_parameters"][band]
    return {
        "bands": selected["bands"],
        "weights": common["subband_weights"],
        "order": common["order"],
        "n_harmonics": selected["n_harmonics"],
        "regularization": common["regularization"],
        "filter_family": common["filter_family"],
        "passband_ripple_db": common["passband_ripple_db"],
        "reproduction_contract": selected["reproduction_contract"],
    }


def test_parent_fbcca_is_formally_incompatible_with_choi_mid_and_high() -> None:
    dataset = _load("configs/data/choi2019.yaml")
    parent = _load("configs/baselines/fbcca_chen2015_m3.yaml")

    for band in ("MID", "HIGH"):
        frequencies = dataset["stimulus_by_band"][band]["frequencies_hz"]
        with pytest.raises(ValueError, match="below Nyquist"):
            make_reference_signals(
                frequencies,
                dataset["raw_sfreq"],
                400,
                n_harmonics=parent["n_harmonics"],
            )


def test_frozen_bandwise_view_is_nyquist_safe_and_covers_low_fundamentals() -> None:
    dataset = _load("configs/data/choi2019.yaml")
    contract = _load("configs/baselines/fbcca_choi2019_bandwise_v1.yaml")
    sfreq = float(contract["sampling_frequency_hz"])

    assert contract["evaluation_view"]["pool_twelve_frequencies"] is False
    assert contract["evaluation_view"]["bands_required"] == ["LOW", "MID", "HIGH"]
    assert contract["support_query"]["budgets_trials_per_class"] == [0, 1, 3, 5]
    assert contract["support_query"]["query_reuse_across_budgets"] == "immutable"

    expected_harmonics = {"LOW": 5, "MID": 4, "HIGH": 2}
    explicit_weight_bytes = np.asarray(
        contract["common_parameters"]["subband_weights"], dtype="<f8"
    ).tobytes()
    assert contract["common_parameters"]["subband_weight_resolution"] == (
        "explicit_vector_only_no_runtime_exponentiation"
    )
    assert contract["common_parameters"]["weight_formula_role"] == ("historical_provenance_only")
    for band, expected in expected_harmonics.items():
        selected = contract["band_parameters"][band]
        frequencies = dataset["stimulus_by_band"][band]["frequencies_hz"]
        assert selected["candidate_frequencies_hz"] == frequencies
        assert selected["n_harmonics"] == expected
        assert max(frequencies) * expected < sfreq / 2.0
        references = make_reference_signals(
            frequencies, sfreq, 400, n_harmonics=selected["n_harmonics"]
        )
        assert references.shape == (4, 2 * expected, 400)
        resolved = resolve_filterbank_parameters(_runtime_filterbank(contract, band), sfreq)
        assert resolved["n_harmonics"] == expected
        assert np.asarray(resolved["weights"], dtype="<f8").tobytes() == explicit_weight_bytes

    low = contract["band_parameters"]["LOW"]
    assert low["bands"][0][0] <= min(low["candidate_frequencies_hz"])
    assert _load("configs/baselines/fbcca_chen2015_m3.yaml")["bands"][0][0] > min(
        low["candidate_frequencies_hz"]
    )
