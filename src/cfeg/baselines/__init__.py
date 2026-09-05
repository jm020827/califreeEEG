from cfeg.baselines.fbcca import (
    FilterBankConfig,
    apply_filterbank,
    cca_score,
    make_reference_signals,
    predict_cca,
    predict_fbcca,
)

__all__ = [
    "CalibrationBaselineOutput",
    "FilterBankConfig",
    "apply_filterbank",
    "cca_score",
    "least_squares_transfer_trials",
    "make_reference_signals",
    "predict_cca",
    "predict_ensemble_trca",
    "predict_fbcca",
    "predict_filterbank_ensemble_trca",
    "predict_lst_filterbank_ensemble_trca",
    "predict_supervised_template_correlation",
]
from cfeg.baselines.calibration import (
    CalibrationBaselineOutput,
    least_squares_transfer_trials,
    predict_ensemble_trca,
    predict_filterbank_ensemble_trca,
    predict_lst_filterbank_ensemble_trca,
    predict_supervised_template_correlation,
)
