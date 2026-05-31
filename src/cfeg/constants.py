DEFAULT_C_MAX = 64
DEFAULT_TARGET_SFREQ = 200.0

CATEGORICAL_VOCABS = {
    "dataset_id": ["unknown", "synthetic", "wang", "beta", "wearable", "openbci"],
    "reference": ["unknown", "average", "linked_mastoids", "cz", "openbci_default"],
    "hardware_id": ["unknown", "public_unknown", "openbci_cyton"],
    "cap_type": ["unknown", "wet_cap", "dry_cap", "wearable"],
    "electrode_type": ["unknown", "wet", "dry", "gel"],
    "reattach_flag": ["unknown", "false", "true"],
}

CONDITION_CATEGORICAL_FIELDS = [
    "dataset_id",
    "reference",
    "hardware_id",
    "electrode_type",
    "cap_type",
    "reattach_flag",
]

LEAKAGE_FIELDS = {
    "label",
    "class_id",
    "stimulus_frequency_hz",
    "stimulus_phase_rad",
    "trial_id",
    "subject_id",
    "session_id",
    "source_file",
}

