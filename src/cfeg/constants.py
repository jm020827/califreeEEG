DEFAULT_C_MAX = 64
DEFAULT_TARGET_SFREQ = 200.0
UNKNOWN_CATEGORY = "unknown"

CATEGORICAL_VOCABS = {
    "dataset_id": [
        UNKNOWN_CATEGORY,
        "synthetic",
        "wang",
        "beta",
        "dong2023",
        "wearable",
        "openbci",
    ],
    "reference": [
        UNKNOWN_CATEGORY,
        "average",
        "linked_mastoids",
        "cz",
        "forehead",
        "fp1",
        "openbci_default",
    ],
    "hardware_id": [
        UNKNOWN_CATEGORY,
        "public_unknown",
        "neuroscan_synamp2",
        "neuracle_neusenw",
        "openbci_cyton",
    ],
    "cap_type": [UNKNOWN_CATEGORY, "wet_cap", "dry_cap", "wearable"],
    "electrode_type": [
        UNKNOWN_CATEGORY,
        "wet",
        "dry",
        "gel",
        "pregelled_semidry",
    ],
    "reattach_flag": [UNKNOWN_CATEGORY, "false", "true"],
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
