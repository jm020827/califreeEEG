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
        # New fallback categories are append-only so legacy checkpoint IDs stay stable.
        "synthetic_quality",
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

METADATA_CONTRACT_LEGACY = "legacy"
METADATA_CONTRACT_V04_DEV = "0.4-dev"
QUERY_QC_EXTRACTOR_V1 = "filtered_cropped_pre_zscore_channel_std_median_v1"
EXTERNAL_CONTINUOUS_SCHEMA_V1 = "impedance_mean_max_v1"

# Protocol 0.4-dev asks only whether portable, pre-query physical descriptors
# add information beyond the waveform and common observation/QC contract.
PROTOCOL_V04_EXTERNAL_CATEGORICAL_FIELDS = [
    "reference",
    "electrode_type",
    "cap_type",
]
PROTOCOL_V04_EXTERNAL_CONTINUOUS_FIELDS = [
    "impedance_mean_kohm",
    "impedance_max_kohm",
]
PROTOCOL_V04_QUERY_QC_FIELDS = ["query_signal_std"]

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
