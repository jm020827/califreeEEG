# Data policy

This repository stores only code, configs, and tiny synthetic fixtures. Raw EEG,
processed HDF5/parquet datasets, OpenBCI recordings, REVE weights, checkpoints,
and experiment logs are external assets and are ignored by Git.

Use environment variables to point to real storage:

```bash
export EEG_DATA_ROOT=/mnt/eeg_data
export EEG_MODEL_ROOT=/mnt/eeg_models
export HF_HOME=$EEG_MODEL_ROOT/huggingface
export MNE_DATA=$EEG_DATA_ROOT/mne_data
```

