# Data policy

This repository stores only code, configs, and tiny synthetic fixtures. Raw EEG,
processed HDF5/parquet datasets, OpenBCI recordings, REVE weights, checkpoints,
and experiment logs are external assets and are ignored by Git.

Use separate persistent data and Hugging Face Hub cache paths:

```bash
export HF_HOME=/mnt/pvc/hf
export HF_HUB_CACHE=/mnt/pvc/hf/hub
export EEG_DATA_ROOT=/mnt/pvc/eeg
export MNE_DATA=$EEG_DATA_ROOT/mne_data
```

On the jm020827 interns cluster, source `scripts/env_k8s_interns.sh`. The legacy
`eeg_models/` directory is not used; Hugging Face repositories live
under `HF_HUB_CACHE`, and prepared EEG datasets live under `EEG_DATA_ROOT`.
