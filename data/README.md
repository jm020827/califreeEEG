# Data policy

This repository stores only code, configs, and tiny synthetic fixtures. Raw EEG,
processed HDF5/parquet datasets, OpenBCI recordings, REVE weights, checkpoints,
and experiment logs are external assets and are ignored by Git.

On the current server, load the persistent data and model-cache paths:

```bash
source scripts/env_server.sh
```

The legacy `eeg_models/` directory is not used; Hugging Face repositories live
under `HF_HUB_CACHE`, and prepared EEG datasets live under `EEG_DATA_ROOT`.
