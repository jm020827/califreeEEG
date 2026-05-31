# Codex 구현 지시서 v2: 획득 조건 임베딩 기반 Calibration-Free EEG Decoding 모델

이 문서는 첨부 연구계획서의 아이디어를 **실제로 코드로 구현 가능한 개발 계획**으로 재구성한 것이다. Codex는 이 문서를 그대로 읽고 repository를 생성한 뒤, MVP부터 단계적으로 구현한다.

이 v2 문서에는 큰 데이터셋, REVE backbone/checkpoint, OpenBCI 자체 수집 데이터, 학습 checkpoint처럼 repository에 직접 넣으면 안 되는 외부 asset 관리 계획을 포함한다. 핵심 원칙은 **코드는 작게 유지하고, 데이터/모델 weight는 명시적인 외부 asset으로 관리**하는 것이다.

개인 연락처, 학번 등 연구계획서 신청서상의 개인정보는 구현 지시서에 포함하지 않는다.

---

## 0. 연구계획서에서 코드 요구사항으로 변환한 핵심

연구계획서의 목표는 EEG 신호 자체만 보는 모델이 아니라, EEG가 **어떤 획득 조건에서 측정되었는지**를 함께 입력받아 새로운 사용자, 새로운 세션, 다른 채널 구성에서도 calibration 부담을 낮추는 decoding 모델을 만드는 것이다.

구현해야 할 핵심 구조는 다음과 같다.

```text
Acquisition condition metadata m
      -> Metadata/Condition Encoder E_m
      -> learnable prompt tokens P

EEG signal X + prompt P
      -> Decoding model
         - frozen REVE backbone, if available
         - trainable adapter
         - fallback TinyEEGTransformer backbone, always runnable
      -> representation h
      -> classification head
      -> SSVEP class logits

Train only:
  latent encoder q_phi estimates latent nuisance z
  use z, classification loss, and consistency loss for robustness

Inference:
  no per-user calibration
  use z = 0 or train-set mean z
```

구현상 반드시 지킬 원칙은 다음이다.

1. **REVE가 없어도 전체 pipeline이 돌아가야 한다.** `REVEBackbone` wrapper는 optional이고, 기본 smoke-test 모델은 `TinyEEGTransformerBackbone`이다.
2. **큰 데이터셋과 model weight를 repository에 넣지 않는다.** REVE weight, Wang/BETA/Wearable raw data, OpenBCI recording, processed HDF5, checkpoint는 모두 외부 asset이다.
3. **Condition metadata는 EEG 획득 조건만 포함한다.** stimulus frequency, class id, phase 등 label 정보는 ConditionEncoder 입력으로 절대 넣지 않는다.
4. **Subject id, trial id는 split/evaluation용 metadata로만 사용한다.** 새로운 사용자 일반화 실험에서 subject id embedding은 leakage가 될 수 있으므로 기본 모델 입력에서 제외한다.
5. **OpenBCI 8채널 자체 데이터는 외부 검증용으로 설계한다.** 처음부터 40-class SSVEP를 요구하지 말고 4-class 또는 8-class subset으로 시작한다.
6. 모든 기능은 synthetic data에서 먼저 통과해야 한다.
7. asset download는 import 시점에 절대 일어나면 안 된다. 다운로드는 명시적 CLI에서만 허용한다.

---

## 1. 외부 asset 정책

### 1.1 Repository에 넣지 말 것

다음 파일은 절대 Git commit 대상이 아니다.

```text
raw EEG datasets
processed EEG datasets
REVE model weights
Hugging Face model cache
OpenBCI recordings
subject-level private metadata
HDF5/NPY/NPZ/MAT/FIF/EDF/BDF files
training checkpoints
TensorBoard/W&B logs
ablation result artifact bundles
```

Codex는 큰 파일을 repository에 복사하거나 sample로 포함하지 않는다. 테스트용 fixture는 synthetic non-human signal만 허용하고, 전체 fixture 크기는 1 MB 이하로 유지한다.

### 1.2 외부 경로는 환경 변수로 관리

모든 큰 asset은 환경 변수로 지정된 위치에 저장한다.

```bash
export EEG_DATA_ROOT=/mnt/eeg_data
export EEG_MODEL_ROOT=/mnt/eeg_models
export HF_HOME=$EEG_MODEL_ROOT/huggingface
export HF_TOKEN=<only-if-needed>
export MNE_DATA=$EEG_DATA_ROOT/mne_data
```

`.env.example` 파일에는 변수 이름만 넣고 실제 token이나 개인 경로는 넣지 않는다.

```dotenv
EEG_DATA_ROOT=/absolute/path/to/eeg_data
EEG_MODEL_ROOT=/absolute/path/to/eeg_models
HF_HOME=/absolute/path/to/eeg_models/huggingface
HF_TOKEN=
MNE_DATA=/absolute/path/to/eeg_data/mne_data
```

### 1.3 다운로드는 explicit CLI로만 수행

금지:

```python
# Bad: importing a module downloads model/data
from cfeg.models.backbones.reve import REVEBackbone
model = REVEBackbone()  # hidden download during import or normal training
```

허용:

```bash
python scripts/fetch_reve.py --model brain-bzh/reve-base --positions brain-bzh/reve-positions
python scripts/fetch_dataset.py --dataset beta
python scripts/verify_assets.py --dataset beta
python scripts/prepare_dataset.py --dataset beta --config configs/data/beta.yaml
```

학습/평가 시 asset이 없으면 자동 다운로드하지 말고, 다음처럼 명확한 error를 낸다.

```text
Missing REVE asset: brain-bzh/reve-base was not found in local cache.
Run:
  huggingface-cli login
  python scripts/fetch_reve.py --model brain-bzh/reve-base --positions brain-bzh/reve-positions
or set model.backbone=tiny_transformer for a no-REVE smoke run.
```

### 1.4 REVE weight 재배포 금지

REVE는 Hugging Face에서 접근 조건을 확인하고 받아야 하는 외부 backbone으로 취급한다. Codex는 다음을 구현해야 한다.

- REVE weight를 repository에 vendor하지 않는다.
- REVE source/remote code를 repository에 복사하지 않는다.
- 각 실행 환경에서 Hugging Face login/token으로 접근하게 한다.
- training config의 기본값은 `allow_download: false`, `local_files_only: true`로 둔다.
- 다운로드가 필요하면 `scripts/fetch_reve.py`를 먼저 실행하게 한다.
- REVE 접근 실패 시 fallback TinyEEGTransformer로 smoke test가 가능해야 한다.

---

## 2. 외부 source snapshot

Codex는 아래 source를 기준으로 asset loader를 설계한다. 이 목록은 코드에 hard-coding하기보다 `configs/assets.yaml`의 기본값으로 둔다.

```yaml
external_sources:
  reve_base:
    repo_id: brain-bzh/reve-base
    type: huggingface_model
    note: REVE backbone; use transformers.AutoModel.from_pretrained(..., trust_remote_code=True)
    required_sample_rate_hz: 200
    url: https://huggingface.co/brain-bzh/reve-base

  reve_positions:
    repo_id: brain-bzh/reve-positions
    type: huggingface_model
    note: position bank mapping electrode names to positions
    required_sample_rate_hz: 200
    url: https://huggingface.co/brain-bzh/reve-positions

  beta:
    repo_id: Bingchuan/beta
    type: huggingface_dataset_or_manual
    note: 64-channel EEG, 70 subjects, 40-target SSVEP
    url: https://huggingface.co/datasets/Bingchuan/BETA

  wang:
    type: moabb_or_manual
    moabb_dataset: Wang2016
    note: 35 subjects, 64-channel EEG, 40-target SSVEP benchmark
    url: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Wang2016.html

  wearable:
    type: figshare_or_manual
    note: 8-channel wearable SSVEP, 102 subjects, 12-target, wet/dry electrodes and impedance metadata
    url: https://figshare.com/articles/dataset/An_Open_Dataset_for_Wearable_SSVEP-Based_Brain-Computer_Interfaces/13560281

  openbci_cyton:
    type: local_private_recording
    note: 8-channel Cyton board; raw sampling rate is 250 Hz per channel; resample to experiment target_sfreq during preprocessing
    url: https://shop.openbci.com/products/cyton-biosensing-board-8-channel
```

중요한 sampling-rate 규칙:

```text
OpenBCI Cyton raw: 250 Hz
REVE expected input: 200 Hz
Therefore:
  - raw OpenBCI files keep sfreq_original=250.0
  - REVE experiments must resample to sfreq_processed=200.0
  - TinyEEGTransformer can run at 200 or 250, but config must record it explicitly
```

---

## 3. Repository 구조

다음 구조로 구현한다.

```text
calibration_free_eeg/
  README.md
  pyproject.toml
  requirements.txt
  requirements-dev.txt
  .gitignore
  .env.example

  configs/
    assets.yaml
    default.yaml
    canonical_channels.yaml
    channel_sets.yaml
    data/
      synthetic.yaml
      wang.yaml
      beta.yaml
      wearable.yaml
      openbci.yaml
    model/
      tiny_transformer.yaml
      prompt_adapter.yaml
      full_latent.yaml
      reve_wrapper.yaml
    train/
      debug.yaml
      ssvep_pretrain.yaml
      ablation.yaml
    eval/
      cross_subject.yaml
      cross_dataset.yaml
      channel_stress.yaml
      openbci_external.yaml

  data/
    README.md
    raw/.gitkeep
    processed/.gitkeep
    manifests/.gitkeep

  checkpoints/
    .gitkeep

  scripts/
    fetch_reve.py
    fetch_dataset.py
    verify_assets.py
    prepare_synthetic.py
    prepare_dataset.py
    inspect_manifest.py
    train.py
    evaluate.py
    run_ablation.py
    export_results_table.py
    openbci_record.py
    openbci_convert.py

  src/
    cfeg/
      __init__.py
      constants.py
      seed.py

      assets/
        __init__.py
        paths.py
        hf.py
        registry.py
        verify.py
        errors.py

      data/
        __init__.py
        schema.py
        io_hdf5.py
        preprocess.py
        ssvep_synthetic.py
        label_mapping.py
        datasets.py
        collate.py
        splits.py
        transforms.py
        prepare_wang.py
        prepare_beta.py
        prepare_wearable.py
        prepare_openbci.py

      models/
        __init__.py
        condition_encoder.py
        eeg_tokenizer.py
        adapters.py
        latent_nuisance.py
        heads.py
        full_model.py
        backbones/
          __init__.py
          base.py
          tiny_transformer.py
          reve.py

      baselines/
        __init__.py
        fbcca.py
        simple_eegnet.py

      losses.py
      metrics.py
      train_loop.py
      eval_loop.py
      utils/
        config.py
        logging.py
        checkpoint.py
        params.py
        tensorboard.py

  tests/
    test_assets.py
    test_schema.py
    test_condition_encoder.py
    test_model_shapes.py
    test_transforms.py
    test_losses.py
    test_synthetic_overfit.py

  outputs/.gitkeep
```

### 3.1 `.gitignore` 필수 내용

```gitignore
# Large data and generated artifacts
data/raw/*
data/processed/*
data/manifests/*
checkpoints/*
outputs/*
runs/*
wandb/*
artifacts/*

# EEG/data formats
*.h5
*.hdf5
*.npz
*.npy
*.mat
*.edf
*.bdf
*.fif
*.csv.gz
*.parquet

# Model/checkpoint formats
*.ckpt
*.pt
*.pth
*.safetensors

# Secrets/local paths
.env
.env.*
!.env.example

# Python/cache
__pycache__/
.pytest_cache/
.ruff_cache/
*.egg-info/
.venv/

# Keep directory placeholders
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/manifests/.gitkeep
!checkpoints/.gitkeep
!outputs/.gitkeep
```

---

## 4. 환경과 기본 실행 목표

`requirements.txt` 최소 구성:

```txt
numpy>=1.24
scipy>=1.10
pandas>=2.0
pyarrow>=14.0
h5py>=3.9
mne>=1.6
scikit-learn>=1.3
PyYAML>=6.0
tqdm>=4.66
einops>=0.7
torch>=2.1
tensorboard>=2.14
pytest>=7.4
ruff>=0.4
transformers>=4.45
huggingface_hub>=0.25
datasets>=2.20
```

Optional extras는 `pyproject.toml`에 분리한다.

```toml
[project.optional-dependencies]
openbci = ["brainflow>=5.12"]
dvc = ["dvc>=3.50"]
moabb = ["moabb>=1.1"]
dev = ["pytest>=7.4", "ruff>=0.4"]
```

처음 구현이 끝났을 때 아래 명령이 모두 작동해야 한다.

```bash
python -m pip install -r requirements.txt
python scripts/prepare_synthetic.py --out_dir data/processed/synthetic --n_subjects 8 --n_trials_per_class 20 --n_classes 4 --target_sfreq 200
python scripts/inspect_manifest.py --processed_dir data/processed/synthetic
python scripts/train.py --config configs/train/debug.yaml
python scripts/evaluate.py --config configs/eval/channel_stress.yaml --ckpt outputs/debug/best.pt
pytest -q
```

---

## 5. `configs/assets.yaml`

Repository에는 실제 asset이 아니라 asset을 찾고 검증하는 설정만 둔다.

```yaml
paths:
  data_root: ${env:EEG_DATA_ROOT}
  model_root: ${env:EEG_MODEL_ROOT}
  hf_home: ${env:HF_HOME}
  mne_data: ${env:MNE_DATA}

policies:
  allow_download_during_train: false
  allow_download_during_import: false
  require_explicit_fetch: true
  local_files_only_by_default: true
  max_test_fixture_mb: 1

models:
  reve_base:
    type: huggingface_model
    repo_id: brain-bzh/reve-base
    positions_repo_id: brain-bzh/reve-positions
    cache_dir: ${paths.hf_home}
    requires_hf_login: true
    trust_remote_code: true
    local_files_only: true
    allow_download: false
    required_sample_rate_hz: 200
    freeze: true
    no_redistribution: true

  tiny_eeg_transformer:
    type: local_code
    requires_external_asset: false

datasets:
  synthetic:
    type: generated
    raw_dir: null
    processed_dir: ${paths.data_root}/processed/synthetic_v1
    target_sfreq: 200
    public: false

  beta:
    type: huggingface_dataset
    repo_id: Bingchuan/beta
    raw_dir: ${paths.data_root}/raw/beta
    processed_dir: ${paths.data_root}/processed/beta_v1
    target_sfreq: 200
    expected:
      n_subjects: 70
      n_channels: 64
      n_targets: 40
    fetch:
      method: datasets.load_dataset
      allow_manual_fallback: true

  wang:
    type: moabb_or_manual
    moabb_dataset: Wang2016
    raw_dir: ${paths.data_root}/raw/wang
    processed_dir: ${paths.data_root}/processed/wang_v1
    target_sfreq: 200
    expected:
      n_subjects: 35
      n_channels: 64
      n_targets: 40
    fetch:
      method: moabb
      allow_manual_fallback: true

  wearable:
    type: figshare_or_manual
    raw_dir: ${paths.data_root}/raw/wearable
    processed_dir: ${paths.data_root}/processed/wearable_v1
    target_sfreq: 200
    expected:
      n_subjects: 102
      n_channels: 8
      n_targets: 12
      has_wet_dry: true
      has_impedance: true
    fetch:
      method: manual_or_figshare
      allow_manual_fallback: true

  openbci:
    type: local_private
    raw_dir: ${paths.data_root}/raw/openbci
    processed_dir: ${paths.data_root}/processed/openbci_v1
    target_sfreq: 200
    expected:
      n_channels: 8
      raw_sfreq: 250
    public: false
```

Codex는 `${env:VAR}` interpolation을 `src/cfeg/assets/paths.py` 또는 config loader에서 처리한다.

---

## 6. Asset CLI 구현

### 6.1 `scripts/fetch_reve.py`

목적: REVE model과 position bank를 Hugging Face cache에 받는다. 이 스크립트에서만 다운로드를 허용한다.

CLI:

```bash
huggingface-cli login
python scripts/fetch_reve.py \
  --model brain-bzh/reve-base \
  --positions brain-bzh/reve-positions \
  --cache-dir $HF_HOME
```

구현 지침:

```python
from huggingface_hub import snapshot_download


def fetch_reve(model_id: str, positions_id: str, cache_dir: str | None) -> None:
    snapshot_download(repo_id=positions_id, cache_dir=cache_dir, repo_type="model")
    snapshot_download(repo_id=model_id, cache_dir=cache_dir, repo_type="model")
```

요구사항:

- `HF_TOKEN`이 없고 access error가 나면 login 방법을 안내한다.
- 다운로드한 경로를 출력한다.
- weight를 repository 하위로 복사하지 않는다.
- `--dry-run` option은 cache 존재 여부만 확인한다.

### 6.2 `scripts/fetch_dataset.py`

목적: 공개 dataset을 명시적으로 다운로드하거나, 자동 다운로드가 어려운 dataset은 수동 다운로드 지시를 출력한다.

CLI:

```bash
python scripts/fetch_dataset.py --dataset beta --assets-config configs/assets.yaml
python scripts/fetch_dataset.py --dataset wang --assets-config configs/assets.yaml --method moabb
python scripts/fetch_dataset.py --dataset wearable --assets-config configs/assets.yaml
```

Dataset별 처리:

```text
beta:
  - datasets.load_dataset("Bingchuan/beta") 사용 가능하면 raw_dir/cache에 저장
  - 실패하면 Figshare/Tsinghua manual download 안내

wang:
  - moabb optional dependency가 있으면 Wang2016 loader 사용
  - moabb가 없거나 source가 바뀌면 manual download 안내

wearable:
  - Figshare source를 안내하거나 manual raw_dir 구조를 검증
  - automatic download가 안정적이지 않으면 download하지 말고 clear instruction 출력

openbci:
  - fetch 대상 아님
  - local private recording이므로 openbci_convert.py 안내
```

### 6.3 `scripts/verify_assets.py`

목적: raw/processed/model asset이 준비되어 있는지 확인한다.

CLI:

```bash
python scripts/verify_assets.py --all
python scripts/verify_assets.py --model reve_base
python scripts/verify_assets.py --dataset beta --stage raw
python scripts/verify_assets.py --dataset beta --stage processed
```

검증 항목:

```text
models/reve_base:
  - local HF cache에 model files가 있는지
  - positions repo도 있는지
  - AutoModel.from_pretrained(..., local_files_only=True, trust_remote_code=True)가 가능한지 optional smoke check

datasets/raw:
  - raw_dir 존재
  - 최소 file count
  - expected metadata 또는 shape 확인 가능하면 확인

datasets/processed:
  - signals.h5 존재
  - manifest.parquet 존재
  - class_map.json 존재
  - preprocess_config.yaml 존재
  - schema validator 통과
  - sample 몇 개를 열어 shape/dtype/missing 확인
```

### 6.4 Asset error class

`src/cfeg/assets/errors.py`:

```python
class MissingAssetError(RuntimeError):
    pass

class AssetVerificationError(RuntimeError):
    pass
```

모든 missing asset error에는 실행할 명령어를 포함한다.

---

## 7. DVC/Git LFS 정책

기본 정책은 **Git에는 코드와 작은 config만 저장**하는 것이다.

### 7.1 DVC 사용 시

팀 내부에서 processed dataset이나 직접 학습한 checkpoint를 공유해야 할 때만 DVC를 선택적으로 붙인다.

허용:

```bash
dvc init
dvc remote add -d eegstore s3://<bucket-or-team-storage>/calibration_free_eeg
dvc add data/processed/beta_v1
dvc add checkpoints/public_pretrain/best.pt
git add data/processed/beta_v1.dvc checkpoints/public_pretrain/best.pt.dvc .dvc/config
```

주의:

- REVE weight는 DVC remote에 올리지 않는다.
- OpenBCI 원본 데이터는 피험자 동의/비식별화 정책이 정리되기 전까지 DVC remote에 올리지 않는다.
- 공개 dataset도 license가 허용하는 경우에만 team remote에 cache한다.

### 7.2 Git LFS 사용 시

Git LFS는 pointer 파일로 큰 파일을 다룰 수 있지만, 이 프로젝트에서는 기본적으로 사용하지 않는다. 사용하더라도 다음에만 제한한다.

허용 가능:

```text
작은 공개 가능 demo artifact
논문/포스터용 figure 원본
직접 학습한 공개 가능한 small checkpoint
```

금지:

```text
REVE weight
접근 조건이 있는 Hugging Face model weight
OpenBCI 개인/원본 recording
license가 불명확한 raw public dataset copy
```

---

## 8. 데이터 포맷 설계

### 8.1 공통 processed format

모든 raw dataset은 아래 형식으로 변환한다.

```text
{EEG_DATA_ROOT}/processed/{dataset_name}_v{version}/
  signals.h5
  manifest.parquet
  manifest.jsonl              # optional fallback/debug
  class_map.json
  preprocess_config.yaml
  asset_info.json
```

`signals.h5` datasets:

```text
/x              float32 [N, C_max, T]
/channel_mask   bool    [N, C_max]
/y              int64   [N]
```

`asset_info.json` 예시:

```json
{
  "dataset_id": "beta",
  "raw_dir": "/mnt/eeg_data/raw/beta",
  "processed_dir": "/mnt/eeg_data/processed/beta_v1",
  "created_by": "scripts/prepare_dataset.py",
  "target_sfreq": 200.0,
  "preprocess_hash": "...",
  "source": "Bingchuan/beta or manual",
  "notes": "No raw data is stored in repository."
}
```

`manifest.parquet` columns:

```text
sample_id: str
h5_index: int
dataset_id: str
subject_id: str                  # split only, not model input by default
session_id: str                  # split/eval only
run_id: str
trial_id: str
label: int
stimulus_frequency_hz: float     # eval/class mapping only, never ConditionEncoder input
stimulus_phase_rad: float|null   # eval/class mapping only, never ConditionEncoder input
sfreq_original: float
sfreq_processed: float
window_start_sec: float
window_duration_sec: float
reference: str|null
hardware_id: str|null
cap_type: str|null
electrode_type: str|null
n_channels_original: int
n_channels_used: int
channel_names_original: list[str]
channel_names_used: list[str]
canonical_channel_ids: list[int]
impedance_mean_kohm: float|null
impedance_max_kohm: float|null
reattach_flag: bool|null
time_since_last_session_hours: float|null
environment_note_code: str|null  # unknown/quiet/noisy etc. free text 금지
source_file: str
```

Parquet list column 처리가 불편하면 `manifest.jsonl`을 함께 저장해도 된다. 단, 학습 코드는 `manifest.parquet`를 우선 사용한다.

### 8.2 Label leakage 규칙

다음 field는 manifest에는 저장하지만 condition input에는 넣지 않는다.

```text
label
class_id
stimulus_frequency_hz
stimulus_phase_rad
trial_id
subject_id
session_id, by default
source_file
```

`session_id`는 연구 질문에 따라 session-level metadata ablation에서만 사용할 수 있다. 기본 모델에서는 leakage 위험을 줄이기 위해 제외한다.

---

## 9. Canonical channel map

`configs/canonical_channels.yaml`을 둔다. 처음에는 전체 10-20/10-10 system을 완벽히 넣지 않아도 되지만, SSVEP에서 자주 쓰는 occipital/parietal channel은 반드시 포함한다.

```yaml
unknown_id: 0
channels:
  - id: 1
    name: Pz
    aliases: [PZ, pz]
    xyz: null
  - id: 2
    name: PO3
    aliases: [po3]
    xyz: null
  - id: 3
    name: PO4
    aliases: [po4]
    xyz: null
  - id: 4
    name: POz
    aliases: [POZ, poz]
    xyz: null
  - id: 5
    name: PO7
    aliases: [po7]
    xyz: null
  - id: 6
    name: PO8
    aliases: [po8]
    xyz: null
  - id: 7
    name: O1
    aliases: [o1]
    xyz: null
  - id: 8
    name: Oz
    aliases: [OZ, oz]
    xyz: null
  - id: 9
    name: O2
    aliases: [o2]
    xyz: null
```

Unknown channel은 `unknown_id=0`으로 mapping하고 warning을 남긴다.

### 9.1 Channel subset config

`configs/channel_sets.yaml`:

```yaml
occipital_8:
  - Pz
  - PO3
  - PO4
  - POz
  - PO7
  - O1
  - Oz
  - O2
occipital_4:
  - POz
  - O1
  - Oz
  - O2
occipital_2:
  - O1
  - O2
openbci_default_8:
  - Pz
  - PO3
  - PO4
  - POz
  - PO7
  - O1
  - Oz
  - O2
```

평가 시 없는 channel은 skip하고, 남은 channel 수와 mask를 정확히 기록한다.

---

## 10. 전처리 구현

`src/cfeg/data/preprocess.py`:

```python
from dataclasses import dataclass

@dataclass
class PreprocessConfig:
    target_sfreq: float = 200.0
    window_start_sec: float = 0.14
    window_duration_sec: float = 2.0
    bandpass_low_hz: float = 6.0
    bandpass_high_hz: float = 90.0
    notch_hz: float | None = None
    normalize: str = "per_trial_channel_zscore"  # none | per_trial_channel_zscore
    c_max: int = 64
```

필수 함수:

```python
def bandpass_and_resample(x: np.ndarray, sfreq: float, cfg: PreprocessConfig) -> tuple[np.ndarray, float]: ...
def crop_window(x: np.ndarray, sfreq: float, t_start: float, duration: float) -> np.ndarray: ...
def normalize_trial(x: np.ndarray, eps: float = 1e-6) -> np.ndarray: ...
def place_on_canonical_channels(
    x: np.ndarray,
    ch_names: list[str],
    canonical_map: CanonicalChannelMap,
    c_max: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]: ...
```

Shape convention:

```text
raw x:         [C_raw, T_raw]
processed x:   [C_max, T]
channel_mask:  [C_max]
```

전처리 규칙:

1. bandpass와 notch는 config로 끄고 켤 수 있게 한다.
2. 모든 trial은 `target_sfreq`로 resample한다.
3. REVE backbone을 사용할 때는 `target_sfreq == 200.0`을 assert한다.
4. `window_start_sec`, `window_duration_sec` 기준으로 crop한다.
5. 채널별 trial z-score를 기본으로 한다.
6. `C_max`에 맞춰 canonical channel 위치에 신호를 놓고, 없는 channel은 0으로 채운다.
7. label/stimulus 정보는 `y`와 manifest에만 저장한다.

---

## 11. Dataset preparer

공통 CLI:

```bash
python scripts/prepare_dataset.py \
  --dataset wang \
  --raw_dir $EEG_DATA_ROOT/raw/wang \
  --out_dir $EEG_DATA_ROOT/processed/wang_v1 \
  --config configs/data/wang.yaml
```

Dataset별 adapter:

```text
src/cfeg/data/prepare_wang.py
src/cfeg/data/prepare_beta.py
src/cfeg/data/prepare_wearable.py
src/cfeg/data/prepare_openbci.py
```

각 adapter는 다음 함수를 제공한다.

```python
def prepare(raw_dir: Path, out_dir: Path, cfg: dict) -> None: ...
```

실제 공개 데이터 `.mat` 구조가 버전에 따라 다를 수 있으므로, preparer는 다음처럼 robust하게 만든다.

- `scipy.io.loadmat`과 `h5py.File` 둘 다 시도한다.
- 예상 key가 없으면 사용 가능한 key list를 error message에 출력한다.
- data shape 후보 `[channels, time, trials, blocks]`, `[trials, channels, time]` 등을 감지하고 명시적으로 transpose한다.
- channel name 파일이 없으면 config의 `channel_names`를 사용한다.
- metadata가 없는 항목은 `unknown` 또는 `null`로 채운다.
- `sfreq_original`과 `sfreq_processed`를 모두 manifest에 저장한다.
- raw file을 processed output에 복사하지 않는다.

### 11.1 BETA preparer

BETA는 기본적으로 `datasets.load_dataset("Bingchuan/beta")` 또는 manual raw_dir을 지원한다.

```bash
python scripts/fetch_dataset.py --dataset beta
python scripts/prepare_dataset.py --dataset beta --config configs/data/beta.yaml
```

Acceptance:

```text
- expected subject count를 최대한 검증한다.
- 64-channel/40-target class_map이 생성된다.
- 없는 metadata는 unknown/null로 채운다.
- processed output에는 HDF5/manifest/class_map/preprocess_config만 둔다.
```

### 11.2 Wang preparer

Wang은 MOABB loader 또는 manual `.mat` raw_dir을 지원한다.

```bash
pip install -e '.[moabb]'
python scripts/fetch_dataset.py --dataset wang --method moabb
python scripts/prepare_dataset.py --dataset wang --config configs/data/wang.yaml
```

MOABB가 설치되어 있지 않으면 다음 메시지를 출력한다.

```text
MOABB is not installed. Install with:
  pip install -e '.[moabb]'
or manually place Wang2016 raw files under $EEG_DATA_ROOT/raw/wang and rerun prepare_dataset.py.
```

### 11.3 Wearable SSVEP preparer

Wearable SSVEP는 manual/Figshare source를 우선 지원한다.

```bash
python scripts/fetch_dataset.py --dataset wearable
python scripts/verify_assets.py --dataset wearable --stage raw
python scripts/prepare_dataset.py --dataset wearable --config configs/data/wearable.yaml
```

이 dataset은 wet/dry electrode와 impedance metadata가 중요하므로 아래 field를 최대한 채운다.

```text
electrode_type: wet | dry | unknown
cap_type: wearable | unknown
impedance_mean_kohm
impedance_max_kohm
```

### 11.4 OpenBCI preparer

OpenBCI는 공개 fetch 대상이 아니다. `data/raw/openbci/{session}` 구조의 local private data를 변환한다.

```bash
python scripts/openbci_convert.py \
  --raw_session_dir $EEG_DATA_ROOT/raw/openbci/sub001_ses001 \
  --out_dir $EEG_DATA_ROOT/processed/openbci_v1
```

---

## 12. Synthetic SSVEP data

외부 데이터 없이 CI와 smoke test를 통과시키기 위한 synthetic generator를 반드시 구현한다.

`src/cfeg/data/ssvep_synthetic.py`:

- class frequency 예: `[8.0, 10.0, 12.0, 15.0]`
- 각 trial: sine + 2nd/3rd harmonic + channel-specific amplitude + Gaussian noise
- subject별 phase lag, noise level, amplitude scale이 다름
- session별 drift, channel dropout, reference offset을 다르게 부여
- occipital channel에 SSVEP signal을 강하게 넣고 non-occipital channel은 약하게 넣는다
- synthetic data는 실제 인간 EEG가 아니며 repository에 작은 fixture로만 허용한다

CLI:

```bash
python scripts/prepare_synthetic.py \
  --out_dir data/processed/synthetic \
  --n_subjects 8 \
  --n_trials_per_class 20 \
  --n_classes 4 \
  --target_sfreq 200 \
  --duration_sec 2.0
```

Acceptance:

- synthetic data로 `train.py` 실행 시 validation accuracy가 80% 이상까지 올라가야 한다.
- `pytest tests/test_synthetic_overfit.py`에서 작은 model이 2~3 epoch 내 mini dataset을 overfit해야 한다.

---

## 13. DataLoader와 batch schema

`src/cfeg/data/schema.py`:

```python
from dataclasses import dataclass

@dataclass
class EEGSample:
    x: np.ndarray
    y: int
    sample_id: str
    dataset_id: str
    subject_id: str
    session_id: str
    channel_mask: np.ndarray
    canonical_channel_ids: np.ndarray
    sfreq: float
    reference: str | None
    hardware_id: str | None
    electrode_type: str | None
    cap_type: str | None
    n_channels_used: int
    impedance_mean_kohm: float | None
    impedance_max_kohm: float | None
    reattach_flag: bool | None
    time_since_last_session_hours: float | None
```

`src/cfeg/data/collate.py`는 batch를 아래 dict로 만든다.

```python
batch = {
    "x": FloatTensor[B, C_max, T],
    "y": LongTensor[B],
    "sample_id": list[str],
    "split_meta": {
        "dataset_id_str": list[str],
        "subject_id": list[str],
        "session_id": list[str],
    },
    "cond": {
        "dataset_id": LongTensor[B],
        "reference": LongTensor[B],
        "hardware_id": LongTensor[B],
        "electrode_type": LongTensor[B],
        "cap_type": LongTensor[B],
        "reattach_flag": LongTensor[B],
        "channel_ids": LongTensor[B, C_max],
        "channel_mask": BoolTensor[B, C_max],
        "continuous": FloatTensor[B, F_cont],
        "continuous_missing": BoolTensor[B, F_cont],
    },
}
```

Continuous feature order:

```text
0: sfreq_processed / 250.0
1: log1p(n_channels_used) / log1p(C_max)
2: impedance_mean_kohm normalized, missing이면 0
3: impedance_max_kohm normalized, missing이면 0
4: log1p(time_since_last_session_hours) normalized, missing이면 0
```

Missing continuous value는 0으로 채우고, missing 여부는 `continuous_missing`에 별도로 넣는다.

---

## 14. Split 설계

`src/cfeg/data/splits.py`:

```python
def make_cross_subject_split(manifest: pd.DataFrame, seed: int, val_ratio: float, test_ratio: float) -> SplitIndices: ...
def make_within_dataset_leave_subjects_out(manifest: pd.DataFrame, dataset_id: str, seed: int) -> SplitIndices: ...
def make_cross_dataset_split(manifest: pd.DataFrame, train_datasets: list[str], test_datasets: list[str]) -> SplitIndices: ...
def make_openbci_external_split(manifest: pd.DataFrame) -> SplitIndices: ...
```

금지 사항:

- cross-subject 실험에서 같은 subject가 train/test 양쪽에 들어가면 안 된다.
- augmentation view가 서로 다른 split에 들어가면 안 된다.
- trial id나 session id를 model condition input으로 넣어 split leakage를 만들면 안 된다.
- public pretrain 후 OpenBCI external validation에서는 OpenBCI trial로 model update를 하지 않는다. calibration comparison 실험만 별도 config에서 허용한다.

---

## 15. Augmentation과 consistency learning

`src/cfeg/data/transforms.py`:

```python
class ChannelSubset:
    def __init__(self, subset_names: list[str], p: float): ...
    def __call__(self, x, cond) -> tuple[torch.Tensor, dict]: ...

class RandomChannelDropout:
    def __init__(self, drop_prob: float, min_channels: int): ...

class GaussianNoise:
    def __init__(self, std_range: tuple[float, float], p: float): ...

class TimeShift:
    def __init__(self, max_shift_samples: int, p: float): ...
```

Transforms는 `x`만 바꾸면 안 된다. 반드시 아래 condition fields도 같이 업데이트한다.

```text
channel_mask
channel_ids
continuous[n_channels_used]
continuous_missing
```

학습에서는 같은 trial에서 두 view를 만든다.

```python
view1 = weak_transform(sample)
view2 = strong_transform(sample)  # channel subset/dropout/noise/time shift
```

Loss:

```python
L_cls = CE(logits1, y) + CE(logits2, y)
L_cons = MSE(normalize(h1), normalize(h2))
L_logit_cons = symmetric_KL(softmax(logits1), softmax(logits2))  # optional
loss = L_cls + lambda_cons * L_cons + lambda_logit_cons * L_logit_cons + beta_kl * L_kl
```

---

## 16. 모델 구현

### 16.1 Backbone interface

`src/cfeg/models/backbones/base.py`:

```python
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class BackboneOutput:
    h: torch.Tensor
    tokens: torch.Tensor | None
    aux: dict[str, torch.Tensor]

class EEGBackbone(nn.Module):
    d_model: int
    supports_prompt_tokens: bool

    def forward(
        self,
        x: torch.Tensor,
        cond: dict[str, torch.Tensor],
        prompt_tokens: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> BackboneOutput:
        raise NotImplementedError
```

### 16.2 TinyEEGTransformerBackbone

`src/cfeg/models/backbones/tiny_transformer.py`는 기본 runnable model이다.

입력:

```text
x: [B, C_max, T]
cond["channel_ids"]: [B, C_max]
cond["channel_mask"]: [B, C_max]
prompt_tokens: [B, P, D] or None
```

구조:

1. EEG time axis를 patch로 분할한다.
2. patch를 `Linear(patch_size -> d_model)`로 token화한다.
3. channel embedding과 time embedding을 더한다.
4. `[CLS] + prompt_tokens + eeg_tokens` sequence를 만든다.
5. TransformerEncoder를 통과시킨다.
6. CLS token을 representation `h`로 사용한다.

Pseudo-code:

```python
patches = unfold_time(x, patch_size)                 # [B, C, Np, patch]
tok = self.patch_embed(patches)                      # [B, C, Np, D]
tok = tok + channel_embedding(channel_ids)[:, :, None, :]
tok = tok + time_embedding[:, None, :Np, :]
tok = tok.reshape(B, C * Np, D)

cls = self.cls.expand(B, 1, D)
seq = concat([cls, prompt_tokens, tok], dim=1)
key_padding_mask = build_mask_from_channel_mask(channel_mask, Np, prompt_len=P)
out = self.encoder(seq, src_key_padding_mask=key_padding_mask)
h = out[:, 0]
```

주의: PyTorch `src_key_padding_mask`는 `True`가 ignore/pad 위치이다.

### 16.3 REVEBackbone wrapper

`src/cfeg/models/backbones/reve.py`:

REVE는 optional external backbone이다. 기본 구현은 Hugging Face `AutoModel`을 사용한다. REVE와 position bank 모두 local cache에서 로드하며, training/eval script가 자동 다운로드하지 않도록 기본값은 `local_files_only=True`이다.

Config 예시:

```yaml
model:
  backbone:
    name: reve
    hf_model: brain-bzh/reve-base
    hf_positions: brain-bzh/reve-positions
    cache_dir: ${env:HF_HOME}
    trust_remote_code: true
    local_files_only: true
    allow_download: false
    freeze: true
    required_sample_rate_hz: 200
    fallback: tiny_transformer
```

Wrapper skeleton:

```python
class REVEBackbone(EEGBackbone):
    supports_prompt_tokens = False

    def __init__(self, cfg: dict):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as e:
            raise MissingAssetError(
                "transformers is required for REVE. Install requirements or use model.backbone=tiny_transformer."
            ) from e

        self.pos_bank = AutoModel.from_pretrained(
            cfg["hf_positions"],
            cache_dir=cfg.get("cache_dir"),
            trust_remote_code=cfg.get("trust_remote_code", True),
            local_files_only=cfg.get("local_files_only", True),
        )
        self.reve = AutoModel.from_pretrained(
            cfg["hf_model"],
            cache_dir=cfg.get("cache_dir"),
            trust_remote_code=cfg.get("trust_remote_code", True),
            local_files_only=cfg.get("local_files_only", True),
            dtype="auto",
        )
        for p in self.reve.parameters():
            p.requires_grad = False

    def forward(self, x, cond, prompt_tokens=None, return_tokens=False):
        # x: [B, C, T], must be sampled at 200 Hz for REVE configs
        sfreq = cond.get("sfreq_processed_float")
        # caller should already have asserted sfreq == 200.0
        electrode_names = cond.get("channel_names_used")
        positions = self.pos_bank(electrode_names)
        positions = positions.expand(x.size(0), -1, -1)
        out = self.reve(x, positions)
        h = extract_reve_representation(out)
        return BackboneOutput(h=h, tokens=None, aux={})
```

실제 remote code의 output field 이름이 다를 수 있으므로 `extract_reve_representation(out)` helper를 구현한다.

```python
def extract_reve_representation(out):
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state.mean(dim=1)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for key in ["pooler_output", "last_hidden_state", "embeddings", "h"]:
            if key in out:
                value = out[key]
                return value.mean(dim=1) if value.ndim == 3 else value
    raise RuntimeError(f"Cannot extract representation from REVE output type: {type(out)}")
```

REVE가 prompt token prepend를 지원하지 않으면, prompt는 backbone 입력 앞에 붙이지 않고 다음 중 하나로 처리한다.

1. TinyTransformer에서는 prompt token prepend를 사용한다.
2. REVE에서는 condition-dependent feature adapter를 `h` 뒤에 적용한다.
3. REVE adapter는 `h + Adapter(h, cond_vec)` 방식으로 시작한다.

REVE asset이 없을 때의 error:

```text
REVE package/checkpoint not found in local Hugging Face cache.
Run:
  huggingface-cli login
  python scripts/fetch_reve.py --model brain-bzh/reve-base --positions brain-bzh/reve-positions
or use:
  model.backbone=tiny_transformer
```

### 16.4 ConditionEncoder

`src/cfeg/models/condition_encoder.py`:

```python
class ConditionEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_prompt_tokens: int,
        vocab_sizes: dict[str, int],
        n_cont_features: int,
        channel_vocab_size: int,
        dropout: float = 0.1,
    ): ...

    def forward(self, cond: dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor]:
        # returns prompt_tokens [B, P, D] or None, cond_vec [B, D]
```

입력 feature:

- categorical embeddings:
  - dataset_id
  - reference
  - hardware_id
  - electrode_type
  - cap_type
  - reattach_flag
- continuous MLP:
  - `[continuous, continuous_missing.float()]`
- channel summary:
  - `channel_embed(channel_ids)`를 mask-aware mean pooling

Fusion pseudo-code:

```python
cat_vec = sum(embedding_i(cond[name]) for name in cat_names)
cont_vec = cont_mlp(torch.cat([continuous, continuous_missing.float()], dim=-1))
ch_vec = masked_mean(channel_embed(channel_ids), channel_mask)
cond_vec = fuse_mlp(torch.cat([cat_vec, cont_vec, ch_vec], dim=-1))
prompt = to_prompt(cond_vec).view(B, n_prompt_tokens, d_model) if n_prompt_tokens > 0 else None
return prompt, cond_vec
```

Acceptance:

- missing metadata는 unknown category 또는 missing mask로 처리된다.
- `n_prompt_tokens=0`이면 prompt 없는 baseline이 동작한다.
- condition encoder만 켜고 EEG 신호를 넣지 않는 모델은 만들지 않는다. condition은 보조 입력이다.

### 16.5 Adapter

`src/cfeg/models/adapters.py`:

```python
class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.up(self.act(self.down(self.ln(x)))))
```

MVP에서는 backbone output `h`에 feature adapter만 적용한다. 이후 TinyTransformer layer 내부 token adapter를 option으로 추가한다.

Condition-aware adapter option:

```python
class ConditionedAdapter(nn.Module):
    def forward(self, h, cond_vec):
        gate = torch.sigmoid(self.gate(cond_vec))
        return h + gate * self.adapter(h)
```

### 16.6 LatentNuisanceEncoder

`src/cfeg/models/latent_nuisance.py`:

```python
class LatentNuisanceEncoder(nn.Module):
    def __init__(self, h_dim: int, cond_dim: int, z_dim: int = 16): ...

    def forward(self, h: torch.Tensor, cond_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h.detach()를 기본 사용
        inp = torch.cat([h.detach(), cond_vec], dim=-1)
        mu, logvar = self.net(inp).chunk(2, dim=-1)
        return mu, logvar

    def sample(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)
```

Inference mismatch를 줄이기 위해 train 중에도 zero latent branch를 함께 학습한다.

```python
if training and use_latent:
    mu, logvar = q_phi(h, cond_vec)
    z = sample(mu, logvar)
else:
    z = zeros([B, z_dim])

logits_z = head(concat([h, z]))
logits_zero = head(concat([h, zeros_like(z)]))
loss_cls = CE(logits_z, y) + ce_zero_weight * CE(logits_zero, y)
```

기본 config:

```yaml
latent:
  enabled: true
  z_dim: 16
  beta_kl: 0.001
  ce_zero_weight: 0.5
  z_dropout: 0.5
  inference_mode: zero  # zero | train_mean
```

### 16.7 Full model

`src/cfeg/models/full_model.py`:

```python
from dataclasses import dataclass

@dataclass
class ModelOutput:
    logits: torch.Tensor
    logits_zero: torch.Tensor | None
    h: torch.Tensor
    prompt_tokens: torch.Tensor | None
    cond_vec: torch.Tensor | None
    z: torch.Tensor | None
    mu: torch.Tensor | None
    logvar: torch.Tensor | None
    aux: dict[str, torch.Tensor]

class ConditionedEEGDecoder(nn.Module):
    def forward(self, x, cond, use_latent: bool | None = None, return_repr: bool = True) -> ModelOutput:
        prompt, cond_vec = condition_encoder(cond) if enabled else (None, None)
        backbone_out = backbone(x, cond=cond, prompt_tokens=prompt)
        h = backbone_out.h
        h = adapter(h) if adapter is not None else h
        z, mu, logvar = maybe_latent(h, cond_vec, use_latent)
        logits = head(h, z)
        logits_zero = head(h, zero_z) if latent_enabled else None
        return ModelOutput(...)
```

Classification head:

```python
class ClassificationHead(nn.Module):
    def __init__(self, h_dim: int, n_classes: int, z_dim: int = 0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(h_dim + z_dim),
            nn.Linear(h_dim + z_dim, h_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h_dim, n_classes),
        )
```

---

## 17. Losses

`src/cfeg/losses.py`:

```python
def representation_consistency_loss(h1, h2):
    h1 = F.normalize(h1, dim=-1)
    h2 = F.normalize(h2, dim=-1)
    return F.mse_loss(h1, h2)


def symmetric_kl_logits(logits_a, logits_b, temperature=1.0): ...


def kl_normal(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
```

Training step pseudo-code:

```python
out1 = model(x1, cond1, use_latent=True)
out2 = model(x2, cond2, use_latent=True)

loss_cls = CE(out1.logits, y) + CE(out2.logits, y)
if out1.logits_zero is not None:
    loss_cls = loss_cls + ce_zero_weight * (CE(out1.logits_zero, y) + CE(out2.logits_zero, y))

loss_cons = representation_consistency_loss(out1.h, out2.h)
loss_logit_cons = symmetric_kl_logits(out1.logits, out2.logits)
loss_kl = kl_normal(out1.mu, out1.logvar) + kl_normal(out2.mu, out2.logvar) if latent else 0

loss = loss_cls + lambda_cons * loss_cons + lambda_logit_cons * loss_logit_cons + beta_kl * loss_kl
```

---

## 18. Training loop

`src/cfeg/train_loop.py`는 pure PyTorch로 작성한다. Lightning은 사용하지 않는다.

기능:

- seed 고정
- AMP optional
- gradient clipping
- AdamW optimizer
- cosine or step scheduler optional
- early stopping
- best checkpoint 저장
- TensorBoard logging
- trainable parameter count logging
- config, vocab, class_map 저장
- asset_info 저장
- REVE 사용 시 frozen parameter count와 trainable adapter/condition/head parameter count를 따로 logging

CLI:

```bash
python scripts/train.py --config configs/train/debug.yaml
python scripts/train.py --config configs/train/ssvep_pretrain.yaml data.processed_dirs='[$EEG_DATA_ROOT/processed/wang_v1,$EEG_DATA_ROOT/processed/beta_v1]'
python scripts/train.py --config configs/train/ablation.yaml model.variant=A4_full_latent
```

`configs/train/debug.yaml` 예시:

```yaml
seed: 42
run_name: debug
output_dir: outputs/debug
assets_config: configs/assets.yaml

data:
  processed_dirs:
    - data/processed/synthetic
  split: cross_subject
  batch_size: 16
  num_workers: 2

model:
  n_classes: 4
  backbone:
    name: tiny_transformer
  c_max: 64
  target_sfreq: 200
  t_len: 400
  d_model: 128
  patch_size: 20
  depth: 4
  n_heads: 4
  condition_encoder:
    enabled: true
    n_prompt_tokens: 4
  adapter:
    enabled: true
    bottleneck_dim: 32
  latent:
    enabled: true
    z_dim: 16
    z_dropout: 0.5

train:
  epochs: 20
  lr: 0.0003
  weight_decay: 0.01
  amp: true
  grad_clip_norm: 1.0
  early_stop_patience: 10

loss:
  lambda_cons: 0.1
  lambda_logit_cons: 0.05
  beta_kl: 0.001
  ce_zero_weight: 0.5

augment:
  make_two_views: true
  channel_dropout_prob: 0.2
  min_channels: 4
  noise_std_range: [0.01, 0.05]
  time_shift_samples: 8
```

`configs/model/reve_wrapper.yaml` 예시:

```yaml
model:
  backbone:
    name: reve
    hf_model: brain-bzh/reve-base
    hf_positions: brain-bzh/reve-positions
    cache_dir: ${env:HF_HOME}
    trust_remote_code: true
    local_files_only: true
    allow_download: false
    freeze: true
    fallback: tiny_transformer
  target_sfreq: 200
  condition_encoder:
    enabled: true
    n_prompt_tokens: 4
  adapter:
    enabled: true
    type: conditioned_feature_adapter
  latent:
    enabled: true
```

---

## 19. Evaluation

`src/cfeg/eval_loop.py`와 `scripts/evaluate.py` 구현.

### 19.1 Metrics

`src/cfeg/metrics.py`:

- accuracy
- balanced accuracy
- macro F1
- top-k accuracy optional
- negative log-likelihood
- expected calibration error, ECE
- confusion matrix CSV
- ITR bits/min optional

ITR:

```python
def itr_bits_per_min(n_classes: int, acc: float, trial_time_sec: float) -> float:
    # handle chance-level, acc <= 0, acc >= 1 edge cases
```

### 19.2 평가 시나리오

1. **Cross-subject within dataset**
   - train subject와 test subject를 분리한다.
   - test subject data로 fine-tuning하지 않는다.

2. **Channel stress test**
   - 같은 checkpoint로 test input channel을 64/32/16/8/4/2 또는 `channel_sets.yaml` subset으로 줄인다.
   - 결과 CSV:

```text
model_variant,dataset,split,channel_set,n_channels,accuracy,balanced_acc,macro_f1,nll,ece
```

3. **Cross-dataset**
   - train: Wang + BETA
   - test: Wearable 또는 OpenBCI
   - class label overlap만 평가한다.

4. **Calibration comparison**
   - k=0: calibration-free, 핵심 결과
   - k=1,3,5: target subject에서 class당 k개 trial로 head/adapter만 fine-tune하는 비교 실험

5. **OpenBCI external validation**
   - public dataset으로 학습한 checkpoint를 OpenBCI processed data에 적용한다.
   - 초기에는 4-class 또는 8-class subset만 평가한다.

### 19.3 Ablation variants

`configs/train/ablation.yaml` 또는 `scripts/run_ablation.py`에서 다음 variant를 자동 실행한다.

```text
A0_eeg_only:
  condition_encoder: off
  adapter: off
  latent: off
  consistency: off

A1_dataset_id_prompt:
  condition_encoder: on, but only dataset_id category
  adapter: on
  latent: off
  consistency: off

A2_structured_condition_prompt:
  condition_encoder: full metadata
  adapter: on
  latent: off
  consistency: off

A3_prompt_adapter_consistency:
  condition_encoder: full metadata
  adapter: on
  latent: off
  consistency: on

A4_full_latent:
  condition_encoder: full metadata
  adapter: on
  latent: on
  consistency: on

A5_full_finetune_optional:
  unfreeze backbone
  condition_encoder: full metadata
  adapter: on
  latent: on
  consistency: on
```

A5는 시간/GPU가 부족하면 skip 가능하다. 단, results summary에 skip 이유를 기록한다.

---

## 20. Classical baseline: FBCCA

SSVEP는 calibration-free classical baseline이 있어야 한다.

`src/cfeg/baselines/fbcca.py`:

```python
def make_reference_signals(freqs, sfreq, n_samples, n_harmonics=3): ...
def cca_score(x, ref): ...
def predict_fbcca(x, freqs, sfreq, filterbank=None): ...
```

주의:

- FBCCA는 후보 stimulus frequency를 알고 class score를 계산하는 baseline이다.
- Neural model의 ConditionEncoder에는 stimulus frequency를 넣으면 안 된다.
- Neural model과 같은 split에서 FBCCA 결과를 CSV로 저장한다.

---

## 21. OpenBCI 데이터 수집/변환

### 21.1 초기 protocol

처음부터 40-class를 구현하지 않는다. 다음 중 하나로 시작한다.

```text
4-class: 8, 10, 12, 15 Hz
8-class: 8, 9, 10, 11, 12, 13, 14, 15 Hz
```

Trial structure:

```text
1.0 sec fixation/rest
2.0 sec flicker stimulus
1.0 sec rest
```

권장 세션 규모:

```text
4-class: class당 20 trials = 80 trials/session
8-class: class당 10 trials = 80 trials/session
```

### 21.2 Raw folder format

```text
$EEG_DATA_ROOT/raw/openbci/sub001_ses001/
  eeg.csv
  events.csv
  session_meta.json
```

`session_meta.json` 예시:

```json
{
  "dataset_id": "openbci",
  "hardware_id": "openbci_cyton",
  "sfreq": 250.0,
  "reference": "openbci_default",
  "cap_type": "wet_cap",
  "electrode_type": "gel",
  "channel_names": ["Pz", "PO3", "PO4", "POz", "PO7", "O1", "Oz", "O2"],
  "session_id": "sub001_ses001",
  "subject_id": "sub001",
  "reattach_flag": true,
  "time_since_last_session_hours": null,
  "environment_note_code": "quiet",
  "impedance_kohm_by_channel": {
    "Pz": 12.0,
    "PO3": 10.5,
    "PO4": 11.2,
    "POz": 9.8,
    "PO7": 15.0,
    "O1": 13.1,
    "Oz": 10.0,
    "O2": 12.4
  }
}
```

`events.csv`:

```csv
trial_id,onset_sec,duration_sec,class_id,stimulus_frequency_hz,stimulus_phase_rad
0001,12.500,2.0,0,8.0,0.0
0002,17.200,2.0,1,10.0,0.0
```

`openbci_convert.py`:

```bash
python scripts/openbci_convert.py \
  --raw_session_dir $EEG_DATA_ROOT/raw/openbci/sub001_ses001 \
  --out_dir $EEG_DATA_ROOT/processed/openbci_v1 \
  --target_sfreq 200
```

`openbci_record.py`는 BrainFlow가 설치되어 있을 때만 hardware recording을 지원한다. BrainFlow 미설치 시 다음 메시지와 함께 종료한다.

```text
BrainFlow is not installed. Install brainflow or export CSV from OpenBCI GUI and use scripts/openbci_convert.py.
```

### 21.3 개인정보와 연구윤리

OpenBCI 데이터는 local private asset이다.

- `subject_id`는 pseudonym만 사용한다.
- 이름, 학번, 이메일, 전화번호를 metadata에 넣지 않는다.
- raw recording과 consent 관련 문서는 repository에 넣지 않는다.
- 외부 공유 전에는 별도 비식별화/동의 범위를 확인한다.

---

## 22. Config system

Hydra 대신 YAML + argparse override를 구현한다.

`src/cfeg/utils/config.py`:

```python
def load_config(path: str) -> dict: ...
def merge_overrides(cfg: dict, overrides: list[str]) -> dict:
    # support nested.key=value
```

CLI:

```bash
python scripts/train.py --config configs/train/debug.yaml train.epochs=5 model.latent.enabled=false
```

Override parser는 list/dict parsing이 복잡하면 처음에는 JSON string만 지원해도 된다.

환경 변수 interpolation:

```python
${env:EEG_DATA_ROOT}
${env:EEG_MODEL_ROOT}
${env:HF_HOME}
```

환경 변수가 없으면 명확한 error를 낸다.

```text
Environment variable EEG_DATA_ROOT is not set.
Set it before using public datasets:
  export EEG_DATA_ROOT=/path/to/eeg_data
```

---

## 23. Vocabularies

ConditionEncoder category vocab은 train split에서 만들고 `unknown` token을 포함한다.

기본 category:

```text
dataset_id: unknown, synthetic, wang, beta, wearable, openbci
reference: unknown, average, linked_mastoids, cz, openbci_default
hardware_id: unknown, public_unknown, openbci_cyton
cap_type: unknown, wet_cap, dry_cap, wearable
electrode_type: unknown, wet, dry, gel
reattach_flag: unknown, false, true
```

Test에서 처음 보는 category는 unknown으로 mapping한다.

---

## 24. Checkpoint format

`src/cfeg/utils/checkpoint.py`:

```python
ckpt = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict() if scheduler else None,
    "config": cfg,
    "epoch": epoch,
    "best_metric": best_metric,
    "vocabularies": vocabularies,
    "class_map": class_map,
    "asset_info": asset_info,
    "train_z_mean": train_z_mean if latent_enabled else None,
}
```

Evaluation script는 checkpoint 안의 config/class_map/vocab을 우선 사용한다.

Checkpoint도 기본적으로 Git에 commit하지 않는다. 연구 결과 공유가 필요하면 DVC remote 또는 별도 artifact storage를 사용한다.

---

## 25. Logging과 outputs

각 run output:

```text
outputs/{run_name}/
  config.yaml
  asset_info.json
  vocab.json
  class_map.json
  train.log
  metrics_train.csv
  metrics_val.csv
  best.pt
  last.pt
  tensorboard/
  eval/
    test_metrics.csv
    channel_stress.csv
    cross_dataset.csv
    confusion_matrix.csv
```

`export_results_table.py`는 여러 run의 CSV를 합쳐 다음 파일을 만든다.

```text
outputs/summary/ablation_results.csv
outputs/summary/channel_stress_results.csv
outputs/summary/openbci_external_results.csv
```

---

## 26. 테스트 계획

### 26.1 Unit tests

`tests/test_assets.py`

- env var interpolation 검사
- missing asset error message에 해결 명령이 포함되는지 검사
- verify_assets가 synthetic processed dir을 통과시키는지 검사
- REVE가 없을 때 tests 전체가 fail하지 않고 skip 또는 fallback하는지 검사

`tests/test_schema.py`

- manifest required columns 검사
- label leakage field가 condition input에 들어가지 않는지 검사

`tests/test_condition_encoder.py`

- missing category/continuous 처리
- output shape `[B, P, D]`
- `n_prompt_tokens=0` 처리

`tests/test_model_shapes.py`

- TinyTransformer forward shape
- Full model forward shape
- latent on/off shape
- channel_mask에 pad channel이 있어도 forward 성공

`tests/test_transforms.py`

- ChannelSubset 후 mask와 n_channels_used 일치
- label이 변하지 않음
- all channels dropout 방지

`tests/test_losses.py`

- consistency loss finite
- KL finite
- symmetric KL finite

`tests/test_synthetic_overfit.py`

- synthetic tiny dataset에서 loss가 감소해야 함
- debug model이 mini-batch를 overfit할 수 있어야 함

### 26.2 Smoke tests

```bash
python scripts/prepare_synthetic.py --out_dir /tmp/cfeg_synth --n_subjects 4 --n_trials_per_class 4 --n_classes 4 --target_sfreq 200
python scripts/train.py --config configs/train/debug.yaml data.processed_dirs='[/tmp/cfeg_synth]' train.epochs=2
python scripts/evaluate.py --ckpt outputs/debug/best.pt --config configs/eval/channel_stress.yaml
```

---

## 27. 구현 단계별 작업 목록

### Phase 0: repository skeleton

- [ ] 파일/폴더 구조 생성
- [ ] requirements, pyproject, README 작성
- [ ] `.env.example`, `.gitignore`, `data/README.md` 작성
- [ ] config loader 구현
- [ ] env var interpolation 구현
- [ ] seed/logging/checkpoint utility 구현
- [ ] asset error class 구현
- [ ] pytest가 import error 없이 실행되게 만들기

Acceptance:

```bash
pytest -q
```

이 단계에서는 아직 일부 test는 skip 가능하지만 import error는 없어야 한다.

### Phase 1: asset config + synthetic data + processed format

- [ ] `configs/assets.yaml` 구현
- [ ] `scripts/verify_assets.py` skeleton 구현
- [ ] Canonical channel map loader
- [ ] HDF5 writer/reader
- [ ] manifest schema validator
- [ ] synthetic SSVEP generator
- [ ] `prepare_synthetic.py`
- [ ] `inspect_manifest.py`

Acceptance:

```bash
python scripts/prepare_synthetic.py --out_dir data/processed/synthetic --n_subjects 4 --n_trials_per_class 5 --target_sfreq 200
python scripts/inspect_manifest.py --processed_dir data/processed/synthetic
python scripts/verify_assets.py --dataset synthetic --stage processed
pytest tests/test_schema.py tests/test_assets.py -q
```

### Phase 2: DataLoader + transforms

- [ ] `EEGProcessedDataset`
- [ ] collate function
- [ ] train/val/test split functions
- [ ] ChannelSubset, RandomChannelDropout, GaussianNoise, TimeShift
- [ ] two-view generation

Acceptance:

```bash
pytest tests/test_transforms.py -q
```

### Phase 3: model MVP

- [ ] ConditionEncoder
- [ ] TinyEEGTransformerBackbone
- [ ] Adapter
- [ ] ClassificationHead
- [ ] Full model wrapper
- [ ] losses

Acceptance:

```bash
pytest tests/test_condition_encoder.py tests/test_model_shapes.py tests/test_losses.py -q
```

### Phase 4: train/eval loop

- [ ] train.py
- [ ] evaluate.py
- [ ] metrics
- [ ] TensorBoard logging
- [ ] checkpoint save/load
- [ ] synthetic overfit test

Acceptance:

```bash
python scripts/train.py --config configs/train/debug.yaml
python scripts/evaluate.py --config configs/eval/channel_stress.yaml --ckpt outputs/debug/best.pt
pytest tests/test_synthetic_overfit.py -q
```

### Phase 5: latent nuisance + consistency

- [ ] LatentNuisanceEncoder
- [ ] zero latent branch
- [ ] KL loss
- [ ] z dropout
- [ ] consistency pair loss
- [ ] ablation configs A0~A4

Acceptance:

```bash
python scripts/run_ablation.py --config configs/train/ablation.yaml --only synthetic
python scripts/export_results_table.py --runs outputs/ablation_* --out outputs/summary/ablation_results.csv
```

### Phase 6: external asset scripts

- [ ] `scripts/fetch_reve.py`
- [ ] `scripts/fetch_dataset.py`
- [ ] `scripts/verify_assets.py` full implementation
- [ ] HF cache/local_files_only logic
- [ ] missing asset error messages
- [ ] docs for asset setup

Acceptance:

```bash
python scripts/verify_assets.py --all
python scripts/fetch_reve.py --dry-run --model brain-bzh/reve-base --positions brain-bzh/reve-positions
```

Dry run은 실제 REVE가 없어도 clear missing message를 내야 한다.

### Phase 7: REVE wrapper

- [ ] REVEBackbone wrapper
- [ ] position bank integration
- [ ] 200 Hz assertion
- [ ] local_files_only default
- [ ] fallback TinyTransformer behavior
- [ ] REVE unavailable시 tests skip/fallback

Acceptance:

```bash
python scripts/train.py --config configs/train/debug.yaml model.backbone.name=tiny_transformer
python scripts/train.py --config configs/train/debug.yaml model.backbone.name=reve train.epochs=1
```

두 번째 명령은 REVE asset이 준비된 환경에서만 통과하면 된다. REVE asset이 없으면 friendly error를 내야 한다.

### Phase 8: public dataset preparers

- [ ] Wang preparer
- [ ] BETA preparer
- [ ] Wearable preparer
- [ ] class_map alignment
- [ ] cross-subject split
- [ ] cross-dataset split

Acceptance depends on raw data availability. Without raw data, preparers should fail with clear instructions, not with cryptic stack traces.

### Phase 9: OpenBCI external validation

- [ ] openbci raw folder schema
- [ ] openbci_convert.py
- [ ] optional openbci_record.py
- [ ] openbci external eval config
- [ ] metadata fields for impedance/reattach/environment
- [ ] subject pseudonym validation

Acceptance:

```bash
python scripts/openbci_convert.py --raw_session_dir $EEG_DATA_ROOT/raw/openbci/example_session --out_dir $EEG_DATA_ROOT/processed/openbci_v1 --target_sfreq 200
python scripts/evaluate.py --config configs/eval/openbci_external.yaml --ckpt outputs/public_pretrain/best.pt
```

---

## 28. README에 반드시 포함할 내용

README에는 다음을 포함한다.

1. 연구 목표 요약
2. 모델 구조 diagram text
3. 설치 방법
4. external asset policy
5. environment variable setup
6. REVE download/cache 방법
7. synthetic data quickstart
8. public data 준비 방법
9. training command
10. evaluation command
11. ablation command
12. OpenBCI data format
13. leakage 방지 규칙
14. REVE wrapper 사용법과 fallback 안내
15. DVC/Git LFS 사용 여부와 금지 대상

Quickstart 예시:

```bash
git clone <repo>
cd calibration_free_eeg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export EEG_DATA_ROOT=$PWD/.local/eeg_data
export EEG_MODEL_ROOT=$PWD/.local/eeg_models
export HF_HOME=$EEG_MODEL_ROOT/huggingface
mkdir -p $EEG_DATA_ROOT $EEG_MODEL_ROOT

python scripts/prepare_synthetic.py --out_dir data/processed/synthetic --n_subjects 8 --n_trials_per_class 20 --target_sfreq 200
python scripts/train.py --config configs/train/debug.yaml
python scripts/evaluate.py --config configs/eval/channel_stress.yaml --ckpt outputs/debug/best.pt
```

REVE setup 예시:

```bash
huggingface-cli login
python scripts/fetch_reve.py --model brain-bzh/reve-base --positions brain-bzh/reve-positions --cache-dir $HF_HOME
python scripts/verify_assets.py --model reve_base
python scripts/train.py --config configs/train/ssvep_pretrain.yaml model.backbone.name=reve
```

Public dataset setup 예시:

```bash
python scripts/fetch_dataset.py --dataset beta
python scripts/verify_assets.py --dataset beta --stage raw
python scripts/prepare_dataset.py --dataset beta --config configs/data/beta.yaml
python scripts/verify_assets.py --dataset beta --stage processed
```

---

## 29. 성공 기준

MVP 성공 기준:

- synthetic SSVEP에서 end-to-end 학습/평가 가능
- model variants A0~A4가 모두 실행 가능
- channel stress evaluation이 CSV로 출력됨
- condition metadata missing 상황에서도 학습이 깨지지 않음
- inference에서 target subject calibration data 없이 prediction 가능
- REVE 없이도 fallback model이 작동함
- asset이 없을 때 cryptic stack trace가 아니라 해결 명령어가 포함된 error가 출력됨

연구 실험 성공 기준:

- Wang/BETA/Wearable 중 최소 1개 공개 데이터셋에서 processed format 변환 성공
- cross-subject split 결과 보고 가능
- 8/4/2 channel reduction 결과 보고 가능
- prompt/adapter/latent ablation 결과 보고 가능
- OpenBCI 4-class 또는 8-class external dataset ingestion 가능
- REVE backbone 사용 환경에서는 200 Hz input assertion과 position bank integration이 통과됨

---

## 30. 구현 시 주의할 함정

1. **Label leakage**
   - stimulus_frequency_hz, stimulus_phase_rad, class_id를 ConditionEncoder에 넣지 않는다.

2. **Subject leakage**
   - subject_id를 condition embedding으로 넣지 않는다.
   - cross-subject split에서 subject가 섞이지 않게 group split을 쓴다.

3. **Asset leakage / repository bloat**
   - raw data, processed data, REVE weight, checkpoint를 Git에 넣지 않는다.
   - 작은 synthetic fixture만 허용한다.

4. **Hidden download**
   - import나 train/eval script에서 몰래 다운로드하지 않는다.
   - 다운로드는 `fetch_*` CLI에서만 수행한다.

5. **Prompt가 EEG를 대체하지 않게 하기**
   - condition-only 성능이 비정상적으로 높으면 leakage를 의심한다.

6. **Latent z inference mismatch**
   - train에서는 z를 쓰지만 inference에서는 z=0일 수 있다.
   - zero latent branch CE를 반드시 둔다.

7. **Dataset별 class mismatch**
   - public datasets와 OpenBCI class frequency가 다르면 common subset만 평가한다.

8. **Channel order mismatch**
   - raw channel order를 그대로 믿지 말고 channel name -> canonical id mapping을 사용한다.

9. **Sampling-rate mismatch**
   - OpenBCI raw는 250 Hz일 수 있지만 REVE는 200 Hz input을 기대한다.
   - processed manifest에 original/processed sfreq를 모두 저장하고, REVE config에서 200 Hz를 assert한다.

10. **OpenBCI protocol overreach**
    - 초기 validation은 4-class/8-class로 제한한다.

11. **REVE dependency block**
    - REVE가 없어도 codebase가 완전히 작동해야 한다.

12. **License/access violation**
    - REVE weight를 재배포하지 않는다.
    - 공개 dataset raw copy를 재업로드하지 않는다.
    - OpenBCI private data를 동의 없이 공유하지 않는다.

---

## 31. Codex 작업 순서 요약

Codex는 다음 순서대로 구현한다.

1. Repository skeleton, `.gitignore`, `.env.example`, config utility를 만든다.
2. Asset registry, env var interpolation, friendly missing-asset errors를 만든다.
3. Synthetic SSVEP generator와 HDF5/manifest format을 만든다.
4. DataLoader, collate, split, transforms를 만든다.
5. TinyEEGTransformer, ConditionEncoder, Adapter, Full model을 만든다.
6. Loss, train loop, eval loop를 만든다.
7. Synthetic overfit와 channel stress evaluation을 통과시킨다.
8. LatentNuisanceEncoder와 consistency loss를 붙인다.
9. Ablation runner를 만든다.
10. `fetch_reve.py`, `fetch_dataset.py`, `verify_assets.py`를 만든다.
11. REVE wrapper를 만들되 REVE가 없어도 fallback이 동작하게 한다.
12. Public dataset preparer skeleton을 만든다.
13. OpenBCI converter를 만든다.
14. README quickstart와 테스트를 완성한다.

최종적으로 아래 명령이 통과하면 MVP 완료로 본다.

```bash
python scripts/prepare_synthetic.py --out_dir data/processed/synthetic --n_subjects 8 --n_trials_per_class 20 --n_classes 4 --target_sfreq 200
python scripts/train.py --config configs/train/debug.yaml
python scripts/evaluate.py --config configs/eval/channel_stress.yaml --ckpt outputs/debug/best.pt
python scripts/run_ablation.py --config configs/train/ablation.yaml --only synthetic
python scripts/verify_assets.py --dataset synthetic --stage processed
pytest -q
```

REVE와 public dataset이 준비된 환경에서는 추가로 아래가 통과해야 한다.

```bash
python scripts/fetch_reve.py --dry-run --model brain-bzh/reve-base --positions brain-bzh/reve-positions
python scripts/verify_assets.py --model reve_base
python scripts/fetch_dataset.py --dataset beta
python scripts/prepare_dataset.py --dataset beta --config configs/data/beta.yaml
python scripts/train.py --config configs/train/ssvep_pretrain.yaml model.backbone.name=reve
```
