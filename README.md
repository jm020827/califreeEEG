# Calibration-Free EEG Decoding

EEG와 획득조건 metadata를 함께 사용해 unseen subject, dataset, channel layout, wet/dry electrode에서 calibration-free SSVEP decoding을 평가하는 연구 코드다. Tiny Transformer는 smoke용이고 최종 구성은 frozen REVE token과 metadata prompt를 trainable cross-attention으로 결합한다.

## 현재 서버 quickstart

~~~bash
git clone https://github.com/jm020827/califreeEEG.git
# private repository 또는 SSH key를 쓰면:
# git clone git@github.com:jm020827/califreeEEG.git
cd califreeEEG

# 현재 서버의 데이터, 모델 cache, W&B 경로 설정
source scripts/env_server.sh

bash scripts/cfeg.sh setup
bash scripts/cfeg.sh assets synthetic
bash scripts/cfeg.sh smoke
bash scripts/cfeg.sh help
~~~

Setup은 의존성만 준비하고 데이터와 weight를 받지 않는다.

## 서버 저장 경로

`scripts/env_server.sh`는 현재 서버의 대용량 자산을 repository 밖에 둔다.

| 용도 | 경로 |
|---|---|
| Hugging Face 상위 설정 | `/mnt/ssd3/jm020827/cache/huggingface` |
| 실제 Hub model/dataset cache | `/mnt/ssd3/jm020827/cache/huggingface/hub` |
| EEG raw/processed/MNE | `/mnt/ssd3/jm020827/califreeEEG/eeg_data` |
| W&B 지속 로그 | `/mnt/ssd3/jm020827/califreeEEG/wandb` |
| 실험 결과와 checkpoint | `/mnt/ssd3/jm020827/califreeEEG/experiments` |
| 임시 파일 | `/mnt/ssd3/jm020827/califreeEEG/tmp` |
| pip cache | `/mnt/ssd3/jm020827/cache/pip/califreeEEG` |

`HF_HOME`은 Hugging Face 전체 상위 경로이고 `HF_HUB_CACHE=$HF_HOME/hub`가
`models--*`, `datasets--*`, `.locks`의 실제 위치다. 예전 코드가 만든 빈
`eeg_models/`와 repository 내부 `.local/`은 사용하지 않는다. 다른 서버에서는
`CFEG_SERVER_ROOT`와 `CFEG_SERVER_CACHE_ROOT`만 재정의하면 된다.

## HF와 W&B

Secret은 환경변수나 각 CLI의 저장된 인증을 사용한다.

~~~bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
export WANDB_MODE=online
export WANDB_PROJECT=calibration-free-eeg
export WANDB_ENTITY=jm020827
~~~

WANDB_API_KEY가 있으면 scripts/cfeg.sh는 online logging을 켜고, 없으면 disabled다. WANDB_MODE=offline도 지원한다.

Token은 Git에 저장하지 않는다.

## Asset

REVE gated access 승인 후:

~~~bash
bash scripts/cfeg.sh assets reve
CFEG_BETA_SUBJECTS=1,2 bash scripts/cfeg.sh assets beta
bash scripts/cfeg.sh assets beta

CFEG_ENABLE_MOABB=1 bash scripts/cfeg.sh setup
CFEG_WANG_SUBJECTS=1,2 bash scripts/cfeg.sh assets wang
bash scripts/cfeg.sh assets wang
~~~

Wearable은 https://figshare.com/articles/dataset/13560281 에서 받고 S001.mat부터 S102.mat, Impedance.mat을 EEG_DATA_ROOT/raw/wearable 아래 둔다.

~~~bash
bash scripts/cfeg.sh assets wearable
~~~

전용 parser가 [channel,time,electrode,block,target], dry/wet, block, impedance, 공식 8채널과 9.25–14.75Hz 12개 target을 읽는다.

## 기존 Wang/BETA label 변환

예전 processed asset에서 `Class ... conflicting frequencies` 오류가 나면 신호를 다시
다운로드하거나 전처리하지 않는다. 주파수 기준으로 manifest, `signals.h5/y`,
`class_map.json`만 변환한다. 원래 label metadata와 y는 processed 폴더 안의
`.label-alignment-backup-*`에 보존된다.

~~~bash
bash scripts/cfeg.sh migrate-labels          # dry-run
bash scripts/cfeg.sh migrate-labels --apply  # 실제 변환, 한 번만
~~~

## Train

~~~bash
CFEG_BACKBONE=tiny_transformer bash scripts/cfeg.sh train wang-to-beta
CFEG_BACKBONE=reve WANDB_MODE=online bash scripts/cfeg.sh train wang-to-beta
~~~

Preset은 wang-to-beta, beta-to-wang, wearable-loso, wearable-dry-to-wet, wearable-wet-to-dry, joint, synthetic이다.

각 run은 `split.csv`, source-validation checkpoint, held-out `metrics_test.json`을 저장한다. Target test는 checkpoint 선택에 절대 사용하지 않으며, source validation이 불가능할 만큼 피험자가 적으면 실행을 중단한다.

Wang과 BETA label은 raw index가 아니라 stimulus frequency로 canonical 40-class 8.0, 8.2, ..., 15.8Hz에 정렬된다. 학습 strong view는 8/4/2채널 subset을 명시적으로 포함한다. Source validation만 checkpoint 선택에 쓰며 target은 test-only다.

## Evaluate, robustness, calibration, inference

~~~bash
bash scripts/cfeg.sh eval wang-to-beta $CFEG_EXPERIMENT_ROOT/research/wang_to_beta/best.pt
bash scripts/cfeg.sh eval beta-to-wang $CFEG_EXPERIMENT_ROOT/research/beta_to_wang/best.pt

bash scripts/cfeg.sh channel-stress $CFEG_EXPERIMENT_ROOT/research/wang_to_beta/best.pt \
  "$EEG_DATA_ROOT/processed/beta_v1"
bash scripts/cfeg.sh robustness $CFEG_EXPERIMENT_ROOT/research/wang_to_beta/best.pt \
  "$EEG_DATA_ROOT/processed/beta_v1"

bash scripts/cfeg.sh calibration $CFEG_EXPERIMENT_ROOT/research/wearable_dry_to_wet/best.pt
bash scripts/cfeg.sh predict <checkpoint.pt> <processed-dir> $CFEG_EXPERIMENT_ROOT/predictions.csv
~~~

Robustness는 metadata 결측 25/50/75/100%, 그룹별 제거, shuffle, downsample, re-reference, broadband/band-limited noise와 복합 4채널 조건을 평가한다. Channel metadata를 가려도 backbone의 실제 electrode 위치 입력은 보존한다. CSV에는 accuracy, balanced accuracy, macro-F1, NLL, ECE, ITR, 기준 대비 절대 저하와 상대 저하율, confusion matrix가 저장된다. Calibration은 피험자별 k=0/1/3/5다.

## Ablation과 전체 suite

~~~bash
bash scripts/cfeg.sh ablation
bash scripts/cfeg.sh ablation A0_eeg_only,A4_full_latent
python scripts/run_ablation.py --include-optional --continue-on-error
CFEG_BACKBONE=reve WANDB_MODE=online bash scripts/cfeg.sh research
~~~

Research는 Wang→BETA와 BETA→Wang의 zero-shot·채널·강건성 평가를 실행한다. wearable_v1이 있으면 leave-subject-out, dry→wet, wet→dry, k=0/1/3/5 calibration도 실행한다. Ablation은 별도 한 번의 명령으로 A0-A4를 Wang→BETA에 수행하며 `CFEG_BACKBONE`과 W&B 환경변수를 그대로 따른다.

## 규칙

- Label, stimulus frequency/phase, subject/session/trial의 원시 식별자와 source file은 ConditionEncoder 입력이 아니다. 세션 조건은 reattach 여부와 경과시간처럼 일반화 가능한 파생 metadata로만 사용한다.
- Dataset-ID-only ablation은 continuous/channel metadata도 사용하지 않는다. Structured-without-ID ablation은 dataset_id도 제거한다.
- Frequency overlap이 없으면 unrelated class id 비교를 거부한다.
- k=0이 핵심 calibration-free 결과다.
- Raw/processed EEG, REVE weight, checkpoint, token, W&B log는 Git 제외다.
- Download는 명시적인 assets 명령에서만 일어난다.
- Frozen REVE는 checkpoint에 복제하지 않고 `HF_HUB_CACHE`에서 다시 읽는다.

상세 완료/남은 실험은 calibration_free_eeg_codex_implementation_plan.md에 있다.
