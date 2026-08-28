# Calibration-Free EEG Decoding

EEG와 획득조건 metadata를 함께 사용해 unseen subject, dataset, channel layout, wet/dry electrode에서 calibration-free SSVEP decoding을 평가하는 연구 코드다. Tiny Transformer는 smoke용이고 최종 구성은 frozen REVE token과 metadata prompt를 trainable cross-attention으로 결합한다.

현재 protocol 0.3의 primary는 **wearable 5-fold outer participant CV에서 wet/dry를 모두 source participant에게 보여 준 뒤, 처음 보는 participant를 k=0으로 분류하는 A0 metadata-null 대 A2 structured-metadata 비교**다. 두 모델은 같은 condition encoder와 adapter 및 같은 parameter 수를 쓰고 A0만 metadata를 모두 missing으로 강제한다. 모든 102명은 한 번씩 test가 되며 optimization seed와 split seed를 분리한다. Wang↔BETA는 공개 acquisition metadata가 거의 같아 boundary test로 쓰고, exact 40-class지만 실제 획득조건이 다른 Dong2023을 acquisition replication에 추가한다. 자세한 사전지정과 수정 이유는 [연구 실행 계획](docs/research_execution_plan.md)과 [연구설계 감사](docs/research_design_analysis.md)에 있다.

## Kubernetes quickstart

~~~bash
git clone https://github.com/jm020827/califreeEEG.git
# private repository 또는 SSH key를 쓰면:
# git clone git@github.com:jm020827/califreeEEG.git
cd califreeEEG

# jm020827 interns cluster: exact NVMe/DDN paths
source scripts/env_k8s_interns.sh

# Existing legacy cache/data: inspect first, then migrate once.
bash scripts/migrate_server_storage.sh
bash scripts/migrate_server_storage.sh --apply

bash scripts/cfeg.sh setup
bash scripts/cfeg.sh assets synthetic
bash scripts/cfeg.sh smoke
bash scripts/cfeg.sh help
~~~

Setup은 의존성만 준비하고 데이터와 weight를 받지 않는다.

## 서버 저장 경로

`scripts/env_k8s_interns.sh`는 현재 interns Kubernetes mount를 다음처럼 고정한다.

| 용도 | 경로 |
|---|---|
| Hugging Face 상위 설정 | `/mnt/nvme/cache/interns/hf` |
| 실제 Hub model/dataset cache | `/mnt/nvme/cache/interns/hf/hub` |
| EEG raw/processed/MNE | `/mnt/ddn/prod-runs/interns/jm020827/califreeEEG/storage/eeg_data` |
| W&B 지속 로그 | `/mnt/ddn/prod-runs/interns/jm020827/califreeEEG/storage/wandb` |
| 임시 파일 | `/mnt/nvme/cache/interns/tmp/jm020827/califreeEEG` |
| pip cache | `/mnt/nvme/cache/interns/pip/jm020827/califreeEEG` |

`HF_HOME`은 Hugging Face 전체 상위 경로이고 `HF_HUB_CACHE=$HF_HOME/hub`가
`models--*`, `datasets--*`, `.locks`의 실제 위치다. 예전 코드가 만든 빈
`eeg_models/`는 사용하지 않는다. 기존 HF 루트의 REVE cache와 DDN의
`.local/eeg_data`는 `migrate_server_storage.sh`가 대상 덮어쓰기나 파일시스템 간
이동 없이 정리한다. 기본 실행은 dry-run이고 `--apply`에서만 `mv`한다.

일반 PVC 환경은 서버 프로필 대신 직접 지정한다.

~~~bash
export HF_HOME=/mnt/pvc/hf
export HF_HUB_CACHE=/mnt/pvc/hf/hub
export CFEG_HF_ROOT=/mnt/pvc/hf
export EEG_DATA_ROOT=/mnt/pvc/eeg
export WANDB_DIR=/mnt/pvc/wandb
~~~

## HF와 W&B

Secret은 Pod 환경변수로 주입한다.

~~~bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
export WANDB_MODE=online
export WANDB_PROJECT=calibration-free-eeg
export WANDB_ENTITY=jm020827
~~~

WANDB_API_KEY가 있으면 scripts/cfeg.sh는 online logging을 켜고, 없으면 disabled다. WANDB_MODE=offline도 지원한다.

~~~bash
kubectl -n <namespace> create secret generic califree-credentials \
  --from-literal=HF_TOKEN='<token>' \
  --from-literal=WANDB_API_KEY='<key>'
~~~

Pod spec에는 secretRef로 연결한다. Token은 Git에 저장하지 않는다.

## Asset

REVE gated access 승인 후:

~~~bash
bash scripts/cfeg.sh assets reve
CFEG_BETA_SUBJECTS=1,2 bash scripts/cfeg.sh assets beta
bash scripts/cfeg.sh assets beta

CFEG_ENABLE_MOABB=1 bash scripts/cfeg.sh setup
CFEG_WANG_SUBJECTS=1,2 bash scripts/cfeg.sh assets wang
bash scripts/cfeg.sh assets wang

# Wang/BETA와 exact 40-class이면서 8-channel semi-dry acquisition인 Dong2023
CFEG_DONG2023_SUBJECTS=1,2 bash scripts/cfeg.sh assets dong2023
bash scripts/cfeg.sh assets dong2023
~~~

Wearable은 Figshare에서 subject 파일과 Impedance·공식 설명 파일을 선택 다운로드하며 크기와 MD5를 검증한다.

~~~bash
# 원격 파일 목록만 확인
python scripts/fetch_dataset.py --dataset wearable --subjects 1,2,3 --probe-remote
# 3명 pilot 또는 전체 102명
python scripts/fetch_dataset.py --dataset wearable --subjects 1,2,3
bash scripts/cfeg.sh assets wearable
~~~

전용 parser가 `[channel,time,electrode,block,target]`, dry/wet, block, impedance, 공식 8채널과 9.25–14.75Hz 12개 target을 읽는다. EEG electrode 축은 `[dry, wet]`이지만 Impedance 축은 `[wet, dry]`이며, target frequency/phase도 단순 오름차순이 아니다. 현재 config는 공식 stimulation table 순서를 사용한다. **2026-08-29 이전 wearable config로 만든 processed asset과 그 결과는 삭제 또는 별도 보관하고 반드시 다시 전처리해야 한다.** `window_start_sec=0.64`는 시작 offset이고 실제 window는 2.0초다.

Wang/BETA 전처리는 알려진 원본 tensor schema, target/block 수, 유한값과 공식 channel order를 검증한다. schema가 다르면 heuristic으로 축을 추측하지 않고 중단한다. Wang/BETA의 electrode/cap 재질은 공개 근거가 없어 `unknown`으로 기록한다.

Dong2023은 versioned Zenodo mirror의 size/MD5를 검증하고 `[8,1250,40,4]`를 canonical 8.0–15.8 Hz 40-class로 전처리한다. NEMAR 배포 license는 CC BY-NC 4.0이다.

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

Preset은 wang-to-beta, beta-to-wang, wearable-loso(legacy command name; 실제 config는 outer fold 0), wearable-dry-to-wet, wearable-wet-to-dry, joint, synthetic이다.

각 run은 `split.csv`, source-validation checkpoint, held-out `metrics_test.json`을 저장한다. Target test는 checkpoint 선택에 절대 사용하지 않으며, source validation이 불가능할 만큼 피험자가 적으면 실행을 중단한다.

Wang과 BETA label은 raw index가 아니라 stimulus frequency로 canonical 40-class 8.0, 8.2, ..., 15.8Hz에 정렬된다. 학습 strong view는 8/4/2채널 subset을 명시적으로 포함한다. Source validation만 checkpoint 선택에 쓰며 target은 test-only다.

## Evaluate, robustness, calibration, inference

~~~bash
bash scripts/cfeg.sh eval wang-to-beta outputs/research/wang_to_beta/best.pt
bash scripts/cfeg.sh eval beta-to-wang outputs/research/beta_to_wang/best.pt

bash scripts/cfeg.sh channel-stress outputs/research/wang_to_beta/best.pt \
  "$EEG_DATA_ROOT/processed/beta_v1"
bash scripts/cfeg.sh robustness outputs/research/wang_to_beta/best.pt \
  "$EEG_DATA_ROOT/processed/beta_v1"

bash scripts/cfeg.sh calibration outputs/research/wearable_dry_to_wet/best.pt
bash scripts/cfeg.sh predict <checkpoint.pt> <processed-dir> outputs/predictions.csv
~~~

Robustness는 metadata 결측 25/50/75/100%, 그룹별 제거, shuffle, downsample, re-reference, broadband/band-limited noise와 복합 4채널 조건을 평가한다. Channel metadata를 가려도 backbone의 실제 electrode 위치 입력은 보존한다. checkpoint 옆 `split.csv`의 held-out test ID 또는 명시적 target filter가 없으면 실행을 거부한다. CSV에는 accuracy, balanced accuracy, macro-F1, NLL, ECE, theoretical ITR, 기준 대비 절대 저하와 상대 저하율, confusion matrix가 저장된다.

Calibration은 피험자별 k=0/1/3/5이며 모든 k에서 같은 fixed query를 사용하고 support와 query를 분리한다. partition CSV, query hash, subject metric, sample prediction을 함께 저장한다.

Training-free CCA/FBCCA baseline은 실제 canonical correlation과 sub-band filter bank를 계산한다.

~~~bash
python scripts/evaluate_frequency_baseline.py \
  --processed-dir "$EEG_DATA_ROOT/processed/wearable_v2" \
  --method fbcca --channel-set wearable_8 \
  --selection-csv outputs/confirmatory/wearable_subject_dg/fold0/seed42/A0_eeg_only/split.csv \
  --out-prefix outputs/baselines/wearable_fbcca
~~~

## Ablation과 전체 suite

~~~bash
bash scripts/cfeg.sh ablation
bash scripts/cfeg.sh ablation A0_eeg_only,A4_full_latent
python scripts/run_ablation.py --include-optional --continue-on-error
# primary wearable의 parameter-matched F0/A0/A2: 아래를 fold=0..4, seed=42..에 반복
python scripts/run_ablation.py \
  --base-config configs/train/wearable_loso.yaml \
  --output-root outputs/confirmatory/wearable_subject_dg/fold0/seed42 \
  --only F0_backbone_linear,A0_eeg_only,A2_structured_condition_prompt \
  --override data.fold_index=0 --override seed=42
CFEG_BACKBONE=reve WANDB_MODE=online bash scripts/cfeg.sh research
~~~

`run_ablation.py`는 `--base-config`, `--output-root`, 반복 가능한 `--override KEY=VALUE`를 지원하고 optimization seed·split seed·outer fold·base config·parameter 수·결과를 기록한다. `bash scripts/cfeg.sh research`는 전체 suite 편의 명령이고, confirmatory 결과는 위처럼 fold/seed별 output root를 분리해 실행한다. 5개 fold의 test subject 합집합이 102명이며 seed를 독립 subject로 세지 않는다.

두 모델의 prediction CSV가 준비되면 사전지정 threshold에서 subject×condition cell의 retained/learned/forgotten/unresolved와 paired bootstrap/sign-flip 결과를 만든다.

~~~bash
python scripts/analyze_ood_coverage.py \
  --baseline outputs/a0/predictions.csv \
  --candidate outputs/a2/predictions.csv \
  --processed-dir "$EEG_DATA_ROOT/processed/wearable_v2" \
  --success-threshold <preregistered-BA> \
  --out-prefix outputs/analysis/a0_vs_a2
~~~

## 규칙

- Label, stimulus frequency/phase, subject/session/trial의 원시 식별자와 source file은 ConditionEncoder 입력이 아니다. 세션 조건은 reattach 여부와 경과시간처럼 일반화 가능한 파생 metadata로만 사용한다.
- Dataset-ID-only ablation은 continuous/channel metadata도 사용하지 않는다. Structured-without-ID ablation은 dataset_id도 제거한다.
- Frequency overlap이 없으면 unrelated class id 비교를 거부한다.
- k=0이 핵심 calibration-free 결과다.
- `dataset_id`와 opaque hardware ID는 primary A2에서 제외하고, unseen category는 source train vocabulary의 학습된 `UNK=0`으로 보낸다.
- A0와 A2는 동일 구조·parameter 수를 사용한다. A0를 condition encoder가 없는 작은 모델로 바꾸지 않는다.
- 평균 성능 개선과 OOD coverage 확장을 구분한다. `learned > forgotten` 및 worst-group 개선 전에는 “적용 범위가 확장됐다”고 쓰지 않는다.
- Raw/processed EEG, REVE weight, checkpoint, token, W&B log는 Git 제외다.
- Download는 명시적인 assets 명령에서만 일어난다.
- Frozen REVE는 checkpoint에 복제하지 않고 `HF_HUB_CACHE`에서 다시 읽는다.

상세 완료/남은 실험은 calibration_free_eeg_codex_implementation_plan.md에 있다.
