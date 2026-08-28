# Calibration-Free EEG 구현·실험 체크리스트 v5

업데이트: 2026-08-29

예전 구현 지시서 v2를 현재 상태에 맞춘 체크리스트로 대체한다. 체크 완료는 코드·설정·테스트 경로가 repository에 있다는 뜻이다. 외부 데이터/REVE 항목은 서버 실험을 끝내야 연구 결과까지 완료된다.

## 연구 질문

1. Metadata prompt가 EEG-only보다 unseen subject/dataset/electrode 일반화를 개선하는가?
2. Latent nuisance와 consistency가 채널·sampling·reference·noise·metadata 결측 저하를 줄이는가?
3. Calibration-free k=0이 k=1/3/5와 비교해 어느 accuracy/ITR을 유지하는가?
4. Primary wearable subject DG와 Wang↔BETA external-domain boundary에서 어떤 경향인가?

Protocol 0.3에서 연구 질문은 유지하되 primary를 Wang→BETA에서 wearable 5-fold outer participant wet+dry DG로 옮겼다. Wang/BETA는 공개된 장비·reference·channel 조건이 거의 같아 structured metadata의 주효과를 식별하기 어렵기 때문이다. exact 40-class이면서 acquisition이 다른 Dong2023을 replication으로 추가하고, dry→wet/wet→dry는 unseen categorical level이 `UNK`가 되는 joint subject-condition stress로 해석한다.

## 구현 완료

### 데이터

- [x] 공통 HDF5/manifest/class-map과 leakage 차단
- [x] dataset_id::subject_id split과 train/val/test 고정 `split.csv`
- [x] Wang/BETA 원본 class 순서와 frequency 기반 canonical 40-class 재매핑
- [x] 충돌 class map 결합 차단
- [x] wearable [8,710,2,10,12] 전용 parser
- [x] dry/wet, block, impedance, 공식 8채널/12-class/2초 창(0.64초 onset offset)
- [x] Wang raw 64채널 순서와 BETA 750/1000-sample schema 사전 검증
- [x] wearable 공식 target frequency/phase 순서와 EEG/impedance wet-dry 축 검증
- [x] Figshare wearable subject 선택 다운로드, 크기·MD5 검증과 provenance 저장
- [x] Dong2023 Zenodo subject 선택 다운로드, MD5, `[8,1250,40,4]`, 40-class adapter
- [x] 모든 대용량 asset의 외부 경로

### 모델

- [x] Tiny prompt prepend
- [x] frozen REVE/position bank
- [x] REVE token과 metadata prompt cross-attention
- [x] Transformer condition encoder/condition-gated adapter
- [x] dataset-ID-only continuous/channel 차단
- [x] source-train-only categorical vocabulary, trained `UNK=0`, training-only metadata dropout
- [x] primary A2에서 dataset/opaque hardware ID 제외
- [x] A0/A2 동일 condition encoder·adapter·parameter 수; A0만 metadata 강제 missing
- [x] latent posterior, KL, z-dropout, inference z=0와 zero-latent CE

### 학습·평가

- [x] cross-subject와 학습 중 8/4/2채널 strong views
- [x] Wang→BETA / BETA→Wang source-validation + target-test
- [x] wearable participant-fold DG와 participant-disjoint dry→wet / wet→dry source-validation + target-test
- [x] wearable 5-fold outer participant CV와 fold 내부 source-only validation
- [x] optimization seed와 split seed 분리
- [x] accuracy, balanced accuracy, macro-F1, NLL, ECE, confusion matrix
- [x] ITR와 accuracy/ITR 절대 저하·상대 일반화 저하율
- [x] 64→8→4→2 channel stress와 metadata 25/50/75/100% missing/shuffle
- [x] metadata 그룹별 제거, downsample, re-reference, broadband/band-limited noise, 복합 변환
- [x] 피험자별 k=0/1/3/5 fixed query, disjoint support, partition/query hash
- [x] held-out split 밖의 robustness/calibration 실행 거부
- [x] 실제 regularized CCA와 filter-bank FBCCA 및 training-free 평가 artifact
- [x] subject-level prediction/metric과 learned/forgotten OOD coverage 분석
- [x] EEG-only, dataset-ID-only, structured-without-ID를 포함한 A0-A4 자동 ablation과 선택형 A5
- [x] processed inference, final test metrics, W&B

### 실행

- [x] clone 위치 독립 경로와 PVC override
- [x] `HF_HOME/hub` 표준 cache와 interns NVMe/DDN env profile
- [x] legacy HF root/`.local/eeg_data` 안전한 dry-run migration
- [x] legacy Wang/BETA label을 신호 재처리 없이 canonical frequency로 migration
- [x] optional dependency 분리
- [x] 다운로드 없는 bootstrap과 선택형 asset download
- [x] scripts/cfeg.sh 단일 entrypoint
- [x] base config/output root/override를 받는 재현 가능한 ablation runner
- [x] 93개 전체 unit test, targeted Ruff, compileall, YAML/shell syntax 검증
- [x] 대용량/secret Git 제외

## 서버에서 남은 일

- [ ] REVE gated access와 Kubernetes secrets
- [x] BETA S1/S16과 wearable S1–S3 공개 pilot 전처리·baseline 검증
- [ ] BETA/Wang/Wearable 전체 전처리 검증
- [ ] 실제 REVE 1-epoch smoke
- [ ] primary wearable 5 folds × A0/A2 3–5 seed와 102명 subject-level paired 통계
- [ ] Dong/Wang/BETA leave-one-acquisition-out replication
- [ ] Wang↔BETA boundary 및 joint condition stress 각각 3개 이상 seed
- [ ] A3/A4와 robustness 전체
- [ ] k=0/1/3/5 curve
- [ ] 평균·표준편차와 paired 통계
- [ ] OpenBCI 동의·비식별화 후 외부 검증
- [ ] 최종 표/그림/보고서

## 실행

~~~bash
git clone https://github.com/jm020827/califreeEEG.git
# private repository/SSH 환경이면 git@github.com:jm020827/califreeEEG.git 사용
cd califreeEEG
source scripts/env_k8s_interns.sh
bash scripts/migrate_server_storage.sh
bash scripts/migrate_server_storage.sh --apply  # legacy asset이 있을 때 한 번만
export HF_TOKEN=<secret>
export WANDB_API_KEY=<secret>
export WANDB_MODE=online

bash scripts/cfeg.sh setup
bash scripts/cfeg.sh assets synthetic
bash scripts/cfeg.sh smoke
bash scripts/cfeg.sh assets reve beta
CFEG_ENABLE_MOABB=1 bash scripts/cfeg.sh setup
bash scripts/cfeg.sh assets wang
CFEG_BACKBONE=reve bash scripts/cfeg.sh research
python scripts/run_ablation.py \
  --base-config configs/train/wearable_loso.yaml \
  --output-root outputs/confirmatory/wearable_subject_dg/fold0/seed42 \
  --only F0_backbone_linear,A0_eeg_only,A2_structured_condition_prompt \
  --override data.fold_index=0 --override seed=42
~~~

Wearable 원본은 자동 다운로드하거나 기존 파일을 검증한 뒤 전처리한다.

~~~bash
python scripts/fetch_dataset.py --dataset wearable --subjects 1,2,3 --probe-remote
python scripts/fetch_dataset.py --dataset wearable
bash scripts/cfeg.sh assets wearable
CFEG_BACKBONE=reve bash scripts/cfeg.sh train wearable-loso
bash scripts/cfeg.sh calibration outputs/research/wearable_dry_to_wet/best.pt
~~~

## 판정 규칙

- 핵심 결과는 wearable 5-fold subject DG의 k=0 A2−A0 subject-level paired difference다.
- Wang↔BETA와 dry↔wet 방향은 별도 boundary/stress 결과로 보고한다.
- Wearable은 독립 12-class head다.
- Target test는 early stopping에 사용하지 않으며 source validation이 비면 실행을 거부한다.
- 표에 seed, backbone, source/target, channel, perturbation, budget을 기록한다.
- Tiny는 pipeline/baseline이고 REVE와 구분한다.
- Frequency overlap 없는 class id와 label-frequency가 어긋난 예전 processed asset을 거부한다.

외부 자산을 실행하지 않은 상태에서는 연구 결과 완료로 표시하지 않는다.

서버 표준값은 `HF_HOME=/mnt/nvme/cache/interns/hf`,
`HF_HUB_CACHE=/mnt/nvme/cache/interns/hf/hub`,
`EEG_DATA_ROOT=/mnt/ddn/prod-runs/interns/jm020827/califreeEEG/storage/eeg_data`다.
`eeg_models/`는 더 이상 생성하거나 사용하지 않는다.
