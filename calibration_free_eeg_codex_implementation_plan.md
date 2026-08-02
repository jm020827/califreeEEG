# Calibration-Free EEG 구현·실험 체크리스트 v4

업데이트: 2026-08-02

예전 구현 지시서 v2를 현재 상태에 맞춘 체크리스트로 대체한다. 체크 완료는 코드·설정·테스트 경로가 repository에 있다는 뜻이다. 외부 데이터/REVE 항목은 서버 실험을 끝내야 연구 결과까지 완료된다.

## 연구 질문

1. Metadata prompt가 EEG-only보다 unseen subject/dataset/electrode 일반화를 개선하는가?
2. Latent nuisance와 consistency가 채널·sampling·reference·noise·metadata 결측 저하를 줄이는가?
3. Calibration-free k=0이 k=1/3/5와 비교해 어느 accuracy/ITR을 유지하는가?
4. Wang↔BETA 40-class와 wearable dry↔wet 12-class에서 같은 경향인가?

## 구현 완료

### 데이터

- [x] 공통 HDF5/manifest/class-map과 leakage 차단
- [x] dataset_id::subject_id split과 train/val/test 고정 `split.csv`
- [x] Wang/BETA 원본 class 순서와 frequency 기반 canonical 40-class 재매핑
- [x] 충돌 class map 결합 차단
- [x] wearable [8,710,2,10,12] 전용 parser
- [x] dry/wet, block, impedance, 공식 8채널/12-class/0.64초 창
- [x] 모든 대용량 asset의 외부 경로

### 모델

- [x] Tiny prompt prepend
- [x] frozen REVE/position bank
- [x] REVE token과 metadata prompt cross-attention
- [x] Transformer condition encoder/condition-gated adapter
- [x] dataset-ID-only continuous/channel 차단
- [x] latent posterior, KL, z-dropout, inference z=0와 zero-latent CE

### 학습·평가

- [x] cross-subject와 학습 중 8/4/2채널 strong views
- [x] Wang→BETA / BETA→Wang source-validation + target-test
- [x] wearable leave-subject-out와 dry→wet / wet→dry source-validation + target-test
- [x] accuracy, balanced accuracy, macro-F1, NLL, ECE, confusion matrix
- [x] ITR와 accuracy/ITR 절대 저하·상대 일반화 저하율
- [x] 64→8→4→2 channel stress와 metadata 25/50/75/100% missing/shuffle
- [x] metadata 그룹별 제거, downsample, re-reference, broadband/band-limited noise, 복합 변환
- [x] 피험자별 k=0/1/3/5
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
- [x] 대용량/secret Git 제외

## 서버에서 남은 일

- [ ] REVE gated access와 Kubernetes secrets
- [ ] BETA/Wang/Wearable 전체 전처리 검증
- [ ] 실제 REVE 1-epoch smoke
- [ ] 네 전이 방향 각각 3개 이상 seed
- [ ] A0-A4 반복과 robustness 전체
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
CFEG_BACKBONE=reve bash scripts/cfeg.sh ablation
~~~

Wearable 원본 배치 후:

~~~bash
bash scripts/cfeg.sh assets wearable
CFEG_BACKBONE=reve bash scripts/cfeg.sh train wearable-dry-to-wet
bash scripts/cfeg.sh calibration outputs/research/wearable_dry_to_wet/best.pt
~~~

## 판정 규칙

- 핵심 결과는 k=0이며 양방향 결과를 별도 보고한다.
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
