# Calibration-Free EEG 구현·실험 체크리스트 v5 (historical)

업데이트: 2026-09-01

> **2026-09-04 superseded:** 이 문서는 retired `physical_hybrid_v1`과 당시 strict-k0
> `A2−A0` 설계의 구현·실행 기록이다. 현재 체크리스트나 실행 권한이 아니다. 현재 연구는
> labeled target calibration 부담을 줄이는 `A_QM−A_Q`, k=0/1/3/5 설계이며
> [새 설계](docs/metadata_calibration_efficiency_design.md),
> [machine-readable plan](configs/analysis/metadata_calibration_efficiency_v1.yaml),
> [프로토콜 상태](docs/research_protocol_status.md)를 우선한다.

> DEC-20260830-001에 따라 S1–S3는 영구 development-only다. DEC-20260830-002에 따라 S4–S102 N=99는 39명 confirmatory training과 60명 independent lockbox로 고정됐고, 이전 N=99 5-fold primary는 superseded됐다. DEC-20260901-004의 physical reveal #2 exact restart는 18/18 training·3/3 intervention·atomic publication까지 완료됐으며, 유효 assay에서 substantive gate 네 개가 실패해 diagnostic no-go다. 현 `physical_hybrid_v1`의 39/60 confirmatory는 차단한다. 전체 N=102·24,480행은 자산 계약이다. 현재 권한·gate는 [프로토콜 상태](docs/research_protocol_status.md)와 [append-only 연구일지](docs/research_log.md)를 우선한다.

이 문서는 예전 구현 지시서 v2를 2026-09-01 당시 상태에 맞춰 대체했던 체크리스트다. 체크 완료는 당시 코드·설정·테스트 경로가 repository에 있었다는 뜻이며 현재 후보의 완료 또는 실행 승인을 뜻하지 않는다.

## 연구 질문

1. Query 전에 관측한 물리 metadata의 factorized FiLM/gain 결합이 external-null EEG control보다 unseen participant 성능을 개선하는가?
2. Latent nuisance와 consistency가 채널·sampling·reference·noise·metadata 결측 저하를 줄이는가?
3. Calibration-free k=0이 k=1/3/5와 비교해 어느 accuracy/ITR을 유지하는가?
4. Primary wearable subject DG와 Wang↔BETA external-domain boundary에서 어떤 경향인가?

Protocol 0.3에서 primary를 Wang→BETA에서 wearable participant DG로 옮겼고, 0.4-lockbox에서 통계적으로 취약한 5-fold primary를 39-train/60-lockbox 독립 검정으로 교체했다. 연구 질문은 그대로다. Wang/BETA는 공개된 장비·reference·channel 조건이 거의 같아 structured metadata의 주효과를 식별하기 어렵다. Exact 40-class이면서 acquisition이 다른 Dong2023은 replication이고, dry→wet/wet→dry는 unseen categorical level이 `UNK`가 되는 joint subject-condition stress다.

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

- [x] Tiny prompt prepend(legacy secondary)
- [x] frozen REVE/position bank
- [x] REVE token과 metadata prompt cross-attention(legacy secondary)
- [x] Transformer condition encoder/condition-gated adapter(legacy secondary)
- [x] dataset-ID-only continuous/channel 차단
- [x] source-train-only categorical vocabulary와 physical primary의 fixed-neutral `UNK=0`
- [x] primary A2에서 dataset/opaque hardware ID 제외
- [x] primary `physical_hybrid_v1`: 공통 query-QC FiLM, global electrode/impedance FiLM, per-channel impedance gain
- [x] A0/A2 동일 physical module graph·parameter 수·초기 state; A0 external branch는 exact identity
- [x] reference/cap primary 제외, legacy prompt/adapter와 latent 비활성
- [x] latent posterior, KL, z-dropout, inference z=0와 zero-latent CE

### 학습·평가

- [x] cross-subject와 학습 중 8/4/2채널 strong views
- [x] Wang→BETA / BETA→Wang source-validation + target-test
- [x] wearable participant-fold DG와 participant-disjoint dry→wet / wet→dry source-validation + target-test
- [x] wearable 5-fold outer participant CV 구현(당시 39/60 physical primary에서 이미 superseded됐던 경로)
- [x] exact 39명 train/60명 independent lockbox allocation과 single-fold no-validation execution
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
- [x] wearable_v3 config opt-out 불가 governance, S1–S3 dataset view, confirmatory dry-run 차단
- [x] governed training test-loader 제거와 registered confirmatory prediction action
- [x] plan/role/cohort/source hash 및 clean tagged source freeze 계약
- [x] canonical six-job manifest/run root, create-exclusive lifecycle, manifest-bound hidden staging
- [x] confirmatory generic-eval 차단과 completion epoch/runtime/resume 및 final bundle 재검증
- [x] S1–S3 physical metadata-only와 split-local acquisition-block shuffle-train control, clean validation 평가
- [x] natural missingness pattern 하나인 missingness-only를 `invalid_assay`로 명시
- [x] control donor-map potency/hash 및 parent-checkpoint evaluation provenance sidecar
- [x] physical six-role×3-fold 18-job canonical manifest, 전역 reveal ledger와 generic 실행/eval 우회 차단
- [x] `physical-reveal2-freeze-20260901-r1`에서 18/18 train·3/3 intervention·reveal #2 atomic publication
- [x] potency-valid 10-gate 결과 판정: A0 0.1292, Full A2 0.1194, Δ −0.0097, substantive 4개 실패로 diagnostic no-go

### 실행

- [x] clone 위치 독립 경로와 PVC override
- [x] `HF_HOME/hub` 표준 cache와 interns NVMe/DDN env profile
- [x] legacy HF root/`.local/eeg_data` 안전한 dry-run migration
- [x] legacy Wang/BETA label을 신호 재처리 없이 canonical frequency로 migration
- [x] optional dependency 분리
- [x] 다운로드 없는 bootstrap과 선택형 asset download
- [x] scripts/cfeg.sh 단일 entrypoint
- [x] base config/output root/override를 받는 재현 가능한 ablation runner
- [x] 전용 CUDA 환경 전체 unit suite, targeted Ruff, compileall, YAML/shell syntax 검증(최신 수치는 연구일지 참조)
- [x] Python 3.10.12/Torch 2.2.2+cu121 isolated runtime과 실제 CUDA backward fail-fast probe
- [x] 실제 device·동기화 elapsed·peak allocated/reserved·OOM 상태 runtime sidecar
- [x] 대용량/secret Git 제외

## 당시 서버에서 남았던 일 (historical)

- [ ] REVE gated access와 Kubernetes secrets
- [x] BETA S1/S16과 wearable S1–S3 공개 pilot 전처리·baseline 검증
- [x] BETA/Wang/Wearable/Dong2023 공개자료 전처리·자산 검증
- [ ] 실제 REVE 1-epoch smoke
- [x] **차단 적용:** 현 physical A2로 39명 A0/A2×seeds `[42,43,44]` 6 jobs 및 독립 60명 lockbox 통계를 실행하지 않음
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
# Physical reveal #2는 완료·소비됐다. 아래 dry-run만 새 output에 outcome 없이 사용할 수 있다.
EEG_DATA_ROOT=/path/to/eeg-data python scripts/run_ablation.py \
  --config configs/train/wearable_physical_mechanism.yaml \
  --output-root outputs/development/wearable_v3/physical_contract_dryrun \
  --only A0_eeg_only,A2_structured_condition_prompt,M1_global_only,M2_channel_only,M3_full_shuffle_train,M4_metadata_only \
  --dry-run

# 첫 prompt grid와 두 번째 physical grid는 완료된 역사적 artifact다.
# S1–S3 train/reveal을 다시 실행하지 않는다. 공개 결과만 읽는다.
jq '.mean_metrics, .predeclared_gate_evaluation' \
  outputs/development-loso/physical-mechanism-v2/reveal-bundle/analysis/revealed_summary.json
~~~

Wearable 원본은 자동 다운로드하거나 기존 파일을 검증한 뒤 전처리한다.

~~~bash
python scripts/fetch_dataset.py --dataset wearable --subjects 1,2,3 --probe-remote
python scripts/fetch_dataset.py --dataset wearable
bash scripts/cfeg.sh assets wearable
# 현 candidate의 confirmatory 명령은 substantive no-go로 차단된다.
# 새 독립 development study와 별도 decision contract 전에는 실행하지 않는다.
~~~

## 당시 판정 규칙 (superseded)

- 당시 계획의 핵심 결과는 모델 학습·선택에서 제외된 wearable 60명 lockbox의 k=0 A2−A0 subject-level paired difference였다. 이 contrast는 실행되지 않았고 현재 primary가 아니다.
- Wang↔BETA와 dry↔wet 방향은 별도 boundary/stress 결과로 보고한다.
- Wearable은 독립 12-class head다.
- Governed wearable training은 target test loader를 만들지 않는다. 동결 뒤 별도 prediction action에서만 outer test를 연다.
- 표에 seed, backbone, source/target, channel, perturbation, budget을 기록한다.
- Tiny는 pipeline/baseline이고 REVE와 구분한다.
- Frequency overlap 없는 class id와 label-frequency가 어긋난 예전 processed asset을 거부한다.

Physical N=3 개발 진단 결과는 완료됐지만 전체 연구의 confirmatory 결과는 없다. 외부 replication이나 새 독립 development evidence를 실행하지 않은 상태에서 broad acquisition-OOD 또는 모집단 metadata benefit을 완료된 연구 결과로 표시하지 않는다.

서버 표준값은 `HF_HOME=/mnt/nvme/cache/interns/hf`,
`HF_HUB_CACHE=/mnt/nvme/cache/interns/hf/hub`,
`EEG_DATA_ROOT=/mnt/ddn/prod-runs/interns/jm020827/califreeEEG/storage/eeg_data`다.
`eeg_models/`는 더 이상 생성하거나 사용하지 않는다.
