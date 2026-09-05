# Calibration-Efficient SSVEP EEG Decoding

이 저장소의 상위 목표는 **처음 보는 사용자가 쓸 만한 closed-set SSVEP 성능에 도달하는 데 필요한 labeled target calibration을 최소화하는 것**이다. `k=0`은 calibration-free anchor이고 `k=1/3/5`는 명시적인 low-calibration 자원점이다. Metadata는 연구목표가 아니라 이 부담을 줄이기 위한 수단이며, 현재 직접 시험하는 metadata도 개인정보나 dataset ID가 아니라 query 전에 관측되는 wet/dry interface와 block별 채널 impedance다.

새 설계 초안 `metadata-calibration-efficiency-v1`은 같은 완전한 calibration block과 같은 고정 query에서, EEG·구조·signal-derived QC만 쓰는 `A_Q`보다 pre-query acquisition context를 추가한 `A_QM`이 `k=0/1/3` early-budget curve를 개선하는지 묻는다. Wearable에서는 `k=1`이 **deployment interface당** 12개 class를 한 번씩 포함한 한 block, 즉 12 labeled trial이다. Primary는 participant-level `eAUC(A_QM)-eAUC(A_Q)`이고, “두 block(36→12 trial) 절약”은 별도의 사전 비열등성 contrast가 통과할 때만 주장한다. 전체 설명은 [metadata-assisted low-calibration 설계](docs/metadata_calibration_efficiency_design.md), 현재 상태는 [연구 프로토콜 현재 상태](docs/research_protocol_status.md), 결정 이력은 [append-only 연구일지](docs/research_log.md)에 있다.

기존 `query-reliability-spatial-v1`은 사람 EEG outcome을 열지 않은 채 primary에서 내려와 frozen query-only baseline 후보로 남는다. 그 구현·hash·CUDA forward 증거는 보존하지만 기존 BETA 35/20 outcome plan은 실행하지 않는다. `physical_hybrid_v1` no-go, S1–S3 추가 outcome 금지와 `reliability-spatial-v1` Stage-0 terminal failure도 그대로 유지한다.

봉인된 선행 confirmatory 설계는 **사전 할당한 39명에서 A0/A2를 학습하고 독립 60명 lockbox에서 paired balanced-accuracy 차이를 검정**하는 것이었다. 원시 신호는 전체 자산 무결성 감사에서 읽혔지만 S4–S102 성능과 60명 lockbox prediction은 계속 미접근이다. 이 역사적 계약은 [A2 physical-hybrid 설계](docs/a2_physical_hybrid_design.md)에 보존돼 있으나, 값을 채우거나 코드를 다시 고정하는 것만으로 재개할 수 없다.

> **P0 revision 경계:** 배포 Readme의 impedance-axis 문구는 실제 수치·원 논문 Figure 9의 조건 평균과 모순된다. 수치상 axis 0/1은 각각 dry/wet(`261.67/19.63 kΩ`)이므로 코드가 이 signature를 fail-closed로 검증한다. 반대로 결합된 기존 `wearable_v2`와 그 A2 결과는 무효이며, raw-to-processed deep audit를 통과한 `wearable_v3`만 Protocol 0.4-dev 실행에 사용한다.

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

Setup은 의존성만 준비하고 데이터와 weight를 받지 않는다. 프로젝트 전용 Python 3.10 환경에 NumPy 1.26.4와 Torch 2.2.2+cu121을 설치하고 실제 CUDA forward/backward probe가 실패하면 즉시 중단한다. Ambient/system Torch는 상속하지 않는다.

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
bash scripts/cfeg.sh assets beta

# Wang은 MOABB adapter의 S35 누락을 피하기 위해 공개 Zenodo MOABB re-upload를 직접 사용한다.
bash scripts/cfeg.sh assets wang

# Wang/BETA와 exact 40-class이면서 8-channel semi-dry acquisition인 Dong2023
bash scripts/cfeg.sh assets dong2023
~~~

`scripts/cfeg.sh assets`는 full-cohort 준비와 검증만 수행한다. 소수 subject 개발용 raw가 필요하면 별도 pilot directory에 `scripts/fetch_dataset.py --subjects ...`를 직접 실행한다. 대량 공개자료는 `CFEG_FETCH_WORKERS=8`로 파일 단위 병렬 다운로드할 수 있다. BETA는 Figshare v3, Wearable은 Figshare v4, Wang은 Zenodo record 14865172, Dong2023은 Zenodo record 18847318에 고정하며 각 배포 API가 제시한 size와 MD5를 검증한다.

Wearable은 Figshare에서 subject 파일과 Impedance·공식 설명 파일을 선택 다운로드하며 크기와 MD5를 검증한다.

~~~bash
# 원격 파일 목록만 확인
python scripts/fetch_dataset.py --dataset wearable --subjects 1,2,3 --probe-remote
# 3명 raw pilot은 full cohort와 분리
python scripts/fetch_dataset.py --dataset wearable --subjects 1,2,3 \
  --raw-dir "$EEG_DATA_ROOT/raw/wearable_pilot_s001_s003"
# full 102명 준비·검증
bash scripts/cfeg.sh assets wearable
python scripts/audit_wearable_processed.py \
  --raw-dir "$EEG_DATA_ROOT/raw/wearable" \
  --processed-dir "$EEG_DATA_ROOT/processed/wearable_v3" \
  --expected-subjects 102
~~~

전용 parser가 `[channel,time,electrode,block,target]`, dry/wet, block, 8채널 impedance, headband 착용순서, 공식 8채널과 9.25–14.75Hz 12개 target을 읽는다. EEG axis는 배포 Readme와 독립 공개 SSVEP-DAN pipeline으로 `[dry, wet]`을 교차확인했다. Impedance axis는 Readme 문구의 오류를 실제 수치와 논문 조건 평균으로 교정해 `[dry, wet]`으로 읽는다. Deep audit는 Figshare v4 파일 MD5, 전체 raw EEG→HDF5/query-QC, raw impedance→채널 vector/scalar, headband order, exact trial grid를 대조하고 receipt를 남긴다. 기존 `wearable_v2`와 그 A2 결과는 제외한다. `window_start_sec=0.64`는 시작 offset이고 실제 window는 2.0초다.

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

지원하는 mutable train preset은 `wang-to-beta`, `beta-to-wang`, `joint`, `synthetic`뿐이다. Wearable development/LOSO/dry↔wet preset과 `controls`/`research` shortcut은 폐기했으며 호출하면 fail-closed다. 첫 prompt-based grid와 두 번째 physical six-role grid는 모두 완료된 역사적 S1–S3 개발 실험이다. Physical reveal #2는 `DEC-20260901-004`와 clean annotated source tag에서 실행·소비됐으므로 다시 실행하지 않는다. Governed wearable은 전용 manifest orchestrator만 사용한다.

Legacy/외부 데이터 run은 `split.csv`, source-validation checkpoint, held-out metric을 저장한다. Governed wearable run은 학습 중 test loader나 `metrics_test.json`을 만들지 않는다. 역사적 development는 S1–S3 train/val만 썼고, retired `physical_hybrid_v1`의 confirmatory outer-test prediction은 substantive no-go로 봉인돼 있다. 새 `metadata-calibration-efficiency-v1`은 아직 executable preset이나 outcome 권한이 없다.

Wang과 BETA label은 raw index가 아니라 stimulus frequency로 canonical 40-class 8.0, 8.2, ..., 15.8Hz에 정렬된다. 학습 strong view는 8/4/2채널 subset을 명시적으로 포함한다. Source validation만 checkpoint 선택에 쓰며 target은 test-only다.

## Evaluate, robustness, calibration, inference

~~~bash
bash scripts/cfeg.sh eval wang-to-beta outputs/research/wang_to_beta/best.pt
bash scripts/cfeg.sh eval beta-to-wang outputs/research/beta_to_wang/best.pt

bash scripts/cfeg.sh channel-stress outputs/research/wang_to_beta/best.pt \
  "$EEG_DATA_ROOT/processed/beta_v1"
bash scripts/cfeg.sh robustness outputs/research/wang_to_beta/best.pt \
  "$EEG_DATA_ROOT/processed/beta_v1"

# Outcome-gated physical S1-S3 checkpoint는 generic predict가 아니라
# run_physical_mechanism_loso.py의 manifest-bound atomic reveal만 사용한다.
~~~

Robustness는 external metadata 결측 25/50/75/100%, acquisition-block 단위 derangement shuffle, global/channel/query-QC 분리 제거, downsample, re-reference, broadband/band-limited noise와 복합 4채널 조건을 평가한다. External missing/shuffle은 channel ID/mask, sampling/time grid, query QC를 바꾸지 않는다. Shuffle은 donor block 하나의 12개 label을 label/window 정렬로 함께 교환하고 donor mapping CSV·mapping hash·실제 metadata 변경률을 저장하므로 불가능한 row mosaic와 상수-metadata no-op을 구분한다. Signal perturbation 뒤에는 저장된 query QC를 invalid 처리한다. checkpoint 옆 `split.csv`의 held-out test ID 또는 명시적 target filter가 없으면 실행을 거부한다.

기존 generic calibration 유틸리티는 피험자·class별 k=0/1/3/5와 fixed query를 지원하지만 label별 sample을 따로 고르므로 서로 다른 block을 섞을 수 있다. 새 low-calibration primary에서는 각 interface의 block 1–5를 nested support, block 6–10을 모든 역할·budget에서 동일한 query로 두고 complete block만 선택해야 한다. 이 block-coherent runner와 새 owner freeze가 없으므로 wearable_v3 support label과 query outcome 접근은 현재 거부된다. S1–S3를 다시 calibration 개발자료로 쓰지 않는다.

Training-free CCA/FBCCA baseline은 실제 canonical correlation과 sub-band filter bank를 계산한다. 다만 wearable S1–S3 outcome을 새로 계산하는 예시는 reveal budget 소진 뒤 제거했다. 현재 candidate에서 S1–S3 baseline을 추가 실행하지 않으며, confirmatory baseline action도 등록되지 않아 S4–S102를 열 수 없다.

## Historical physical ablation과 전체 suite

~~~bash
bash scripts/cfeg.sh ablation
bash scripts/cfeg.sh ablation A0_eeg_only,A4_full_latent
python scripts/run_ablation.py --include-optional --continue-on-error
# 과거 prompt v1 manifest/result는 immutable historical artifact다.
# run_development_loso.py prepare와 mutable run_ablation 재생성은 명시적으로 거부된다.

# physical reveal #2는 완료·소비됐으며 재실행하지 않는다.
# 공개 결과: outputs/development-loso/physical-mechanism-v2/reveal-bundle/analysis/revealed_summary.json

# 과거 physical 6개 역할의 outcome-free contract regression만 허용
python scripts/run_ablation.py \
  --config configs/train/wearable_physical_mechanism.yaml \
  --output-root outputs/development/wearable_v3/physical_contract_dryrun \
  --only A0_eeg_only,A2_structured_condition_prompt,M1_global_only,M2_channel_only,M3_full_shuffle_train,M4_metadata_only \
  --dry-run
# REVE는 snapshot-byte binding과 serial-layout 성능 gate 전까지 exploratory secondary다.
# 완료된 S1-S3 prompt grid를 cfeg.sh research로 다시 실행하지 않는다.
~~~

`run_ablation.py`는 `--config`, `--base-config`, `--output-root`, 반복 가능한 `--override KEY=VALUE`를 지원한다. 당시 physical primary A0/A2는 `physical_hybrid_v1`의 전체 module graph와 parameter 수, protocol/fairness, parameter schema, 초기 trainable state, split, vocabulary, exact asset bytes, source tree, 실행환경 hash가 모두 같아야 한 pair로 완료됐다. 안정적인 artifact ID `A2_structured_condition_prompt`는 역사적으로 남지만 새 primary가 아니다. Development control family도 이 계약을 공유하며, dynamic elapsed/peak-memory 값은 동일성 조건이 아니라 별도 `runtime_metrics.json` 증거로 남긴다. Metadata-only는 EEG·structure·query-QC를 우회하는 shortcut 진단이고, 자연 missingness pattern이 하나뿐인 missingness-only는 invalid assay였다. Shuffle-train과 wet↔dry counterfactual은 사전 계약대로 평가됐지만 correct metadata reliance gate를 통과하지 못했다. Retired physical 39명 training과 60명 lockbox는 no-go이며 새 후보는 별도 계약을 요구한다.

`analyze_ood_coverage.py`는 당시 physical 후보의 한-seed 진단용 비교다. CSV sidecar, checkpoint, split, analysis manifest, source revision이 모두 연결된 prediction bundle만 받는다. 당시 최종 primary 계약은 A0/A2×seeds `[42,43,44]`의 6 jobs를 모아 lockbox subject 안에서 seed BA를 먼저 평균한 N=60 표였으며 현재 실행하지 않는다.

~~~bash
python scripts/analyze_ood_coverage.py \
  --baseline outputs/a0/predictions.csv \
  --candidate outputs/a2/predictions.csv \
  --processed-dir "$EEG_DATA_ROOT/processed/wearable_v3" \
  --success-threshold <preregistered-BA> \
  --out-prefix outputs/analysis/a0_vs_a2

# physical development 결과는 이 공개 bundle에서 읽는다. Reveal #2는 이미 소비됐다.
jq '.mean_metrics, .predeclared_gate_evaluation' \
  outputs/development-loso/physical-mechanism-v2/reveal-bundle/analysis/revealed_summary.json

# 이 명령은 retired physical_hybrid_v1의 historical confirmatory 경로이며 금지된다.
# 새 metadata 후보에는 별도 candidate, runner, freeze와 owner execution decision이 필요하다.
~~~

집계기는 누락·중복 run, 60명 lockbox drift, 변조된 CSV/sidecar/checkpoint/train-metric/split/execution manifest/reveal receipt, 현재 plan/role/source와 다른 contract를 거부한다. Plan은 canonical manifest와 run root를 하나만 허용하고 manifest는 create-exclusive다. 여섯 completion의 exact epoch/runtime/resume 계약이 맞기 전에는 lockbox가 열리지 않으며, generic evaluation과 canonical hidden staging 밖 개별 prediction은 reveal 이후에도 dataset 생성 전에 실패한다. 6개 prediction은 staging과 공개 후 final 위치에서 모두 검증된다. 이 보호는 같은 OS 사용자의 raw-file 읽기까지 막는 암호학적 봉인이 아니라 application/procedural gate와 hash provenance다. Physical margin과 `development_gate_only`는 이미 동결됐고 사전 substantive gate가 실패했다. Confirmatory 값과 source tag를 채우는 것만으로는 현 no-go를 해제할 수 없다. Seed는 독립 피험자로 세지 않으며 S1–S3와 39명 training participant는 primary table에 포함하지 않는다.

## 규칙

- Label, stimulus frequency/phase, dataset/hardware/subject/session/trial ID와 source file은 primary conditioner 입력이 아니다. headband order, condition period, reattach와 경과시간은 분석 covariate/confound로만 보존한다.
- Dataset-ID-only ablation은 continuous/channel metadata도 사용하지 않는다. Structured-without-ID ablation은 dataset_id도 제거한다.
- Frequency overlap이 없으면 unrelated class id 비교를 거부한다.
- k=0은 calibration-free anchor이고 k=1/3/5는 low-calibration 결과다. k>0을 calibration-free라고 부르지 않는다.
- Wearable primary에서 k는 participant·electrode condition당 완전한 labeled block 수이며 한 block은 12 trial이다.
- Primary metadata contrast는 동일 Q·support·query·adaptation 권한에서 `A_QM−A_Q`다. Query-derived QC를 external metadata로 세지 않는다.
- `dataset_id`, opaque hardware ID, constant reference/cap은 새 primary의 external context `M`이 아니다. 현재 식별 가능한 `M`은 wet/dry interface와 block 전 채널별 impedance/availability뿐이다.
- 새 primary `A_QM−A_Q`는 하나의 composite source checkpoint에서 bounded metadata residual만 on/off한다. Common backbone과 Q-only calibration estimator는 M을 fit하기 전에 고정하며, all-missing M은 동일한 frozen `A_Q` 계산으로 정확히 돌아가야 한다.
- 평균 성능 개선과 OOD coverage 확장을 구분한다. `learned > forgotten` 및 worst-group 개선 전에는 “적용 범위가 확장됐다”고 쓰지 않는다.
- Raw/processed EEG, REVE weight, checkpoint, token, W&B log는 Git 제외다.
- Download는 명시적인 assets 명령에서만 일어난다.
- Frozen REVE는 checkpoint에 복제하지 않고 `HF_HUB_CACHE`에서 다시 읽는다.

현재 설계와 실행 차단조건은 [metadata-assisted calibration-efficient 설계](docs/metadata_calibration_efficiency_design.md)와 [프로토콜 상태](docs/research_protocol_status.md)를 따른다. 과거 physical strict-k0 구현 기록은 [역사적 체크리스트](calibration_free_eeg_codex_implementation_plan.md)에만 보존한다.
