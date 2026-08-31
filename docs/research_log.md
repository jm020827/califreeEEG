# Calibration-Free EEG 연구일지

이 파일은 append-only 기록이다. 기존 항목을 고치지 않고 오류 정정·결정 변경은 새 ID로 추가하며 `supersedes`를 명시한다. 각 항목은 관찰, 해석, 정보 접근 범위, 증거, 다음 gate를 분리한다. 아래 과거 항목은 2026-08-30에 저장소 artifact를 근거로 backfill했다.

## AUD-20260830-001 — wearable_v3 전체 자산 감사

- 상태: completed, backfilled
- 관찰: Figshare v4 wearable 전체 102명·24,480행, dry/wet 각 12,240행, dry-first/wet-first 53/49를 확인했다. Raw EEG→processed signal, query-QC, impedance/channel alignment 최대 절대오차는 모두 0이었다.
- 정보 접근: 전체 manifest와 raw/processed 자산의 무결성·metadata를 감사했다. A0/A2 성능 비교는 열지 않았다.
- 증거: `wearable_v3_audit_receipt.json`; receipt SHA-256 `1e2c10d7922c329d8d28cd8bd3cb8ed94e0b22f6a2f3432e95c77deab3f0bc9b`; `docs/data_acquisition_2026-08-30.md`.
- 해석: 전체 자산 계약은 N=102/24,480으로 유지한다. 이는 primary 표본 수와 별개다.
- 다음 gate: cohort 역할과 성능 접근 권한 고정.

## INV-20260830-001 — wearable_v1/v2 결과 무효화

- 상태: accepted, backfilled
- 관찰: v1은 class frequency order가 잘못됐고 v2는 impedance condition axis가 EEG condition과 반대로 결합됐다.
- 정보 접근: S1–S3 pilot artifact를 확인했다.
- 결정: v1/v2와 모든 파생 A2 결과를 연구 결과에서 제외한다. raw-to-processed deep audit를 통과한 wearable_v3만 사용한다.
- 해석: v2 A2 성능은 metadata 효과의 양·음 증거가 아니다.
- 증거: `docs/research_execution_plan.md`, `docs/protocol_0_4_dev_implementation.md`, wearable numeric-signature regression tests.
- 다음 gate: v3에서 schema/fairness를 다시 검증.

## EXP-20260830-001 — S1–S3 사전 성능 노출

- 상태: completed, backfilled
- 관찰: S1–S3에서 native-8 CCA BA 0.6125, generic FBCCA BA 0.5750을 확인했다. v2 Tiny A0/A2는 모두 BA 0.091667이었으나 impedance 결합 오류 때문에 무효다.
- 정보 접근: S1–S3의 baseline 및 무효 v2 성능을 열었다. S4–S102 성능은 열지 않았다.
- 해석: 결과의 유효성 여부와 무관하게 S1–S3의 눈가림은 회복되지 않는다.
- 증거: `outputs/pilot/`의 wearable S1–S3 artifact와 `docs/research_execution_plan.md`의 dated execution record.
- 다음 gate: S1–S3를 영구 development로 격리.

## DEC-20260830-001 — cohort 역할 동결

- 상태: frozen
- 결정: S1–S3(3명, 720행)는 영구 development-only다. S4–S102(99명, 23,760행)는 confirmatory primary다. 전체 N=102 분석은 primary 보고서가 잠긴 뒤 exploratory로만 허용한다.
- 근거: EXP-20260830-001의 사전 노출에 대한 보수적 조치이며, 사후 sensitivity로 눈가림을 복구할 수 없기 때문이다.
- 정보 접근: cohort ID와 표본 수만 고정했으며 N=99 성능은 열지 않았다.
- 기계 계약: `configs/governance/wearable_cohort_roles.yaml`; `configs/analysis/wearable_primary.yaml`.
- supersedes: 과거 `N=102 primary + N=99 sensitivity` 결정.
- 다음 gate: 모든 official train/eval/predict/baseline/aggregate 경로의 fail-closed 적용.

## IMP-20260830-001 — target-lock 및 provenance 구현

- 상태: implemented, verification ongoing
- 관찰: wearable_v3 auto-gate, confirmatory dry-run 차단, canonical plan/role 검증, S1–S3 dataset view, test-loader 제거, fresh output guard, registered confirmatory action, current plan/role/cohort/source hash 비교를 구현했다.
- 정보 접근: 단위시험과 source inspection만 수행했다. N=99 성능은 열지 않았다.
- 검증: 프로젝트 CUDA 환경에서 전체 test suite 158 passed(중간 checkpoint); 이후 추가 governance tests는 별도 최종 verification 항목으로 기록한다.
- 증거: `src/cfeg/governance.py`, `src/cfeg/train_loop.py`, `src/cfeg/eval_loop.py`, `src/cfeg/prediction.py`, `src/cfeg/baselines/evaluate.py`, `tests/test_governance.py`.
- 다음 gate: 실제 S1–S3 dry-run/1-epoch와 최종 전체 test.

## ENV-20260830-001 — 전용 CUDA runtime 검증

- 상태: completed
- 관찰: 프로젝트 전용 `.venv`를 Python 3.10.12, NumPy 1.26.4, Torch 2.2.2+cu121, Torch CUDA 12.1로 생성했다. RTX 4090 compute capability 8.9에서 matrix multiply, backward, finite gradient, synchronize가 통과했고 probe loss는 77.5였다.
- 정보 접근: 모델/EEG 성능 없음.
- 증거: `requirements-cuda121.txt`, `scripts/bootstrap_k8s.sh`, `scripts/verify_cuda.py`.
- 해석: ambient CPU Torch 오염을 제거했다. 다른 GPU에서는 device name/compute capability가 달라도 tensor/backward 계약을 만족하면 된다.
- 다음 gate: 실제 development cohort forward.

## DEV-20260830-001 — S1–S3 A0/A2 CUDA dry-run

- 상태: completed
- 명령: `run_ablation.py` + `configs/train/wearable_development.yaml`, A0/A2, `--dry-run`, CUDA.
- 관찰: 양쪽 모두 execution phase `development`, cohort SHA `f2de24e99cf3003cced773a87143c4c7a050ef3ffacec6d7609842c75824cea0`, 총/학습 parameter 6,125,900, outer-test access `false`, runtime pair `verified_equal`이었다. elapsed time은 A0 18.10초, A2 17.33초였다.
- 정보 접근: S1–S3의 train batch forward만 수행했다. validation/test 성능은 계산하지 않았고 S4–S102 EEG sample은 dataset view의 `__getitem__` 대상이 아니었다.
- 증거: `outputs/development/wearable_v3/governance_dry_run_20260830/summary.csv`.
- 해석: schema·parameter·split/provenance 경로의 smoke이며 metadata 효과 결과가 아니다.
- 다음 gate: S1–S3 1-epoch development smoke와 negative controls.

## GATE-20260830-001 — confirmatory 봉인 상태

- 상태: active
- 관찰: `configs/analysis/wearable_primary.yaml`은 `dev_not_frozen`이다. success threshold, MCID, alpha/alternative, multiplicity, primary fairness hash, source freeze tag와 implementation hash가 비어 있다.
- 결정: 이 값들을 source-only로 정하고 clean Git commit/tag를 만들기 전에는 confirmatory dry-run·training·evaluation·baseline·aggregation을 수행하지 않는다.
- 정보 접근: local repository 범위에서 confirmatory 결과 artifact를 만들지 않았다. 외부 시스템의 부재까지 증명한 것은 아니다.
- 다음 gate: S1–S3 개발 실험 → 통계/모델 규칙 동결 → tagged freeze.

## FAIL-20260830-001 — 첫 S1–S3 1-epoch smoke 직렬화 실패

- 상태: failed, retained
- 관찰: A0 model forward 뒤 `config.yaml`을 저장할 때 Torch 2.2.2의 `TorchVersion` 객체를 PyYAML이 표현하지 못해 중단됐다. Epoch training과 validation metric 계산 전 실패했고 A2는 시작하지 않았다.
- 정보 접근: S1–S3 train batch forward만 수행했다. N=99와 test/validation 성능은 열지 않았다.
- 조치: 실패 output `outputs/development/wearable_v3/a0_a2_smoke_20260830/`를 삭제·재사용하지 않았다. runtime environment의 Torch/CUDA version을 명시적 문자열로 직렬화하고 회귀시험을 추가했다.
- 해석: 연구 가설 실패가 아니라 CUDA runtime provenance 직렬화 결함이다.
- 다음 gate: 새 output 경로에서 동일 1-epoch smoke 재실행.

## FAIL-20260830-002 — batch 128 개발 smoke CUDA OOM

- 상태: failed, retained
- 관찰: 직렬화 수정 뒤 A0의 1-epoch training 4 batches는 완료했지만 첫 validation batch에서 6.30 GiB 추가 할당이 필요해 RTX 4090 OOM으로 중단됐다. 속도를 위해 base config 64보다 크게 덮어쓴 batch 128이 원인이었다.
- 정보 접근: S1–S3 training loss만 계산됐고 validation metric은 생성되지 않았다. A2와 N=99는 열지 않았다.
- 조치: 실패 output `outputs/development/wearable_v3/a0_a2_smoke_20260830_v2/`를 삭제·재사용하지 않는다. 새 경로에서 batch 16으로 재시도한다.
- 해석: 연구 가설이나 metadata 효과의 실패가 아니라 runtime capacity 실패다.
- 다음 gate: batch 16의 A0/A2 1-epoch 완료와 peak-memory 측정 개선.

## DEV-20260830-002 — S1–S3 A0/A2 1-epoch smoke 완료

- 상태: completed
- 실행: Tiny Transformer, batch 16, 1 epoch, seed 42, S1–S3 development train/val, CUDA. 각 모델의 split은 train 480행/2명, val 240행/1명, test 0행이었다.
- 관찰: A0 validation accuracy 0.083333, A2 0.087500이었다. 두 모델은 총/학습 parameter 6,125,900, cohort 3명/720행, runtime pair `verified_equal`, outer-test access `false`였다. elapsed time은 A0 95.88초, A2 95.54초였다.
- 정보 접근: 허용된 S1–S3 validation 성능만 열었다. `metrics_test.json`과 test column은 생성되지 않았고 N=99는 열지 않았다.
- 해석: 12-class chance 부근의 1-epoch pipeline smoke다. A2의 0.004167 차이는 연구 효과나 모델 선택 근거로 해석하지 않는다.
- 증거: `outputs/development/wearable_v3/a0_a2_smoke_20260830_v3/summary.csv`, SHA-256 `7f555e238dbe508e866eff57d4a8b301831add5bfa80aa23128bf1fdaa873868`; 각 run의 `research_access.json`, `split.csv`, `metrics_val.csv`, checkpoints.
- 다음 gate: 충분한 development epochs를 쓰기 전에 metadata-only/missingness-only/shuffle/wrong control과 메모리 계측을 구현한다.

## VER-20260830-001 — governance/CUDA 통합 검증

- 상태: completed
- 관찰: 프로젝트 전용 CUDA 환경에서 전체 unit suite `170 passed`, CUDA tensor/backward probe, Python compileall, 수정 범위 Ruff, YAML parse, shell syntax와 `git diff --check`를 실행했다. Confirmatory dry-run을 존재하지 않는 data root로 실행했을 때 dataset path 오류보다 먼저 `plan.status=frozen` gate에서 중단됐다.
- 정보 접근: 단위·구조 검증과 S1–S3 개발 artifact만 사용했다. N=99 성능은 열지 않았다.
- 해석: 현재 official train/eval/predict/baseline/aggregate 경로의 target-lock과 개발 CUDA 경로는 연구 시작에 충분한 구현 검증을 갖췄다. 저장소 전체 Ruff에는 이번 변경과 무관한 기존 style debt가 남아 있다.
- 다음 gate: negative controls 구현 → S1–S3 개발 반복 → plan 수치와 source tag 동결.

## IMP-20260830-002 — development negative-control 및 runtime telemetry 구현

- 상태: implemented
- 범위: `cfeg.development-control.v1`; S1–S3 development-only. Metadata-only와 missingness-only는 waveform, channel structure/time grid, query-QC를 제거하고 condition representation만 분류한다. Within-class shuffle은 train/val 각각의 dataset×label 안에서 manifest-only donor를 고정한다. Wet↔dry counterfactual은 같은 subject×label×block×window의 반대 electrode external bundle을 정상 A2 validation에만 주며 재학습은 차단한다.
- 불변 정보: donor EEG·label·query-QC는 읽거나 교환하지 않는다. Target x/y/sample ID/channel structure/query-QC는 보존하고 reference/electrode/cap, impedance value와 missing mask만 원자적으로 교환한다.
- provenance: control seed/scope, fixed point, pair coverage, mapping hash, bundle 및 field별 변경률을 저장한다. Evaluation은 parent checkpoint, S1–S3 cohort, 동일 sample identity, donor와 모든 artifact hash를 `cfeg.evaluation-provenance.v1`에 묶는다.
- runtime: 실제 선택 device, synchronized elapsed, peak allocated/reserved, CUDA OOM과 오류 유형을 checkpoint contract와 분리한 `cfeg.runtime-metrics.v1` sidecar로 저장한다. Dynamic peak/time은 A0/A2 또는 control-family equality 조건에 넣지 않는다.
- 정보 접근: 구현·단위시험 중 N=99 성능을 열지 않았다.
- 증거: `src/cfeg/data/metadata_controls.py`, `configs/train/wearable_development_controls.yaml`, `configs/eval/wearable_development_controls.yaml`, `tests/test_metadata_controls.py`.
- 다음 gate: 실제 S1–S3 dry-run, 1-epoch backward/checkpoint, validation-only intervention.

## DEV-20260830-003 — S1–S3 negative-control CUDA dry-run 및 1-epoch smoke

- 상태: completed
- 실행: Tiny Transformer, batch 16, seed 42, train 480행/2명, validation 240행/1명, test 0행. Correct A2, metadata-only, missingness-only, split-local within-class shuffle를 같은 parameter/schema/init/split/vocab/asset/source/environment 계약으로 실행했다.
- dry-run 관찰: 네 variant 모두 총/학습 parameter 6,125,900, cohort 3명/720행, outer-test access `false`, control-family runtime `verified_equal`이었다. Shuffle은 720/720 pair, fixed point 0, external bundle 변경률 0.995833, electrode flip 0.511111이었다.
- 1-epoch 관찰: validation accuracy는 correct A2 0.087500, metadata-only 0.083333, missingness-only 0.083333, shuffled-training 0.091667이었다. 12-class chance는 0.083333이다.
- GPU 관찰: correct A2 peak allocated/reserved는 3,648,479,232/6,824,132,608 bytes, shuffled-training은 3,648,479,232/6,826,229,760 bytes였다. Backbone을 우회한 metadata-only와 missingness-only는 각각 63,173,632/85,983,232 bytes였다. 이는 batch 16의 실행 증거이며 더 큰 batch의 안전 보장은 아니다.
- 정보 접근: 허용된 S1–S3 train/validation만 사용했다. 어느 run에도 test loader, test column, `metrics_test.json`이 없으며 N=99 성능은 열지 않았다.
- 해석: 모든 값이 chance 부근인 1-epoch pipeline smoke다. Shuffled가 correct보다 0.004167 높거나 두 shortcut control이 chance인 사실을 기전 성공/실패 또는 모델 선택 근거로 해석하지 않는다.
- 증거: dry-run summary `outputs/development/wearable_v3/controls_dryrun_20260830_v1/summary.csv`, SHA-256 `f69224133f2c6080bea70b9ccfa4c5f1193059212ae3273300f5d7f9b6bf23de`; 1-epoch summary `outputs/development/wearable_v3/controls_smoke_20260830_v1/summary.csv`, SHA-256 `068005d8d1b361be09448737089d4a092a249656bd1821fcfc8a77aab555834f`.
- 다음 gate: shortcut assay validity와 충분한-but-bounded S1–S3 learning curve를 정한 뒤 의미 있는 development epoch를 실행한다.

## DEV-20260830-004 — correct-A2 validation-only shuffle/counterfactual intervention

- 상태: completed, pipeline smoke only
- parent: A2 1-epoch checkpoint SHA-256 `6272e04fdc83d6627a4641b54b7d7b0bbe9212a6240b727e8edce3e02d473f9b`.
- 관찰: 동일 S1–S3 validation 240행의 clean, within-class shuffle, counterfactual wet↔dry accuracy가 모두 0.087500이었다. 세 scenario의 sample-identity SHA-256은 모두 `0bcad3ce32fb0fcc47807fb9238d34b6e4854bfa764284b657be01fecd7e0fe7`였다.
- donor potency: validation shuffle은 bundle 변경률 1.0, electrode flip 0.516667, mapping SHA-256 `d134b8de0239a9599a16a74f4a70c9c2763969915a9bfffbda4c1ce7c5750d3a`; counterfactual은 240/240 opposite-electrode pair, bundle/impedance/electrode 변경률 1.0, mapping SHA-256 `06fd1f95b5bc34e137c38ae576ea29e7de6727ea813be24abec948d18db29502`였다.
- 정보 접근: development validation만 열었다. Evaluation dataset view는 S1–S3 720행으로 제한됐고 confirmatory claim flag는 false였다. N=99 EEG 성능은 열지 않았다.
- 해석: parent 모델 자체가 chance 수준이므로 metadata 민감도가 0인 것은 control 기전의 반증이 아니라 `inconclusive/undertrained assay`다. Counterfactual이 같다는 결과로 metadata가 불필요하다고 결론 내리지 않는다.
- 증거: `outputs/development/wearable_v3/controls_eval_20260830_v2/robustness.csv`, SHA-256 `3258e217fd3a526e1938ab9d09e1e2d8cb9851e365bca0b2b3d1ab7062bc0b0d`; provenance SHA-256 `f28c86abd1045e175d188c92f1c149712c60e8c6eb98f17d51b94f1caaee4de6`.
- 다음 gate: 정상 A2가 학습 능력을 보인 뒤 같은 checkpoint/sample IDs로 intervention을 다시 실행한다.

## GATE-20260830-002 — negative-control 판정 규칙 미동결

- 상태: active, confirmatory blocker
- 관찰: S1–S3의 허용 external field에는 자연 missingness pattern이 하나뿐이다. 따라서 missingness-only accuracy가 chance여도 shortcut 부재를 입증하지 못하며 `invalid_assay/not-testable`로 판정한다.
- 결정: metadata-only/missingness-only는 단순 비유의성이 아니라 chance-equivalence upper bound로 판정한다. Shuffle은 `correct − shuffle`과 `shuffle ≈ A0`, counterfactual은 `correct − wrong`의 기전 민감도를 본다. 동시에 `wrong − A0` 허용손실을 별도 safety gate로 둔다. Wrong의 큰 하락을 자동 성공으로 세지 않는다.
- 미동결 값: shortcut equivalence margin, pairing/counterfactual mechanism margin, wrong-metadata safety harm margin, donor potency 최소치, `development_gate_only|confirmatory_once` 정책. 이 값은 `configs/analysis/wearable_primary.yaml`에서 null이며 frozen plan validator가 요구한다.
- 정보 접근: 위 결정은 S1–S3 smoke, 코드 감사와 선행연구 설계 원칙만 사용했다. N=99 성능은 열지 않았다.
- 다음 gate: target-free 문헌/시뮬레이션/배포 요구로 margin과 policy를 정하고, 시도 횟수가 제한된 S1–S3 learning curve 및 baseline을 완료한다.

## VER-20260830-002 — negative-control/telemetry 최종 회귀검증

- 상태: completed
- 관찰: 프로젝트 전용 환경에서 전체 unit suite `181 passed`를 확인했다. 수정 범위 Ruff, Python compileall, 36개 YAML parse, 모든 shell script syntax와 `git diff --check`도 통과했다. Governed evaluation의 기존 output-prefix 재사용을 거부하는 회귀시험을 추가했다.
- confirmatory seal: 존재하지 않는 `EEG_DATA_ROOT`로 N=99 dry-run을 요청해도 dataset 경로를 열기 전에 `plan.status=frozen` gate에서 중단됐다.
- style 범위: 저장소 전체 Ruff는 이번 변경 범위 밖의 기존 실행권한·style debt 22건을 보고했다. 이번 negative-control/runtime/governance 변경 파일 묶음은 clean이다.
- 정보 접근: 단위·정적 검증과 기존 S1–S3 development artifact만 사용했다. S4–S102 성능은 열지 않았다.
- 해석: 구현 루프는 재현 가능한 상태지만 연구 판정 루프는 아직 시작 조건을 만족하지 않는다. GATE-20260830-002의 margin·potency·confirmatory policy와 bounded learning-curve 규칙을 먼저 동결해야 한다.
- 다음 gate: S1–S3 learning-curve protocol과 control 판정 수치 확정 → 충분히 학습된 A0/A2/control 재실행 → source tag/plan freeze 검토.

## DEV-20260830-005 — source-only backbone 학습 가능성 진단

- 상태: completed, development evidence only
- 가설: 기존 chance 결과가 전체 pipeline 고장인지, time-domain Tiny의 SSVEP 유도편향 부족인지 구분한다.
- 관찰: Tiny는 synthetic 12-class 10 epoch에서 validation BA 0.96875였지만 Wang 28-train/7-val no-augmentation 3 epoch에서는 BA 0.025에 머물렀다. One-batch 50-step은 accuracy 0.515625로 자체 0.95 gate를 통과하지 못했다. Query-local complex RFFT의 real/imaginary/log-magnitude spectral Transformer는 같은 Wang split 10 epoch에서 BA 0.8220, condition/adapter/latent를 끈 pure F0는 0.8107이었다. 약한 augmentation 후보 0.8143은 no-augmentation보다 낮아 제외했다.
- scope-matched baseline: 동일 Wang validation 7명·1,680 trial에서 CCA H=5 BA 0.7875, strict Chen-2015 M3 FBCCA BA 0.7768이었다. 이전 CCA 0.854는 한 명 scope여서 직접 비교에서 제외한다.
- 해석: complex spectral backbone을 S1–S3 fixed-epoch grid의 공통 A0/A2 backbone으로 승격한다. 이는 metadata 이득이나 confirmatory 우월성의 증거가 아니다.
- 정보 접근: Wang public source-learning asset과 synthetic/S1–S3 기존 개발 artifact만 사용했다. `N99_accessed=false`; S4–S102 outcome은 열지 않았다.
- 증거: `outputs/development/source-learning/wang-spectral-e10-exact/metrics_val.csv` SHA-256 `868d7d38ef90e5a31484ed822a849e377a043c84376a9679cd7a11a0f372d205`; pure F0 SHA-256 `14ca4f0a72543c08aa38d1767e8aaae36ef9ba39952886c4a84b63c42af4e228`; CCA summary SHA-256 `7a4caceb7bbc2e5bb162463675a145876574f20e7a0215e84452e56fecf3c7ac`; FBCCA summary SHA-256 `1924b4e4656df17016f9f38987761d67243fa139a8d072cc20f55fdaf394c94d`.
- 다음 gate: spectral, 10 epoch, no augmentation, no validation selection으로 S1–S3 A0/A2 LOSO 여섯 job을 한 번 실행한다.

## SIM-20260830-001 — 기존 5-fold primary inference 반증

- 상태: failed design diagnostic
- 가설: N=99 outer-fold subject difference를 iid sign-flip/bootstrap해도 nominal Type-I가 유지되는가?
- 관찰: target-free synthetic simulation에서 independent setting은 허용 범위였지만 fold ICC 0.05와 overlapping training effect에서 null rejection 약 0.122, combined stress에서 약 0.204였다.
- 해석: 같은 fold participant가 공유하는 model error와 training-fold overlap을 무시한 기존 N=99 subject inference는 primary로 사용할 수 없다.
- 정보 접근: synthetic delta와 plan 숫자만 사용했다. EEG/prediction file은 읽지 않았고 `N99_accessed=false`다.
- 결정: 독립 lockbox 대안을 설계·검증한다.

## DEC-20260830-002 — 39명 training / 60명 independent lockbox로 primary 개정

- 상태: frozen allocation, confirmatory plan은 미동결
- 결정: S4–S102의 정렬된 99 ID를 `numpy.default_rng(42)`로 한 번 shuffle하고 앞 39명을 confirmatory training, 나머지 60명을 primary lockbox로 고정한다. 정확한 ID는 `configs/analysis/wearable_primary.yaml`에 기록했다.
- 실행 규칙: A0/A2×seeds `[42,43,44]`의 6 jobs, spectral backbone, 10 fixed epochs, validation/early stopping 없음. 여섯 training completion 뒤 60명 clean prediction을 한 번만 공개한다.
- 추론: lockbox participant별로 세 seed BA를 먼저 평균한 paired A2−A0 전체 contrast 하나만 primary다. Dry/wet condition은 descriptive이며 missing run/subject를 complete-case로 구제하지 않는다.
- 근거: SIM-20260830-001과 k-fold variance dependence 문헌. 독립 inference를 위해 training sample 수를 줄이는 trade-off는 사전 공개한다.
- 정보 접근: subject ID와 행 수만 사용했다. EEG outcome은 읽지 않았고 `N99_accessed=false`, `lockbox_outcomes_accessed=false`다.
- supersedes: N=99 5-fold×3-seed primary와 condition별 inferential family. 5-fold N=99는 primary report 뒤 exploratory로만 허용한다.
- 다음 gate: 독립 lockbox inference의 target-free acceptance simulation.

## SIM-20260830-002 — independent-lockbox inference acceptance

- 상태: accepted, target-free
- 관찰: N=60, 3-seed reduction, 10,000 outer simulation과 10,000 sign-flip/bootstrap resamples를 사용했다. 네 core DGP의 Type-I Wilson 95% upper는 최대 0.0581, one-sided coverage lower는 최소 0.9413, SESOI+0.03 power lower는 최소 0.8769였다. Negative-responder mixture는 report-only stress로 보존했다.
- 판정: 사전 기준 Type-I upper ≤0.06, coverage lower ≥0.94, power lower ≥0.80을 모두 통과했다.
- 정보 접근: synthetic data only, target outcome file list empty. `N99_accessed=false`, `lockbox_outcomes_accessed=false`다.
- 증거: `configs/analysis/wearable_lockbox_inference_simulation.yaml` SHA-256 `e89af8364c4a5c09da0dfa274d92e82059be023137ab4fda23ce5098f3bf595e`; compact receipt `configs/governance/wearable_lockbox_inference_receipt.json` SHA-256 `ecb00eed7441fc5ac23d2d8cade798c76e1f1f7a09c1f4ea318e459f874073c0`; full output receipt SHA-256 `1f287ccd6a482eee8f18a2a3839361f2939564465fee15d16cd3ac21c277c2e5`.
- 다음 gate: execution/resume/provenance implementation을 회귀검증한다.

## IMP-20260830-003 — lockbox execution, exact resume, provenance 완성

- 상태: completed, confirmatory still sealed
- 구현: exact 39/60 split, no-validation `fixed_epoch_sealed`, train-only HDF preload, six-job execution manifest, job/execution-contract hash, completion-bound prediction, reveal-once ledger, exact-grid aggregate를 추가했다. Two-slot atomic resume journal은 Python/NumPy/Torch/CUDA/DataLoader RNG와 terminal stop reason을 저장한다.
- 검증: CPU synthetic training을 epoch-commit 직후 중단한 뒤 exact resume한 final model/optimizer/state/metrics가 uninterrupted run과 tensor-identical했다. Early-stop terminal commit 직후 crash도 추가 epoch 없이 완료됐다. 전체 unit suite는 `224 passed`다.
- 정보 접근: synthetic/unit fixtures와 source code만 사용했다. Confirmatory dry-run은 plan status에서 dataset 생성 전에 중단된다. `N99_accessed=false`다.
- 다음 gate: bounded S1–S3 spectral grid를 동결하고 한 번 공개한다.

## LIT-20260830-001 — spectral 및 inference 인접연구 갱신

- 상태: completed, evidence workspace rendered
- 관찰: SSVEPformer와 user-independent complex-spectrum CNN은 spectral/complex input의 직접 근거를 제공한다. Bengio–Grandvalet의 k-fold variance 결과는 overlapping CV outputs를 단순 iid로 다루지 말아야 한다는 설계 근거다. `arXiv:2608.11829`는 직접 EEG 근거가 아니라 learned/retained/forgotten 보조 해석의 analogy로만 유지한다.
- 한계: spectral paper 두 건과 k-fold card는 현재 abstract/metadata 수준이므로 protocol-matched numerical comparator로 과장하지 않는다.
- 증거: `/home/whwovy/research-spaces/califree-eeg-experiment-design/`의 신규 paper cards 3건, claims 2건, design gap 1건과 rendered landscape.
- 정보 접근: 학술 metadata/abstract와 기존 source-only artifact만 사용했다. `N99_accessed=false`다.
- 다음 gate: S1–S3 결과 후에도 연구 질문을 바꾸지 말고 red-flag 여부만 판정한다.

## DEV-20260830-006 — S1–S3 fixed-epoch blind spectral grid 첫 공개

- 상태: technical-valid, red-flag-for-metadata, population-inconclusive
- 사전 고정: spectral Transformer, real/imaginary/log-magnitude 6–60 Hz, seed 42, 3 LOSO folds, A0/A2, 10 epochs, no augmentation, no validation/checkpoint selection. 여섯 training completion 전에는 결과를 공개하지 않았다.
- 관찰: 720 samples/role에서 mean A0 BA 0.122222, mean A2 BA 0.104167, mean delta −0.018056이었다. Subject delta는 sub001 −0.016667, sub002 −0.004167, sub003 −0.033333이었다. 여섯 job 모두 `max_epochs`, final fixed-epoch checkpoint로 완료됐고 대표 peak CUDA allocated memory는 약 0.94 GB였다.
- 해석: 이 grid는 metadata 이득을 지지하지 않으며 세 participant 모두 음수라는 점은 경고다. 그러나 각 fold의 train participant가 2명뿐이고 N=3 exact one-sided 최소 p가 0.125라 39-subject training의 모집단 효과나 무효를 판정할 수 없다. Epoch/architecture를 사후 변경해 양의 결과를 찾지 않는다.
- 정보 접근: S1–S3 validation 720행만 첫 grid로 공개했다. S4–S102 EEG outcome, 39명 confirmatory training outcome, 60명 lockbox outcome은 모두 미접근이다. `N99_accessed=false`.
- 증거: `outputs/development-loso/spectral-fixed-e10-v1/revealed_summary.json` SHA-256 `19fd2fae6d57fee4f6692df9e7dd95df06885265649c27412ae6bcb54a43e88b`; `revealed_subjects.csv` SHA-256 `a392d7dbfb4118bb0f462ca536d69132cae17ca2fd03031b5027f4d45c7b5044`; grid manifest SHA-256 `eed8b6f57957d29f90ea53d1e4346f2f2756ac7dd37e8506e529bb893597720a`; reveal ledger SHA-256 `2e26a963e1e5b5ec2c50e6d48dcaa4a283523391af8c9e7cc714e30322d1f553`.
- 결정: confirmatory plan은 `dev_not_frozen`으로 유지한다. Recommended 두 번째 S1–S3 reveal은 새 predeclared mechanism question 또는 추가 wearable-like development 근거가 생길 때까지 보존한다.
- 다음 gate: 외부 승인값(SESOI/alpha/control margin)과 clean source ownership을 해결하고, 음의 development delta를 해소할 독립 근거를 확보한다.

## COR-20260830-001 — 접근 상태·개발 판정 정정 및 단일 confirmatory 실행 경계

- 상태: completed, confirmatory still sealed
- 용어 정정: `DEV-20260830-006`의 제목에 쓴 `blind`는 **outcome-gated**로 읽는다. 여섯 job 완료 전 결과를 보지 않았다는 뜻이지 raw signal에 접근하지 않았다는 뜻이 아니다. 같은 항목의 `red-flag-for-metadata`는 formal cutoff가 사전 지정되지 않았으므로 **directional-warning / benefit-not-demonstrated / population-inconclusive**가 현재 판정이다. 이 항목이 이전 용어를 supersede한다.
- 접근 상태 정정: 과거 여러 항목의 `N99_accessed=false`는 성능/outcome 미접근을 뜻했으나 raw byte 접근과 혼동될 수 있다. 현재 기계적 상태는 `raw_audit_accessed=true`, `confirmatory_training_executed=false`, `lockbox_prediction_revealed=false`, `primary_report_locked=false`다. S4–S102 raw/processed byte와 metadata는 전체 자산 무결성 감사에서 읽혔지만 모델 학습·선택·성능 계산에는 쓰지 않았다.
- 구현: analysis plan에 canonical execution manifest와 run root를 결합했다. 다른 경로의 manifest/run tree를 거부하고, canonical manifest는 create-exclusive로 생성한다. 실행 산출물이 없는 unauthorized dev draft만 명시적 `--replace-unexecuted-draft`로 원자 교체할 수 있으며 frozen/authorized manifest는 교체할 수 없다.
- lockbox 경계: confirmatory checkpoint는 generic evaluation route에서 항상 거부한다. 등록된 predictor도 exact clean/test config, canonical manifest/reveal receipt, job checkpoint, manifest가 계산한 숨김 staging output만 허용한다. Reveal 뒤에도 robustness·perturbation·임의 output을 통한 개별 공개는 허용되지 않는다.
- 완료·출판 검증: completion receipt는 final/train-metric/split hash에 더해 exact 10 epoch, `max_epochs`, runtime-contract hash, exact-resume generation/state와 checkpoint 계약을 검증한다. 재시작 staging과 이미 공개된 final prediction grid 모두 CSV/sidecar/checkpoint/split/job/manifest/reveal 계약을 다시 검증하며 partial·unexpected file은 fail-closed다.
- 검증: 프로젝트 전용 환경에서 전체 unit suite `237 passed`를 확인했다. 수정 범위 Ruff, Python compileall, 80개 YAML parse, 모든 shell script syntax, `git diff --check`, Python 3.10.12/Torch 2.2.2+cu121의 RTX 4090 CUDA backward probe가 통과했다. 저장소 전체 Ruff에는 이 hardening 범위 밖의 기존 style debt 21건이 남아 있다.
- 연구 판정: S1–S3 첫 grid는 권고 2회·절대 3회 outcome reveal budget 중 1회를 사용했다. 두 번째 reveal이나 confirmatory 실행을 자동 승인하지 않는다. Plan은 `dev_not_frozen`이고 SESOI·alpha/alternative·operational threshold·control margins/policy·fairness/source freeze가 여전히 비어 있다.
- hand-off 절차: 이 append-only 항목을 기록한 뒤 현재 source contract로 canonical **execution-disallowed dev draft**만 교체·검증한다. Confirmatory training, prediction, reveal receipt, primary aggregation은 생성하지 않는다.

## IMP-20260831-004 — physical A2 전환 및 두 번째 assay 사전결과 계약 완성

- 상태: outcome-free implementation completed, reveal #2와 confirmatory는 미승인·봉인 유지
- A2 전환: primary A2를 legacy prompt/adapter에서 `physical_hybrid_v1`으로 바꿨다. A0/A2는 같은 graph·parameter schema·초기상태를 공유하고 external metadata mode만 null/observed다. 공통 query-QC는 bounded residual FiLM, electrode type과 impedance mean/max는 external-global FiLM, per-channel impedance/availability는 channel identity embedding 전 bounded gain으로 들어간다. Reference/cap은 이 asset에서 상수라 primary treatment에서 제외했다.
- 두 번째 assay: A0, Full A2, global-only, channel-only, shuffle-train→clean-validation, metadata-only의 exact 6 roles×3 folds=18 jobs와 각 fold당 Full A2 intervention bundle 하나(총 세 개)를 canonical grid로 구현했다. Natural missingness-only는 한 pattern뿐이라 `invalid_assay`다.
- 판정 규칙: 사전에 채울 shortcut, train/inference pairing, wet↔dry counterfactual, wrong-metadata safety, train/inference bundle potency, condition-flip potency의 8개 gate를 실제 aggregate에 연결했다. 각 gate는 세 observed subject/fold의 least-favourable 값으로 판정한다. 이는 N=3 모집단 신뢰구간이나 confirmatory 진행 권한이 아니다.
- provenance/reveal hardening: 모든 governed development evaluation을 mutable boolean과 무관하게 canonical grid authorization으로 강제했다. Prediction은 exact checkpoint, resolved config, final/last·train metric·split·development-control completion receipt, grid manifest, immutable reveal receipt에 결합한다. Main Full A2 clean과 intervention clean을 sample/label/prediction 단위로 대조한다. Reveal은 root-wide exclusive lock 안에서 registration→private staging→검증→directory rename을 수행하고, receipt는 후속 ledger append에 영향받지 않는 chained event hash에 묶인다.
- confirmatory 사전검증: lockbox reveal receipt를 만들기 전에 A0/A2 pair-wide parameter schema, initial state, split/runtime/source/control fairness를 검사한다. 구현되지 않은 `confirmatory_once` control policy는 freeze 단계에서 거부하고 현재는 `development_gate_only`만 허용한다.
- outcome-free CUDA 증거: Python 3.10.12, Torch 2.2.2+cu121, CUDA 12.1, RTX 4090에서 여섯 역할의 실제 wearable batch forward를 완료했다. 각 역할은 총/학습 parameter 2,864,027, 동일 parameter schema·초기상태·split·vocabulary·asset hash를 가졌다. M3 donor mapping은 480행, fixed point 0, coverage 1.0, external bundle changed fraction 1.0이었다. 결과 성능은 계산하지 않았다. `outputs/development/wearable_v3/physical_contract_dryrun_20260831_v2/summary.csv` SHA-256은 `52c54fa0553a419f3db9b923e95b66fdf68a45ded0fda164ac39212144f30ece`다.
- 검증: 전체 unit suite `273 passed`(26개의 알려진 Torch transformer warning), 수정 범위 Ruff, Python compileall, 43개 YAML parse, 모든 shell syntax, `git diff --check`가 통과했다. 저장소 전체 Ruff에는 이번 범위 밖의 기존 style debt 21건이 남아 있다.
- 정보 접근: S1–S3 processed signal/metadata와 label-aligned donor 정보는 forward/schema/control audit에 사용했지만 validation accuracy·BA·loss 같은 outcome은 계산·공개하지 않았다. S4–S102는 기존 full-asset 무결성 감사 외에 모델 학습·선택·성능 계산에 쓰지 않았다. Physical reveal #2 ledger/receipt와 39명 confirmatory checkpoint, 60명 lockbox prediction은 생성하지 않았다.
- 남은 blocker: owner가 reveal #2, 6개 numeric mechanism/safety/potency margin, 중단 규칙을 승인해야 한다. Physical 결과 검토 뒤에도 SESOI, alpha/alternative, operational threshold, `development_gate_only` 정책, clean commit/tag와 implementation/fairness hash를 별도로 동결해야 한다.

## DEC-20260831-003 — physical A2 development reveal #2 승인 및 기준 동결

- 상태: explicit owner approval recorded, outcome 미접근, confirmatory/lockbox 봉인 유지
- 승인 근거: active Codex session에서 사용자가 “응 너가 여러가지를 잘 설계해서 시작해보자. 승인.”이라고 명시 승인했다. 기록 시각은 `2026-08-30T17:52:16+00:00`이다. Repository-local receipt는 이 승인을 hash로 결합하지만 외부 전자서명이나 실명 인증은 아니다.
- 승인 범위: `wearable-v3-s1-s3-physical-mechanism-v2`, seed 42, folds 0–2, six roles×3 folds=18 fixed-10-epoch jobs, 세 Full-A2 intervention bundle, outcome reveal index 2, 사전 aggregate와 gate 판정까지다. 39명 confirmatory training, 60명 lockbox prediction, 모집단 추론, outcome을 본 S1–S3 재튜닝은 승인하지 않았다.
- 수치 동결: clean A2 mean minimum delta `0.0`; clean per-subject harm `0.03`; metadata-only observed upper screen `0.03`; train/inference pairing minimum `0.0125`; wet↔dry counterfactual minimum `0.0125`; wrong-metadata harm maximum `0.03`; bundle changed fraction minimum `0.95`; condition flip minimum `1.0`; confirmatory control policy `development_gate_only`다.
- 이산 해석: 각 fold는 12 class×20=240행이라 BA 한 step은 `1/240=0.0041667`이다. `0.0125`는 정답 3개, `0.03` upper bound는 정답 7개까지에 해당한다. 정확한 3/240 float 경계는 `1e-12` comparison tolerance로 판정하고 artifact에 tolerance를 기록한다.
- gate 개정: 기존 8개 mechanism/safety/potency gate에 `mean(Full A2−A0)>=0`과 clean per-subject harm audit를 추가해 10개로 만든다. 후자는 현재 counterfactual `3/240`과 wrong-metadata harm `7/240`을 함께 통과하면 `A0−clean<=4/240`가 되어 3% 한계가 수학적으로 중복임을 명시한다. 통계적 equivalence, 유의성, 모집단 안전성 주장은 금지한다.
- 원자 공개 개정: 기존 분리된 `predict→aggregate`는 사용하지 않는다. 단일 `reveal` transaction이 prediction·intervention·aggregate·10-gate 결과를 mode-0700 staging에서 완성하고 file-tree hash를 확인한다. 그 digest를 global ledger와 precommit receipt에 먼저 기록해 reveal #2를 소비한 다음 전체 directory를 atomic rename하고 parent directory를 fsync한다. 공개 tree를 다시 검증한 뒤 별도 final publication receipt를 쓴다. 따라서 rename 전 장애는 같은 precommitted digest만 재개할 수 있고, rename 후 장애는 최종 영수증만 복구한다.
- 중단 규칙: 학습 중단은 같은 frozen job의 exact resume만 허용한다. Private prediction/intervention의 알려진 partial artifact는 같은 manifest·checkpoint·seed로만 deterministic recompute하며 미신고 파일이나 digest drift는 거부한다. Source/config 변경이 필요하면 prediction 없이 중단하고 새 owner 결정을 요구한다. Provenance/coverage/identity/nonfinite 같은 기술 무결성 실패는 fail-closed이며 substantive 판정을 내리지 않는다. Potency gate 실패는 별도 `assay_valid=false`로 기록하며 음성 mechanism 증거가 아니다. 유효 assay에서 substantive gate 하나라도 실패하면 confirmatory를 차단하고 같은 S1–S3 threshold 조정·재학습을 금지한다. 전부 통과해도 별도 owner review만 가능하다. Reveal #2 뒤 추가 S1–S3 outcome reveal은 기본 금지다.
- 기준원: `configs/governance/wearable_physical_reveal2_decision.json` SHA-256 `15fb6de894d173dd26061e6ddf52b07708d498fc569ee7a3ab18916a30e22eee`. 다음 gate는 전체 테스트와 outcome-free CUDA probe를 통과한 clean commit을 annotated tag `physical-reveal2-freeze-20260831`로 고정하는 것이다.

## DEC-20260901-004 — precommit 전 counterfactual validator 기술정정 및 exact restart 승인

- 상태: explicit owner reapproval recorded, reveal #2 미소비, confirmatory/lockbox 봉인 유지
- 중단 사실: `ed5e7dce385fc456b33350079a98baa4aac88ec0`·`physical-reveal2-freeze-20260831`에서 exact 18개 학습은 완료됐지만, 단일 `reveal`은 counterfactual donor scope validator에서 global-ledger precommit과 공개 전에 fail-closed됐다. Global reveal ledger, precommit receipt와 public reveal bundle은 모두 생성되지 않았으므로 reveal #2는 소비되지 않았다.
- 정보 접근 경계: 성능 outcome은 검토하지 않았다. 실패 원인 확인에는 donor mapping·provenance control field만 사용했다. 세 fold 모두 mapping identity/coverage/potency는 유효했고, 실제 counterfactual scope `same_dataset_subject_label_block_window_opposite_electrode`를 validator가 shuffle scope `within_dataset_block_derangement_label_aligned`로 잘못 요구한 것이 원인이었다.
- 격리: 최초 실행 전체는 `/home/whwovy/califreeEEG/outputs/development-loso/quarantine/physical-mechanism-v2-aborted-precommit-ed5e7dc-20260901`로 이동했고 디렉터리 mode는 `0700`이다. 이 기록은 복구 가능한 격리이며 private outcome을 재사용하거나 열람할 권한이 아니다.
- 재승인 근거: active Codex session에서 사용자가 “응”이라고 승인했고 기록 시각은 `2026-08-31T17:38:18+00:00`이다. `DEC-20260901-004`는 `DEC-20260831-003`을 supersede한다.
- 정정 범위: scenario별 exact donor scope와 canonical manifest 기반 mapping/potency 재구성, 해당 회귀 테스트, private tree와 parent fsync 후 ledger precommit, full-ledger/publication status 검증과 symlink 거부, 문서 정합성만 바꾼다. 모델·dataset·18-job grid·seed 42·fold 0–2·10 epoch·8개 numeric margin·10-gate 판정은 그대로다. Canonical output에서 18 jobs를 처음부터 재학습하고 같은 reveal index 2를 한 번 실행한다.
- outcome-free 재검증: 프로젝트 `.venv`의 Python 3.10.12·Torch 2.2.2+cu121·CUDA 12.1·RTX 4090 backward probe가 통과했고 전체 suite는 `294 passed`(26개 알려진 Transformer warning)였다. 이 검증은 S1–S3 성능을 계산하거나 격리 outcome을 열지 않았다.
- 기준원: `configs/governance/wearable_physical_reveal2_decision.json` SHA-256 `ae7a6cdc49009462f0da836dafb81c1b4ffd3434c4db6a2f62f793d5dc4b1b59`. 다음 gate는 전체 회귀검증을 통과한 clean commit을 새 annotated tag `physical-reveal2-freeze-20260901-r1`로 고정하는 것이다.

## DEV-20260901-005 — physical A2 reveal #2 완료: valid assay / diagnostic no-go

- 상태: 공개 summary의 top-level scope는 `descriptive_mechanism_assay_only`, 사전 gate status는 `assay_valid=true`, `diagnostic_no_go_one_or_more_substantive_gates_failed`, `confirmatory_execution_authorized=false`, `population_inference_allowed=false`다. Physical reveal #2는 소비됐고 public bundle publication까지 완료됐다.
- 실행 기준원: commit `7bb8afc64c02e45beda7e245370563eac0e021e2`, annotated tag `physical-reveal2-freeze-20260901-r1`, decision `DEC-20260901-004`, decision receipt SHA-256 `ae7a6cdc49009462f0da836dafb81c1b4ffd3434c4db6a2f62f793d5dc4b1b59`다. 동일한 six-role×3-fold, seed 42, fixed 10 epoch의 18/18 training job과 3/3 Full-A2 intervention bundle이 완성됐다.
- 원자 공개: private tree 검증과 file/directory/staging-parent fsync 뒤 global ledger precommit으로 reveal budget을 소비했고, 같은 digest를 directory atomic rename·parent fsync로 공개한 다음 final receipt를 검증했다. 등록 시각은 `2026-08-31T18:24:36.017850+00:00`, reveal event SHA-256은 `0857c1d4f5ef4f2198224495fa03a4d1be7b754e349ae91901ba4f87a622a3f2`, grid content SHA-256은 `233b54ea91477b8912cd3d5def91af388d24e2b08566f6b1da74ad1065c696b0`, bundle content SHA-256은 `7761c8181e5bc4326dd5b4bf31999df04a536dd65d46bb8ea7befed07ef1cd7f`다.
- Clean 결과: mean A0 BA `0.1291667`, Full A2 BA `0.1194444`, Full−A0 `−0.0097222`였다. Fold/held-out subject별 Full−A0는 sub001 `−0.0125000`, sub002 `0.0000000`, sub003 `−0.0166667`이다. 따라서 사전등록한 mean direction `>=0`을 통과하지 못했다.
- 역할별 평균: global-only BA `0.1208333`(A0 대비 `−0.0083333`), channel-only `0.1305556`(`+0.0013889`), shuffle-train `0.1194444`(Full과 사실상 동일), metadata-only `0.0833333`(12-class chance와 동일)였다. Full−global은 `−0.0013889`, Full−channel은 `−0.0111111`이다.
- 기전 판정: substantive gate 네 개가 실패했다. `clean_a2_directional_mean` observed mean `−0.0097222`; `counterfactual_reliance` least-favourable `−0.0041667 < 0.0125`; `inference_pairing_shuffle` `0.0000000 < 0.0125`; `training_pairing_shuffle` `−0.0041667 < 0.0125`다. 즉 현재 구현은 correct metadata의 성능 이득이나 pairing/counterfactual reliance를 보이지 않았다.
- 유효성 판정: counterfactual flip fraction은 세 fold 모두 `1.0`, train/inference bundle changed fraction도 모두 `1.0`으로 potency gate를 통과했다. Metadata-only excess는 세 fold 모두 `0`, clean observed subject harm 최악값은 `0.0166667 <= 0.03`, wrong-metadata harm 최악값은 `0.0125000 <= 0.03`으로 shortcut·observed safety screen도 통과했다. Natural missingness-only는 사전 선언대로 `invalid_single_natural_missingness_pattern`이며 physical mechanism bundle 자체는 valid다. 따라서 이번 no-go는 조작 실패로 무효인 결과가 아니라, 유효한 N=3 진단에서 substantive 조건이 실패한 결과다.
- 개입 평균: clean `0.1194444`, all external missing `0.1250000`, block shuffle `0.1166667`, channel impedance missing `0.1208333`, common query-QC missing `0.1236111`, electrode missing `0.1250000`, global external missing `0.1236111`, global impedance-summary missing `0.1194444`, joint wet/dry counterfactual `0.1194444`였다. 제거·교환이 clean보다 일관되게 나쁘지 않았고 counterfactual 평균은 clean과 같았다.
- 해석 한계: 이는 영구 development-only S1–S3 세 명의 wiring/reliance/shortcut/safety assay다. 모집단 효과가 없다고 단정하거나, 통과한 observed safety screen을 모집단 안전성으로 일반화하지 않는다. 다만 potency가 충분한 상태에서 네 substantive gate가 실패했으므로 현재 frozen stopping rule 아래서는 진단적 no-go가 명확하다.
- 중단 결정: 동일 S1–S3에서 threshold·architecture·seed·epoch를 결과에 맞춰 바꾸거나 세 번째 reveal을 실행하지 않는다. 별도 owner review는 기록을 확인하는 절차이지 실패한 gate를 통과로 바꾸는 권한이 아니다. 현재 `physical_hybrid_v1`을 그대로 사용한 39/60 confirmatory training·lockbox prediction은 차단·봉인한다. 진행하려면 현재 실험의 재실행이 아니라, 독립 외부 development data 또는 새 사전근거에 기반한 별도의 모델/연구설계와 새 decision contract가 필요하다.
- 공개 증거: `outputs/development-loso/physical-mechanism-v2/reveal-bundle/analysis/revealed_summary.json`; grid manifest file SHA-256 `a7eb382300f3ca6b3513c4cd31fb6e62dba7f00217ac8167486e360f219ddbe0`; precommit receipt `73c7be68cf1b59f15c002984c169c1488a449708f7a5472d6dfdc078526ba2f6`; final receipt `4aef2b7a82302651fdf45d1de05b97a049f676ae1d4e8020b41bba191b51bad9`; bundle manifest file `7d1a0ff35f036ab159ef48bc6d210b74a55e3aaba0b30fb4e105af93fa89f9b5`; global ledger file `6093c85635cead106643664823c8c6f5285bfb949d28f2b0bb1b91cb636ee82d`.
- 검증: 실행 전 프로젝트 `.venv` 전체 suite `294 passed`(26개 알려진 Transformer warning)와 Python 3.10.12·Torch 2.2.2+cu121·CUDA 12.1·RTX 4090 backward probe를 통과했다. 공개 뒤 read-only `status`가 18/18 trained, 18/18 predicted, 3/3 intervention, reveal consumed, outcomes revealed, public bundle published와 전체 donor/provenance 검증을 재확인했다.
- 다음 gate: current A2 confirmatory가 아니라 연구 방향 결정이다. 현 경로를 종료할지, outcome-blind 독립 데이터와 외부 근거로 새로운 development study를 사전등록할지 owner가 결정한다. S4–S102 성능과 60명 lockbox prediction은 계속 미접근이다.
