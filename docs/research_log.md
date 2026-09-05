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

## DEC-20260903-005 — physical-hybrid-v1 영구 종료와 outcome 경계

- 상태: owner-approved retirement / deny overlay active.
- 결정: valid S1–S3 reveal #2에서 clean 방향, counterfactual reliance, inference pairing, training pairing의 네 substantive gate가 실패했으므로 `physical_hybrid_v1`을 `retired_no_go`로 종료한다. 과거 `A0_eeg_only`/`A2_structured_condition_prompt` artifact와 결정은 역사적 read-only 증거로만 보존한다.
- 금지: S1–S3의 추가 training, prediction, 새 metric, model selection, reveal #3 및 outcome을 본 tuning을 모두 금지한다. 과거 39명 confirmatory training과 60명 lockbox prediction도 실행하지 않는다.
- 허용: 기존 artifact 무결성 감사, 역사적 상태 조회, outcome-free contract regression뿐이다.
- 승인 근거: active Codex session에서 사용자가 “응 그렇게 하자.”라고 승인했다. 이는 workspace owner 기록이며 외부 전자서명이나 IRB 승인을 뜻하지 않는다.
- 기준원: `configs/governance/wearable_s1_s3_retirement.json`, SHA-256 `a2fcfb53e4ed1edeb21eca13d0f8851fdd6478834106a966ef83bae903f32710`.

## LIT-20260903-002 — no-go 이후 고수준 원문 검토와 가설 축소

- 상태: completed targeted full-text synthesis; systematic review 아님.
- 범위: conditioning/FiLM, electrode impedance와 환경·증폭기 상호작용, SSVEP reliable components, query-conditioned dynamic spatial filtering, missing-modality robustness, domain generalization/model selection, learned/retained/forgotten 분석을 연결해 핵심 9편의 Methods·Results·Discussion·Limitations를 판독했다.
- 결론: 연구목표는 유지하되 metadata를 class semantics conditioner로 쓰는 가설은 버린다. 현재 query EEG의 label-free reliability `Q`가 주 경로이고, impedance/interface `M`은 입력단 spatial operator에 작은 residual prior만 더하는 가설이 문헌과 기존 no-go에 가장 잘 맞는다.
- 설계 영향: primary를 `A_QM−A_Q`로 두고, metadata missing/stale/wrong, correct pairing, metadata-only shortcut과 source-only model-selection parity를 필수화했다. Static impedance는 system·frequency·environment 의존적이므로 calibrated monotonic quality score로 간주하지 않는다.
- OOD 경계: 참가자 수가 많다는 사실만으로 device/site/acquisition-domain OOD를 주장하지 않는다. 합성 실험은 wiring과 identifiability만 검증하며, human/OOD 승격에는 factor가 독립적으로 반복·조작된 새 acquisition development data가 필요하다.
- `arXiv:2608.11829`: learned/retained/forgotten은 나중의 descriptive capability redistribution 틀로만 보존하며, EEG mechanism 근거·endpoint·승격 gate로 쓰지 않는다.
- 증거: `/home/whwovy/research-spaces/califree-eeg-experiment-design/post_no_go_high_level_read_2026-09-02.md`; 당시 evidence workspace는 766 papers, 49 searches, 24 paper cards, 17 deep reads를 포함했다.

## DEC-20260903-006 — reliability-spatial-v1 합성 engineering 후보 동결

- 상태: owner-approved / frozen for synthetic engineering only.
- 연구 질문: 처음 보는 participant의 labeled calibration 없이, pre-query acquisition metadata가 현재 EEG에서 직접 계산한 query-local channel-quality 정보 너머의 순증분을 주는가?
- 후보: `X'=(I+ΔQ+αΔM)X`; symmetric diagonal+low-rank operator, `||ΔQ||F≤0.20`, `||ΔM||F≤0.05`, `|α|≤1`, `α=0` 초기화다. Metadata의 일반 decoder-vector route는 금지하고 bounded additive residual만 허용한다.
- Stage-0: 8 synthetic participant, 256 trial, 2,048 sample×active-channel truth row의 participant-disjoint nested LOSO identifiability assay다. Quality truth는 probe에서만 쓰고 decoder input에서는 격리한다. 첫 pass/fail outcome은 exclusive reservation 뒤 한 번만 atomic publish하며 overwrite/supersede하지 않는다.
- 조건부 Stage-1: Stage-0 PASS일 때만 `A0/A_M/A_Q/A_QM` exact 2×2를 seed 42, 같은 graph/init/split/asset/selection/runtime 계약으로 함께 CUDA 실행한다. Primary는 `A_QM−A_Q`, secondary는 `A_M−A0`, 수치는 checkpoint-selection validation subject 두 명의 descriptive engineering 결과다.
- 개입/안전: 같은 `A_QM` checkpoint에 correct/missing/stale-block/wrong-interface M bundle을 적용하고, 별도 `A_Q`와도 참가자별 기술 비교한다. Metadata-only probe는 ID/label 없이 participant-disjoint ridge로 고정했다.
- 중단: Stage-0 실패면 Stage-1을 실행·해석하지 않는다. Stage-1 실패/음성 결과로 S1–S3를 재튜닝하지 않는다. Human study 승격은 독립 데이터와 새 owner decision 없이는 금지한다.
- frozen digests: candidate plan `b75020eeaf5782f92079622078e3715ad9670644c7a9469b606e09323345b911`; Stage-0 plan `985b4f84ec8110d8e7a711b1d1bc0a873d81eb8a0bc5c5cf49febca79b945e91`; normalized full base contract `723277da27e46ea1a7406283eee369c1b699d900ae8491e379b2c54eb856d3f2`; exact suite file `4656941a49120003a60f01df38c9c9fdbf53ff6569c0286281bcb3bc6786558e`.

## IMP-20260903-007 — 합성 reliability 실행·분석·one-shot provenance 구현

- 상태: outcome-free implementation complete; canonical asset/Stage-0/Stage-1 실행 전.
- 구현: block-coherent quality DGP와 truth isolation, Q/M allowlisted probe, nested participant LOSO Stage-0, bounded reliability spatial operator, exact four-arm family, validation prediction/participant contrast, full-M intervention, deterministic metadata-only shortcut probe를 추가했다.
- 실행 안전: asset generation, Stage-0와 Stage-1은 fixed-name atomic reservation으로 동시 실행을 막는다. Stage-0는 terminal outcome 뒤에만 예약을 해제한다. Stage-1은 mode-0700 hidden root에 네 arm과 분석을 완성한 뒤 suite root 전체를 atomic rename하고 성공 뒤에만 예약을 해제한다. 실패 attempt는 비정식 root와 receipt를 남기며 자동 retry를 막는다.
- 최종 binding: Stage-1 공개 직전 candidate/source/implementation, Stage-0와 여덟 asset, 네 checkpoint, split, validation metric, runtime sidecar, nested runtime attempt를 다시 hash한다. Checkpoint epoch/metric은 `metrics_val.csv`의 첫 accuracy 최댓값과 일치해야 한다. `summary.csv`도 exact 네 completed role, canonical path와 analysis receipt hash를 검증한다.
- 회귀검증: 프로젝트 `.venv` 전체 suite `347 passed`(41개 알려진 PyTorch Transformer warning)였다. Focused Stage-0/Stage-1 atomicity, failure, reservation, hash-drift, main orchestration test도 통과했다. 새 dataset fallback vocabulary는 기존 Wang/BETA numeric ID를 바꾸지 않도록 append-only로 정정했다.
- 정보 접근: source/config/unit fixture와 기존 공개 historical artifact만 사용했다. 새 synthetic gate outcome, S1–S3 새 outcome, S4–S102 성능, 39/60 execution은 아직 수행하지 않았다.
- 다음 gate: 최종 정적 검증 → fresh canonical synthetic asset 생성 → one-shot Stage-0. PASS일 때만 exact CUDA Stage-1을 한 번 실행한다.

## COR-20260903-008 — scalar state-hash 정정과 최종 실행 동결

- 상태: outcome-free correction completed; canonical Stage-0/Stage-1 실행 직전 동결.
- dry-run 정정: 최초 비정식 dry-run `outputs/engineering/reliability-spatial-v1/stage1-dryrun-final`은 scalar metadata alpha parameter를 바로 byte view하는 초기상태 hash 코드에서 중단됐다. 이 실행은 Stage-0 이전의 noncanonical contract probe였고 decoder 성능 outcome은 만들지 않았다. Scalar를 먼저 1차원으로 reshape하는 `_trainable_state_sha256`로 정정하고 회귀 테스트를 추가했다.
- 성공한 CUDA 계약 probe: 전용 비정식 namespace `outputs/dry-runs/reliability-spatial-v1/stage1-contract-v1`에서 A0/A_M/A_Q/A_QM 네 역할 모두 `dry_run`을 완료했다. 각 역할은 total/trainable parameter `469,669`, resolved device `cuda:0`, `cuda_oom=false`였고 runtime family는 `verified_equal`이었다. `summary.csv` SHA-256은 `a790bc4d5352d2fbea217dd43e1d8fc3f4aaa05d6b5dfb5c30745495068fc955`다.
- 실행 경계: dry-run은 fresh `stage1-contract-*` direct-child만 허용하고 canonical asset/Stage-0/Stage-1/reservation namespace와 분리했다. Direct partial-arm training은 in-process private authorization으로 차단했다. Stage-0/Stage-1은 fixed reservation 뒤 canonical 부재를 다시 검사하며, Stage-1 summary·analysis·atomic publication 실패는 noncanonical attempt receipt 또는 보존된 reservation으로 fail-closed한다.
- 최종 기준원: candidate plan SHA-256 `6227f2e06db09dec0c18768454f18c8e4866f7ef0b6d29e18aef71c00ad63b39`; Stage-0 plan `985b4f84ec8110d8e7a711b1d1bc0a873d81eb8a0bc5c5cf49febca79b945e91`; normalized full base contract `723277da27e46ea1a7406283eee369c1b699d900ae8491e379b2c54eb856d3f2`; suite file `4656941a49120003a60f01df38c9c9fdbf53ff6569c0286281bcb3bc6786558e`; retirement receipt `a2fcfb53e4ed1edeb21eca13d0f8851fdd6478834106a966ef83bae903f32710`; reliability family `b9320ba600b8ea6912783c757cfd169ae5c87fda92716f5fa2660e0bb0e0ebef`; implementation contract `8c33e4697aac92210b61f3f871dd30dbea8de39c0c6fe312a19c52088dc8a713`다. `DEC-20260903-006`에 적힌 이전 candidate digest는 그 시점의 역사적 값이며 이 항목이 실행 기준 digest를 supersede한다.
- 검증: 프로젝트 `.venv` 전체 suite `351 passed`(41개 알려진 PyTorch Transformer warning)였다. 최종 독립 읽기 전용 감사에서 candidate/Stage-0/suite/base binding, reservation 후 재검사, atomic publication과 실패 영수증을 확인했고 남은 execution-blocking P0/P1은 없었다.
- 다음 gate: 현재 source/config를 더 수정하지 않고 canonical Stage-0를 정확히 한 번 실행한다. PASS일 때만 같은 동결 상태에서 exact CUDA Stage-1 4-arm을 실행한다.

## ENG-20260903-009 — reliability Stage-0 terminal FAIL / Stage-1 미실행

- 상태: canonical one-shot Stage-0 `failed`; conditional Stage-1 prohibited and not executed.
- 실행: exact 8-file `synthetic_quality_v1` asset의 8 participant, 256 trial, 2,048 sample×active-channel row에 participant-disjoint nested LOSO ridge assay를 한 번 실행했다. 결과는 fresh hidden staging에서 검증된 뒤 canonical root에 원자 공개됐고 terminal publication 뒤 Stage-0 reservation이 정상 해제됐다.
- 무결성: asset inventory/fingerprint, generator contract, JSONL↔Parquet semantic identity, active-channel truth coverage 1.0, balanced complete blocks, pre-query metadata timing, predictor allowlist와 truth-model 격리가 모두 통과했다. Label-only subject-macro R² `−0.137111 ≤ 0.01`, class target mean range `0.002847 ≤ 0.02`로 leakage/balance audit도 통과했다.
- 통과한 signal/pairing 항목: Q held-subject mean R²는 `0.973349 > 0`이었다. M의 Q 초과 partial R² 평균은 `0.043065 ≥ 0.02`였다. Correct metadata의 block-shuffle 대비 relative SSE gain은 평균 `0.152518`, channel-permutation 대비 `0.127443`이고 둘 다 8/8 participant에서 양수였다.
- 실패한 substantive 항목: M partial R²가 양수인 participant는 `5/8 < 6/8`; Q→QM channel-rank Spearman 증분 평균은 `0.000651 < 0.02`; rank 증분 양수 participant도 `5/8 < 6/8`이었다. 세 consistency/size check가 실패해 전체 gate가 terminal `failed`다.
- 해석: simulator에 M-corruption 연결과 correct pairing reliance는 존재하지만, query-local Q가 이미 realized quality를 거의 모두 설명하는 상황에서 M이 held-out participant 전반의 품질 예측과 채널 순위를 안정적으로 개선하지 못했다. 이는 human metadata 전체의 무용성을 보인 것이 아니라 동결된 DGP·target·feature에서 H2의 incremental-identifiability contract가 실패한 것이다. H1은 synthetic 범위에서 통과했고 H3 decoder benefit/H4 safety는 미검증이다.
- 중단 준수: canonical Stage-1 root와 Stage-1 reservation은 생성되지 않았다. Passing-receipt validator가 이 결과를 `Reliability Stage-0 receipt failed its frozen gate contract`로 거부함을 확인했다. 따라서 CUDA 4-arm decoder BA, `A_QM−A_Q`, missing/stale/wrong intervention outcome은 존재하지 않는다. 같은 candidate의 Stage-0 재실행·덮어쓰기·threshold/DGP/feature 사후조정은 금지한다.
- 증거: `outputs/engineering/reliability-spatial-v1/stage0/receipt.json` SHA-256 `e8f3f87bbe47a8c8d5825f7d0525c318e87d42c6ecdd2840de21b1c657eefc35`; artifact SHA-256은 receipt에 결속됐다. 실행 source-tree SHA-256은 `f3a6a21264d86c09ca2af2a84b5cde7099aa2c3c97f8631f74bdbdd51bb832f2`, implementation contract는 `8c33e4697aac92210b61f3f871dd30dbea8de39c0c6fe312a19c52088dc8a713`다.
- dirty-source 보존: 실행 직후 동일 source-tree hash를 재확인한 뒤 tracked+untracked source snapshot을 `/home/whwovy/research-spaces/califree-eeg-experiment-design/provenance/stage0_execution_source_f3a6a21264d86c09.tar.gz`에 보존했다. Archive SHA-256은 `136b1a5fdb96f2bd66c6302d6562409eb3f08918f4739d6d7d81f2321f86c3c0`다. 이후 문서 갱신은 실행 snapshot과 분리된다.
- 다음 gate: 연구목표는 유지하되 현재 candidate는 종료 상태로 보존한다. 같은 participant/task에서 interface·contact/impedance·motion·time drift를 반복 또는 무작위화한 독립 human development data와 새 사전등록·owner decision 없이는 새 metadata 후보나 untouched human cohort로 승격하지 않는다.

## DEC-20260904-001 — query-only 후속 연구 방향 승인 / outcome bundle 미승인

- 상태: owner direction approved / implementation draft / human outcome blocked.
- 결정: 상위 응용목표인 unseen-participant strict inductive k=0 SSVEP는 유지한다. 반면 “외부 impedance/interface metadata가 현재 query 품질정보 너머의 순증분을 준다”는 구체 가설은 terminal predecessor failure와 함께 종료한다. 별도 후속 후보 `query-reliability-spatial-v1`은 외부 metadata 없이 현재 single query에서 얻는 Q bundle로 bounded spatial operator를 만드는 질문이다.
- 승인 범위: architecture·contract·unit/integration test, 자산·split·sample identity 무결성 감사, outcome-free power simulation, 사람 BETA training batch의 forward-only CUDA graph probe만 허용한다. Human training, loss/accuracy 계산, lockbox prediction/reduction/reveal, Dong 실행은 승인하지 않았다.
- 비교 경계: primary `Q1_AUG−Q0_AUG`는 동일 corruption augmentation 조건의 exact Q-bundle-conditioned system increment다. Corrupted `Q1_AUG−Q0_CLEAN`에 별도 utility gate를 동결하지 않으면 표준 clean-trained identity보다 우월하다는 주장은 허용하지 않는다.
- 승인 근거: active workspace owner의 “응 그렇게 해보자. 시작.” 이는 방향과 outcome-free 구현 범위 승인이지 외부 전자서명, IRB 승인, exact outcome bundle freeze가 아니다.
- 기준원: `configs/analysis/query_reliability_spatial_v1.yaml` 현재 file SHA-256 `1d2b8c92e454f3ecb05869f3f12596f53facdcba8ff31fd539b23421a855012f`; `exact_outcome_bundle_approved=false`, 모든 human outcome authority=false다.

## LIT-20260904-002 — query reliability 인접연구·신규성 경계 갱신

- 상태: evidence workspace updated / systematic-review completeness not claimed.
- 문헌 지도: `/home/whwovy/research-spaces/califree-eeg-experiment-design`를 860 papers, 55 search runs, 26 paper cards, 27 deep-read items(17 completed, 10 queued), 20 open gaps 상태로 다시 렌더했다. Query-only technique 카드와 N=20/freeze-blocker gap을 갱신했다.
- 가장 가까운 선행: MAPS-CS는 SSVEP query-quality channel selection, Dynamic Spatial Filtering은 current-window-conditioned EEG spatial operator, Fast SSVEP는 query-local alignment/spectral denoising, SSVEPPoolformer는 adaptive denoising/cross-channel pooling의 경계를 이미 제시한다. TMW-CCA와 EMD-QC도 quality-aware SSVEP의 최초 주장을 막는다.
- 증거 수준: DSF full text와 Fast SSVEP preprint는 판독했지만, MAPS-CS와 SSVEPPoolformer는 공식 abstract 수준이며 exact information rights·split·implementation은 미확정이다. TMW-CCA/EMD-QC와 protocol-matched baseline port도 미완료다. Semantic Scholar 429 source failure가 있어 검색 완전성을 주장하지 않는다.
- 해석: 현 기여는 Q feature의 발명이나 순수 feature 인과효과가 아니라, source-trained seven-input Q bundle, bounded operator, exact identity arm, participant-disjoint/stress/localization/shortcut 계약을 묶은 조건부 system evaluation이다. `arXiv:2608.11829`의 learned/retained/forgotten은 lower-tail·cell transition을 보는 analogy일 뿐 EEG mechanism 근거나 primary gate가 아니다.
- 다음 문헌 gate: MAPS-CS·SSVEPPoolformer·TMW-CCA·EMD-QC의 합법적 full text/code/information-rights를 확인하고, 구현 가능하면 같은 BETA split/window/channel/corruption 계약으로 port한다. 그 전에는 `first`나 broad OOD novelty claim을 금지한다.

## IMP-20260904-003 — query-only 2×2 구현 draft와 최종 outcome-free 검증

- 상태: implementation and outcome-free verification complete / exact human outcome runner not implemented or authorized.
- 구현: `Q0_CLEAN/Q1_CLEAN/Q0_AUG/Q1_AUG` exact 2×2, seven-input Q extractor, identity-initialized diagonal-plus-low-rank bounded operator, deterministic nested corruption, fixed BETA 35/20 split, lockbox generic-access deny overlay, complete-grid probability reducer, participant-level paired statistics, semantic table/input/mask hashing을 추가했다.
- 메커니즘 준비: 같은 Q1 checkpoint의 exact-identity와 wrong-query model hook을 구현했다. Wrong-query donor는 같은 participant·class 안의 다른 trial로 고정한 3,200-row nonself bijection이며 mapping SHA-256은 `e1d7be88b55a1ad35d975787b95183ff65909507eace648109e069164344027f`다. 이는 다른 target trial과 offline label pairing을 쓰므로 strict-k0 배포 estimand가 아니고 standard prediction과 분리된 분석 전용 개입이다.
- 분석 출력: corrupted `Q1_AUG−Q0_CLEAN`, clean/corrupted 2×2 interaction, participant×cell, 역할별 worst cell, participant 10/25/50% quantile, class×cell과 50-Hz harmonic flag를 필수 descriptive report로 추가했다. Primary success만으로 모든 cell, unseen corruption, contact/motion, device/site/domain OOD를 주장하지 않는다.
- power audit: N=20, paired SD 0.04에서 true primary mean 0.03의 단일 superiority power는 약 0.944지만, true clean delta 0에서 margin −0.01인 clean NI 하나의 power는 약 0.286이다. 80%에는 SD 약 0.0173 이하 또는 SD 0.04에서 약 N=101이 필요하다. `+0.02`는 inferential null이 아니라 observed promotion screen이고 12/20은 비추론적 consistency screen이다.
- 검증: focused query-reliability suite `50 passed`; 전체 repository suite `402 passed`와 54개 알려진 Transformer warning; 변경된 Python 파일 ruff, `compileall`, `git diff --check`가 통과했다. Repository 전체 ruff에는 이번 변경 밖의 기존 lint debt가 남아 있어 변경 파일 기준 통과를 별도 확인했다.
- 실제 CUDA forward-only: seed 11의 네 역할 모두 RTX 4090 `cuda:0`, `cuda_oom=false`, 11,200-row asset view, 한 training batch logits shape `[256,40]`, total/trainable parameters `2,553,279`로 완료했다. Family SHA-256은 `97a037a826531d9c5d3c0d62e8a28c206642aeea4a0fc04241c8a2dc7e6bfceb`; 공통 parameter schema `52975abb0fff32ff722820862c9728bd8d889d776c8442d7c0ec66b430eee9ce`; initial trainable state `086fb508a8fdd60dce6465834fa191e611e179d277240416a79739e33ae0b7cf`; runtime split `d0afab8d17be77f80b3666722dc71036569b93b9370af6bbfeab9e0e7f14570c`; asset provenance `2f5e9dd2062e6102518e5adfcdec5ff1892387fcd79b0f935cec303034a85568`다.
- outcome 경계: 위 CUDA 작업은 optimizer/epoch에 진입하지 않은 forward-only contract probe다. Human training, checkpoint, loss, accuracy, prediction, participant contrast, lockbox outcome은 생성·열람하지 않았다.
- 동결 blocker: 실제 tensor/checkpoint/runtime artifact를 재해시하는 mode-0700 atomic runner, owner decision receipt와 clean annotated tag를 요구하는 별도 frozen-state validator, intervention/probe/ablation artifact schema·threshold, utility-based numeric margin, corrupted deployment gate 선택, baseline full-text/port가 남았다. 독립 사전등록 Dong replication을 주장하려면 exact recipe·estimand·threshold·stopping/atomic contract를 BETA reveal 전에 동결해야 한다.

## COR-20260904-004 — primary claim·target-population 문구 정정과 최종 family 재검증

- 상태: outcome-free semantic correction / `IMP-20260904-003`의 이전 plan·family hash만 supersede.
- 정정: primary를 순수 architecture 또는 reliability-feature 효과가 아니라 `Q-bundle-conditioned system utility`로 통일했다. 비교는 동일 corruption augmentation의 identity arm이고, target population은 현재 0.63–2.63초 crop이 유효한 BETA 3초 자극 적격 S16–S70로 한정했다. S1–S15나 전체 BETA로 외삽하지 않는다.
- 안전성 용어: clean NI 실패 stopping key를 `clean_noninferiority_failure: clean_safety_not_established`로 바꿨다. 실패는 harm 증거가 아니라 clean safety 미확립이다.
- 문헌 범위: metadata predecessor의 no-go는 frozen synthetic Stage-0 DGP·target·feature 계약에 한정했으며 human metadata 일반 결론으로 확장하지 않았다. Evidence workspace technique 카드와 렌더를 같은 문구로 갱신했다.
- 최신 기준원: candidate plan file SHA-256은 `52a9289cfae872d89f87d8279163fb3b53dd1dc926b4e146d7e86d9c3d90dfc2`, family SHA-256은 `4bf5e8043272fa273b1fb77eb077d8b06200e663f1551c6c62c19bebc69009a5`다. `DEC-20260904-001`과 `IMP-20260904-003`에 기록된 이전 plan/family 값은 그 시점의 draft history이며 실행 기준이 아니다.
- 최종 CUDA 재검증: 최신 family에서 seed 11의 Q0_CLEAN/Q1_CLEAN/Q0_AUG/Q1_AUG 네 역할 모두 RTX 4090 `cuda:0` forward-only를 다시 통과했고 `cuda_oom=false`였다. 공통 logits `[256,40]`, total/trainable parameters `2,553,279`, parameter schema `52975abb0fff32ff722820862c9728bd8d889d776c8442d7c0ec66b430eee9ce`, initial state `086fb508a8fdd60dce6465834fa191e611e179d277240416a79739e33ae0b7cf`, split `d0afab8d17be77f80b3666722dc71036569b93b9370af6bbfeab9e0e7f14570c`, asset provenance `2f5e9dd2062e6102518e5adfcdec5ff1892387fcd79b0f935cec303034a85568`가 동일했다.
- outcome 경계: 이 정정과 재검증도 optimizer/epoch, loss/accuracy, checkpoint, lockbox prediction/reveal을 만들지 않았다. Human outcome 권한은 계속 false다.

## DEC-20260904-005 — calibration burden 목표 복구 / metadata를 수단으로 재정의

- 상태: owner direction approved / 새 exact outcome bundle은 미승인.
- 결정: 상위 연구목표를 “처음 보는 사용자가 유용한 closed-set SSVEP 성능을 얻는 데 필요한 labeled target calibration 최소화”로 복구한다. `k=0`은 calibration-free anchor, `k>0`은 low-calibration regime으로 구분한다.
- Metadata 정의: 개인정보나 dataset ID가 아니라 query 전에 관측 가능한 acquisition context로 제한한다. 현재 primary treatment 후보는 wet/dry interface와 block 전 per-channel impedance다. Query에서 계산한 scale·variance·spectrum·covariance는 signal-derived Q이며 external metadata로 세지 않는다.
- 새 후보: `metadata-calibration-efficiency-v1`. Primary pair는 같은 graph·source·support·query·adaptation을 쓰는 A_Q와 A_QM이고, primary estimand 초안은 `k={0,1,3}` participant eAUC의 `A_QM−A_Q`다. 정확한 method formula, SESOI와 outcome runner는 미동결이다.
- 이전 후보 경계: `physical_hybrid_v1` retirement와 `reliability-spatial-v1` terminal Stage-0 failure는 변경하지 않는다. `query-reliability-spatial-v1`은 사람 outcome 전 철회된 frozen baseline-only이며 기존 BETA 35/20 outcome plan을 실행하지 않는다.
- 승인 근거: active workspace owner의 “응 다시 원래 목표인 적은 callibration SSVEP EEG를 위한 방법으로의 metadata? 라고 해야할까? 로 돌아가자.” 이는 연구 방향과 outcome-free 설계 작업의 승인이지 human EEG training, support-label access, prediction 또는 reveal 승인이 아니다.
- 기준원: `configs/analysis/metadata_calibration_efficiency_v1.yaml`과 `docs/metadata_calibration_efficiency_design.md`.

## AUD-20260904-006 — local acquisition-metadata 식별가능성 감사

- 상태: completed / manifest·config·asset metadata only.
- Wearable: 102명×2 interface×10 block×12 class = 24,480행이다. 204 participant×interface session과 2,040 participant×interface×block이 있으며, block 전 8채널 impedance가 그 block의 12개 class에 공통으로 결합된다. Wet/dry와 impedance는 변하지만 hardware, reference, montage와 site는 고정이다.
- Wang/BETA/Dong: 총 164명·29,040행이지만 각 corpus 안의 physical descriptor가 고정되고 실제 participant/block impedance는 null이다. 이 세 corpus는 signal-only calibration baseline이나 transport sensitivity에는 쓸 수 있으나 dataset 간 차이로 metadata factor 효과를 식별할 수 없다.
- 구조/Q 경계: local canonical xyz는 모두 null이고 channel name/ID만 있다. `query_signal_std`와 channel vector는 EEG-derived QC이므로 A_Q와 A_QM이 동일하게 받아야 한다.
- governance: S1–S3 추가 outcome은 영구 금지다. S4–S102 성능은 미접근이므로 새 후보에서 기존 outcome-blind 39/60 allocation을 보존하는 안은 가능하지만, 새 candidate ID, exact block partition, atomic runner와 별도 owner freeze가 필요하다. K>0 target은 strict-k0 lockbox가 아니라 sealed calibration-support/held-query 평가다.
- 추가 데이터: broad device/site/reference claim에는 같은 participant의 randomized interface×device/reference, repeated session·reattachment, block impedance, 3D coordinates와 실제 onboarding 시간을 기록한 prospective 자료가 필요하다.
- 정보 접근: `signals.h5` array, model output, loss, accuracy, support label과 query outcome을 읽거나 계산하지 않았다.

## DES-20260904-007 — complete-block low-calibration estimand와 power 초안

- 상태: outcome-free design completed / numeric SESOI·NI margin과 execution은 미동결.
- k 정의: wearable에서 k는 participant·electrode condition당 완전한 labeled block 수다. 한 block은 12 class×1 trial이므로 k=0/1/3/5는 0/12/36/60 labeled trial이다.
- partition: block 1–5를 support pool, block 6–10을 모든 role·k의 immutable query로 둔다. `S0⊂S1⊂S3⊂S5`이고 wet/dry를 별도로 보정·평가한 뒤 participant 안에서 equal-average한다. 현재 label별 samplewise calibration utility는 block을 섞으므로 새 primary runner로 사용할 수 없다.
- primary 초안: `eAUC=Y0/6+Y1/2+Y3/3`의 participant-level A_QM−A_Q difference 하나를 one-sided alpha 0.05로 검정한다. Seed/class/block/support draw는 독립 N이 아니다. k5와 각 k contrast는 secondary다.
- savings: A_QM(k0) 대 A_Q(k1), A_QM(k1) 대 A_Q(k3), A_QM(k3) 대 A_Q(k5)를 사전 비열등성 fixed sequence로만 해석한다. Margin은 실제 deployment utility로 정하며 inverse interpolation은 descriptive다.
- power: N=60, paired SD 0.06, true eAUC difference +0.02일 때 analytic one-sided power는 약 0.818이다. 같은 N에서도 +0.01 또는 비열등성은 부족할 수 있어 SD·상관·heavy-tail·20% harmed mixture를 포함한 동결 simulation이 필요하다.
- 문헌 경계: one-shot CSDuDoFN, SSVEP-DAN, cross-domain LST/eTRCA/TDCA가 low-calibration의 직접 선행이다. 조건부 신규성은 few-shot 자체가 아니라 pre-query physical context의 Q 초과 증분, correct/shuffled/missing control과 complete-block savings를 결합한 평가다.
- 정보 접근: 코드·config·manifest와 outcome-free analytic 계산만 사용했다. 사람 EEG training/prediction/metric은 수행하지 않았다.

## IMP-20260904-008 — 목표 복구 계약과 complete-block partition 검증

- 상태: outcome-free design contract implemented and verified / human outcome execution remains unauthorized.
- 구현: `metadata_calibration_contract.py`에 새 plan의 header·정보권한·predecessor 경계·data scope·A_Q/A_QM fairness·k=0/1/3/5 complete-block support·single eAUC primary를 fail-closed로 검증하는 계약을 추가했다. Manifest-only partition builder는 한 participant×interface의 block01–05 support와 block06–10 query, block당 12개 label의 완전성, nested support, query 비중첩과 sample-identity hash를 검사한다.
- 실제 manifest 감사: `wearable_v3/manifest.parquet` 24,480행의 102명×wet/dry=204 participant-condition group 모두가 exact ten-block 계약을 통과했다. 각 group의 support 수는 k=0/1/3/5에서 0/12/36/60이고 query는 60행이다. 모든 group query-identity digest를 정렬 결합한 SHA-256은 `e953efb20e0ca25528e3e9f1fdf10d080100c30818b0d38079ad4b8e27b15e5a`다.
- 기준원: concept plan SHA-256은 `0f90a639268ade06182a4e04b673182f1318427c18d72e09e9faaa82a81d6a15`다. `exact_outcome_bundle_approved=false`이고 code validator의 `outcome_execution_authorized`도 false다.
- 문서 정합성: 철회된 query-only 문서의 과거 strict-k0/metadata-종료 현재형을 historical wording으로 바꾸고, academic evidence workspace의 제목·목적·cutoff를 현재 calibration-efficient acquisition-context 질문과 `2026-09-04`로 갱신한 뒤 렌더했다.
- 검증: 새 focused suite `4 passed`; 전체 repository suite `406 passed`와 54개 알려진 PyTorch Transformer warning; 수정 Python Ruff, 51개 YAML parse와 `git diff --check`가 통과했다.
- 정보 접근: 이 검증은 config, source, tests와 manifest의 sample/block/label schema만 읽었다. `signals.h5`, EEG tensor, optimizer/epoch, model output, loss/accuracy, held support label access와 query prediction/reveal은 수행하지 않았다.

## COR-20260904-009 — nested-prior 명료화와 generic wearable outcome 우회 차단

- 상태: outcome-free P0/P1 audit correction complete / `IMP-20260904-008`의 plan hash와 406-test count superseded / human outcome authority remains false.
- P0 발견과 수정: legacy 또는 public checkpoint의 train config만으로 ungoverned 판정을 받은 뒤 eval config의 `processed_dirs`를 `wearable_v3`로 바꾸면 generic calibration/evaluation으로 진입할 수 있는 경로를 읽기 전용 감사에서 발견했다. Effective evaluation target의 path, expected revision과 `asset_info.json`을 검사하는 새 preflight를 `load_evaluation_context`의 manifest·dataset 생성 전에 연결했다. Generic evaluation, prediction과 calibration은 향후 전용 atomic runner와 exact owner decision 없이는 wearable_v3를 열 수 없다.
- 방법 정정: A_Q와 A_QM을 별도 end-to-end checkpoint로 학습하면서 missing A_QM=A_Q라고 쓰는 모순을 제거했다. Stage 1에서 common spectral backbone과 Q-only calibration estimator를 학습·동결하고, Stage 2에서 공통 경로에 M gradient/statistic을 금지한 채 positive-definite precision 또는 bounded shrinkage의 M residual만 fit한다. A_Q는 같은 composite checkpoint의 residual exact-off, A_QM은 residual-on이다. 따라서 all-missing은 metadata 영향을 받지 않은 frozen A_Q path와 exact 같아야 한다.
- predecessor 경계: 새 후보는 target-support calibration prior만 바꾸며 query waveform을 직접 변환하거나 채널을 섞지 않는다. Terminal `reliability-spatial-v1`의 M-beyond-Q 실패는 여전히 negative prior evidence다. Exact 39명 source-development에서 사전 수치화할 eAUC increment, correct>shuffle/stale, missing/wrong safety와 shortcut gate를 모두 통과해야 60명을 열 수 있다. 그 수치와 runner는 아직 미동결이므로 39명과 60명 outcome 모두 금지다.
- allocation 결속: 새 plan은 `DEC-20260830-002`의 exact 39/60 ID, seed 42, algorithm과 allocation SHA-256 `0c5b7cd1ca8eedafbe9290833a283160f7bfe89897be8923b21568f70bb18898`을 자체 검증한다. Source 78 participant×interface query group은 4,680행, digest `224d8239aef5d0f9dd04c388f8ed8d2f961fc150c1cf72741f5ab22b756f80de`; held 120 group은 7,200행, digest `feffdcc005f2f4818bb496c61b683dc7afd735166c2f50cf7892262ef55d2a16`에 결속했다.
- claim 정정: k=1의 12 trial은 deployment interface당 수치다. Wet/dry를 모두 평가하는 연구 전체 support 접근은 k=1/3/5에서 participant당 24/72/120 trial이다. 현재 자료는 same-device observational predictive utility를 식별하며 interface나 impedance의 독립 인과효과 또는 broad device/site OOD를 허용하지 않는다.
- 최신 기준원: `configs/analysis/metadata_calibration_efficiency_v1.yaml` SHA-256은 `4c82e28c4d60f05a88e118e645bd7b04d8dfa2205df21419e0dd9e2ab11004df`다. Academic workspace technique `technique:1c40921ceed610a6`을 nested calibration-prior로 갱신하고 formula-distinction gap `gap:6d3481c95a3535ce`를 추가해 렌더했다.
- 검증: focused metadata/eval/governance/query gate suite `66 passed`; 최종 전체 repository suite `409 passed`와 54개 알려진 PyTorch Transformer warning; 수정 Python Ruff, 51개 YAML parse, stale-current wording search와 `git diff --check`가 통과했다.
- 정보 접근: 코드, config, asset identity와 manifest block/label schema만 사용했다. `signals.h5`, EEG tensor, optimizer/epoch, loss/accuracy, model prediction, held support adaptation과 held query reveal은 수행하지 않았다.

## COR-20260904-010 — 목표 문서 정합성·alias hardening·canonical cohort binding

- 상태: outcome-free final audit correction / human outcome authority remains false.
- 문서 정합성: README와 과거 strict-k0 체크리스트, metadata protocol amendment, deep-read audit, A2 계약, historical execution plan에 superseded/당시 표현을 보강했다. 현재 질문은 `A_QM−A_Q` early calibration curve이며 과거 `physical_hybrid_v1`의 `A2−A0`는 retired historical contrast다. 과거 physical power receipt가 새 eAUC·calibration-savings power 승인이 아니라는 점도 명시했다.
- alias hardening: generic train/eval preflight가 path basename·expected revision·정상 `asset_info.json`뿐 아니라 manifest의 `dataset_id` fallback으로도 wearable을 탐지한다. Renamed copy에서 asset receipt가 없거나 malformed여도 `signals.h5` 또는 generic manifest loader 전에 차단하는 회귀시험을 추가했다. 이는 same-OS raw-file adversary를 막는 암호학적 봉인이 아니라 accidental stale copy/receipt-loss에 대한 fail-closed hardening이다.
- cohort binding: complete-block cohort validator에서 caller-supplied subject IDs, revision assertion과 optional digest를 제거했다. `source_development`와 `held_participant_evaluation` role만 허용하고 canonical plan에서 정확한 role별 IDs와 query digest를 직접 읽어 검증한다. 이 helper는 outcome-free manifest contract이며, 아직 없는 future atomic runner가 asset receipt 검증 뒤 dataset 생성 전에 호출해야 한다.
- plan validator: same composite checkpoint, frozen common Q path, exact missing fallback, method implementation gates, condition/reset rules, participant-level one-sided eAUC inference와 unresolved SESOI/NI margin을 추가로 fail-closed 검증한다.
- 최신 기준원: plan SHA-256은 변경 없이 `4c82e28c4d60f05a88e118e645bd7b04d8dfa2205df21419e0dd9e2ab11004df`; validator의 `outcome_execution_authorized=false`다.
- 검증: focused metadata/governance suite `38 passed`; 전체 repository suite `410 passed, 3 skipped`와 55개 알려진 Transformer warning. 세 skip은 현재 unit-test Python의 CUDA AMP/integration opt-in 경로다. 수정 Python Ruff, 51개 config YAML parse와 `git diff --check`가 통과했다.
- 정보 접근: 코드, config와 synthetic manifest fixture만 사용했다. Human `signals.h5`, support label, optimizer/epoch, loss/accuracy, prediction과 reveal은 수행하지 않았다.

## COR-20260904-011 — 모든 generic entrypoint alias 차단과 최종 독립 재감사

- 상태: outcome-free contract audit complete / independent read-only audit P0=0, P1=0 / human outcome authority remains false.
- 추가 hardening: training과 neural evaluation뿐 아니라 CCA/FBCCA baseline evaluator와 retired OOD coverage analysis도 같은 manifest-aware wearable detector를 사용한다. Renamed wearable copy의 `asset_info.json`이 valid/missing/malformed인 세 경우 모두 baseline dataset construction 전에 차단하는 회귀시험을 추가했다.
- 독립 재감사: alias 차단, canonical 39/60 ID·query digest 결속, A_Q/A_QM fairness, participant-level eAUC estimand와 현행/역사 문서 구분에서 잔존 P0/P1이 없음을 읽기 전용으로 확인했다. Human signal, support label, prediction과 outcome 봉인은 유지된다.
- 최신 검증: 전체 repository suite `412 passed, 3 skipped`, 55개 알려진 Transformer warning; 세 skip은 현재 unit-test Python에서 CUDA AMP/integration opt-in이 비활성인 경로다. 변경 Python Ruff와 `git diff --check`가 통과했다. Plan SHA-256은 `4c82e28c4d60f05a88e118e645bd7b04d8dfa2205df21419e0dd9e2ab11004df`, `outcome_execution_authorized=false`다.
- 남은 freeze blocker: exact positive-definite/bounded prior 수식과 parameter bound, 39명 수치 mechanism/safety gate, SESOI·calibration-savings 비열등성 margin, 새 power simulation receipt, complete-block full-probability atomic runner, clean source freeze와 exact owner decision receipt다.

## IMP-20260904-012 — calibration-efficient metadata 방법·통계·checkpoint 계약 구현

- 상태: outcome-free implementation advanced / human EEG execution remains blocked.
- 방법: frozen spectral embedding 위의 diagonal-Gaussian target-calibration prior를 구현했다. Q는 precision을 base의 0.25–4배, M은 Q precision을 0.8–1.25배만 바꾸며 absolute precision은 0.05–20이다. M은 source anchor나 waveform을 직접 바꾸지 않는다. k=0에서도 M은 query predictive variance만 바꿀 수 있어 A_QM(k0) 대 A_Q(k1)의 savings 질문이 수학적으로 가능하다.
- 정보권한: Q builder는 공식 8채널×7 특징=56차원, M builder는 interface 2+impedance 8+availability 8=18차원으로 분리했다. A_Q off-path는 M을 읽지 않고 A_M off-path는 waveform/Q를 읽지 않는다. Generic electrode numeric ID, label, participant/session/file/sample ID는 builder API에 없다.
- composite/checkpoint: spectral backbone에는 channel ID/mask만 전달한다. Stage 2 freeze 뒤 trainable parameter는 `prior.metadata_precision_encoder.*`뿐이다. Full-state checkpoint writer는 stage-1 common-state hash와 final common-state hash의 exact equality, parameter schema, M initial/final state, plan/decision/asset/class-map/source binding을 검증하며 trainable-only 저장을 거부한다.
- 개입: correct/all-missing, within-participant×interface 및 support/query pool 내부 block derangement, true previous-block stale(block01 missing), same-participant×block opposite-interface mapping을 label 없이 구현했다. Wearable의 자연 impedance availability는 사실상 전부 available이므로 availability 효과는 engineering intervention 이상으로 주장하지 않는다.
- 평가 누출 차단: held core를 A_0/A_M/A_Q와 A_QM correct/missing/shuffle/stale/wrong의 8개 cell로 고정했다. Private staging에는 query label/prediction/correct/loss/accuracy가 들어갈 수 없고, complete grid·binding·support identity 검증 뒤에만 label sidecar를 join한다. Development checkpoint는 `(seed, outer_fold)`, held는 `(seed)`로 결속한다.
- 통계: 39명 outer fold를 exact 3×13으로 강제하고 metadata-only screen을 participant equal-dry/wet BA의 one-sided 90% Student-t upper bound로 구현했다. Held wrong-context의 severe harm(<−0.05) 최대 6명과 catastrophic harm(<−0.10) 금지는 반드시 공개하는 participant-tail 안전성 보고 항목으로 두되, 평균 NI와 중복된 fixed-sequence gate로 쓰지 않는다. `1/60` margin은 interface당 평균 한 오답, dry+wet 전체 participant outcome에서는 평균 두 오답이라는 의미다.
- source recipe: 39명의 exact 13/13/13 outer folds와 각 fold의 20 inner-fit/6 inner-validation을 outcome-blind로 고정했다. Inner epoch 선택 뒤 non-outer 26명을 refit하고 outer outcome은 tuning에 쓰지 않는다. Stage 1은 common+Q, Stage 2는 M residual만 fit하며 held participant에서는 closed-form support update 외 gradient를 금지한다.
- baseline: mandatory information-rights ledger를 만들고 k=0 FBCCA, k=1/3/5 supervised template correlation, k=3/5 ensemble TRCA, A_Q를 고정했다. Single-trial eTRCA는 퇴화하므로 금지한다. Template/TRCA core와 full score/probability 테스트를 구현했으며 atomic adapter와 TDCA 등 direct comparator 포팅은 남았다.
- power: normal-theory t power, heavy-tail/20%-harmed marginal simulation, 7-endpoint fixed-sequence joint simulation, 39명 all-gate promotion simulation과 deterministic receipt generator를 구현했다. Inferential rejection과 observed SESOI screen을 합친 full-claim probability를 분리한다. 최종 200,000-draw receipt는 source 문서·상태 동결 뒤 생성한다.
- 검증: 이 항목 작성 전 focused method/feature/composite/checkpoint/intervention/reducer/power/baseline tests가 모두 통과했다. 사람 `signals.h5`, optimizer, support label, prediction, loss/accuracy 또는 query outcome은 읽거나 생성하지 않았다.
- 남은 blocker: final power receipt, mandatory baseline atomic/direct adapter, full human atomic runner, clean source tag/hash, exact owner decision receipt.

## COR-20260904-013 — H1/H2 절약계층·최소 development gate·opaque execution boundary 확정

- 상태: outcome-free design/statistics/private-boundary correction complete / human EEG execution remains blocked. 이 항목은 `IMP-20260904-012`의 7-endpoint fixed sequence, metadata-only t-UCB, TDCA 우선순위, final receipt 대기와 baseline adapter 대기 상태를 supersede한다.
- Confirmatory hierarchy: H1은 participant-level `eAUC(A_QM)-eAUC(A_Q)>0`의 one-sided alpha .05와 관측 평균≥.020이다. H1 뒤 H2a `A_Q(k3)-A_Q(k1)>0`·관측 평균≥.020과 H2b `A_QM(k1)-A_Q(k3)>-1/60`의 one-sided alpha .025·두 endpoint 평균 BA≥.50를 intersection-union으로 함께 요구한다. 통과 시 equal-weight wet/dry estimand에서 interface별 보정 schedule 36→12, 즉 24 labeled trials/two blocks 절약만 주장한다. k0→k1, k3→k5는 descriptive다. Wet/dry 각각의 비열등성은 주장하지 않는다.
- Multiplicity와 안전성: H1→H2 serial gatekeeping은 alpha split 없이 성립하고, core 뒤 correct>shuffle/stale 두 mechanism endpoint는 Holm .05를 쓴다. Held wrong-context mean/tail은 efficacy core를 뒤집지 않는 별도 deployment qualification이다. Participant contrast evaluator는 raw BA `[0,1]`, eAUC·mechanism·wrong·세 savings와 calibration-value 항등식을 row별 `atol=1e-12`로 재검증한다.
- Development gate 정정: 39명의 hard promotion은 A_Q(k5) viability, eAUC mean/fold, correct>shuffle mean/fold와 exact-missing·metadata-only structural proof의 5 stochastic+2 invariant만 쓴다. Median/positive count, stale, wrong/correct harm tail은 필수 보고·배포자격으로 보존하되 중복 efficacy gate로 세지 않는다. 계획 DGP에서 hard pass는 normal/t5/20%-harmed `0.784285/0.790335/0.786315`, 모든 17 check 동시 통과는 `0.564815/0.534605/0.567185`, null hard promotion은 `0.00021`이다. Metadata-only는 M block-constant, block당 12 class 각 1회, row ID/order 부재가 BA=1/12를 강제하는 deterministic proof다.
- Power receipt: 사람 EEG outcome을 쓰지 않은 200,000-draw final receipt를 재생성했다. Shared-A_Q(k3) planning에서 H1 `0.903335`, full H2 saving `0.656915`; independent/adverse/t5/harmed는 `0.654560/0.657260/0.660955/0.655990`이다. H2a/H2b partial-null false full claim은 `0.004350/0.013715`; N=60/75/80/83/90 full saving은 `0.656915/0.761365/0.788870/0.805065/0.833565`다. Held Holm mechanism 두 endpoint 동시 claim은 planning sensitivity에서 약 `0.946–0.957`이나 core와의 joint pipeline 확률은 아니다. Plan SHA-256 `7a1156ec4e796d4fcc44f12cf095b486e62402db19bf1001389319252c2fe037`, receipt payload `d4d6d6ee8976e0d03b1787a4245109341682e1e4cae9c2cbf866c7d948344446`, receipt file `76677bf26ff1887054a2add91a4d7ccf9415710dadf9fa7149b124ee7f1fa88a`다.
- Baseline: strict FBCCA k0, target template k1/3/5, target filter-bank eTRCA k3/5, same-checkpoint A_Q와 Chiang-2021 LST+filter-bank eTRCA k1/3/5를 mandatory로 확정했다. LST clean-room core와 label-free query adapter가 구현됐고 filter-bank runtime payload는 실제 config file/hash와 일치해야 한다. LST k1은 protocol-adapted이며 target-only TRCA k1은 금지, plain TDCA k3/5와 CSDuDoFN/SSVEP-DAN은 optional이다.
- Query/support leakage correction: production complete-block partition은 exact allowlist의 opaque q_/s_ token과 block/count만 읽으며 label·raw sample ID·HDF5 index·stimulus/order proxy를 거부한다. 기존 raw sample-ID query bundle 검사는 `audit_complete_block_cohort_partition`이라는 privileged historical input-sealer 전용 경계로 격리했다. Metadata intervention은 sample ID 없는 participant×interface×block 한 행 mapping으로 바꿨다. Unlabeled HDF5 reader는 y를 읽지 않고 nullable Q/impedance를 zero-filled value+boolean missing mask로 전달한다. Support token은 다른 participant/interface에서 재사용할 수 없다.
- Private staging: candidate의 opaque query grid, sample-level nested support rehash와 mandatory baseline raw-score grid seal은 구현됐다. 그러나 in-memory seal을 실제 producer/checkpoint receipt에 묶고 query-label sidecar를 한 번만 여는 production finalizer, fsync/precommit/atomic rename/crash recovery와 full source-development/held runner는 아직 없다.
- Runtime: 이 항목의 최종 preflight에서 CUDA 1개와 `NVIDIA GeForce RTX 4090`이 확인됐다. 이는 forward/training 실행 권한이 아니라 환경 가용성 확인이다.
- 남은 blocker: privileged input sealer+receipt, producer/job/checkpoint/manifest binding을 포함한 mandatory baseline execution seal, all-role×budget×condition runner와 production finalizer, clean source tag/hash, 새 exact owner decision와 authorization envelope다.
- 정보 접근: config, source, synthetic/unit fixtures와 outcome-free Monte Carlo만 사용했다. S4–S102 `signals.h5`, 39/60 support label, optimizer/loss/accuracy, query prediction·metric·reveal은 읽거나 생성하지 않았다.

## COR-20260904-014 — label-free collator·cell dispatch 결속과 power receipt 재동기화

- 상태: outcome-free execution-contract increment / `COR-20260904-013`의 plan·receipt hash를 supersede / human EEG execution remains blocked.
- 입력 경계: `UnlabeledQueryReader`가 HDF5의 `x`와 channel mask만 읽은 뒤, 새 query collator가 raw nonnegative query-QC와 impedance를 동결식 `log1p(value)/log(101)`로 한 번만 정규화한다. Missing value는 0과 별도 boolean mask로 표현한다. Model batch에는 `y`나 raw `sample_id`가 없으며 opaque query token은 prediction row를 이후 sidecar와 결속하는 데만 쓴다.
- 실행 분기: held 8-cell과 source-development 전용 4 diagnostic cell을 immutable dispatch table로 만들었다. 각 role/context는 query/support Q mode, M mode, feature ablation과 intervention에 유일하게 대응하며, source 진단 cell을 held phase에 넣으면 실패한다. Phase별 dispatch SHA-256은 source-development `c85096bd1d16da7474cc572bc2da996364c43cc8b571aabceb7023f26895be82`, held `24119fba7f1455597842648efe4543a5054a5bdef0aa418cbe1ae59ac477f564`다. 향후 producer receipt가 이 hash를 반드시 결속한다.
- Power 재동기화: plan 변경 뒤 사람 outcome 없이 200,000-draw receipt를 다시 만들었다. 확률값은 `COR-20260904-013`과 동일하다. 최신 plan SHA-256은 `5008ff52f6a663ebaa3b39b7b4c932d48f3b47bee8c724ec24c384f4e53a7c41`, receipt payload SHA-256은 `d94c287789899ff0a448ec9de48e5b4e822916fafb33bb51cf65b12b0ff517bd`, receipt file SHA-256은 `d833913f3f97e072bf31ed889852c862514f4885ab7e4b2add19c3992a3ee0b8`다.
- Count·safety 용어 정정: `COR-20260904-013`의 “모든 17 check”는 일곱 hard gate와 열한 required report를 합친 **18 check**로 정정한다. Median/positive-count와 stale 세 지표는 diagnostic report이고, wrong-context 평균·두 tail과 correct-context 두 tail의 다섯 check만 robustness deployment qualification을 정한다. 이 열한 report는 held efficacy 접근 gate가 아니다.
- Runtime: Torch `2.2.2+cu121`, CUDA device 1개, `NVIDIA GeForce RTX 4090`을 확인했다. 이는 outcome 실행 승인이나 학습 완료를 의미하지 않는다.
- 남은 blocker: privileged input sealer와 artifact receipt, model/baseline producer receipt 및 execution-manifest 결속, one-time label join production finalizer, all-role×budget×condition runner, clean source freeze와 exact owner authorization envelope다.
- 정보 접근: config, source, synthetic/unit fixture와 outcome-free Monte Carlo만 사용했다. Human `signals.h5`, support label, optimizer/loss/accuracy, query prediction·metric·reveal은 읽거나 생성하지 않았다.

## COR-20260904-015 — label-proxy 제거·2단계 권한·power null 해석 교정

- 상태: outcome-free design/security/statistics correction complete / `COR-20260904-014`의 plan·receipt hash와 `COR-20260904-013`의 null 해석을 supersede / human input sealer와 outcome execution은 모두 unauthorized.
- Query label-proxy 제거: wearable 원본은 target loop가 0–11 순서라 source `h5_index`와 행 위치가 label을 드러낸다. Downstream reader schema에서 raw `h5_index`를 금지하고, HMAC query token 오름차순으로 행과 signal을 함께 재작성한 local `signal_index=0..N−1`만 허용했다. Query HDF5는 exact `x`·`channel_mask` 두 dataset만 가져야 하며 `y`나 추가 dataset이 있으면 거부한다. Canonical channel ID는 integer cast 전에 정수성을 검사하고 official 8-active collator→composite k0 E2E test를 추가했다.
- 권한 순서: input seal 이전에는 support/query label과 raw human signal 접근이 금지되므로, seal-only owner decision template를 별도로 만들었다. 첫 receipt는 봉인 artifact 생성만 허용하고 training, score, loss/accuracy와 query-label 공개를 금지한다. 실제 outcome execution은 생성된 input-seal receipt를 execution manifest와 함께 결속한 두 번째 owner receipt가 있어야 한다. 현재 둘 다 template일 뿐 승인이나 실행이 아니다.
- Phase-derived grid: held 8-cell과 source-development 추가 4-cell을 반환하는 factory를 만들고, production bundle spec은 phase에서 context와 checkpoint group을 재구성하게 했다. Dispatch hash payload에는 schema, candidate와 phase를 포함한다. 최신 source-development dispatch SHA-256은 `d092c5b6b911c688e05d69b520422cb355627d7ec693eaaf0c59f5a4e5a82987`, held는 `aad87d4c8b2cfa5dcb159e3b37afad299e1000d19b3ae7fe0ffde224cfe5f3f2`다. Production manifest/finalizer 연결은 아직 남았다.
- Development report 구현: gate boolean뿐 아니라 A_Q(k5) 평균, eAUC·shuffle·stale의 평균/median/양수 수, 세 outer-fold 평균, wrong/correct tail의 실제 count와 minimum을 `observed_statistics`로 반환한다. 일곱 hard gate와 열한 diagnostic/deployment report 역할은 분리돼 있다.
- Power 교정: H1 eAUC-difference SD grid와 H2b BA-difference SD grid를 의미상 분리하고, config-driven development scenario에 정상 baseline 0.55의 metadata-null을 추가했다. 그 hard promotion 확률은 `0.009530`; baseline까지 0.48인 joint-bad null은 `0.000210`이다. 따라서 false-go 설명에는 전자를 쓴다. Partial-null full 확률은 유한 대립값을 둔 configured scenario일 뿐 union-null supremum이 아니며, strong control은 H2a/H2b local component test와 intersection-union 구조에서 온다. H2 full sensitivity는 두 endpoint BA≥0.50을 가정한 조건부 contrast simulation이고 mechanism power는 core와 별도로 생성한 standalone sensitivity다.
- Receipt hardening: validator가 receipt self-hash 외에 현재 power-config hash, plan, script/power/reducer module, 전체 held core/sample-size/mechanism/development scenario의 DGP, draw 수, deterministic cell seed와 MC error를 재검증한다. 재해시한 DGP tampering 회귀시험도 추가했다.
- 최신 기준원: plan SHA-256 `2e9860a66eb5df239d12aad0a9ea97ec710e7c3edd165111a541afd8eb8d4296`, receipt payload SHA-256 `76123bb348e076727b60f123e6ed842e98bcaf9b57feafa7bc89785a2b84f682`, receipt file SHA-256 `cfb264e6e3f757b836260c0c168f737b9efef8eacc323487c33c9038e53c428d`다.
- 검증: metadata-calibration focused suite `81 passed`; 관련 Python Ruff와 `git diff --check`가 통과했다. 이 단계에서 human raw manifest/signal, support/query label, optimizer/loss/accuracy, query prediction·metric·reveal은 읽거나 생성하지 않았다.
- 남은 P0: 실제 privileged input sealer/receipt, sealed context에 적용된 intervention tensor hash, model/baseline producer receipt와 checkpoint/execution-manifest 결속, path-only production finalizer·precommit/recovery, full source-development/held runner, clean source tag와 두 단계 exact owner authorization이다.

## COR-20260905-016 — phase 실행계약·governed metadata 경로·직접 선행 비교 정합화

- 상태: outcome-free execution-contract/security/design increment complete / `COR-20260904-015`의 plan·receipt hash와 comparator 상태를 supersede / actual execution manifest와 human outcome authority는 여전히 없음.
- 목표와 주장 경계: 연구목표는 바꾸지 않았다. H1은 participant-level `eAUC(A_QM)-eAUC(A_Q)`에 대해 population mean `>0`을 one-sided alpha .05로 검정하고 관측 평균 `≥.020` screen을 함께 요구하지만, population effect가 최소 .020임을 검정하는 것은 아니다. 따라서 H1만으로는 external acquisition context의 양의 순증분만 주장한다. H1 뒤 H2a의 Q-only `k3-k1` calibration value와 H2b의 `A_QM(k1)` 대 `A_Q(k3)` 비열등성·BA floor가 모두 통과해야 interface별 `36→12`, 즉 24 trials/two blocks 절약을 주장한다.
- 실행계약: canonical plan, source recipe, baseline ledger, input-artifact contract, power receipt와 phase별 cell dispatch에서 현재 mandatory-core source-development candidate 9+baseline 12=21 jobs와 held candidate 3+baseline 4=7 jobs를 유도하는 self-hashed contract를 구현했다. Source job은 exact outer fold를, held job은 `outer_fold=null`을 갖고 job/output/receipt path가 모두 유일해야 한다. Held gate/private-seal 값은 아직 actual execution manifest가 채워야 하는 명시적 placeholder이며, contract 자체는 `execution_authorized=false`다. CSDuDoFN/OS-SSVEP 또는 SSVEP-DAN이 feasible하면 adapter job을 추가해 count를 다시 동결한다. Source/held contract SHA-256은 각각 `2340e942afe6afffb2fa372c80963920b3c3bca16105b2e43b8ed093cb9b649d`와 `046fe2bfaea1ca140e9e845d983fb84471dade1ea0c620d0e47ca74be6ebda27`다.
- metadata 사용 경로: production candidate는 caller가 interface·Q/M mode·ablation을 따로 고르는 low-level forward가 아니라 canonical cell과 resolved context를 받는 `MetadataCalibrationComposite.forward_cell`을 사용한다. Resolver가 정한 correct/shuffle/stale/wrong interface와 8채널 impedance/missing 값을 query와 support에 즉시 적용하고 phase/cell drift를 거부한다. Query HDF5는 local dataset 두 개만, exact float32 `[N,64,400]`와 bool `[N,64]`만 허용하며 초기화 뒤 signal 또는 mask byte가 바뀌면 row hash로 실패한다. 다만 source-fit `f_` 경로와 실제 producer/finalizer에서 sealed input으로 context usage hash를 재계산하는 경계는 아직 남았다.
- 입력·권한 계약: input-artifact contract 전체 파일 bytes를 fail-closed hash로 고정하고, source target 39명은 3×13 cross-fit, source seal의 fold별 fit pool은 26명으로 제한하며 39명 전체 final refit은 source phase에서 명시적으로 금지했다. Held seal만 target 60명과 held-domain final-refit source 39명을 별도 phase domain으로 허용한다. Candidate A_Q와 Chiang-LST ledger도 source fold에서는 non-outer 26명, passing gate 뒤 held refit에서만 39명을 쓰도록 validator로 강제했다. Label sidecar는 같은 Unix user의 mode-0600만으로 격리하지 않으며, owner가 separate OS principals 또는 finalizer public-key encryption 중 하나를 선택해야 한다. 권한은 `source seal decision/receipt → source outcome decision/execution → immutable gate/private seal → PASS 시 held seal decision/receipt → held outcome decision/execution` 순으로 phase별 반복한다. Template에는 prior-phase receipt, manifest payload, trusted signing key/signature, sample-size acknowledgement와 두 high-value direct comparator의 run/not-run resolution을 추가했지만 실제 signature/envelope validator나 승인 receipt는 없다.
- 직접 선행 비교: CSDuDoFN arXiv와 peer-reviewed OS-SSVEP는 같은 one-shot lineage이므로 하나의 k=1 comparator로 관리한다. SSVEP-DAN 원법은 class당 최소 2 labeled trial을 요구하므로 현재 complete-block grid의 k=1에는 부적합하고 k=3/5에서만 재현 가능하다. 두 방법 모두 owner freeze 전에 author code, license와 exact target-information rights를 판정하고, 실행하지 못하면 이유를 공개한다. Chiang-2021 LST+filter-bank eTRCA는 현재 mandatory direct comparator이고 TDCA는 optional이다. 887-paper/57-search academic workspace의 좁은 prior-art 경계는 유지하며, nested-prior formula gap `gap:6d3481c95a3535ce`는 구현 완료로 닫았다. 이는 systematic-first 또는 SOTA 주장을 허용하지 않는다.
- Power·기준원: 계획 중심값에서 H1 통과 `0.903335`, N=60 full-H2 조건부 민감도 `0.656915`, N=83은 `0.805065`다. Owner는 N=60의 약 .657을 명시적으로 수용하거나, 승인하지 않고 약 N=83의 새 allocation·plan·power receipt를 만들어야 한다. 최신 plan SHA-256은 `f45845272e9851b42fce15b8024d07801dbf54609611a4ad3699cafde1aa5de0`, source recipe `85a2f69d166e98bc89e0abd1bba25a327d56871193c34857fd022112001f8f0b`, baseline ledger `785b0e7823579c482ef98d51d0fb09c4390f8f54812b43c99f25383656720dba`, input-artifact contract `6ffb8f49b53682e886ff0c3692e8c13c76016b71b69af4845256b47cfe4b2658`, power receipt payload `c08e440317113ecef5d421c5b3d977a412e8afee89e3764aa7d348a2e4111f6b`, receipt file `b51cce2f9d508de5f555777c7ee8a9611c7c3b598ff664973d199231c304d3ec`이다.
- 검증·경계: 새 focused contract/security suite는 `44 passed`; 전체 repository suite는 `508 passed, 3 skipped`와 66개 알려진 Transformer warning으로 통과했다. 관련 Python Ruff, `compileall`, YAML 55개·JSON 8개 parse와 `git diff --check`가 통과했다. Full-repository Ruff의 이번 변경 밖 기존 19개 lint debt는 별도로 남아 있다. 이 작업은 config, source, 문헌 workspace, synthetic/unit fixture와 outcome-free Monte Carlo receipt만 사용했다. S4–S102 raw signal, 39/60 support/query label, optimizer/loss/accuracy, model prediction·metric·reveal은 읽거나 생성하지 않았다.
- 남은 P0: high-value direct comparator code·license·정보권한 resolution, privileged sealer와 실제 signed input-seal receipt, actual execution manifest, owner-decision/authorization-envelope validator, checkpoint·intervention/context·output을 manifest 기대값에 묶는 producer receipt, source-fit governed path, path-only one-time-label-join finalizer와 atomic full runner, clean source tag가 필요하다. 이들이 닫히기 전에는 39명 source outcome도 실행하지 않는다.

## COR-20260905-017 — one-shot 실행 승인·최종 인프라 봉인·source→conditional-held 동결

- 상태: outcome-free implementation/freeze candidate complete / user가 추가 데이터와 운영 결정을 위임하고 source부터 조건부 held까지 실험 종료를 승인함 / 이 항목 작성 시점에는 human outcome claim 없음.
- 연구목표 유지: 질문은 여전히 `적은 target calibration에서 EEG-derived Q를 넘는 pre-query acquisition context M이 participant-level calibration curve를 개선하는가`이다. H1 `eAUC(A_QM)-eAUC(A_Q)`, H2a Q-only calibration value, H2b `A_QM(k1)` 대 `A_Q(k3)` 비열등성과 BA floor를 그대로 사용한다. Same-device observational predictive utility만 식별하며 impedance/interface 인과효과나 broad device/site OOD는 주장하지 않는다.
- 문헌·추가 데이터 결정: academic workspace는 cutoff 2026-09-04 기준 887 papers, 57 searches, 28 cards, 29 deep reads(19 complete/10 queued)다. Strict-k0 SSVEP에서 impedance/acquisition context를 Q 초과 조건부로 검정한 직접 선행은 찾지 못했고, 문헌은 impedance가 context-dependent proxy라 Q와 중복될 수 있음을 지지한다. 따라서 지금의 가장 값싼 불확실성 해소는 추가 검색보다 correct/shuffled/missing/wrong intervention을 포함한 사전 동결 실험이다. Ke2025 24명은 primary metadata-effect 검정에 합치지 않고 primary 종료 뒤 signal/generalization 진단에만 사용한다.
- 직접 비교군 결정: strict FBCCA k0, target template k1/3/5, target filter-bank eTRCA k3/5, SAME component k1, Chiang-2021 LST+FB-eTRCA k1/3/5를 실행한다. CSDuDoFN/OS-SSVEP와 SSVEP-DAN은 공개 재현 코드의 명시적 license 및 현 complete-block 정보권한·protocol 부적합 때문에 숫자 없이 audited `NOT_RUN`으로 고정했다. 이는 few/one-shot 최초 주장을 하지 않기 위한 투명한 범위 제한이다.
- 데이터·실행량: wearable_v3 24,480 rows/102 participants에서 exact source 39명과 held 60명을 사용한다. Source는 candidate 9+baseline 15=24 jobs, source gate가 PASS일 때만 held candidate 3+baseline 5=8 jobs다. N=60에서 계획 DGP의 full-H2 조건부 민감도 약 0.657을 수용했으며, PASS가 아니면 held query outcome은 열지 않는다.
- 권한·원자성: input sealer, least-capability bubblewrap worker, encrypted finalizer sidecar, private-grid full-chain validator, permanent outcome-access registry와 no-replace claim을 구현했다. Lifecycle completion은 canonical 공개 전 signed pending 전체를 검증하고 두 생성 private key 삭제와 parent fsync 뒤에만 publish한다. Key 삭제 후 publication만 실패한 경우에는 signed pending recovery만 허용한다. Outcome claim 이후의 다른 실패는 signed redacted failure를 남기고 임의 재시도하지 않는다. 같은 Unix UID의 악의적 프로세스까지 격리하는 보장은 없으므로 실행 중 해당 UID의 비협조 프로세스가 없다는 운영 가정을 둔다.
- 계산 경로: query EEG embedding과 Q를 prepared batch당 한 번 계산하고 모든 governed cell·budget에서 재사용한다. Cache에는 raw M이나 forbidden future field를 넣지 않고, 입력·모델·embedding·Q signature와 opaque row binding을 매 호출 검증한다. Shared job은 Q-off cell에서도 Q를 미리 계산할 수 있으나 Q-off logits에는 exact zero만 전달되어 수치적으로 Q에 의존하지 않는다. Source/held의 모든 80 cell×budget 조합에서 production과 같은 `no_grad` 조건의 uncached/cached logits와 probabilities가 bitwise 동일했다.
- 기준원: plan SHA-256 `04d77058dbec3d82d11ac15c8ad25f32fb3a3c102644e8a92a76cf759e22515a`, source recipe `48be9d7df0badd5f5fb38589e3d92d6f28385de44d330ce58f9e8068f3f735c6`, baseline ledger `6c3058e0d704cb5dcea045ebb12362590362ff2d66d54250e6ba42aba9cbf42d`, input-artifact contract `6443956372afb318a21ec38776ea02e6470fad14310fe9c6b724632de6b840ca`, power receipt payload `5d8313b95d46f9859c36ddbb71a86cc20fec2e452516a53814581e1d5c8d9f53`, receipt file `b54f5064af62250192daed3f3babb88058f1a9d41e2785b91c2288588ac2835a`다.
- 검증: metadata-calibration suite `109 passed`, 전체 repository suite `536 passed`와 68개 알려진 PyTorch Transformer warning, 변경 Python Ruff, 전체 config YAML/JSON parse, `git diff --check`가 통과했다. Full-repository Ruff에는 이번 변경 밖의 기존 19개 lint debt가 남아 있다. Repository에 private-key/API-token pattern이 없고 untracked 연구 파일은 모두 124 KiB 이하임을 확인했다.
- 실행 전 마지막 gate: 모든 의도된 연구 변경을 한 clean commit으로 만들고 annotated freeze tag로 봉인한 뒤 strict preflight를 통과해야 한다. 그 전까지 signals/training/prediction/query metric은 생성하지 않으며, 실행 중 source claim 이후 held 오류는 안전상 terminal이라는 정책을 명시적으로 수용한다.

## RES-20260906-018 — metadata-calibration-efficiency-v1 source no-go와 held 미개봉 종료

- 실행 완결: commit `56bff34dda154e12771b198cc4d316df2b3b33ee`, annotated tag `metadata-calibration-efficiency-v1-freeze-20260905-r1`, plan `04d77058dbec3d82d11ac15c8ad25f32fb3a3c102644e8a92a76cf759e22515a`의 clean snapshot에서 strict preflight를 통과했다. RTX 4090/CUDA, Bubblewrap, asset/plan/power/baseline/input hash와 빈 registry를 확인한 뒤 새 run root `/home/whwovy/eeg-results/metadata-calibration-efficiency-v1-20260905-run1`에서 lifecycle을 한 번 실행했다.
- 수량: source candidate 9+baseline 15=24/24 jobs, mandatory private artifact 75/75, 39 participants×wet/dry, 고정 query 4,680 rows를 완료했다. 모든 job return code는 0이고 독립 reset·job capability·producer receipt와 private grid 검증을 통과했다. Query probability는 세 seed를 먼저 평균한 뒤 participant BA로 줄였으며 seed/interface를 독립 N으로 세지 않았다.
- 사전 판정: source gate는 `development_no_go`다. Hard gate 관측값은 A_Q k5 viability `0.476282<0.50`, A_QM−A_Q early eAUC mean `−0.004843<+0.010`, outer-fold means `−0.004274/−0.004808/−0.005449`, correct−block-shuffle eAUC `0<+0.010`이다. Shuffle fold nonnegative는 모두 0으로 통과했고 all-missing=A_Q probability exact fallback과 metadata-only structural BA `1/12=0.083333`도 통과했다. Participant primary contrast는 6 improved/7 tied/26 worsened, median `−0.004167`, SD `0.006251`이다.
- Calibration 병목: 평균 BA curve A_0 `0.485684/0.482051/0.485470/0.486111`, A_M `0.485897/0.474359/0.480342/0.485043`, A_Q `0.484402/0.404915/0.463889/0.476282`, A_QM-correct `0.484615/0.396581/0.461752/0.472863`이다. A_QM−A_Q는 k0/1/3/5에서 `+0.000214/−0.008333/−0.002137/−0.003419`; Q-only도 A_0보다 early eAUC `−0.045976` 낮았다. 따라서 M 증분뿐 아니라 현재 Q/support one-block posterior update가 큰 공통 병목이다. `A_QM(k0)>A_Q(k1)`은 metadata 절약이 아니라 k1 악화로 생긴 역전이므로 savings로 주장하지 않는다.
- Mechanism: correct와 block-shuffle participant BA/eAUC는 전원·전 budget에서 같고 correct−stale은 `−0.004380`이었다. Wrong-context bounded-harm 조건은 평균 `−0.004950`, 최악 `−0.022222`, severe harm 0명으로 통과했지만 efficacy나 배포 적합성 증거가 아니다. M reliance가 거의 없어 harm도 작았을 수 있다.
- 사후 label-free implementation 진단: signed result 공개 뒤 source private probability만 비교했다. Correct 대 block-shuffle의 mean absolute probability difference는 k1 `1.06e-5`이고 argmax 변경은 9 seed-fold의 14,040 row 중 0이었다. Correct 대 all-missing은 k1 `0.01422`, argmax 431/14,040이어서 M branch 자체는 죽지 않았다. Interface-only, 즉 impedance 제거는 k1 `0.01114`였지만 impedance-only, 즉 interface 제거는 `0.000190`이었다. Shuffle donor와 8채널 impedance가 완전히 같은 source block은 780개 중 6개뿐이므로 intervention no-op 설명은 맞지 않는다. 거의 항상 observed인 availability/presence의 비특이적 precision offset을 배웠다는 해석이 가장 잘 맞는다. 이 진단은 사전 endpoint가 아닌 exploratory evidence다.
- 학습 선택 단서: 아홉 stage-2 job이 모두 최소 허용 epoch 5를 선택했고 그 시점 inner-validation M increment는 모두 음수, correct−shuffle은 사실상 0이었다. Zero-M epoch 0을 선택 후보로 허용하지 않아 `M을 쓰지 않기`로 abstain할 수 없었다. 결과를 본 뒤 이를 고쳐 같은 39명으로 v1을 재실행하지 않는다.
- Baseline: strict FBCCA k0 `0.677564`, SAME3 FB-eTRCA component k1 `0.560043`, target FB-eTRCA k3/5 `0.316026/0.387179`, template k1/3/5 `0.141239/0.166453/0.189316`, Chiang-LST+FB-eTRCA k1/3/5 `0.090812/0.095513/0.098291`이다. Strong classical baseline이 높으므로 dataset decoding 불가능성보다는 candidate backbone/Q updater와 pairing-insensitive M operator의 실패가 더 타당하다. Cross-method 순위는 정보권한·source training이 달라 descriptive다.
- Held 보존: source FAIL 직후 lifecycle은 `held_not_opened`로 정상 종료했다. Run root에 `held/`가 없고 registry에는 source claim 하나만 있으며 held claim은 0이다. Held completion/result hash는 null, generated private keys는 canonical path에 0이고 encrypted source sidecar만 남는다. 같은 candidate/cohort 자동 재시도는 금지한다.
- 무결성: 동결 snapshot validator로 lifecycle, source completion, private seal, 24-job grid, permanent claim과 공개 7 files를 독립 재검증했다. Result bundle `63bf33ae213bb5ae5ec37fcdb235e0769b23e3924e54d6d3c75e31be66203b77`, source completion signed record `c89d00ed01dff0a6c574fe2378fc830cb238d8a09e962e8a16c10eb9a545145e`, lifecycle signed record `86c4ba772539b6419053e8938c5454be2f7f93ffc47da42b59d680e8a7bfdd58`이며 registry integrity failure는 없다. Snapshot의 executable mode-only 100755→100644 표시는 read-only hardening 결과이고 content/tag/tree hash drift는 없다. Canonical path key absence는 확인했지만 Python memory나 외부 복사본의 forensic zeroization은 보장하지 않는다.
- Reporting gap: primary/gate는 완결됐지만 public result bundle은 wet/dry별 participant summary와 seed 안정성을 내보내지 않았다. Private query label key가 삭제됐으므로 현 one-shot result에서 이를 사후 복구하지 않는다. 다음 finalizer에는 condition·seed summary와 probability-level reliance를 opaque aggregate로 포함한다.
- 추가 데이터 감사: local BETA/Wang은 interface `unknown`, Dong2023은 단일 `pregelled_semidry`이고 모두 impedance가 없다. Ke2025 24명은 두 session·32-channel AR SSVEP지만 electrode impedance가 전부 `n/a`이고 interface contrast가 없어 primary M 검정에 합치지 않는다. 이 자료들은 signal/backbone/OOD 진단에만 쓸 수 있다.
- 후속 결정: 연구목표는 유지하되 v1 후보는 종료한다. V2는 (1) k1 no-harm/gated A_Q common path와 stronger harmonic/spectral anchor, (2) epoch-0 exact-M-off abstention, (3) constant availability shortcut 제거, (4) query-support context similarity와 correct-vs-shuffle mechanism-aware source loss, (5) 완전한 condition/seed reporting을 새 candidate로 사전등록한다. 기존 39명은 exploratory engineering에만 쓰고 unbiased promotion에는 새 독립 wearable-like cohort가 필요하다. 기존 held 60명은 미개봉 보존하며 H2 약 0.80 민감도를 원하면 동일 protocol held 약 23명을 추가해 N=83을 새로 power/freeze한다.
- 해석 문서: `docs/metadata_calibration_efficiency_results.md`. 이 결과는 static impedance/interface metadata 일반의 무효가 아니라 현재 bounded precision-residual operator에 대한 강한 negative development evidence다.

## RES-20260906-019 — 최종 보고·학술 근거 지도·사후 재검증

- 결과 보고: README, 현재 protocol, 설계·적합성·데이터 문서에 source no-go와 held 미개봉 상태를 반영했고, 수치·메커니즘·baseline·주장 경계·V2 데이터 요청을 `docs/metadata_calibration_efficiency_results.md`에 통합했다.
- 학술 workspace: `califree-eeg-experiment-design`에 gate JSON SHA-256 `ad5bab8a55dc66e42c03fcaee9ce18e9857555b1ede200bec591cd52ab9288fb`를 근거로 candidate-specific source no-go measurement claim을 `verified`로 등록했다. 독립 acquisition-context cohort와 pairing-sensitive operator를 priority-1 gap으로 추가했고 workspace를 다시 render했다. 문헌 cutoff는 2026-09-04이며 이 사후 measurement는 새 문헌 주장이 아니다.
- 재검증: canonical lifecycle validator가 signed record `86c4ba772539b6419053e8938c5454be2f7f93ffc47da42b59d680e8a7bfdd58`를 다시 인증했다. Permanent registry는 current run source claim만 보유하고 held claim은 없으며 integrity failure는 false다. Metadata suite `109 passed`, 전체 suite `536 passed`(68개 기존 Transformer warning)를 실험 종료 후 다시 통과했다.
- 종료 경계: V1 실험은 완결됐다. 새 피험자 수집·IRB·장비 운영과 V2 실행은 이 결과를 덮어쓰는 후속 재시도가 아니라 별도 candidate/data contract를 필요로 하는 새 연구 루프다.

## DEV-20260906-020 — legacy spectral backbone 진단의 범위 정리

- 목적·범위: V1/V2 confirmatory outcome과 무관한 기존 Wang development 진단을 재감사했다. 이 항목은 새 방법의 efficacy 증거가 아니라 source-learning 구현 병목을 찾기 위한 legacy engineering evidence다.
- time-domain tiny 진단: augmentation을 모두 끈 Wang 3-epoch run도 loss `3.7059→3.6950`, validation BA `0.025`에 머물렀다. one-batch 진단은 50 step 뒤 loss가 `3.728→1.560`, accuracy가 `0.515625`였으나 사전 성공 기준 `0.95`에는 못 미쳤다. 따라서 optimizer/gradient 경로가 완전히 끊긴 것은 아니지만 Wang의 40개 근접 주파수에 대한 현재 tiny backbone의 유도편향이 부족하다는 해석이 가장 일관된다.
- 최소 spectral 진단: label별 주파수나 정답 codebook을 입력하지 않는 generic complex-RFFT spectral backbone은 Wang 28명 train/7명 validation, augmentation 없음, 10 epoch에서 validation BA `0.025→0.4554→0.7226→0.8220`을 보였다. 3,980,968 parameters, RTX 4090 wall time `30.73 s`, peak allocated 약 `0.94 GB`다.
- 비교 경계: 위 spectral run은 `legacy`, ungoverned, dirty-source development evidence다. Wang CCA `0.854`는 validation 7명이 아니라 `--max-subjects 1`로 택한 sub006 한 명·240 trial 결과이므로 직접 순위 비교하지 않는다. V2는 이 학습 backbone을 confirmatory 경로로 가져오지 않고 frozen strict-FBCCA anchor를 사용한다.
- 정보권한: generic spectral 입력은 현재 query EEG뿐이며 정답 class frequency/phase를 전달하지 않는다. 향후 all-candidate sinusoid bank를 쓰려면 모든 query에 동일한 frozen full codebook만 허용하고 correct-class-only 전달을 금지하는 별도 protocol amendment가 필요하다.

## DES-20260906-021 — V2 방법·외부 할당·합성 lockbox와 구현 경계 동결

- 연구질문: 상위 목표는 그대로다. 처음 보는 사용자의 labeled target calibration을 줄이되, `A_Q`가 strict FBCCA보다 실제로 유용하고 안전한지 먼저 검정하고, 그 위에서만 pre-query acquisition context를 쓰는 `A_QM−A_Q` 증분과 `k1 대 k3` 절약을 검정한다. Few/one-shot 최초를 주장하지 않는다.
- 후보·게이트: 두 support operator(P1 score-prototype shrinkage, P2 filterbank target-template residual), `lambda_max={.10,.20,.30}`, P1 pseudocount `{1,4,16}`의 12개 후보를 고정했다. k3/k5 local gate는 미래 query를 쓰지 않고 chronological prequential prefix만 사용하며 실패 시 strict anchor로 정확히 복귀한다. 최종 endpoint 예산은 `{0,1,3,5}`다.
- 데이터 분리: V1 source outcome에 노출된 participant를 제외하고 BETA 23명+Dong 19명을 development, BETA 45명+Dong 39명을 independent A_Q gate로 사전 난수 할당했다. allocation canonical SHA-256은 `3722449446183a7d5a4b7006b6a28c38cbcf536bbeac7bc74a07317143077cf4`, receipt file SHA-256은 `af6cfe5200f1239b1893b61d0ab169e5523155e63b5e5f219c330772269f039c`다. Independent scorer는 development selection completion receipt에 든 단 하나의 후보와 A_Q만 허용한다.
- 합성 분리: 7개 beneficial/null/adversarial family, keyed PCG64DXSM, development seed `20260906`, one-time lockbox seed `20260907`을 결과 전에 동결했다. r2 plan SHA-256은 `5b7bb7d091e5ce937d4818659931cf70f83e198836370ee748556933698d11bd`, freeze receipt SHA-256은 `1b6242a3254cd5fd2f9d9204539aa1b39dce875d8b0797735151f7f0b1aaac73`다. 이 항목 작성 시 두 reserved seed 모두 미실행이다.
- 구현 상태: safe operator는 exact fallback, probability bound, context pairing capability, P2 provenance와 intermediate depth 2/4 전용 prequential API를 구현했다. 외부 runner는 participant-level BA/eAUC, development selection, independent one-sided gate, label-join receipt와 request/receipt binding을 구현했다. 현재 방법 기준 commit은 `c003ada85d39fdb48d47edc964bfa643ff6f3f4f`; 전체 suite는 `.venv`에서 `588 passed`, 68개 기존 Transformer warning으로 통과했고 PyTorch `2.2.2+cu121`, RTX 4090 CUDA 사용 가능을 확인했다.
- 1-ULP 감사와 r6 수정: repo `.venv`의 NumPy 1.26은 다섯 번째 weight를 `0.38374806099528436`(`0x1.88f5406f833c4p-2`)으로, system Python의 NumPy 2.2는 같은 vector power 식을 `0.3837480609952844`(`0x1.88f5406f833c5p-2`)으로 계산한다. V2 outcome 전 이를 발견했으므로 기존 frozen operator 값 7개를 새 V2-only filter-bank config에 명시해 환경 의존 계산을 제거한다. 수치 후보 자체는 `.venv`에서의 기존 r5와 동일하며 method revision만 `r6_pre_outcome_explicit_filterbank_weights`로 올린다. 최종 master plan SHA-256은 `81e8dcdb58f1950f9c57677a2bedff1ffccf7f45989a3cfa481692cab3fd4975`, explicit filter-bank SHA-256은 `b8c1ce4477d980b40f359bc3dc97f3125313191d24401f29a7e2dc473d0d1380`이다.
- 결과·봉인 경계: 이 시점까지 V2 human EEG outcome, BETA/Dong query label join, synthetic reserved-seed outcome, Choi classification outcome은 생성하거나 열지 않았다. wearable held 60은 모든 synthetic·external·replication gate와 별도의 새 서명 권한 전까지 금지한다. Choi2019은 다운로드·schema/hash/partition 감사를 마친 뒤 결과 전에 별도 cross-day replication contract로 동결한다.
