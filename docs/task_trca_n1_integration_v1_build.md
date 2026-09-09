# N1 인공 학습기 통합 v1 — 사전 설계와 작업 계약

2026-09-09. 사용자 “응 계속.”을 직전 제안인 **인공 학습기 통합 검증** 승인으로 적용한다.
원 저보정 SSVEP 목표는 유지하며, 기존 종료 C1/C2와 별개의 `task-trca-n1-integration-v1`이다.
[기계 계약](../configs/analysis/task_trca_n1_integration_v1.json)이 수식·격자·판정·예산 기준이다.

## 문제와 범위

표현은 Q15·숫자M2→채널규제R→일반화고유필터→시간중심 SSVEP 점수다. 병목은 지난 N1의
단일 행렬/미분 검증이 실제 optimizer·nested 선택·입력역할·저장복원 경로까지 검증하지 않았다는 점이다.
허용 변경은 새 이름의 학습/평가/reader/runner/auditor에 N1을 연결하는 것이다. C2의 pair-S
가설, 특징이나 scaler로 바꾸지 않는다. 실패 신호는 수치/미분/역할/파일/완료 경로 불일치이며
인공 QM 정확도 우위를 통과 조건으로 삼지 않는다. 사람 EEG/M/query·held60·외부요청·유료·GPU0이다.

N1의 원 raw-input 검사를 평균대칭화 전에 수행한다. 정확한 tau는
`.1*lambda_min(C)/(16/9)`이지 `trace(C)/8`이 아니다. N1이 직접 노출하지 않는 옛
`tr(CK)` 검사는 `tr(CF)/tr(BF)`라는 사전 명시한 동등식 평가로 대체한다. 이를 옛 수치
정책과 byte-identical이라고 하지 않는다. N1·기존operator·기존 모든 runtime/config는 수정하지 않는다.

## 실행과 중단

새 seed20260913, 생성9명×2interface×N17×k3/5를 한 번 만든다. 이는 해당 ID 사람의 관측이
아니며,780 routing packet도 전부 생성값이다. 3outer×(3inner×3lambda+1final)×4heads×200steps,
정확히24,000updates다. Q 동결 후 Q2/QM/SHAM 잔차를 학습하고, Q-only 검증손실로만 lambda를
선택한다. 마지막 query는 세 모델 모두 저장·동결한 뒤 연다. 평가10arms·36cells·72query-bearing
calls·325전체access rows를 독립 감사한다. A0는 인공 correlation 자리표시자이므로 보정0개 성능 근거가 아니다.

모든 head에서 실제 gradient와 계수 변화가 관측되는지 사전 수치 하한으로 검사한다.
그 검사는 입력 경로 작동 여부이지 metadata 효용 검정이 아니다. 독립 감사는 저장 통계 이후
NumPy/SciPy 재계산이며 Adam/Q15/raw 전처리의 별도 전체 구현은 아니다.

CPU1·GPU0, fixture/primary/cold 각1회, primary7200초/cold1800초/tests총1800초,
신규출력총3GiB 중 입력+실험2GiB를 예약한다. 등록 실행 후 실패하면 보존·종료하고 science
수리나 재실행을 하지 않는다. 등록 전 toy 수정/검증은 예산 내 허용한다. 성공하면 통합 단계만
닫고 별도 사람효능 프로그램을 제안한다. 이번 승인으로 사람 실험을 자동 시작하지 않는다.

## 소유권·API·통합

Base42333af/main. 가용218,370,292KiB(약208GiB), 기존40trees를 보존한다.
작은 checkout2개는 기존 temporal-runtime/cold tree를 새 branch로 재사용한다. dependency는
root `.venv` read-only, PYTHONPATH는 각 tree의 src, 별도 toy output/cache를 쓴다.
각agent toy≤300초/128MiB, root toy≤300초/128MiB, fulltests≤900초/640MiB다.
실제 등록 실행·독립 cold·최종 통합·공통 계약·SQLite는 root 단독 소유다.

| Lane | 소유 신규 파일 | 종속성 | 통합 |
|---|---|---|---|
| learner / goal_scope_review | src/cfeg/analysis/task_trca_n1_{signfree,learning,batch,evaluation}.py; tests/test_task_trca_n1_{learning,evaluation}.py | 고정 계약/N1 | 1 |
| independent / metadata_loop_history | src/cfeg/analysis/task_trca_n1_{audit,artifact_audit}.py; scripts/audit_task_trca_n1_integration.py; tests/test_task_trca_n1_{audit,cold}.py | 고정 계약/아래 envelope | 2 |
| root | src/cfeg/analysis/task_trca_n1_archive.py; scripts/{prepare,run}_task_trca_n1_integration.py; tests/test_task_trca_n1_runtime.py; 계약/보고/일지 | learner와 auditor API | 최종 |

기존 파일을 새 버전으로 기계 복제한 뒤 import/schema만 명시적으로 전환하는 것은 허용한다.
기존 module globals monkeypatch, 기존 Pipeline alias를 새 학습기로 위장, 기존 과학 계약/실패 덮어쓰기는 금지한다.
그대로 쓰는 pure feature/native/temporal-statistics/byte helpers는 명시적 import로 재사용한다.

Learner API는 기존 temporal의 make_task_case/fit_pipeline/predict/validation_ce/nested_fit 및
evaluation support_state/evaluate/pipeline_from_record/pack_cases와 동일하다. SCHEMA는
`task-trca-n1-integration-v1`, SCORE_SCHEMA는 기존 temporal 점수 문자열 그대로다.
Case/Pipeline은 새 type이며 old record/token을 거부한다. device는 CPU만, batch import도 새 모듈이다.
signfree 어댑터는 bounded_projectors만 새로 구현하고 TemporalGramStatistics/statistics/scorer는
기존 순수 함수를 명시적으로 재사용할 수 있다. N1 진단은 detached 값만 반환하며 F를 detach하지 않는다.

Archive는 GENERATED_PROFILE(seed20260913)만 허용한다. SOURCE_IDS39는 lexical routing용
상수일 뿐 수치 reader의 허용집합은9개다. default도 generated이며 HUMAN_PROFILE을 내보내지 않는다.
Freeze schema=`cfeg.task_trca_n1.all_models_frozen.v1`. Model/source/evaluation NPZ layout은
옛 temporal generated contract 그대로이며 study schema만 새 값이다. status 문자열은
기존 `TEMPORAL_SOURCE_SELECTION_AUDIT_PASS` 등 수학/검사 종류에 한해 재사용한다.

Manifest schema=`cfeg.task_trca_n1.generated_execution.v1`, status=`GENERATED_FROZEN`,
study_id/new profile/seed/generated/score_schema/attempt_id/권한false flags/native_plan/native_root/
native_manifest_sha256/native_weights/code_pins/device=cpu/backend=batch/precision=float64/
cpu_threads=1/output_root/output_budget_bytes/max_seconds/optimizer_updates_max/preflight=null.
`design`은 새 JSON의 절대path/SHA. **program_path/program_sha256/gpu_memory_fraction은 없다.**
옛 program/C1과 human preflight는 새 authority가 아니다. Output basename prefix는
`task-trca-n1-integration-`; native plan은 오직 native_root/native_plan.json. Root의9code
수치감사 pins와는 별개로 아래 실행 dependency 전체를 새 code_pins에 정확히 고정한다.

Cold CLI `--output --manifest --manifest-sha256`는 새 generated-only manifest만 받는다.
Reader/learner/producer/N1 import 없이 새 독립 audit 모듈+옛 독립 pure helpers만 사용한다.
원 temporal cold의 exact inventory/model-source/freezepin/access/event/aggregate검사를 유지한다.
새 cold 결과에 all30pipeline의 네 head gradient>1e−12/coef>1e−12 coverage와
METADATA_NOT_EVALUATED를 추가한다. 기록누락·reference부재는 통과가 아니다.
Root runner는 clean tracked diff를 요구하지만 기존4개 untracked test dirs는 허용·보존한다.
미계측 warmup optimizer.step은 제거한다. 실제primary 전 scientific/runtime pins를 고정한다.

### 실행 pins와 envelope 보충 — 등록 입력 전 고정

CODE_PATHS는 아래 정확한 집합이다. Runner/cold는 각자 독립 상수로 선언하며 generator는
cold의 상수를 읽어 hash만 만들 수 있다. 이 문서 자체와 JSON의 해시도 포함한다.

```text
configs/analysis/task_trca_n1_integration_v1.json
configs/analysis/metadata_prior_source39_v1.json
docs/task_trca_n1_integration_v1_build.md
scripts/prepare_task_trca_n1_integration.py
scripts/run_task_trca_n1_integration.py
scripts/audit_task_trca_n1_integration.py
src/cfeg/__init__.py
src/cfeg/metrics.py
src/cfeg/analysis/__init__.py
src/cfeg/analysis/ood_coverage.py
src/cfeg/analysis/primary_aggregate.py
src/cfeg/analysis/primary_inference.py
src/cfeg/analysis/provenance.py
src/cfeg/analysis/metadata_prior_validation.py
src/cfeg/analysis/metadata_prior_source.py
src/cfeg/analysis/metadata_trca_prior.py
src/cfeg/analysis/native_support_prefix.py
src/cfeg/analysis/task_trca_shape_inputs.py
src/cfeg/analysis/task_trca_shape_archive.py
src/cfeg/analysis/task_trca_shape_features.py
src/cfeg/analysis/task_trca_shape_operator.py
src/cfeg/analysis/task_trca_shape_signfree.py
src/cfeg/analysis/task_trca_shape_audit.py
src/cfeg/analysis/numerical_stability_operator.py
src/cfeg/analysis/task_trca_n1_signfree.py
src/cfeg/analysis/task_trca_n1_learning.py
src/cfeg/analysis/task_trca_n1_batch.py
src/cfeg/analysis/task_trca_n1_evaluation.py
src/cfeg/analysis/task_trca_n1_audit.py
src/cfeg/analysis/task_trca_n1_archive.py
src/cfeg/analysis/task_trca_n1_artifact_audit.py
```

Native fixture root는 execution output의 sibling `inputs`, manifest basename은 `manifest.json`;
둘의 parent는 root가 mktemp로 만든 fresh `task-trca-n1-integration-*` directory다. Output은
같은 parent의 `task-trca-n1-integration-primary1`이다. 기존 generated-builder의 projection/plan/
start/result/fixture와 S{id}.npz layout을 유지한다. Fixture origin schema/study는 반드시
`GENERATED`와 새 study 문자열을 포함한다. Manifest는 `fixture` descriptor(path/SHA/bytes)도
필수로 포함하고 generator/code/nativeconfig/seed/profile과 일치시킨다. Producer/cold의 숫자 접근은
오직 이 bound fixture/output 자식만 허용한다. Code pin 확인 뒤 fixture에 없는 추가 입력·oldhuman
path·상위root alias/symlink/hardlink는 거부한다.

Archive output의 기존 start/events/access/source0..2/model0..2/globalfreeze/evaluation9개/
scores/result inventory를 유지한다. `fixture`는 입력 provenance이므로 output artifact dictionary에
섞지 않는다. Independent cold의 최종 receipt에 `learning_path_coverage`를 기록한다. 네 head 각각
전체30pipeline 중 최대gradient_norm과 최대abscoef를 집계하며 둘 다1e−12 초과일 때만
`exercised:true`. 이 항목이 false면 성공 문자열을 발행하지 않는다. 새 summary의
`metadata_effect`는 항상 `NOT_EVALUATED`다.

## 연구 근거 적용

`academic-research`의 기존 frontier와 직전 N1의80/120자리 검증을 재사용한다. 이번 불확실성은
새 문헌 수보다 실제 통합 실행으로 더 직접 확인할 수 있어 broadsearch/PDF추가0으로 시작한다.
근거는 numerical→learner integration→human efficacy로 구분해 새 claim으로 남긴다.
`coordinate-worktree-changes`에 따라 계약 전에는 두agent 모두 읽기전용, 계약 고정 후에만
격리 writing lane으로 전환한다. Root reader/runtime와 학습/audit를 같은 tree에서 동시에 쓰지 않는다.
