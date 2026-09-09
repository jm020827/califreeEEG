# N1 source39 v1 — 승인·과학·구현 계약

2026-09-09, 사용자 “응 승인. 계속.”은 직전 제안의 새 source39 개발 후보1회 승인을 뜻한다.
원 저보정 SSVEP metadata 목표와 이전 부정 결과/C1·C2 종료는 유지한다.
[기계 계약](../configs/analysis/task_trca_n1_source39_v1.json)을 사람 자료 접근 및 등록 생성 전에 고정한다.

## 문제 signature와 과학 범위

표현: support EEG→Q15, support acquisition impedance→M2, bounded R→N1 필터→시간중심 점수.
병목: 생성자료에서 작동한 학습이 실제 M에서 Q/공통정보 이상의 성능과 관측 label saving을 내는가.
허용연산: 검증된 순수 N1 학습기 그대로, 새 실험 identity/입력권한/reader/runtime/auditor만 구현.
목적: 원 metadata 증분 및 실제 관측 보정 격자에서의 부담 감소. 자원: CPU1/6시간/출력12GiB의
사람 primary1회, 코드·생성 관문 별도 유한 예산. Feedback: 모든 사전 대조군·624 budget cells,
participant 통계, cost/harm, 수치/접근/완료 감사. 실패: 관문실패 또는 유효 음성 모두 보존·종료한다.

수학 구현을 복제하지 않는다. `task_trca_n1_learning/batch/evaluation/signfree`는 원본 그대로
명시적으로 import해 재사용한다. 그 **algorithm schema**는 `task-trca-n1-integration-v1`을 유지한다.
새 **experiment identity**는 `task-trca-n1-source39-v1`, 새 manifest/freeze/reader token과
source/model byte binding으로 구분한다. 예전 모델을 새 학습기라고 relabel하거나 oldglobals를
monkeypatch하지 않는다. Pipeline class/SCHEMA 재사용은 공개된 수학 포맷 재사용이며 파일 접근 권한이 아니다.
Model26fit/13eval 및 모든window/k 격자, 새manifest/source hash/freeze가 맞아야 사람 query를 허용한다.

Q-only lambda·200step·Q동결·Q2/QM/SHAM 용량·R/tau/N1·score·원 native A0/FULL을 바꾸지 않는다.
새 독립 endpoint는 옛 temporal `summarize`의 순수 통계 계산을 명시적으로 재사용할 수 있으며,
새 summary의 experiment schema와 알고리즘 포맷을 별도 필드로 설명해야 한다.
Q2는 전체 EEG 정보 충분성을 보장하는 대조군이 아니고, CI는 노출39명 개발자료의 기술통계다.
39×2×4=312기본조건, k3/5 포함624평가행이다. access4213/query-bearing1248/events182다.

## 예산·순서·중단

과학 동결→격리 구현 및 toy→전체 회귀→코드pin 동결→fullshape resource1→새 작은 생성 완료경로1
및 독립cold1→새 사람 manifest 동결→human primary1→독립 terminal audit1→보고 순서다.
Resource는 생성26ID×2interface×4window×k3/5=416case의 실제800update 한 번이다.
N500의 독립 잡음6block을 만들고 실제 nested window/support prefix를 취한다. RSS≤4GiB,
`36*fit_seconds+3*preparation_seconds+600≤21600`, `4*RSS≤16GiB`를 사전 관문으로 둔다.
이는 screening이지 실제 실행 시간 보증이 아니므로 human 자체6시간/RSS16GiB/VA32GiB 한도를 적용한다.
앞선 작은593.87초 실행의 약17배 case수 때문에 기존 GPU7200초 한도를 CPU에 그대로 전용하지 않는다.
GPU를 새로 쓰거나 수치 정책을 바꾸는 대신 사람 입력 전 충분한 한도를 명시한다.

새 작은 생성 fixture는 이전 N1 prepare recipe의 seed만20260914 및 새 identity로 바꾼다.
생성9ID/N17/24,000updates, 생성A0자리표시자, 전부M-효능 미평가이며 이전generation을 다시 쓰지 않는다.
Resource seed20260915는 toy에서 사용하지 않는다. Toy seed73, 실패·소요시간·출력은 모두 보존한다.
등록 resource/생성완료/human 실패 후 수리·retry·seed교체·N2대체0이다. 등록 전 toy 수리는 예산 안 허용한다.
각 primary24000/자원800updates. Fulltests900초,전체tests1800초,각lane300초/128MiB,
fulltest640MiB,전체신규출력16GiB다. Root만 실제 등록실행/사람자료/SQLite를 소유한다.

## 입력과 API 계약

새 파일 `src/cfeg/analysis/task_trca_n1_source39_archive.py`는 옛 temporal archive의 byte/selective
reader를 명시적 재사용하되 새 token과 HUMAN39/GENERATED9seed14 프로필만 제공한다.
`SCHEMA`는 알고리즘 포맷, `STUDY_ID`는 새 실험ID, `FREEZE_SCHEMA=cfeg.task_trca_n1_source39.all_models_frozen.v1`.
기존 `verify_freeze`, `NativeArchive`, `SupportMetadata`, `RuntimeProfile` API/layout 유지.
Oldtoken/oldmanifest/held/retired/evaluation block5/query-before-freeze/미래M/파일alias를 거부한다.

새 runtime `scripts/run_task_trca_n1_source39.py`는 `--manifest --manifest-sha256` human entry;
generated CLI 선택은 새generated schema에만 가능하다. `_run(path,pin,profile)`/`run_generated`를 제공한다.
Algorithm import는 기존 N1, reader/artifact-audit import는 새source39. 기존 source/model/eval NPZ layout은 유지.
Result는 `study_id`, `design_sha256`, `algorithm_schema`를 명시하고 oldprogram/candidate_slot은 없다.
Start/freeze/result 모두 새manifest에 결합한다. 미계측optimizer warmup step0이다.

새 `scripts/prepare_task_trca_n1_source39.py`는 fresh `/home/whwovy/task-trca-n1-source39-*` parent
안에 생성전 registration과 생성 inputs/manifest를 만든다. 실제primary basename은
`task-trca-n1-source39-generated1` 또는 `task-trca-n1-source39-primary1`이다. Human manifest는
generated 입력을 가리키지 않고 고정 native-cold-r1/source39 projection을 가리킨다.
Manifest schema=`cfeg.task_trca_n1_source39.execution.v1`(human) 또는
`cfeg.task_trca_n1_source39.generated.v1`; status EXECUTION_FROZEN/GENERATED_FROZEN.
공통필드: schema/study_id/algorithm_schema/score_schema/attempt_id/status/generated/seed/profile/source_ids,
held60_authorized=false/retired1_3_authorized=false/old_attempts_reopened=false,
design{path,SHA}/native_plan{path,SHA}/native_root/native_manifest_sha256/native_weights/code_pins,
device=cpu/backend=batch/precision=float64/cpu_threads1/output_root/output_budget_bytes/max_seconds/
optimizer_updates_max24000/preflight/resource_preflight/fixture.
Human fixture=null, generated preflight=null; 둘 다 resource_preflight가 새resource receipt를 결합한다.
Human preflight는 새same-code generated cold receipt descriptor; generated fixture는 새 exact14input descriptor.
Human 권한은 새design에서 나오며 옛program/manifest/preflight는 권한이 아니다.

Human native manifest는 고정resultSHA로 먼저 결합하고 각 files[]의 raw_receipt도 원출처 metadata로
보존/검사한다. raw MAT를 읽는 허가는 아니다. 전체 원 metadata/다른participant data는 금지한다.
Byte provenance hashing은 숫자 decode와 구분하며 raw/fullM의 path를 따라가지 않는다.

새 `task_trca_n1_source39_artifact_audit.py`는 기존 temporal의 두profile 격자검사를 복제하고
독립 수학은 기존 `task_trca_n1_audit`로 연결한다. Generatedseed14/human39만 허용한다.
새 `task_trca_n1_source39_audit.py`는 새endpoint wrapper로 기존temporal 순수summary를 재사용한다.
새 cold `scripts/audit_task_trca_n1_source39.py`는 CPU/새design/새same-code preflight/resource/입력pin,
전체 exact artifact/event/access/source/eval/model/freeze를 독립 검사한다. Producer/N1 learner/operator import 금지.
Generated는 family coverage를 성공관문으로 유지한다. Human coverage는 기술적 보고이며 zero learning이
유효 음성일 수 있으므로 invalid/retry로 바꾸지 않는다. 최종human metadata 판단은 고정endpoint 그대로다.
`--failure-only`는 첫failure의 byte/provenance/counters/partial journals를 검증하고 성공통계를 발행하지 않는다.

Resource CLI `scripts/check_task_trca_n1_source39_resources.py --output NEW_DIR`는
start.json/pipeline.json/resource.json 또는 failure.json을0400으로 남긴다. 새design/전체code_pins/seed/
actual800updates/timing/RSS/projection/arrays_generated=true/human_reads=false/gpu_used=false가 필수다.
정확 CODE_PATHS는 모든newruntime와 실제 import된 구/신 모듈·새설계·이문서·oldscience/nativeconfig를 포함해
최종 등록생성 전에 root가 동결한다. Freeze 뒤 code 변경0이며 모든 단계가 같은핀을 사용한다.

## 작업 소유권

Base9d82e59/main, 가용198,971,316KiB(약190GiB), availableRAM약47GiB다.
40기존tree/실패출력 보존. 기존 temporal-runtime/cold의 두tree를 새branch로 재사용한다.
증분은 checkout각<64MiB,공유rootvenv읽기전용,각toy128MiB,통합640MiB와실행16GiB로 충분하다.

| Lane | 소유 파일 | 의존·통합 |
|---|---|---|
| resource / goal_scope_review | scripts/check_task_trca_n1_source39_resources.py; tests/test_task_trca_n1_source39_resources.py | 기존N1 순수수학/고정계약; 1 |
| independent / metadata_loop_history | src/cfeg/analysis/task_trca_n1_source39_{audit,artifact_audit}.py; scripts/audit_task_trca_n1_source39.py; tests/test_task_trca_n1_source39_{audit,cold}.py | 고정계약/새manifest; 2 |
| root | 새archive/prepare/run/tests 및공통contracts/codepin/보고/실행/SQLite | 최종통합/E2E |
| read-only / task_shape_reader | 쓰기없음; 전체계약·노출/권한·카운트검토 | sharedrepo |

새수학 lane은 만들지 않는다. 기존순수N1 재사용으로 중복코드와 수치의미 변경을 피한다.
계약/통합/최종실행은 의존성이 있어 root순차 처리하고,독립 두writing lane만 격리한다.
`academic-research`의 기존근거/열린gap을 읽었고, 새문헌검색보다승인된개발실험이현재불확실성에직접답한다.
별도외부연구주장을추가하지않으며 결과후새claim/evidence로남긴다.

## 등록 생성 전 실행 envelope 보충

Resource packet5 pattern: [0,1]=0, [0,3]=NaN, [1,0]=NaN, [3,2]=NaN; order=full39 rank%2.
실제 evaluation.pack_cases(cases) 한 벌을 fit 동안 메모리에 유지한다.
Resource path=fresh /home/whwovy/task-trca-n1-source39-*/resource1.
Generated는 별도 fresh parent의 inputs/generated_manifest.json(파일은 inputs의 sibling), output=task-trca-n1-source39-generated1.
Human은 새 fresh parent의 human_manifest.json/output=task-trca-n1-source39-primary1이다.
Generated inputs14개 구조는 옛 N1 그대로; resource/prior proof는 inputs 밖이다.
Resource receipt 필수: status=GENERATED_RESOURCE_PASS,study_id,design_sha256,code_pins,seed,source_revision,fit_ids,
cases=416,updates_completed=800,preparation_seconds,fit_seconds,elapsed_seconds,peak_rss_bytes,
projected_seconds=36*fit_seconds+3*preparation_seconds+600,arrays_generated=true,human_reads=false,gpu_used=false,
pipeline{path,sha256,bytes},start{path,sha256,bytes}. 추가diagnostics는 허용하며 관문 재계산으로 검사한다.

--failure-only는 성공경로와 분리한다. manifest/start/failure bytebinding, 실제partial파일목록·bytes·SHA,
연속 access seq/허용multiset부분집합/역할·freeze전후권한,완성모델·freeze만결합,
completed≤charged=budgeted≤24000/querycount를검사한다. 미완성NPZ numericdecode/실패수치재실행0.
terminal_audit.json status=FIRST_FAILURE_PROVENANCE_AUDIT_PASS,terminal=VALIDITY_FAILURE,
efficacy=NOT_EVALUATED,calibration=NOT_EVALUATED,numeric_failure_replayed=false와partialinventory/한계.
감사실패는terminal_audit_failure.json,선택한cold mode1회이며retry0이다.

Human isolated zero/nearzero를구조적무작동이라고하지않지만,옛endpoint의 세outer QM잔차전체정확0
인증은 유지한다. 새로운 same-code완료경로는 작은profile, 별도416case검사는fullwindow부하검사다.
이둘을39명fullnested전체생성재현이라고하지않는다. 생성A0와human A0 차이를유지한다.

정확 CODE_PATHS36개는 다음과 같고 runner/resource/cold 각각 literal집합이 일치해야 한다.
Generator는cold상수를 읽어hash만만들수있다. 모든실제import cfeg모듈포함을root가등록전에확인한다.

```text
configs/analysis/task_trca_n1_integration_v1.json
configs/analysis/metadata_prior_source39_v1.json
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
configs/analysis/task_trca_n1_source39_v1.json
configs/analysis/task_trca_temporal_v1_design.json
docs/task_trca_n1_source39_v1_build.md
scripts/prepare_task_trca_n1_source39.py
scripts/run_task_trca_n1_source39.py
scripts/audit_task_trca_n1_source39.py
scripts/check_task_trca_n1_source39_resources.py
src/cfeg/analysis/task_trca_n1_source39_archive.py
src/cfeg/analysis/task_trca_n1_source39_audit.py
src/cfeg/analysis/task_trca_n1_source39_artifact_audit.py
src/cfeg/analysis/task_trca_temporal_audit.py
```
