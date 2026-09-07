# Direct acquisition-M v1 — 입력 연결 오류로 중단

2026-09-07. **상태: infrastructure/data-validation inconclusive. 성능 결과 없음.**
연구목표나 M 가설의 실패가 아니라, 올바른 기존 metadata 파일을 새 코드가 거부한 구현 오류다.
[고정 설계](native_subset_m_source39_v1_design.md)는 보존하며 같은 attempt를 재실행하지 않았다.

## 무엇이 실행됐나

전체 회귀시험1294개(131.33초, 기존Torch warning68개)와 새 집중시험98개(12.43초),
Toolbox 저자 환경의 인공 참가자80fits, 독립 ridge/M/Q 수학 검산 후 실행했다.
그러나 이 시험들은 실제 저장 파일과 소비 코드 사이의 **11개 필드 envelope 연결**을 놓쳤다.
검증 개수가 실제 인터페이스의 정확성을 대신하지 못했다. 구현과 통합 검토의 책임은 이번 작업에 있다.

1. Clean 실행 코드와 계획을 확인하고 `start.json`을 저장했다.
2. 허용된 기존 source39-only metadata 파일을 읽고 전체 byte SHA를 확인했다.
3. 내용 검사 일부는 통과했으나 core의 root-key 검사에서 예외가 발생했다.
4. EEG 읽기, native fit/예측, Q/QM 학습, outer 성능 평가에는 도달하지 않았다.

따라서 **이번 새 시도에 대한 BA·QM−Q·비용 절감·harm·유효성 판단은 모두 미측정**이다.
Metadata bytes는 읽었으므로 'human data 접근0'이라고 부르지 않는다. EEG/held60/retired 접근은 없었다.
실행 시도1회, 완료된 효능 평가0회, 과학적 M 방식 수정0회로 구분한다.

## 정확한 원인

[기존 작성기](../src/cfeg/analysis/context_template_source.py)는 내용6개에 출처5개를 추가한다.
[새 core](../scripts/native_subset_m_core.py)는 정확히 내용6개만 허용했다.
독립 auditor에도 같은 잘못된 구조 가정이 있었다.

| 실제 파일 항목 | 개수 | 확인 |
| --- | ---: | --- |
| manifest_sha256, packets, returned_rows, returned_packets, returned_subject_ids, columns | 6 | 기존 manifest hash,39명 ID,780packets,9360rows,columns 모두 계획과 일치 |
| schema, study_id, plan_sha256, source_commit, start_sha256 | 5 | 원본 작성기가 의도적으로 추가한 출처 기록; 예상 못한 데이터가 아님 |

원본 SHA는 `1082a81d23ebffe26e8451a9e99f1a7d4c84d0ba2d9d00c6695b3a5c6a5114d4`로 계획과 정확히 같다.
첫 packet의6개 key도 예상 구조와 같다. 이 진단은 임피던스 값의 효능 분석이 아니다.
원본 작성기와 실패 호출 순서를 별도 읽기전용 검토자가 확인했다.

기존 fixture는 내용6개만 만든 뒤 lifecycle에서 core를 대체했다. 추가한
[historical failure regression](../tests/test_native_subset_m_projection_failure.py)은 원본 작성기 모양의
**인공11-key 파일 → 실제 loader → 실제 core**를 연결해 똑같은 예외와 start-only 종료를 재현한다.
이는 원인 재현이지 성공하는 수정본이나 재실행 권한이 아니다. 실패한 producer/core/auditor/계획은 바꾸지 않았다.
향후 성공 경로 테스트는 별도로 추가하고, 이 historical test의 종료된 대상 코드는 보존한다.

## 보존한 실행 기록

| 항목 | 값 |
| --- | --- |
| 시작 UTC | 2026-09-07T13:06:59.242843+00:00 |
| 실행 commit | e470d7b4a0d0ce7fb28c009f1df58cd930b171d2 |
| 실행 tree | 473409aca6d83f8e24da11bd42d7f2018d3a2ae9 |
| 과학 계획 SHA | ee9b758f755a18f7540c978e9faf18a5711e652948b94df595377bbfb83cb80a |
| start SHA | 4e4a2d736a2165f3552297931d4cbe0b02ed0b89bd75d367c56148e6cf769f68 |
| 프로세스 | exit1, wall1.04초, user0.95초, system0.09초, maxRSS124048KiB |
| 남은 산출물 | /home/whwovy/native-subset-m-artifacts/source39-v1/start.json,3540bytes,0400 |
| 없는 산출물 | source-projection.json, features.npz, fold-freezes.json, result.json |

위 directory에 실패 보고서를 덧붙이거나 파일을 덮어쓰지 않았다. 실패 설명은 이 저장소 문서와
append-only 연구일지에 남긴다. 정상 완료를 요구하는 전체 artifact auditor의 PASS를 주장하지 않는다.
기존 eTRCA 호환성·context·spatial·reference 실험 결과도 변경하지 않았다.

## 다음에 필요한 조치 — 아직 미승인·미실행

연구목표·Q/M 수식·학습 budget·split·평가 기준은 바꿀 근거가 없다. 입력 연결만 고쳐야 한다.

- 명시적인 새 **attempt_id와 출력 경로**를 고정한다. 기존 경로 재사용·resume은 하지 않는다.
- 원본11-key envelope를 엄격하게 검증하고 전부 보존한 뒤, core에 검증된 내용6개를 명시적으로 전달한다.
  임의의 추가 key를 무조건 허용하거나 출처 기록을 조용히 버리는 수정은 하지 않는다.
- Auditor도 envelope 검증과 내용 추출을 독립 구현한다. 원본 byte SHA, 데이터 범위, 출처 bindings는 유지한다.
- 인공11-key 원본 파일에서 **실제 loader→adapter→core→auditor**의 연결 시험과 잘못된 schema/출처/extra key 거부 시험을 추가한다.
- SHAM의 SHA 입력인 **과학적 study_id/randomization namespace는 유지**한다. 단순 실행 이름 변경으로 대조군 배정이 바뀌지 않게 한다.
- 새 실행 authority가 확정되기 전 실제 EEG 학습·새 attempt 생성은 하지 않는다.

원본 provenance 참고: schema `cfeg.context-template-source.projection.v1`, study `context-template-source39-v1`,
plan SHA `a76f1df0e0d2989f6217013e610d074e50d9efb52b0cf636bbcdf38a905898d3`,
source commit `46b87b7a6c2bae2ca6ac35cdc079c7c8a50ebb0d`,
start SHA `baa7104b659738b79a7cae78b958330195ab17d1adc591c097541c203f4c0125`.
이 값과 기존 파일 SHA만 재사용하며 원본 start·full manifest·Impedance.mat는 추가로 열지 않았다.
