# Known-zero v1 — 같은 답 후보의 이득을 0으로 제한

2026-09-08. 사용자 '응 계속 그렇게하자'에 따른 **한 개의 기작 수정 명세와 인공 검증**이다.
연구목표는 적은 labeled SSVEP calibration을 위한 추가 acquisition metadata의 효용 검정으로 유지한다.
실제 수정 후 성능은 아직 계산하지 않았다. [직전 완료 결과](native_subset_m_envelope_r1_results.md)는 불변이다.

권위는 [고정 명세](../configs/analysis/native_subset_known_zero_v1.json), SHA
`9f3ed2309b9dc2efff81460a91c49e131722595ef1ff8de4b24f9291f66c9e55`다.
이 명세는 인공 검증 단계만 포함한다. 사람 자료를 읽는 실행기·출력 경로·게시 및 감사 계약은 별도로 고정해야 한다.

## 쉽게 설명하면

예를 들어 세 후보가 다음과 같다고 하자. 아래 숫자는 **실제 EEG 결과가 아니라 인공 사례**다.

| 후보 | 최종 답 | 회귀식의 예상 이득 | 수정 후 예상 이득 |
| --- | --- | ---: | ---: |
| 모든 보정 예시 사용(FULL) | A | 0 | 0 |
| 첫 블록 제외 | A | +0.30 | 0 |
| 둘째 블록 제외 | B | +0.20 | +0.20 |

기존 규칙은 +0.30을 보고 첫 블록을 제외하지만, 답은 FULL과 같은 A다.
정답이 무엇이든 두 후보는 같이 맞거나 같이 틀리므로 **정답률 차이는 정확히 0**이다.
새 규칙은 이 사실만 반영해 첫 후보의 이득을 0으로 놓고 B 후보를 선택한다.

정답이 B라면 개선, A라면 악화, C라면 둘 다 오답이다.
따라서 수식의 구조를 바로잡는다고 정확도나 metadata 효과가 반드시 좋아지는 것은 아니다.
이것은 직전 입력 연결 오류와 다른 **추론 정책의 과학적 수정**이며, 이전 결과가 무효라는 뜻도 아니다.

## 정확히 무엇을 바꾸고 유지하나

기존 모델이 내는 10자리 반올림된 예상 이득을 `g_j`, 각 후보의 native 점수 argmax를 `p_j`라 하자.
FULL은 `p_0`이며, 새 이득은 `p_j == p_0`이면 정확히0, 아니면 기존 `g_j` 그대로다.
FULL의 이득0과 함께 최댓값을 고르며, 동률은 FULL→기존 낮은 블록 순서를 유지한다.

- 동일성은 **실제 후보의 native argmax**로 계산한다. Q의 원래18번째 열(index17)은 일치 검사에 사용할 수 있지만,
  표준화된 특징값을 조건으로 쓰거나 M에 따라 mask를 바꾸지 않는다.
- Q, QM, SHAM_REFIT, M_STALE, 각 M_SHUFFLE map에 동일 적용한다.
- 각 M 방식의 기존 결측 fallback mask가 켜진 query는 **수정된 Q 선택**으로 돌아간다.
  M_MISSING도 수정된 Q와 정확히 같아야 한다. 이전 Q로 돌아가면 공정한 비교가 아니다.
- 기존9개 router의 계수·정규화·metadata scaler를 그대로 재사용한다. 학습 목표는 원래부터 맞았으며,
  선형 회귀에 equality 특징이 있다고 그 특징에 따른 정확한0 제약이 자동으로 보장되지는 않는다.
- k3/5의4/6개 후보를 padding 없이 전달한다. 12개 class 점수 자체, 특징, label cost, split,
  window, ridge0.1, fit mass, SHAM namespace·map, 모든 원본 산출물은 변경하지 않는다.
- FULL과 다른 답을 내는 후보들끼리의 중복 제거, 불일치 행만 재학습, 별도 threshold나 tuning은 하지 않는다.

## 어느 경우에 달라질 수 있나

1. 원래 FULL을 선택했다면 FULL을 유지한다.
2. 원래 FULL과 다른 답의 양수 이득 후보를 선택했다면 같은 후보를 유지한다.
3. 원래 FULL과 같은 답 후보를 선택했고 다른 답 후보의 양수 이득이 없다면 FULL로 돌아간다. 답은 같다.
4. 원래 FULL과 같은 답 후보를 선택했고 다른 답 후보의 양수 이득이 있다면 답이 바뀔 수 있다.

이 조건들은 각 route의 결측 fallback **전**에 성립한다. Fallback 후에는 수정된 Q를 상속한다.
같은 규칙을 두 번 적용해도 결과가 같고, query 순서를 바꾸어도 해당 query의 선택은 같아야 한다.
직전의 동일답 후보 선택 수는 기회가 있을 수 있다는 단서일 뿐, 위4번의 빈도나 개선량을 뜻하지 않는다.

## 구현과 인공 검증

[순수 선택 모듈](../scripts/native_subset_known_zero.py)은 파일 읽기·학습·정답 인자가 없다.
Float64의 이미 반올림된 gain, integer class ID 배열, boolean fallback만 받는다.
NaN/Inf는 동일답 후보에 있어도 거부하며, 비정상 값을0으로 가려서 진행하지 않는다.

`route_family`는 Q를 먼저 고정한 다음 모든 M arm과 각 shuffle에 같은 규칙을 적용한다.
Shuffle이 불가능할 때 기존 identity 평가1개를 유지하고 availability는 false로 기록해야 한다.
이 함수는 availability 판단이나 map 생성, metric 평균을 새로 정의하지 않는다.

검증은 다음을 분리한다.

- 모든 가능한12개 정답을 인공 예측에 대입해 동일답의 correctness 차이가0임을 확인한다.
- 원래 ridge 함수에 올바른0 목표의 동일답 인공 행을 넣어도 예상 이득이 양수일 수 있음을 재현한다.
  이 fit은 기존 함수의 인공 시험이며 실제 저장 계수를 읽거나 바꾸지 않는다.
- 원래 선택기는 민감도 사례에서 동일답 후보를 선택하고 새 선택기는 다른 후보를 선택해야 한다.
  개선/악화/오답 간 변경을 모두 포함해 유리한 사례만 성공 기준으로 삼지 않는다.
- 비양수/양수 동률, 기존10자리 반올림, k3/5, query 독립성, 입력 불변성, 반복 적용,
  모든 대조군/각 shuffle/수정된 Q fallback 및 잘못된 입력 거부를 검사한다.
- 독립 검산기는 vectorized projection을 복사하지 않고 query별로 FULL과 같은 답 후보를 건너뛰는
  순차 최댓값 계산을 사용한다. 두 구현의 정확한 선택 일치를 인공 자료에서 대조한다.

**이는 인공 선택 경로 검증이지 실제 metadata 파일→기존 freeze→수정 결과→감사 전체 경로 검증이 아니다.**
그 연결은 다음 실행 단계의 별도 필수 시험으로 남는다. 인공 PASS를 사람 성능 PASS로 부르지 않는다.

## 후속 실제 평가의 사전 요구사항과 중단 기준

다음에는 완료된 envelope-r1의 cache/projection/9개 fit freeze를 정확한 SHA로 묶어 재사용할 수 있다.
새 raw EEG 추출·재학습은 이 수정에 필요하지 않다. 그러나 아직 재사용 입력/새 결과 경로/시작 기록/
no-overwrite/독립 감사의 실행 명세와 연결 구현은 없다. 기존 폐쇄 runner를 수정하거나 호출해서 대체하지 않는다.

고정될 endpoint는 기존과 같다: 전8조건×k3/5,120summary/153contrasts/624diagnostics/1248attainment,
primary **수정 QM3−수정 Q3**, 참고 효과1pp와 기술적CI, conditional shuffle/stale/sham,
36/60labels, 최초관측80%, QM3−Q5의 참고 비열등폭1/60, 개인별 harm과 A0/FULL 비교다.
추가 기작 진단은 방식별 기존 동일답 승자 수, gain/후보/최종 답 변화, 정답 수리·손상이며
유리한 조건이나 사람을 고르기 위해 쓰지 않는다.

다음 결과에서 Q와QM이 비슷하게 좋아지면 **공통 선택기 개선**이다. M 성공으로 부르지 않는다.
후보 번호만 바뀌고 답이 그대로여도 M 효과가 아니다. M 증분·controls·실용성·비용 근거가 부족하면
이 수정 후 해당 구현 탐색을 종료하고 프로그램/독립 paired-M 자료의 필요성을 재평가한다.
긍정 결과가 나와도 반복노출 개발39명이 새 독립 확인으로 바뀌지는 않으며 held60을 자동 개봉하지 않는다.

현재 예산은 **M 방식1개의 완료평가1회 + 단일 수정후보1개 고정/인공 검증 + 수정 후 실제평가0회**다.
다른 수정 후보는 추가하지 않으며, 한 번의 수정 후 실제 평가가 끝나면 결과 부호와 무관하게 프로그램을 점검한다.

## 증거 기록과 작업 분리

`academic-research`로 기존 claim:21d948583be84d7b와 gap:f86524a7700e7501을 재사용했다.
이번 질문은 원문 개수보다 코드·정확한 수학적 항등식·인공 민감도로 확인 가능하므로 새 문헌 검색/PDF 읽기는0이다.
새 방법의 문헌상 최초성이나 metadata 효능은 주장하지 않는다. Signature는 연구공간의
`native_subset_known_zero_signature_20260908.md`에 기록했다.

`coordinate-worktree-changes`: base main2cb18f2, 계약8ee0f22. 기존26worktrees 보존,
가용 약285GiB 대비 추가 checkout12MiB/temp1GiB로 독립 검산용 worktree 하나를 만들었다.
기존 환경은 읽기 전용 재사용하고 SQLite는 main만 쓴다. 다음 소유권은 겹치지 않는다.

| Lane | 소유 경로/자원 | 검증·통합 |
| --- | --- | --- |
| Main | 새 JSON, 선택 module/test, 문서, SQLite | 계약→선택기→검산기 통합→전체 회귀 |
| Auditor | 별도 worktree의 audit_native_subset_known_zero.py/test_native_subset_known_zero_audit.py | 독립 scalar 구현과 인공 테스트만; commit 반환 |
| Skeptic/Builder | 공유 repo 읽기 전용 | 수학·기작·최소 삽입 지점 검토 |

Auditor 경로 `/home/whwovy/califreeEEG-wt-known-zero-audit`, branch `codex/native-known-zero-audit-v1`.
실제 input/결과 접근0이며 기존 worktree/환경/산출물의 삭제·정리·push는 하지 않는다.

## 완료된 검증 기록

- 계약8ee0f22 → main 구현/설계 b51d251 → auditor068cb4a를 edaa9d2로 통합 → main 연결시험2c385ff.
  파일 소유 중복0, three-way merge preview 충돌0, 최종 root 통합이다. Auditor worktree는 clean으로 보존했다.
- Main31tests PASS0.23초, 독립 scalar62tests PASS0.09초. 통합97tests PASS0.28초에는 원본
  `inputs_for_cell`·`predict_gain`→새 family→독립 scalar를 실제로 호출하는 인공 k3/5 시험도 포함한다.
  Q/M 배열과 계수는 인공값이며, 실제 raw 전처리나 사람 fit freeze를 읽은 것이 아니다.
- 읽기전용 reviewer의 별도11,151개 유한 인공 경우에서도 exact action과 분기 불변조건/반복 적용이 일치했다.
- Clean 코드2c385fff4ad9ac8478e4adfb238bfef36adc4472에서 전체 **1442tests PASS151.48초/기존 Torch warnings68개**,
  Ruff4files lint/format 및 diff check PASS. Temp `/tmp/cfeg-kz-check.ykeqKj/full` 보존.
- 순수 모듈 SHA `d9830fe97e358eae7f65305feee5776ac7390b0d8eef7ca67a8c4790fbb952ba`,
  독립 scalar SHA `5c2bb42d0c04a46d316efb940ce58dcf7c65a45b32b2ce75913f2b6951cb2d7a`.
  기존 과학 JSON/core/producer/auditor/envelope 파일은 변경0이다.
- 연구공간 claim:e287f6e9db9486b4(qualified)에 코드·인공 검증과 개선/악화 가능성의 증거를 연결했다.
  Evidence:f6b551b19368ac42/a12a19b42c425c3f/db355dfa827d08fc. Gap은 다음 실제 산출물 연결 단계로 갱신/render했다.
  이 claim은 실제 EEG의 M 효능 claim이 아니며, 프로그램 전체 연구 완료를 뜻하지 않는다.
