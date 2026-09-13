# S001 실제 reference 진단 결과: 12/15 → 14/15

2026-09-13. 단회 실행과 독립 저장 수치 감사를 완료했다.
**지원 기록으로 만든 reference bank가 같은 평가 trial 15개에서 오답 2개를 추가로 맞혔다.**
다만 이것은 개발용 한 명의 reference 민감도 진단이며, 학습된 metadata의 효용이나
보정량 감소를 입증한 실험은 아니다. 원래 저보정 SSVEP 연구목표와 이전 v2 부정 결과는 유지한다.

## 무엇을 비교했나

S001a의 첫 main trial을 class마다 하나씩 골라, 기록된 event sample 간격으로
다섯 reference 주파수를 만들었다. 이를 먼저 고정한 뒤 S001b의 EEG를 열었다.
두 조건 모두 E126 한 채널, 2초 창, 기본 주파수와 2차 고조파, 동일한 투영 점수식을 썼다.
Query DIN에서 추론한 정답은 창·적격성 검사와 최종 평가에만 사용했고, 해당 query의
주파수를 reference에 넣지 않았다. 모든 query가 동일한 전체 다섯 후보 bank를 받았다.

| 조건 | 정답 수 | 정확도 | 사전 개발 관문 |
| --- | ---: | ---: | --- |
| NOMINAL: 명목 주파수 | 12/15 | 80.00% | 통과 |
| SAMPLE_SUPPORT: A의 지원 기록 주파수 | 14/15 | 93.33% | 통과 |

둘 다 맞힘 12개, 지원 bank만 맞힘 2개, nominal만 맞힘 0개, 둘 다 틀림 1개였다.
사전 관문은 12/15 이상이면서 각 class에 최소 한 정답이 있는 것이었다.
이 문턱은 개발 신호 검출용 운영 기준이지, 통계적 유의성 기준은 아니다.

| 명목 class Hz | 지원 bank Hz | NOMINAL 정답 | SAMPLE_SUPPORT 정답 |
| ---: | ---: | ---: | ---: |
| 6.66 | 6.483402 | 3/3 | 3/3 |
| 7.50 | 7.291667 | 3/3 | 3/3 |
| 8.57 | 8.333333 | 2/3 | 3/3 |
| 10.00 | 9.655532 | 2/3 | 3/3 |
| 12.00 | 11.645963 | 2/3 | 2/3 |

## 결과가 뜻하는 것과 뜻하지 않는 것

이 고정 단일채널 조건에서는 실제 EEG로 자극 class를 구별할 수 있었고, A의 기록을
이용한 reference 변경이 B의 예측 일부를 개선했다. 모든 MAMEM 분석이 불가능하거나
모든 reference가 동일하게 동작한다는 해석은 이 관측과 맞지 않는다.

반면 다음 주장은 아직 할 수 없다.

- **Metadata 학습이 성공했다:** 새 학습은 0회다. 이번은 기록된 주파수를 정해진 식으로
  reference에 반영한 비교다. Q/Q2/QM/SHAM 학습 대조를 실행하지 않았다.
- **공통 정보를 넘어 추가 가치가 있다:** 다른 참가자에서 얻은 공통 reference만으로
  같은 개선이 나는지 아직 비교하지 않았다. Nominal과만 비교하면 이 둘을 분리할 수 없다.
- **보정량이 줄었다:** nominal은 원래 사용자 support가 필요 없고 이미 같은 80% 관문을
  통과했다. 지원 bank는 class당 한 trial, 총 5개를 사용했다.
- **B 세션의 clock을 측정했다:** 측정은 A에서 했다. A→B는 cross-run 전달이며,
  B에서 측정한 현재 acquisition context가 아니다. 단일 global clock 원인도 검증하지 않았다.
- **강한 저자 baseline을 복원했다:** E126은 저자 Dataset-I 기본 예제의 선택에서 왔지만,
  무필터·2초·2harmonics·CCA argmax 조합은 우리의 진단 설정이다. 전체 저자 CCA recipe
  미확정 판정은 그대로다. 기존 256채널 방법과의 변화 원인을 분해한 실험도 아니다.
- **독립 참가자 일반화가 확인됐다:** S001은 이미 개발에 사용한 한 명이며 trial들은
  상관되어 있다. DIN 추론 label은 독립적인 실제 자극 정답 검증과 같지 않다.

## 지원 비용

지원 trial 5개의 자극 길이 합은 25초이고, 선택한 DIN 분석창 길이 합은 10초다.
마지막 선택 지원 trial까지 기록의 elapsed prefix는 **445.94초**다.
23개 group의 사후 적격성 검증에는 A 전체 DIN이 필요하며 A 기록 길이는 **471.668초**다.
따라서 25초나 10초를 실제 준비시간으로 바꾸어 보고하지 않는다. 실제 query-ready 시간과
setup 비용은 UNKNOWN이고, 이번에 관측된 보정량·시간 절감은 없다.

## 실행·감사·재현 기록

- 계약 39f1f79 → producer 00f1bbd → 종료 기록 보강 54de7d1 → 격리 auditor
  08281b0을 e2ee6e2로 통합 → 생성 producer/auditor 연결 검증 6a8a2ec.
- 실제 시작 `14:27:47.259799 UTC`, 종료 `14:27:48.737328 UTC`:
  **1.477529초, 1attempt, 30predictions, 150class scores, 0fits**.
- A는 DIN_1/samplingRate만 decode했다. B는 EEG/DIN_1/samplingRate를 전체 decode했지만,
  EEG 수치는 row125의 15×500만 사용했다. A EEG·다른 B 채널·query M2 추출은 0이다.
- 최종 관련 **366 tests PASS, 2.25초**, ruff/diffcheck PASS. 전체 repository suite는 미실행.
  합성 suite 예산 3회: root 첫 20PASS, 독립 감사 첫 25PASS/54FAIL, 최종 통합 366PASS.
  감사 초기 실패는 Python3.10의 한 자리 UTC 소수초 파싱이었다. 수정과 실패 이력을 보존했고,
  실제 결과를 본 뒤 threshold나 과학 설정을 바꾸지 않았다.
- 고정 독립 감사는 실제 저장자료에 **1회** 실행했다. 600 projection scalars의 제곱합,
  150scores, 30argmax, 정답·class/paired counts·판정·비용·지원 주파수 식·6code pins·
  시작/worker/terminal/result 연결이 PASS. 최대 산술 오차 `2.7755575615628914e-17`,
  재합산 near-tie 0이다. 원 EEG→projection을 독립 재구축하거나 OS I/O를 추적한 감사는 아니다.
- `result.json` 22,345bytes, 전체 run 24,429bytes, stdout/stderr 각 0bytes.
  원 결과 SHA `a026ac6314d2e1342bc36e5c0c08e4889ea374587d99513ec7ddad72d773b313`.
  [감사 영수증](reports/mamem_reference_probe_development_v1_audit.json),
  [실행 원본](reports/mamem_reference_probe_development_v1_run/result.json),
  [상태](reports/mamem_reference_probe_development_v1_state.json).

Root와 독립 auditor는 서로 다른 worktree에서 작성했다. 신규 worktree 약36MiB이며,
기존 45개와 미추적 사용자 디렉터리는 보존했다. 새 다운로드·설치·삭제·push·source cohort
수치 읽기·held60 개봉·사람에게 자료 요청·유료 자원은 모두 0이다.

## 다음 결정

이 진단은 완료한다. 지원 bank는 **후속 공통 reference 대조 후보**로만 유지한다.
바로 이전 M2→PSD shrinkage 모델을 재개하지 않는다. 다음에는 공통 주파수 설정의
개선과 잔여 acquisition metadata의 추가 정보를 먼저 분리하고, 학습 후보·Q/Q2/SHAM 및
단순 M 직접 사용 대조·보정 비용·독립 확인 계획을 새 예산 안에서 고정해야 한다.
[후속 초안](mamem_reference_residual_learning_next.md)은 아직 실행하지 않았다.
