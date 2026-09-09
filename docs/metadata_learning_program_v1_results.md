# Metadata 학습 연구 루프 v1 — 두 후보의 결과와 다음 검증

2026-09-09. **유망 후보를 확보하지 못했다. 연구목표는 유지하지만 이번 두 후보로 metadata의
추가 성능 이득이나 보정량 감소를 입증하지 못했다.** 두 후보 모두 유효성 실패로 효능 미평가다.
C2 독립 terminal 감사와 최종 전체2857tests를 완료했고 사전 중단 규칙대로 유한 프로그램을 닫았다.
이는 연구 질문의 해결이나 metadata 효능 성공이 아니라, 고정된 두 슬롯의 검증·보고 완료다.

## 처음의 질문은 그대로다

새 사용자가 적은 보정 EEG로 SSVEP를 잘 분류하도록 할 수 있는가? 특히 EEG 자체에서 얻는
품질 단서 Q와 공통 정보를 똑같이 준 상태에서, **별도로 측정한 acquisition metadata의 숫자**가
학습 결정을 더 잘하게 만들고 실제로 필요한 보정량도 줄이는가?

이번 numerical M은 보정 블록별·채널별 임피던스다. Dry/wet, 측정 순서, 결측 패턴은 Q 쪽에도
제공하므로 QM이 단순히 공통 정보를 더 받았기 때문에 유리해지지 않게 했다. 임피던스가 높으면
반드시 나쁘다고 강제하지 않고, source 분류 오차로 작은 추가 보정을 학습하도록 설계했다.
목표를 손상 EEG 처리·OOD 발견·LLM 활용으로 바꾸지 않았다.

## 실행 전에 고정한 범위

[프로그램](metadata_learning_program_v1.md), [기계 계약](../configs/analysis/metadata_learning_program_v1.json)의
후보2개·예산·순서·중단 규칙을 먼저 고정했다. C2 수식도 C1의 새 사람 학습 전에 확정했다.
작업마다 승인을 다시 요구하지 않았지만, 종료 조건 이후의 새 후보나 과학 설정 변경은 실행하지 않는다.

| 후보 | Metadata로 바꾸려던 학습 결정 | 실제 도달한 단계 | 결과 |
|---|---|---|---|
| C1 temporal-R | 채널별 공간필터 규제의 비중 | source39의30pipelines/120heads/24,000updates, 세 모델 동결 후 최종 평가 일부 | 최종 필수 대조군의 수치 검증 실패; 효능 미평가 |
| C2 pair-S | 서로 다른 보정 반복 두 개의 공통 패턴이 S에 기여하는 비중 | Q/M 특징·미분 연산 구현, 사전 고정 인공 조건수 검사 | 균일 대조군의 수치 검증 실패; 사람 학습 전 종료, 효능 미평가 |

평가 설계는 정확히 허용된39명 개발자료, 두 interface와 네 시간창, 앞3/5블록이었다.
사람 단위 outer3/inner3 분리, Q만으로 lambda를 선택하고 Q를 동결한 뒤 QM/Q2/SHAM을 각각
같은 크기의 head와 같은 학습 예산으로 비교하도록 했다. 세 outer-fold 모델을 고정한 뒤 query6–9를
연다. Source 감독 block5와 target 보정 label의 권한도 구분했다. 이39명은 과거 반복 노출된
개발자료이므로 nested 분리·독립 재계산이 새로운 독립 확증 표본을 만들어 주지는 않는다.

## 성공을 무엇으로 판정하려 했나

- 같은 보정량에서 QM이 Q보다 평균1%p 이상 좋고, 사람별 차이 CI하한도0보다 커야 한다.
- 단순히 head를 추가한 효과인지 확인할 Q2, 다른 사람 M으로 재학습한 SHAM보다 좋아야 한다.
  정확한 prefixmask/order/interface/분할 안에서 donor를 정하고 충분한 M 변화 coverage를 요구한다.
- 보정3블록 QM이80% 이상이면서 Q5·A0·원 FULL·같은 필터의 centered FULL과 비교해
  실질적으로 뒤처지지 않아야 한다. C2에는 UNIFORM_C2 비교도 추가했다.
- 0/3/5블록의 관측 비용0/36/60labels, 48query 중39개 정답 도달 여부로 label절감과 손해를
  함께 본다. 미도달 비용은 null이며 3블록이라는 이유만으로24labels 절감으로 세지 않는다.

이 판정은 관측 grid에서의 보정 label비용이다. 임피던스 측정·장비 준비·실제 시간 절감이나
배포용 stopping policy, one-shot 효과를 증명하는 설계는 아니다.

| 이번 프로그램의 효능 항목 | C1 | C2 |
|---|---|---|
| QM−Q, QM−Q2, QM−SHAM의 완결 성능 비교 | 미평가 | 미평가 |
| 보정3블록80% 도달 및 Q5/A0/FULL 비교 | 미평가 | 미평가 |
| 실제 관측 label절감·추가 도달·악화한 사람 | 미평가 | 미평가 |
| 독립 표본 일반화 | 미실행 | 미실행 |

**미평가는0이 아니다.** C1의 부분 예측으로 유리한 arm/사람/시간창만 추리거나, 미완성 전체 비교를
성능표로 만들지 않는다. C2는 분류 학습·query 자체를 실행하지 않았다.

## 무엇을 실제로 배웠나

[C1 상세](task_trca_temporal_source39_v1_results.md): 생성 완료 경로·독립 cold·전체2578tests를 통과한
실행도 실제 입력에서는 실패할 수 있었다. 587.195초 뒤 `h must be symmetric within the frozen
tolerance`로 멈췄다. 이미 저장된 통계·동결계수만 제한적으로 재구성하니 S063/wet/0.5초/k3의
PERMUTED donorS84 경로가 동일 검사를 거절했다. 허용치의1.05653배였다. 사람 입력 attempt1회와
부분 final-query 모델 개봉1회는 소비한 것으로 센다. 단순 저장 오류가 아니므로 recovery로 재실행하지 않았다.

[C2 상세](task_trca_pair_s_v1_results.md): 같은 한 prefix의 B=C uniform은 통과했으므로 C2에 대한
실패가 바로 증명된 것은 아니었다. 그래서 이미 고정한 다른 기작을 자체 인공 검사까지만 진행했다.
고정16행 중 k5/조건수120000/UNIFORM_C2 한 행에서 원 H 검사가1.03880배로 실패했다. 균일한
비학습 대조군에서도 발생하므로 metadata 효과의 문제가 아니라 **현재 수치 경로의 유효성을 먼저
확보해야 한다는 근거**다. 나머지15행이나 비균일 행의 통과로 필수 대조군 실패를 무시하지 않는다.

두 결과는 수치 민감성과 일치하지만, 수학적 기작 전체의 불가능성이나 metadata의 무용성을
입증하지 않는다. 수치 정책이 잘못됐다고 단정할 수도 없다. 다음 정책은 정확도 결과를 보고
허용치를 늘리는 방식이 아니라, 별도의 오차 기준과 독립 기준 계산으로 먼저 검증해야 한다.

## 예산과 보호 경계

| 항목 | 고정 상한 | 현재 소비/상태 |
|---|---:|---|
| 과학 후보 슬롯 | 2 | C1 실제 유효성 실패, C2 자체 사전검증 실패 |
| 사람 primary | 2 | 1; C2는 선행 관문 실패로 미실행 |
| 사람 총 attempt / 공유 인프라 복구 | 3 / 1 | 1 / 0 |
| 사람 optimizer updates, 부분·복구 포함 | 72,000 | 24,000 |
| 서로 다른 final-query 모델 개봉 | 2 | 1, 부분 개봉 포함 |
| 사람 실행 시간 | 21,600초 | 587.195초 |
| 사람 출력 | 36GiB | 보존 C1 출력1,082,188KiB(파일시스템 사용량) |
| 추가 writing worktree | 2 | 2개 생성 후 C2에 재사용; 추가 생성·삭제0 |
| held60 / 외부 요청 / 유료 자원 | 별도 승인 | 모두0 |

예산은 최대치이며 전부 소진해야 한다는 의무가 아니다. 사전 중단 규칙으로 적격 슬롯이 모두
닫히면 남은 예산을 세 번째 후보·수치정책 변경·실패 대조군 제외에 자동 전용하지 않는다.
Engineering의 측정 receipt 및 단위/전체 tests는 실행 이력에 따로 기록한다. C1 실패 receipt에는
GPUpeak가 없어 확정 peak를 보고하지 않는다. 다른 GPU process와 기존 작업·데이터를 보존했다.

독립 읽기전용 자원 점검에서 명시된 생성 출력·진단·JUnit·main test출력과 두 worktree 전체는
333,748KiB(0.31829GiB)였다. 이전 보존 test도 포함한 보수적 현재 스냅샷이며, 모든 시점의 peak는
아니다. 두 tree의 tracked checkout은11.035/11.094MiB, 측정 test출력은26.617/73.145MiB로
각64/512MiB 상한 안이다. 각 tree를 상한 전체로 치환하고 감사 출력64MiB까지 예약해도1.38711GiB다.
최종 전체 tests는 별도 `/home/whwovy/task-trca-program-final-tests-Uv1w9v`에서 실행한다.
문서로 추적된 main engineering job 경과시간의 부분합은439.939초이며, 모든 agent/test/retry의
통합 CPU/GPU 사용량은 계측하지 않았다. 이 부분합으로 프로그램 전체 compute12시간 준수를
정밀 인증했다고 주장하지 않는다. 최종 감사/회귀 실측은 종료 기록에 추가한다.

## 다음 연구: 같은 목표, 별도 수치 검증 단계부터

다음은 **제안만 하며 이번 프로그램에서 자동 실행하지 않는다.**

1. 별도 generated-only 수치 검증 계획을 고정한다. S/C가 대칭인 수학적 문제와 두 번의
   triangular solve가 만드는 부동소수점 H를 구분하고, 잔차·역오차·고유벡터/부분공간·1차 미분의
   수용 기준을 먼저 정한다. 독립 고정밀/다른 계산식의 기준값과 이미 실패한 사례, 새 미노출 인공
   사례를 포함한다. 수치정책 변경은 새 정책으로 표시하며 C1/C2를 몰래 수리 재개하지 않는다.
2. 누수 없는 실제 prefix구성·Q/M/scaler/donor·네head·CPU/CUDA·nested·완료 archive 감사까지
   통과한 후에만 새 유한 효능 프로그램과 새 model/reveal 예산을 제안한다. 사람 CE나 query를
   추가로 보며 solver/특징/문턱을 고르지 않는다. 이번 C2의 미구현 learner 등은 선행 관문 실패로
   생략된 항목이지, 후보 종료 뒤에도 계속 만들어야 할 의무가 아니다.
3. 새 효능 프로그램에서도 기존 Q/Q2/SHAM/FULL/A0/보정비용/harm 기준을 유지한다. 유효한
   음성 결과가 나오면 그 구현을 닫으며, 유망 개발 후보를 얻은 경우에만 독립 확인을 계획한다.
4. 추가자료는 수치 문제를 해결하는 대체물이 아니다. 독립 효능 확인에는 EEG와 블록·채널·시점이
   대응하는 acquisition-M, 단위/결측 규칙과 사용권한이 필요하다. Dataset ID나 장비명만 추가된
   EEG 모음은 이번 숫자 M 가설의 직접 복제가 아니다. Held60 개봉·제공자 요청 발송·유료 자원은
   실제 실행 전에 별도 승인한다.

## 보존·근거·검증 이력

기존 [metadata-prior 실제 음성 결과](metadata_prior_source39_v1_results.md),
[context-template 결과](context_template_source39_v1_results.md),
[task-shape 유효성 실패](task_trca_shape_source39_v1_results.md) 및 이전 모든 연구일지를 보존했다.
이번 유효성 실패가 이전 유효한 음성 효능 결과를 지우거나, 반대로 이전 음성이 이번 미평가를
숫자0으로 바꾸지는 않는다.

`academic-research`는 기존 문헌 지도를 재사용하고 주장·실측·한계·다음 공백을 분리해 갱신했다.
이번 병목은 자체 수치 검증이므로 새 광범위 검색/PDF정독0이며, 최신 문헌을 완전히 조사했다고
하지 않는다. `coordinate-worktree-changes`는 두 기존 분리 작업공간 재사용, 단독 통합/GPU/SQLite,
읽기전용 교차 검토에 사용했다. Code/test PASS, 실패기록 감사 PASS, 효능 PASS를 구분한다.

최종 독립 감사·전체 테스트와 closure 상태는 다음과 같다.

### 종료 검증 완료

C2 terminal 감사는 source16행/8case를 독립 재구성하고 원 실패를 확인했다.
Receipt SHA `2c48e84e91bbf11c94a69d0d5e978a8cb581cb71d5c2fb3621e5dafb797fbb8a`, .617036초,
`GENERATED_TERMINAL_VALIDITY_FAILURE_VERIFIED`다. Root 통합 auditor toy75tests도7.85초에 통과했다.
최종 전체 회귀는 추가900초/출력2GiB 한도, CPU threads1, 독립 pytest 임시 경로로 실행하며
출력을 주기적으로 확인했다. 전체 **2857tests PASS314.59초/기존68Torchwarnings**, 실패·skip0이다.
JUnit SHA `8525423bcc8cbb402c3e0451cb4eae3ba6c81225d7465a3553cd24a169186c91`를0400보존했다.
실행시점 e96ee82의 runtime/tests와 종료시점 runtime/tests는 같고 이후 변경은 결과 문서다.
새C2의8Python파일 Ruff/format PASS, 원program/C1/C2설계 및 원25C1/15oldcodepins를 보존했다.

최종 test parent는417,820KiB, C2 gate+독립감사 parent는22,080KiB로 측정했다. 같은 명시
경로들과 최종test parent를 다시 측정한 scoped 합계는752,344KiB(약0.718GiB)다.
중간 출력도 주기적으로 확인했고2GiB test상한을 넘는 관측은 없었다. 통합 compute/과거peak의
미계측 한계는 그대로다. 확인된 예산을 넘어 추가사람학습·세번째후보·복구를 실행하지 않았다.

통합 경로는 root 공통계약→C1 runtime/cold→실제C1→제한진단→C2 features5955042→
operator7c1c9d4→gate48407d7→독립audit c506b12→최종회귀였다. 독립 lane의 소유파일은
겹치지 않았고 root가 의미·수식·형식·failure우선권을 검토한 뒤 통합했다. 두 새tree는 여러
단계에 재사용했고 기존40worktrees·branch·untrackedtest출력·실패자료를 삭제하지 않았다.

최종 상태: `CLOSED_CANDIDATE_FAMILY_NO_PROMISING_CANDIDATE`.
모든 효능 null/미평가와 기존 음성 결과를 보존하고, 후속 수치 검증을 제안 상태로 남긴다.

원 요청과 사전 중단 규칙을 대조한 [종료 재점검](metadata_learning_program_v1_completion_audit.md)에서도
같은 판정을 유지했다. 원 산출물·코드 pin·JUnit을 다시 확인했으며, 유한 후보군의 종료와
미해결 효능 질문, 불완전한 누적 자원 계측을 구분했다. 재점검에 따른 새 실험은 없다.
