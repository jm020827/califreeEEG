# 공통 reference는 개선했지만, 고정 baseline의 준비 기준은 미달

2026-09-14 KST / 2026-09-13 UTC. 기존 공개자료 S002–S011b의 같은 EEG 150창을
고정된 두 방식으로 비교했다. **명목 reference 49/150(32.67%) → source 공통 reference
87/150(58.00%)**, 차이는 **+25.33%p**다. 10명 중 8명은 개선됐고 2명은 악화됐다.
사전 판정은 **`NEITHER_BASELINE_READY_STOP`**이다. 이번 비교는 종료한다.

## 쉽게 설명하면

이전에는 “6.66Hz 자극이면 EEG도 그 명목 주파수와 비교하면 된다”는 기준과,
실제 기록 이벤트에서 얻은 주파수를 쓰는 기준 사이에 차이가 있었다.
이번에는 **검사받는 사람을 제외한 다른 9명의 기록**으로 공통 주파수표를 만들었다.
그 사람의 보정 trial이나 query 주파수는 공통표를 만드는 데 넣지 않았다.

동일한 EEG에서 이 공통표만 바꾸었더니, 기존에 틀린 49개를 추가로 맞히고
기존에 맞힌 11개를 놓쳤다. 순증은 38개다. 따라서 **이 고정 개발 비교에서는
공통 설정 자체가 중요한 차이를 만들었다.** 하지만 충분한 성능에는 도달하지 못했다.

이것은 “개인 metadata를 학습해 보정을 줄였다”는 결과가 아니다. 학습할 개인 M의
새 후보를 평가한 실험이 아니라, **모든 후속 대조군이 공유해야 할 공통 정보의
중요성을 확인한 기준선 진단**이다. 학습된 M은 이후에도 Q·공통 정보·추가 모델 용량을
넘어서는 효과를 입증해야 한다.

## 사전 기준과 실제 결과

참가자 준비 기준은 **15개 중 12개 이상 정답이며, 다섯 class 각각 최소 1개 정답**이다.
전체 기준은 **150개 중 120개 이상 정답(80%)이고, 10명 중 7명 이상 준비**다.
이는 실행 전에 정한 개발 운영 문턱이지 통계적 유의성이나 독립 확증 기준이 아니다.

| 항목 | NOMINAL | COMMON |
| --- | ---: | ---: |
| 정답 / 전체 | 49 / 150 | 87 / 150 |
| 정확도 | 32.67% | 58.00% |
| class별 정답, 각 30개 | 19 / 11 / 5 / 6 / 8 | 20 / 15 / 15 / 19 / 18 |
| 준비된 참가자 | 0 / 10 | 3 / 10 |
| 전체 준비 기준 | 미달 | 미달 |

| 참가자 | NOMINAL 정답 / 15 | COMMON 정답 / 15 | 정답 증감 | COMMON 준비 |
| --- | ---: | ---: | ---: | --- |
| S002 | 7 | 12 | +5 | 예 |
| S003 | 5 | 7 | +2 | 아니오 |
| S004 | 3 | 7 | +4 | 아니오 |
| S005 | 4 | 3 | −1 | 아니오 |
| S006 | 2 | 8 | +6 | 아니오 |
| S007 | 2 | 10 | +8 | 아니오 |
| S008 | 3 | 4 | +1 | 아니오 |
| S009 | 6 | 15 | +9 | 예 |
| S010 | 10 | 6 | −4 | 아니오 |
| S011 | 7 | 15 | +8 | 예 |

Paired 결과는 둘 다 정답 38개, COMMON만 정답 49개, NOMINAL만 정답 11개,
둘 다 오답 52개다. 개선 8명·악화 2명·동률 0명을 모두 보존한다.
좋아진 사람만 뽑아 다음 후보의 참가자를 정하지 않는다.

## 무엇을 고정하고 무엇을 바꿨나

- 입력은 이미 개발용으로 개봉한 S002–S011b 10개 MAT, 사람당 15개 창이다.
  E126(Python row125), 250Hz, 2초, 기본파/2차 고조파, 중심화, 무필터·무CAR·무ridge,
  동일한 단일채널 projection-squared-energy operator를 고정했다.
- NOMINAL은 `(6.66, 7.5, 8.57, 10, 12)`Hz다. COMMON은 기존 저장 DIN 결과의
  **다른 9명 B기록 × class당 3반복**의 기하평균이다. 참가자 자신의 A와 B는 모두 제외했다.
  모든 10개 bank를 새 EEG의 byte hash/header/load 전에 manifest에 동결했다.
- 새 MAT 읽기는 파일별 전체 byte hash·header·`eeg` load 각 1회다. 전체 257×T EEG를
  decode했지만 숫자 처리는 선택 row125의 15×500에 한정했다. 다른 행·비선택 EEG의
  수치 처리는 0이다. DIN/rate는 새로 decode하지 않았으며, 동일 MAT hash의 이전 검증을
  계승했다. 전체 raw를 아예 읽지 않았다고 표현하지 않는다.
- Scorer에는 EEG 벡터와 전체 5개 주파수표만 전달했다. 개인 query DIN 주파수·정답·
  group index는 전달하지 않았다. 사람별 점수 파일에는 정답을 넣지 않고, **전체 300개
  예측을 seal한 다음** 저장된 추론 label로 평가했다.
- 정답은 기존 DIN 분할 규칙으로 추론한 class다. 독립된 물리적 정답, 완전한 저자 CCA
  재현, E126의 Oz/최적 채널 mapping이 확인됐다고 주장하지 않는다.

[사전 계약](mamem_common_reference_v1_contract.md),
[원결과](reports/mamem_common_reference_v1_run/result.json),
[상태](reports/mamem_common_reference_v1_state.json)에 세부 범위와 해시를 남겼다.

## 비용과 해석의 경계

두 arm 모두 **평가받는 사람의 labeled support 0 trial / support 자극 0초**다.
COMMON에는 fold당 다른 9명의 기존 source event 135창이 필요하다. 이번 learned-model
fit은 0이지만, 공통값은 **자료에서 추정한 통계**다. “아무것도 추정하지 않았다”,
“source 구축 비용도 0이다” 또는 “처음부터 즉시 온라인 사용 가능하다”는 뜻은 아니다.
Query 기록·분할 비용은 존재하고, query-ready elapsed와 setup 시간은 **UNKNOWN**이다.

같은 10명의 개발자료를 재사용했고 fold들은 source를 공유한다. 독립 cohort 확증,
독립 trial 가정의 통계 검정, CI/p-value, Q/Q2/QM/SHAM 학습 비교와 실제 보정 절감은
이번에 평가하지 않았다. 앞선 S001의 12/15→14/15 개발 관찰은 남기되 일반화하지 않는다.

이번 미달은 **현재 E126·2초·고정 operator**의 미달이다. 모든 MAMEM decoder나 모든
metadata의 실패로 일반화하지 않는다. 이전 M2→PSD 80-fit 부정 결과와 full잔차 A→B
직접전달 실패도 그대로 남긴다. 다른 operator의 이번 결과만으로 이전 실패의 원인을
확정하지 않는다. 결과를 보고 채널·창·필터·계수·참가자를 바꾸는 구제 실험은 하지 않는다.

## 실행과 독립 검산

- 계약 **c8849e7** → producer **a16e4d9** → 생성 producer/auditor 연결 검사
  **b995d19** → 독립 auditor **eeb9041**을 **a7f0e97**로 통합했다. 실제 실행 전에
  고정됐으며, 관측 뒤 코드/threshold 변경·재실행은 0이다.
- 실제 실행 **2026-09-13 15:30:02.290445–15:30:09.668911 UTC**, **7.378466초**,
  1 attempt, 10 MAT / 150 EEG창 / 300 예측 / 1,500 class 점수 / 6,000 projection,
  learned-model fit 0, terminal **COMPLETE**. 고정 actual 1회 예산은 소진했다.
- 합성 suite 3회 예산도 소진했다. Root 최초 **24PASS / 0.13초**, 별도 auditor
  **76PASS / 1.67초**, 최종 통합 **428PASS / 3.00초**(root25 + auditor76 + 이전327).
  합성 실패는 없었다. Root의 초기 ruff 1건을 실제 전에 수정했고, 최종 ruff/diffcheck
  PASS다. 전체 repository suite를 실행한 것은 아니다.
- 정적 검토에서 제안된 디렉터리 fsync를 실제 전에 반영했다. Root는 계약/producer/
  실행/보고를 소유했고, auditor writer는 기존 clean worktree를 새 branch로 재사용했다.
  같은 working tree의 동시 writer는 없었으며 텍스트/의미 충돌 없이 통합했다.
- 실제 [독립 감사](reports/mamem_common_reference_v1_audit.json) 1회는
  **PASS_SAVED_PROJECTION_AND_SOURCE_ONLY_COMMON**, 최대 차이
  **7.105427357601002e−15**, near-tie 0이다. 별도 agent가 작성한 scalar 식으로
  B-only LOPO bank, 6,000 projection→1,500 점수→300 예측, label/count/paired/준비도/
  비용과 19 pins·역할·창·seal/실행수명을 검산했다. 45개 파일 542,789bytes 읽기,
  raw read 0 / producer import 0 / fit 0, 경과 0.019912초다.
- 이는 원 DIN→분할 또는 원 EEG→projection의 독립 재구성, 광학 clock·독립 정답·
  OS I/O 추적 증거는 아니다. 실제 peak RAM도 측정하지 않았다.
- Run은 28파일 / 391,865bytes, result는 195,739bytes, stdout/stderr는 0bytes다.
  Result SHA `ec2ea73e41fcfa980619babddf1b2911442cdbf8d9f547618ab70de7cafee323`.
  시작 시 공간 약 293GiB, 마감 확인 약 292GiB. 기존 46 worktrees와 8 untracked pytest 디렉터리 보존.
  다운로드·추출·네트워크·설치·삭제·push·GPU·held60·외부 사람 요청·유료 사용 0이다.

`academic-research`의 사전 판단·근거 한계·부정 결과 보존 원칙에 따라 새 논문검색이
아닌 고정 실험으로 질문을 좁혔다. Claim `2ad366bb8f60f874`(QUALIFIED, 근거 3개)와
기존 gap `3f973164c88eabce`를 갱신했다. `coordinate-worktree-changes`에 따라
감사 writer를 격리하고 root가 통합·최종 검증을 맡았다.
누적 97 claims / 257 evidence / 45 gaps, 11개 workspace view 재생성과 SQLite
quick_check `ok`를 확인했다. 기존 보호 문서·v2 결과 6개 hash는 불변이다.
별도 read-only reviewer의 결과 해석·후속 권한 문서 최종 검토도 PASS였다.

## 다음 행동

이번 공통 기준선 비교는 **종료**이고 전체 저보정 연구목표는 **미완료 / active**다.
단순히 큰 모델을 붙이거나 이 개발자료에서 잘 되는 조건을 고르는 단계로 넘어가지 않는다.

다음은 [새 공개 paired 데이터의 inventory/schema 확인 초안](mamem_post_common_reference_next.md)이다.
새로운 외부 acquisition 정보가 언제·무엇을 측정하고 EEG와 어떻게 대응되는지 확인한다.
MMV는 과거 공개 registry 성공과 배포 페이지의 로컬 capture 실패가 기록돼 있으므로
비공개라고 단정하지 않는다. 그러나 현재 학습 적격 데이터로 확보한 것도 아니다.
정확한 pair·시간 대응·pre-query 가용성·기작과 비용이 확인될 때만 새 학습 후보 하나를
정의한다. 이번 actual/감사 예산을 그 작업에 전용하지 않으며, held60·사람 요청·유료
자원은 별도 승인 조건을 유지한다.
