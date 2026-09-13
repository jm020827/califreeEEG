# 다음 단계: 개발용 S001 라벨 해석 진단 — 미실행 초안

2026-09-13. 기존 v1 성능 시도는 종료했으며 재시도하지 않는다. 다음에는 성능을
높이는 후보를 더 만들기 전에, 개발 전용 S001a의 label 의미만 확인할 필요가 있다.
사용자가 승인한 연구 루프 범위의 후속 제안이며 현재 문서만으로 실행하지 않는다.
실행 시 별도 config/시간·자원 예산/producer hash를 먼저 고정해야 한다.

## 현재 근거

v1 S001a reader는23groups를 확인했지만 하나의 continuous frequency가 사전
guardband 밖이라 중단했다. 어느 group인지/추정값/실패의 물리적 이유는 영수증에
없다. 이를 근거 없이 ‘손상 EEG’, ‘metadata 없음’, ‘잘못된 원본’으로 부르면 안 된다.
기존 DataAcquisitionDetails p2–4는5nominal frequencies와8adaptation+15main,
각 main class 연속3회라는 구조를 설명한다. Legacy 코드의 label convention과
우리의 좁은 numeric guardband가 같다는 증거는 없으며, 대체 규칙을 즉시 적용하지 않는다.

## 다음 읽기 전에 고정할 진단 범위

- 대상은 이미 개발용으로 제외한 동일 S001a 한 파일. 추가 archive 추출/참가자0.
- EEG는 shape header만; `DIN_1`만 decode하고 descriptor rows는 해석하지 않는다.
- 23group 각각 group index, event 수, sample 경계, timestamp interval의
  mean/median/min/max, continuous f_est, 기존 guardband 포함 여부만 기록한다.
- 모든 group을 동일하게 기록한다. 실패 group만 선택하거나 임계값을 sweep하지 않는다.
- EEG-derived Q, M feature, trial accuracy, λ/model fit0. 새 source/held60/query0.
- 기존 v1 parser/결과 파일은 수정하지 않고 진단 출력은 새로운 provenance에 둔다.
- 제안 상한: 단회 reader1, 2GiBAS/60CPU/90wall, 출력64KiB, network0.
  실제 마감 UTC·해시·exclusive attempt ledger는 다음 시작 시 고정한다.

## 진단 후 분기

1. Grouping/indexing/문서 해석의 구현 오류라면, 개발-only 근거와 생성 회귀검사를
   먼저 만들고 새 parser version을 별도 고정한다. v1 중단을 성공으로 바꾸지 않는다.
2. Adaptation/main의 계약이 다르다는 직접 근거가 나오면 다음 protocol에서 역할을
   명시적으로 구분한다. 단지 원하는 label을 얻기 위한 사후 제외는 허용하지 않는다.
3. Main class label이 독립적으로 정당화되지 않으면 억지로 nearest label이나
   순서를 배정하지 않는다. 이 데이터의 supervised 효능 실험은 보류한다.

새 v2 efficacy는 development 진단만으로 규칙을 고정하고, S002–S011의 EEG·M·
분류 결과를 보지 않은 상태에서 별도 선언해야 한다. 원래 80-fit 시도는0fit로 보존한다.
추정 label에 의존한다는 한계도 독립 ground truth 확인과 분리한다.

이 단계의 성공은 ‘metadata가 도움이 됨’이 아니라 **무엇을 정답으로 비교하는지
설명할 수 있음**이다. 근거가 안 나오면 해당 후보·데이터 연결을 중단하는 것도 결과다.
