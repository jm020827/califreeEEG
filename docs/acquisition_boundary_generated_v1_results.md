# 사전 acquisition context: 시간경계·보정비용 구현 결과

2026-09-10. **기능 검증 완료, 예산 이탈 1종 기록. 실제 metadata 효능은 미평가.**
현재 `계속 연구` goal은 active로 유지한다. 이전 유한 연구 루프의 완료/부정 결과는 별개로 보존한다.

## 무엇이 달라졌나

다음 metadata 실험에서 **미래 정보를 몰래 쓰거나, 실제로 들인 보정시간을 적게 세는 오류**를
검출할 수 있는 별도 경로를 구현했다. Choi 원 전처리·분할·기존 실험 결과는 바꾸지 않았다.

- 시간 구간은 정확한 sample 단위로 표현한다. 예컨대 200 Hz에서 0.14초는28 samples지만,
  0.143초는 조용히 반올림하지 않고 거부한다.
- 사전 특징 계산에 필요한 FIR 과거 샘플과 export의 과거/미래 참조 범위를 함께 검사한다.
  허용 구간 사이의 금지된 틈, 부족한 baseline, 미확정 marker/변환 시간은 읽기 전에 거부한다.
- 허용 구간만 요청하고, 창마다 필터 상태를 새로 구성한다. 인공 export가 실제로 미래1 sample을
  참조하는 경우도 포함해, cutoff 이후를 바꿔도 결과가 같고 마지막 허용 샘플에는 민감함을 확인했다.
- 특징에는 참가자·날짜·band·session뿐 아니라 clock ID, sampling rate, trial onset을 묶는다.
  실제 선택된 support ID와 비용 schedule을 다시 대조해 다른 trial/시간축의 혼입을 거부한다.
- 선택한 예시 수와 실제로 수집한 예시 수·경과시간을 나눠 계산한다. 설치시간과 추가 baseline이
  미확정이면 총시간도 미확정이다. `k=0`이어도 알려진 설치비용이 사라지지는 않는다.

## 인공 예시에서 확인한 것

| 경우 | 기대한 결과와 확인 |
|---|---|
| A,A,B,C,D 순서에서 class당1개 선택 | 선택4개, 실제 수집5개. 통합 fixture 경과10.5초; 설치시간 미상이라 총시간도 미상 |
| 허용 context 구간 외 모든 값을 NaN으로 교체 | 선택된 모든 context와 결합 결과 동일 |
| 일부러 전체 run 평균을 이용한 잘못된 대조 | 미래 값을 바꾸자 과거 특징도 바뀜. 검사가 이런 누수를 탐지함 |
| 같은 참가자지만 다른 clock/rate/onset | 결합 거부 |
| 수집 시작이 sample400인데 context는378부터 필요 | 선행22/200=0.11초 필요. 추가 baseline0/.109초는 거부, .11초는 허용, unknown이면 총시간 unknown |
| k=5 예시가 실제로 모이지 않은 schedule | 관측 수집량은 보존하지만 도달 선택/총비용은 None |

여기의 `attained`는 **k개씩 모였음**이지, 목표 정확도에 도달했다는 뜻이 아니다.
숫자는 인공 schedule의 코드 검사이며 실제 EEG 보정량 감소 측정이 아니다.

## 구현과 검증 범위

- [공통 시간/identity 타입](../src/cfeg/data/acquisition_context_contract.py)
- [사전 의존 구간·특징 추출](../src/cfeg/data/acquisition_context_boundary.py)
- [실제 수집 비용](../src/cfeg/analysis/calibration_collection_cost.py)
- [선택 trial·context·비용 연결](../src/cfeg/analysis/acquisition_context_support.py)

최종 새 모듈 검사 **34 PASS**: temporal14 + cost16 + integration4, 실패/오류/skip0.
그 전 통합 검사56PASS에는 기존 Choi 회귀23개가 포함됐지만 마지막 비용 수리 전의 실행이다.
이를 최종34와 더해 서로 다른 새 실험90개라고 세지 않는다. 전체 저장소 pytest는 실행하지 않았다.
Ruff/서식/diff 검사와 독립 읽기전용 최종 검토를 마쳤다.

독립 검토로 찾고 수리한 세 항목은 다음과 같다.

1. Frozen 객체라고 필드의 유효성을 믿지 않고 중첩 RunKey/시간 구간을 fetch 전에 재검증.
2. 반환 특징과 비용에 clock/rate를 남겨 실제 trial onset과 대조.
3. 수집 시작 이전에 필요했던 context 시간을 baseline 비용에서 누락하지 않음.

이 검사들은 callable reader 내부의 임의 파일 읽기를 막는 OS 보안 장치가 아니다.
실제 device·export의 timing 선언이 참인지도 증명하지 않는다. **사람 파일 reader·Q/QM learner는
이 단계에서 구현/실행하지 않았다.** 새로운 sensor 처리 규칙을 기존 Choi cache에 적용하지 않았다.

## 예산 이탈을 포함한 실행 기록

사전 [설계](acquisition_boundary_generated_v1_design.md)와
[고정 config](../configs/analysis/acquisition_boundary_generated_v1.json)의 상한을 사후 수정하지 않았다.

| 검사 호출 | tests | process elapsed | 결과 |
|---|---:|---:|---|
| root-call1 | 9 | 0.17초 | PASS |
| root-call2 | 37 | 2.18초 | PASS, 아래 legacy 배열상한 이탈 |
| cost-call1 | 16 | 0.68초 | PASS |
| root-call3 | 56 | 2.01초 | PASS, 같은 legacy 배열상한 이탈 |
| root-call4 | 34 | 0.59초 | 최종 새 모듈 PASS |

5/10호출, process elapsed 합5.63초/1800초, 개별최대2.18초/180초였다.
별도 worktree13,240KiB, 테스트 산출물50,212KiB snapshot을 보존했다.
새 source/docs/DB 기록까지 포함할384MiB 예약을 늘리지 않았다.

**예산 준수 전체를 PASS라고 하지 않는다.** 기존 `test_prepare_choi2019.py`가 만드는
39×10,000=390,000원소 배열이 사전200,000원소 상한보다 컸다. 같은 fixture가 root-call2/3에
포함된 것을 뒤늦게 발견했다. 사람이 아닌 인공자료였고 시간/출력 상한은 남았지만 위반은 남긴다.
그 legacy 검사는 더 실행하지 않았고, 이후 새 fixture(최대6,000원소)만으로 비용 수리를 검증했다.
개별 메모리 할당 전체의 최대 원소 수를 계측했다고 주장하지 않는다.

초기 Ruff 스타일 오류는 테스트 전 고쳤고, 읽기전용 research context를 `head`에 연결한 호출은
출력 pipe 종료 오류가 있었다. 이것들은 사람 실험 실패나 데이터 원천 실패가 아니며 검색/학습은 없었다.

## 병렬 작업·학술 기록·다음 행동

`coordinate-worktree-changes`에 따라 공통 계약은 root가 먼저 동결하고,
cost만 별도 tree에서 구현했다. 겹치는 writer 파일은 없었다.
기존 Git의 `merge-tree`는 새 옵션을 지원하지 않아 구형 형식으로 다시 확인했으며,
텍스트 충돌 없음뿐 아니라 clock/unknown/baseline 의미를 root가 통합 검사했다.

Commits: 계약 `9e33c0e` → root temporal `6caf931` → cost branch `b630ff0`를 main `5e242c6`로 통합
→ pairing·baseline `c4942be`. 기존40+새1 worktree와8 untracked 이전 출력을 보존했다.
환경 설치/cleanup/push/GPU/외부발송/유료/사람파형·학습·outcome/held60 모두0.

`academic-research`에는 이번 범위와 한계를 qualified claim `1cf125821296466c`/5evidence로 저장하고,
Choi measurement gap `b654467d1d604de0`는 OPEN으로 갱신했다. 새 검색/PDF0,
cutoff2026-09-04 유지, render/SQLite quick_check 완료다.
스킬은 인공 공학 검증과 실제 연구 효능을 분리해 기록하는 데 사용했다.

**다음은 더 많은 인공 PASS를 쌓는 것이 아니라 실제 입력 근거를 좁게 확인하는 일이다.**
공개 acquisition/export 코드나 공식 문서에서 Gyro 대응과 marker/다운샘플 시간지지를 찾는
별도 유한 read-only 조사를 우선한다. 근거가 없으면 [미발송 질문](choi_aux_context_clarification_draft.md)의
발송 승인 등 필요한 방향을 요청해야 하며, 임의 guard로 unknown을 채우지 않는다.
그 뒤에만 새로운 motion 학습 후보·대조군·분할·실제 비용·fit/reveal 예산을 고정할 수 있다.
Choi 후보의 기존 DEFER와 모든 과거 부정 결과는 유지하며, 현재 넓은 연구 goal은 종료하지 않는다.

상세 SHA와 범위는 [기계 판독 상태](reports/acquisition_boundary_generated_v1_state.json)를 따른다.
