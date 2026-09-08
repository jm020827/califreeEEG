# Temporal Q/QM — 실제 학습기 연결·생성 배열 검증 결과

2026-09-09. **설계와 학습기 통합 검증 완료. 사람 EEG 효능 실험은 아직 실행하지 않았다.**
[고정 설계](task_trca_temporal_v1_design.md), [소유권·API 계약](task_trca_temporal_v1_build.md),
[기계 판독 요약](reports/task_trca_temporal_v1_engineering.json).

## 쉽게 말하면

연구 질문은 바꾸지 않았다. **EEG 자체에서 얻는 Q와 공통 결측·측정 순서 정보에 숫자 임피던스를
추가하면, 새 사용자가 제공해야 하는 정답 EEG 예시를 줄일 수 있는가?**

이번에는 부호에 의존하지 않는 계산 부품을 실제 Q15 학습기에 연결했다. Q의16계수를 먼저
학습·동결하고 QM은 임피던스2특징의 작은3계수 잔차만 학습한다. Q2와 SHAM도 같은 크기로
학습한다. Q/Q2/QM/SHAM에는 같은 새 점수식·규제 한도·학습 횟수를 적용한다.

기존 FULL_NATIVE는 원래 필터와 원래 점수 그대로다. FULL_CENTERED는 **바로 그 필터를
재학습·재정규화하지 않고 점수식만 변경**한다. 새 점수식 자체의 효과와 metadata의 추가 효과를
구분하기 위한 대조군이다. 보정량 감소 판정에는 두 FULL에 대한 기준을 모두 요구한다.
기존 A0 및 Q/Q2/SHAM 비교·비용·손해 기준을 완화하지 않았다.

## 완료한 구현

- 새 schema를 가진 학습기·batch·평가·독립 감사4개 모듈과 생성 실행/독립 cold 검산2개 CLI.
- 실제 Q15, 앞 k블록 M2, fit-only scaler, Q 동결 후 잔차,200step Adam, Q validation CE만으로
  lambda를 고르는 참가자 분리 nested 학습. 옛 global-score checkpoint는 새 모델로 읽지 않는다.
- 평가10arms: FULL_NATIVE/FULL_CENTERED/ISO/Q/Q2/QM/SHAM_REFIT/PERMUTED/STALE/MISSING.
  명시적으로 고정한 평가 ID·조건의 전체 조합을 검사해, 일부 사람을 빼 donor를 바꾸는 입력을 거절한다.
- Native 필터·템플릿은 변경 불가 배열이다. 학습 역할과 평가 역할, source/evaluation schema,
  packet prefix·결측·순서·donor·모델 저장/복원·failure 우선권을 검사한다.
- 독립 감사는 producer를 import하지 않고 NumPy/SciPy로 M·scaler·donor·R·F·점수·선택을 계산한다.
  새 FULL_CENTERED 비열등성 조건은 기존 calibration 후보를 낮출 수만 있고 실패를 성공으로 올리지 못한다.

## 고정 생성 실행에서 확인한 것

Runtime commit `e736a26e357adfac82207530387647c66dcfa6e1`.
고정 seed20260909를 한 번 실행했고, 결과에 따라 seed·난이도·학습 설정을 바꾸지 않았다.
학습용 가상6명과 별도 평가용 가상2명이다. 사람 ID나 EEG를 사용하지 않았다.

| 검증 | 실제 결과 | 의미·범위 |
|---|---|---|
| 실제4heads CPU/CUDA 각200steps | 계수 최대차2.38e−15, 점수2.06e−15, 네 head의 argmax 모두 동일 | 두 장치의 해당 생성 학습 결과 일치 |
| Scalar/batch loss·gradient | 최대차2.08e−17 /9.15e−20 | 같은 목적함수·미분 경로 |
| 생성6명·1조건 nested |10pipelines/40heads/8000updates,33.78초 | 실제 선택 그래프 연결; 사람39명 전체 조건 검증은 아님 |
| 독립 cold 학습 검산 |36개 validation CE 비교, 최대차1.18e−16 | 저장 Q/S/C부터 scaler·M·donor·lambda 선택 검증 |
| 독립 cold 평가 검산 |70개 배열 비교,20개 정수 count 비교,10arms argmax 정확히 일치 | 생성2명×10arms의 점수·정수 판정 정합성 |
| GPU 자원 | peak allocated31,300,608bytes; reserved52,428,800bytes | 이 생성 검사의 관측치, 사람 전체 실행 예측은 아님 |

CPU4head 적합5.77초,CUDA3.04초였지만 단회 작은 생성 사례이므로 일반 GPU 속도 우위를 주장하지 않는다.
CLI 학습량은 nested8000 + CPU parity800 + CUDA parity800 = **9600 생성 updates**다.
단위 테스트의 적합 계산은 이 CLI 합계에 포함하지 않는다. 생성 lambda=.01은 사람 실행에 넘기지 않는다.

평가48행은 **12개 생성 trial을4번 반복한 배열**이며48개의 독립 반복 측정이 아니다.
QM−Q의 R/F/J/score 변화는 두 가상 평가자 모두0이 아니었고 score 최대차는
2.50e−6/3.28e−6이었다. 둘 다 argmax 변화0이다. 전달 경로가 작동함을 확인했을 뿐,
metadata의 분류 효과·무효·보정량 절감 어느 것도 증명하지 않는다.

## 산출물과 재검산 경계

새128개를 포함한 전체 **2376tests PASS,406.08초**, 기존 Torch 경고68개다.
실행·테스트 commit은 위 e736a26이며 결과 문서 갱신 중 runtime 코드는 바꾸지 않았다.
전체 JUnit `/home/whwovy/task-trca-temporal-full-20260909.xml`의 SHA256은
`c417fbf647003fec6d20972beabceb4cdd062d151101710a6726538275d5f239`다.

보존 경로 `/home/whwovy/task-trca-temporal-engineering-dlDeIx/`.
start/parity/source/model/evaluation/receipt/cold_audit는0400이며 덮어쓰지 않았다.
Warm receipt의 `COLD_AUDIT_PENDING`은 당시 상태로 보존하고 별도 cold_audit가 PASS를 기록한다.

- 설계 JSON SHA256: `e456c3bcfc066e6ba3cb95a9c76798a006416bee11d064d083d9bd98ce7b97da`
- Warm receipt SHA256: `88894b4c6396c5fa7cbfa1555199938645049b1987d0be64b2430e250c56a0ee`
- Cold audit SHA256: `4469da60e1bac62c92346e3b9537e7ba01369d757a24309eee89be73faa6d8d5`

Cold 검사는 warm receipt hash, 산출물 이름·크기·hash,14개 code/config pins, 생성 ID·seed를 결합한다.
독립 reviewer도14개 pin과 저장 네 head의 CPU/CUDA 점수·argmax·200step trace를 재계산했다.
이는 **저장된 source Q/S/C부터의 감사**다. EEG에서 Q15를 만드는 전처리, native support fit,
Adam 전체 재생, 실제 사람 자료의 접근·완료 이력을 독립적으로 검증한 것은 아니다.

이번에는 source archive/raw EEG/실제 숫자 M/기존 실패 진단 JSON/최종 사람 query/held60를
열지 않았다. 기존 실패한 source39 후보의15개 code pins와 설계·실행 JSON은 그대로이며,
그 후보의 `VALIDITY_FAILURE`와 이전 metadata 부정 결과를 닫힌 기록으로 유지한다.

## 다음 할 일 — 실제 입력 연결을 별도로 준비

1. 새 schema/scorer에 맞는 역할 제한 archive reader와 실행 manifest를 구현·고정한다.
   대상39명·조건·support prefix·source block5·native weights·전처리·입력/code hash·자원 한도를 묶는다.
2. 먼저 생성 archive로 완료 경로를 검증한다. 모든 모델의 동결 기록과 정확한 모델/산출물 hash,
   read event 전체 조합, 정수 count/최종 판정까지 독립 cold 감사가 직접 결합해야 한다.
   실패 기록은 성공 receipt보다 우선하며, 인메모리 API 검사를 OS sandbox로 부르지 않는다.
3. 그 준비와 실행 범위를 확인한 뒤 별도로 source39 단일 개발 실험을 진행할 수 있다.
   실패하면 예외/대조군/threshold를 바꿔 자동 재시도하지 않는다. held60는 자동 개봉하지 않는다.

즉 다음 병목은 새 문헌 수나 GPU 설치가 아니라 **실제 입력부터 최종 판정까지의 실행 계약·감사 연결**이다.
현재 자료는 좁은 개발 평가에 쓸 수 있지만 반복 노출39명을 독립 확증 표본으로 바꾸지는 않는다.
추가 paired-acquisition 자료는 일반화·독립 확인의 별도 보강 경로이며 이번 구현의 필수 선행조건은 아니다.

`academic-research`의 기존 landscape/frontier를 재사용하고 구현 측정과 효능 미확인을 별도 claim/evidence로
기록한다. 새 broad search/PDF 추가0이며 논문 수를 검증 강도로 대신하지 않는다.
`coordinate-worktree-changes`에 따라 learner와 auditor를 별도 worktree에서 작성하고 main에서 순차 통합했다.
Root가 공통 계약·GPU·SQLite·최종 검증을 소유했고 읽기 전용 reviewer가 별도 검토했다.
기존36개+신규2개 worktree 및 다른 GPU process를 보존했으며 설치·push·cleanup은 하지 않았다.
