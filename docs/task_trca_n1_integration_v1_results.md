# N1 인공 학습기 통합 v1 — 결과와 다음 검증

2026-09-09. **인공 학습기 통합 검증을 완료했다.** 등록 생성·primary·독립 cold 각1회가
통과했다. Metadata가 학습되는 경로는 작동하지만, 실제 EEG 효능·보정량 절감은 미평가다.

## 무엇을 확인하는 단계인가

원래 질문은 그대로다. **외부 acquisition metadata가 EEG 자체에서 얻는 정보 Q와 공통 정보에
더해 도움이 되어, 같은 SSVEP 분류 성능에 필요한 보정 데이터를 줄이는가?** 이번 단계는
그 질문에 답할 학습기가 끝까지 정상 작동하는지 확인하는 공학 검증이다. 사람 EEG 효능 실험은 아니다.

채널별 impedance의 요약값 M을 Q와 함께 학습기에 넣고, 채널별 규제 강도 R을 정한다.
R이 EEG 공간 필터와 분류 점수에 영향을 주며, 분류 손실의 미분이 다시 M을 쓰는 계수로
전달된다. N1은 이 과정의 일반화고유값 계산·미분을 담당하는 새 수치 구현이다.
기존 temporal-R 가설을 유지하며, C2의 pair-S 가설로 바꾸지 않았다.

Q만 학습한 부분은 먼저 동결한다. 이후 EEG 특징을 추가하는 Q2, 올바르게 대응시킨 생성 metadata를
추가하는 QM, 대응을 섞어 다시 학습하는 SHAM을 같은 잔차 용량으로 비교한다.
설정 선택에는 Q-only 검증 손실만 사용한다. Metadata가 유리해 보이는 설정을 고르는 절차가 아니다.

## 사전 고정과 실제 실행

- [설계/예산/소유권](task_trca_n1_integration_v1_build.md), [기계 계약](../configs/analysis/task_trca_n1_integration_v1.json).
- 생성9개 routing ID, 두 interface, N17, k3/5. ID는 사람 관측값이 아니라 생성자료의 표지다.
- seed20260913. 입력은 한 번만 만들며 인공 M과 EEG 사이에 효능 관계를 심지 않았다.
- outer3 × (inner3 × lambda3 + final1) = 30 pipelines; 네 head × 200 step = 24,000 updates.
- 세 최종 모델을 모두 저장·동결한 뒤 36개 평가 조건/10arms를 계산한다.
- 전체72 query-bearing reader calls/325 journal rows를 검사한다. 이는 원 role-limited reader 호출 기록이지 모든 OS 파일 접근의 완전한 로그는 아니다.
- A0는 생성 correlation 자리표시자다. 그 정확도로 실제 zero-calibration 성능이나 label saving을 주장하지 않는다.

실행 구현은 `2126fe264a7de4a10a1604715b1b15f8022d37b9`다. 실제 runtime은 전체 검사를
통과한 f73ee0d와 같고 이후 변경은 사전 검사 기록뿐이다. 등록 생성 전 science와31개
code pin을 고정했다. 입력·실험 parent는 `/home/whwovy/task-trca-n1-integration-ZeJC9H`다.
Manifest SHA는 `b5dd50fa817092523c3557f26e4b196365b75b8900644c8926015d914fa6a414`다.

## 검사 결과

[사전 검사 기록](reports/task_trca_n1_integration_v1_preflight.json): 새 통합298PASS,
전체3300PASS/68기존warnings. 전체 pytest420.06초/process421.28초다. 등록 전 toy의
실패 및 수정 이력도 일지와 원 JUnit에 보존했다. 실패 기대값을 고친 toy를 성공한 첫 실행으로 세지 않는다.

등록 primary는 24,000updates를 모두 수행했고, 592.3815초/process593.87초에 완료했다.
별도 cold는 4.7039초/process5.14초에 `GENERATED_COLD_INDEPENDENT_AUDIT_PASS`를 발행했다.
생성 과정은 .5178초/process.89초다. 등록 실패·재시도·seed/방법/허용오차 수정0이다.

| 학습 경로 | 30개 기록 중 최대 gradient norm | 최대 최종 계수 절댓값 | 연결 검사 |
|---|---:|---:|---|
| Q | 7.9230e−5 | .822205 | PASS |
| Q2 | 8.5984e−5 | .183656 | PASS |
| QM | 9.1047e−5 | .090511 | PASS |
| SHAM_REFIT | 9.1990e−5 | .101047 | PASS |

총30pipelines/120heads/200steps가 모두 기록되어 있다. 세 모델 모두 query 전 동결되었고,
전체325접근행·72query-bearing calls·92진행event가 일치했다. 세 outer fold 모두 Q-only
lambda1e−4를 선택했으며 독립 CE 재계산 최대오차는9.7145e−17이었다.

36평가조건의 전체10arms에서 argmax와 정수 정답 수가 정확히 일치했다. 새 점수 최대절대오차는
1.7764e−15, FULL_NATIVE는3.5528e−15였다. MISSING=Q도 정확히 유지됐다. 이것은 수치 재현
결과이지, 인공 QM의 성능 우위나 실제 label saving 검정 결과가 아니다.

원 [cold receipt](/home/whwovy/task-trca-n1-integration-ZeJC9H/task-trca-n1-integration-primary1/cold_audit.json)의
SHA는 `25107e8586c94e393db61a7f87f617025cda04c7fa9d029913ca97def7440ee9`다.
Primary result SHA는 `6f6a55e234afb3d549433d533e35e6f0d683dbfff4be1eae3d37f4623bcb862d`,
생성 전 registration SHA는 `b3286df1707d531762172107d99c98dc4f31306299dfcd9ddc3cf10b63aeca09`다.
등록 시각 자체의 외부 인증을 주장하지는 않는다. 생성 전에 기록하도록 고정한 코드·commit과
파일 byte binding이 본 workflow 근거다. Result의 `COLD_AUDIT_PENDING`은 감사 전 producer
원본이므로 덮어쓰지 않았으며, 별도 성공 receipt가 최종 감사 근거다.

입력+실험 디렉터리는182.164MiB(du 할당량), 기록한 모든 실험·test 디렉터리 합계는671.309MiB다.
새 source/docs/연구 DB 증분을 위한2GiB 이상 여유가 남는다. 한도인 입력+실험2GiB/전체3GiB,
primary7200초/cold1800초/tests총1800초 안이다. CPU만 사용했고 peak CUDA allocation0,
사람 자료·held60·외부 요청·유료·새 worktree·dependency 설치0이다.

## 해석의 경계

Head 연결 검사는 각 Q/Q2/QM/SHAM **종류별 전체30 pipeline 중** gradient norm과 최종
계수 크기의 최대값이 고정 하한1e−12를 넘는지 확인한다. 모든120 head가 각각 비영이라는
뜻도, M이 유용하다는 뜻도 아니다. 실제 성능 이득은 다른 문제다.
Q2 역시 잔차 용량을 맞춘 대조군이지 EEG가 담을 수 있는 모든 정보를 대변하는 대조군은 아니다.

독립 cold는 저장 Q/S/C/M/Gram/scaler/donor/모델 이후의 NumPy/SciPy 계산으로 모든
대조군·선택·점수·정확한 argmax/counts·동결/권한/파일 결합을 검산한다. Adam 궤적이나 raw
전처리/Q15를 처음부터 별도 구현해 재학습하는 검증은 아니다. 수치 독립성과 미노출 데이터의
독립 확인을 혼동하지 않는다. 본 기록은 재현 가능한 workflow provenance이지 OS sandbox 증명은 아니다.

기존 C1/C2의 종료와 모든 유효한 부정 효능 결과는 그대로 보존한다. 이번 별도 단계의 성공이
기존 실패를 지우거나 원 연구목표를 완료시키지 않는다. 사람 입력·held60·외부 요청·유료·GPU는 범위 밖이다.

## 다음 제안 — 아직 승인·실행하지 않음

최소 다음 범위는 새 후보 `task-trca-n1-source39-v1` 하나의 개발 효능 검증이다. 기존 source39,
두 interface, 기존 네 시간창, k3/k5를 쓰되 별도 사람용 reader/manifest/입력권한/동결 형식을
구현하고 사전 검증해야 한다. 현재 생성 전용 reader를 사람 자료에 그대로 적용할 수 없다.

새 primary1회·새 최종 모델세트 query reveal1회·24,000updates, 실패 시 자동 재실행/N2
전환0을 제안한다. 실제 CPU 시간/메모리/출력 한도는 사람 입력을 열기 전에 full-shape 생성
리허설의 별도 예산과 함께 정해야 한다. 이번 작은 N17 검증으로 실자료 규모의 실행성을 보장하지 않는다.

기존 판단 기준을 유지한다: QM3−Q3 평균≥1pp 및 기술통계용 CI 하한>0, Q2/SHAM 대비 하한>0,
SHAM 특징 변화 coverage≥50%. 보정량 후보에는 QM3 평균≥80%, Q5/A0/FULL_NATIVE3/
FULL_CENTERED3 대비 하한>−1pp, harmed≤helped, 실제 관측한 보정 격자에서 양의 pooled
label saving 및 새 도달≥도달 상실을 추가로 요구한다. 비용은0/36/60labels, 도달은≥39/48이며
미도달을 절감으로 세지 않는다. 생성 A0 대신 원래 고정한 진짜 native A0를 써야 한다.

Source39는 반복해서 노출된 개발자료다. 이 단계가 긍정적이어도 확인적 통계나 독립 재현은 아니다.
held60·외부 요청·유료 사용은 계속 별도 승인이 필요하다.

`academic-research`로 기존 근거를 재사용하고 수치/통합/효능 주장을 분리했다. 새 broad search나
PDF 추가 대신 현재 불확실성에 직접 답하는 고정 실행을 택했다. `coordinate-worktree-changes`로
두 격리 구현 lane과 root 단독 통합/실행/연구 DB 기록을 적용했다.
새 claim `32189cbc583e2dac`(qualified/3evidence), technique `69668107e8ee3d89`를 추가하고,
원 효능 gap `6a8254f849b36b2f`는 open으로 유지했다. 기존 claim/technique의 실패 이력은
덮어쓰지 않았다. 누적51claims/114evidence/14techniques, render 및 SQLite quick_check=ok다.
이번 새 검색/PDF0이며 기존 문헌 cutoff2026-09-04를 최신 완전성이라고 주장하지 않는다.
