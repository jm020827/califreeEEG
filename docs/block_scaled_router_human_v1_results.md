# 실제 EEG 후속 실험: 학습 경로는 작동, metadata 이득은 미확립

2026-09-11 · 실행 동결 `eb6c272` · 단회 완료, 재시도 없음.

## 결론

**이 gyro 기반 source-routing 후보는 종료한다.** 학습 안정화를 적용하자
메타데이터가 다른 사람의 EEG 모델을 빌리는 비율과 예측을 실제로 바꾸었다.
하지만 Q만 사용하는 비교군이나 추가 EEG 특징 Q2보다 저보정 성능이 좋지
않았고, 실제 획득해야 하는 보정 trial 수는 줄지 않았다.

이는 “metadata는 어디에도 쓸모없다”가 아니다. 현재16명의 공개 개발 데이터,
현재 support gyro 특징·차용 구조·유한 학습 조건에서 목표 이득을 입증하지
못했다는 결과다. 같은 사람들에게 학습률/epoch를 계속 바꾸는 구조 구제는
사전 중단 기준에 따라 끝낸다. 원래 저보정 SSVEP 연구 목표는 유지한다.

## 무엇을 바꿨나

기존 데이터·support/query·source bank·Q/Q2/M/SHAM·보정량 정책은 그대로다.
59개 특징 묶음 등이 logit을 과하게 지배하지 않도록 표준화 **후** 블록별
스케일을 적용하고, source 학습 손실이 늘거나 logit이 지나치게 움직이는
업데이트는 축소/거부하는 SAFE 학습기만 바꿨다.

M은 support 구간에서 측정한 gyro 두 특징이다. 자극 주파수·고조파는 모든
비교군의 공통 정보다. Gyro를 display 지연이나 뇌 반응 위상으로 해석하지
않았으며 Neural ODE나 새 파형 특징을 추가하지 않았다.

## 사전에 고정한 비교 결과

수치는 참가자별로 속도·k를 평균한 balanced accuracy다. 저보정은 k1/2,
고보정은 k3/5이며 k는 class당 선택 support 수다.

| 모델 | 저보정 정확도 | 고보정 정확도 | source가 선택한 정책 정확도 |
|---|---:|---:|---:|
| Q + 공통 정보 | 68.771% | 69.342% | 70.551% |
| Q + 추가 EEG 특징 Q2 | 68.973% | 69.400% | 70.782% |
| Q + 실제 gyro M | 68.608% | 69.501% | 70.782% |
| Q + 대응을 바꾼 gyro SHAM | 68.461% | 69.516% | 70.695% |
| 자기 support 모델만 사용 | 67.855% | — | 70.580% (QM의 k에서) |

저보정 QM−Q는 **−0.162 percentage points**, 참가자 paired bootstrap95%구간
**[−0.585,+0.304]pp**다. QM−Q2는−0.365pp, QM−SHAM은+0.148pp지만
구간[−0.721,+1.056]pp로 불확실하다. 자기 모델보다 평균+0.753pp 높은 점은
Q만 사용한 차용도 개선된 것과 구분해야 한다. Metadata 고유 이득은 아니다.

정책의 QM−Q는+0.231pp,[0,+0.579]pp지만 Q2와는 수치정밀도 수준에서 같은
70.782%다. 이것을 독립적인 유의한 발견이라고 선택적으로 강조하지 않는다.
모든 arm/fold의 sourceOOF가80%에 못 미쳐 **모두 k5 fallback**을 선택했다.
평균 chronological acquisition prefix는 전부 **21.854 trials**, ready-time
proxy는203.261초이며 **보정량 절감0%**다. QM의80% 이상 run 비율은43.75%로,
모든 사용자가80%에 도달했다는 뜻이 아니다.

사전 전체 retention 조건 중 실패한 항목은 다음4개다.

- 저보정 QM−Q>=2pp: 실패.
- 저보정 QM>Q2: 실패.
- 선택 정책 QM 평균>=80%: 실패.
- 평균 실제 획득 prefix 절감>=10%: 실패.

그 밖의 안전성·SHAM 평균 대비 조건과 M 작동 조건은 통과했다. 전체 AND
기준은 실패다. Q 비용 역시80% 도달을 관찰한 비용이 아니라 미달 fallback
비용이다. 이 실험에서 전체 setup 시간/첫 온라인 결정 지연 절감은 측정하지
않았다. 고정 후기 query를 사용한 개발 평가이며 개인별 온라인 stopping이 아니다.

## 수치 문제가 해결됐다는 근거와 그 한계

36개 QM source 학습에서 모두 M-only 개입이 gate와 class-margin을 바꿨다.
각 fit의 최대 gate 변화 범위는0.000511–0.831816, 최대 margin 변화는
0.000260–0.462072다. 원래144-fit의 gate 포화/score 무변화와 다른 결과다.
QM 학습 episode의 평균 borrowed-source mass는 fit별0.00303–0.59425로,
모든 비율이 자기 모델1에 고정되지 않았다. 이는 fit내 최대/평균 진단이며
모든 trial 또는 모든 사람에게 유용한 변화가 생겼다는 뜻은 아니다.

144개 학습 모두 source loss가 초기보다 감소했다.14,400 proposals 중
14,038회 전체 수락,309회 축소 수락,53회 거부/parameter 복구가 발생했다.
최대 완료 step 손실 증가0, 최대 centered-logit step0.490787이었다.
이는 안정화 규칙의 실행 증거이지 최적해·일반화·optimizer 보편 우월성 증거가
아니다. 블록 스케일과 safeguards를 함께 바꿨으므로 각각의 인과 효과도
이 사람 실험만으로 분리하지 않는다.

## 실행·검증·보존

- [고정 계약](block_scaled_router_human_v1_contract.md)과
  [설정](../configs/analysis/block_scaled_router_human_v1.json)을 cache 접근 전 commit.
- 생성 시험3/3 호출 소진: 첫 호출은 역할 행 수 오류로0fit에서 실패,
  이후6/6·7/7 통과. 실제 학습은 합계288 generated fits/576 proposals.
  실패도 [시험 기록](reports/block_scaled_router_human_v1_mock_tests.json)에 보존.
- 사람 단회144fits/14,400proposals/576ridge solves,26.0845초(report 쓰기 전).
  후보검사/후보 loss 평가 각15,414, 완료 ordinary loss 평가28,800.
  cache bytes checksum/load 각각1회, raw extraction0, 총 producer15,347,140B.
- 사전72개 context-role/1,344행 역할 ledger,96 inner fit 이후 정책 원자적
  잠금, 각4개 head 동결 이후 query 평가. 이전 역할 보고 덮어쓰기 없음.
- [실행 관측](reports/block_scaled_router_human_v1_report.json),
  [정책](reports/block_scaled_router_human_v1_choices.json),
  [역할](reports/block_scaled_router_human_v1_roles.json),
  [journal](reports/block_scaled_router_human_v1_journal.jsonl) 보존.
  무거운 model/NPZ는 설정에 명시한 로컬 전용 경로에 저장했다.

독립 [수치 감사](reports/block_scaled_router_human_v1_numeric_audit.json)는
116,699검사/실패0으로 통과했다. 저장 NPZ1회 로드,132 saved-head forward,
계산1.035초이며 재학습·ridge 재풀이·원 cache 접근은0이다.
[프로토콜 감사](reports/block_scaled_router_human_v1_protocol_audit.json)도
역할1,344행·12정책·동결 순서·보호된 과거 결과를 검증해 통과했다.
생성 full-flow는
`execute()`의 학습/평가 순서를 검증했으며, 사람 `run()`의 cache 읽기부터
모든 파일 게시까지를 생성자료로 end-to-end 시험했다고 주장하지 않는다.
`cache_checksum_passes`는 구현상 검증 시도 전에 증가하는 legacy 이름이므로
실패 실행에서는 성공 횟수로 읽으면 안 된다. 이번 COMPLETE에서는 실제
해시 일치와 동일 bytes load가 모두 완료됐다. Full repository suite는 미실행.

## 다음 연구 방향의 경계

이 후보를 새 이름으로 반복하지 않는다. 이번까지의 근거는 “M을 학습에
넣을 수 있느냐”보다 **“측정한 M이 Q로 설명되지 않는, 보정에 유용한 차이를
정말 알려주는가”**가 다음 후보의 핵심임을 보여준다.

다음 후보를 열려면 공개적으로 얻을 수 있는 실제 acquisition 측정과 decoder
작동 경로의 연결부터 확인한다. 예컨대 측정된 전극/참조/실제 자극 타이밍과
적합한 관측·정렬 연산의 연결은 검토할 수 있으나, 아직 실행 승인용 계약이나
효능 증거가 아니다. 현재 gyro를 그 측정의 대용으로 이름만 바꾸지 않는다.
새 후보군·예산·중단 기준을 먼저 기록한 별도 단계로 간다. 유망 후보가 없으면
부정 결과 정리 자체가 이번 루프의 종료 결과다. held60·외부 요청·유료는 계속
별도 승인 대상이며 Choi 비공개 경로는 보류한다.
