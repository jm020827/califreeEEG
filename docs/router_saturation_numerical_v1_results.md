# 수치 진단 완료: 유용한 보조정보도 self 포화에 막힐 수 있다

2026-09-11. **12/12 fits·1200 updates·1212 states, 단회 종료. 새 사람 실험0.**
계약7a1adc0 → 정적 수학/코드 검토 → 구현동결bc12b00 → 실제 실행의 순서다.
기록된 수치 시간1.126612초,CPU float64 한 thread였다.

## 무엇을 물었는가

“정답을 이미 아는 간단한 문제에서도 선택기가 잘못 자기 판독기만 고르는가?”를 검사했다.
13개 expert 중 source12개는 서로 같으므로 **self 대 source 총량**의 시험이지 donor12개
구별 능력의 검증은 아니다. 사람의59개 실제 특징을 복제한 것이 아니라 동일 self 특징을
인위적으로59번 복사했다. 정보와 표현 가능한 함수는 같지만 optimizer의 좌표·스케일이 달라진다.

## 모든 결과

각 칸은 최종 self 비중이다. ONE=한 열, SUM59=표준화한59열 합산,
MEAN59=동일 블록을 표준화 **후** /59. QM의 M열은 평균화하지 않았다.

| 인공 조건 | 해석적 최적값 | ONE | SUM59 | MEAN59 |
|---|---:|---:|---:|---:|
| 섞는 것이 최선 | .266667 | .266461 ✓ | 1.0 ✗ | .266462 ✓ |
| self만 쓰는 것이 최선 | 1.0 극한 | .992362 ✗ | 1.0 ✓ | .992362 ✗ |
| 상황별 차이 있음, Q만 | .5 | .499121 ✓ | 1.0 ✗ | .499121 ✓ |
| 상황별 차이 있음, QM | .266667 / .733333 | .270344 / .735595 ✓ | 1.0 / 1.0 ✗ | .270344 / .735595 ✓ |

✓는 사전 기준인 NLL 최적값 차이≤.001 **및** self 비중 오차≤.02를 뜻한다.
총7/12개가 수렴 기준을 만족했다. 경계 조건에서 ONE/MEAN59는 올바른 방향으로 개선했지만
NLL 차이.004519라 고정100updates 내 엄격한 기준에 못 미쳤다. 이 실패도 보존하고 더 돌리지 않았다.

섞어야 하는 조건에서 SUM59의 NLL은 .553525→.857399로 악화, 최적 NLL 대비+.319951였다.
Self−개별 source gap은62.4223,source 총량9.32231e−27이었다. ONE/MEAN59는 최적 NLL 차이
약1.85e−8에 도달했다. 이것은 **잘못된 self 포화가 발생할 수 있다는 통제된 예**다.

상황정보가 인위적으로 유용한 조건에서 QM ONE/MEAN59의 NLL은 약.537452로 Q의.561965보다
낮았다. M만 뒤집으면 gate 최대변화.465251, class margin 변화.558301이었다. 반면 SUM59는
M gate변화약8.03e−28, margin변화0이었다. 모든 조건에서 source 가중치의 정확한0은 없었으며
작은 양수가 최종 혼합에서 흡수되는 경우를 정확한 underflow와 구분한다.

## 결정과 해석 경계

- **유지:** 특징 블록 스케일/optimizer 좌표 의존성은 다음 구현에서 다룰 구체적 가설이다.
- **확인되지 않음:** 실제 사람 실험이 이 원인 하나로 실패했다는 인과 설명, gyro의 유용성,
  보정량 감소, 일반적으로 /59가 최선이라는 주장. 실제59열은 중복열이 아니다.
- **종료:** 이번12-fit 예산. 추가 lr/seed/step 탐색0; 기존144/336-fit 후보 재개0.
- **다음:** 작고 scale-controlled인 실제 형상 router를 먼저 generated 통합 검증한다.
  새 사람 효능 후보는 별도 고정계약/대조군/예산 뒤 판단한다. Neural ODE 도입은 보류한다.

같은 형식의 함수라도 AdamW의 좌표계와 decoupled weight decay, epsilon 영향이 달라지므로
엄밀히 동일한 parameter-space 정규화 문제라고 주장하지 않는다. 저장 gap derivative는
float64 autograd가 낸 값이지 포화에서도 정확한 실수 해석 미분을 보장하는 oracle가 아니다.
인공 CONTEXT M는 획득 센서 측정이 아니며 인공자료 내 NLL 차이는 실제 정확도 이득이 아니다.

## 재현 기록과 검증

[사전계약](router_saturation_numerical_v1_contract.md),
[실행기](../scripts/analysis/run_router_saturation_numerical_v1.py),
[결과](reports/router_saturation_numerical_v1/result.json),
[전체 trajectory](reports/router_saturation_numerical_v1/trajectory.jsonl),
[시작·환경·hash](reports/router_saturation_numerical_v1/start.json).
ResultSHA `4a8e0178ab789246794e9f84bfe12193d5b720ff327e8808483456c66467fd06`,
journalSHA `31e3048a51472d49928fb2e75a41859dbfaa7b864664171e3a9410fa149a8c49`.

Ruff/compile 통과; 초기 import 정렬 오류를 실행 전에 수정했다. 전용 fixture의 균형 NLL와
plain NLL 동치, ONE/SUM59의 생산 Router 동치, 해석 optimum 하한을 실행 중 검사했다.
최대 보고 fixture28000B이며 이는 fixture 텐서량이지 Torch process RSS가 아니다.
전체 repo suite·별도 optimizer 반복·새 패키지/GPU0.
[독립 검산](reports/router_saturation_numerical_v1_independent_audit.json)은8741checks/불일치0,
수치.138556초였다. 저장self로NLL·gap/weight·최종모델M개입을재구성했으며생산helper를
import하거나optimizer를재실행하지않았다. 감사식첫구문오류는수치실행전발생했고보존했다.
이감사는실행counter/trajectory검사이며실제optimizer재실행·원인에대한사람인과검증이아니다.
Choi 보류·held60 미개봉·외부 발송/유료/새 raw/cache 사람자료 읽기0이다.
