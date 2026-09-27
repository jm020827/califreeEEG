# Joint harmonic v1 — 승인된 저장 결과 검산 보완 r1

2026-09-23 KST. 사용자의 직전 제안에 대한 `응 계속하자.`를 검산 코드 수정1회와
저장 결과 재검산1회에 대한 승인으로 기록한다. 과학 후보·모델·학습·튜닝의 재개가 아니다.
원 계획/config/실제 weights/predictions/result 및 실패 audit는 변경하지 않는다.

## 고정 범위

- 코드 보완1회: 독립 auditor와 그 수치 회귀 시험만. 생산자/학습기/reader 수정0.
- 생성 수치 시험 최대2호출/합계120초, optimizer update0. 첫 호출은 인공 배열의
  메모리 배치·float32 집계·NPZ 왕복 문제 재현, 두 번째는 보완 회귀 검증이다.
- 실제 feature cache 추가 load1회, 저장 결과 감사1회/최대600초.
  원 raw/manifest/held60 재접근0, 새 학습0, 기존 결과 검산 밖의 새 예측0.
- 새 자료/사람 연락/외부 요청/유료0. 최종 query를 이용한 모델·정책 변경0.
- 원 `audit_start.json`/`audit.json`을 덮어쓰지 않고 `audit_r1_start.json`/
  `audit_r1.json`에 독립 기록한다. 고정 이름의 exclusive 생성으로 추가 시도를 차단한다.
- 인공 재현 후 실제 재검산 전에 비교 정책을 아래에 기록한다. 숫자가 통과할 때까지
  허용오차를 반복 조절하지 않는다. 이번 검산도 실패하면 실패/한계를 보존하고 멈춘다.

대상: `/home/whwovy/eeg-data/joint-harmonic-human-v1-0ioz9_2y`.
원 result SHA `a02ab289da364c10a2d2b6bf412237cc653439cc8dc7accce66d2a2b3066b664`.
원 실패 audit SHA `0ae4beeb1037345892da3b2d723bce63d0a8130ca1fc6bc6f6b8ed8305c7bf7f`.
원 auditor SHA `e8b37ff1f1db4b75046504501e963f6c1161cbe816a298b8cd0085dead0b7115`.

## 수정 전 가설과 중단 조건

생산자는 추출 후 메모리에 있는 Q/Q2를 float32로 집계하지만 감사자는 NPZ에서 복원한
배열을 집계한다. NPZ는 값은 보존해도 임의의 strides는 보존하지 않으므로 집계 순서가
달라질 수 있다. 아직 실제 불일치 원인의 확정은 아니다. 인공 재현에서 원 값/배치/
집계차를 분리하고 수치 비교 정책을 정한다. 적절한 보완을 생성 단계에서 검증하지
못하면 실제 감사는 실행하지 않는다.

## 생성 진단 뒤 고정한 수치 정책 — 실제 재감사 전

첫 생성 호출(1PASS, 외부1.06초, 새 학습0)에서 MAT/F 경로의 Q prefix 평균이 NPZ 왕복
후 최대3.5762786865234375e-7, Q2는1.1920928955078125e-7 달라졌다. 저장 전후 원소는
정확히 동일했다. C 경로에서는 그 진단의 평균 차이가0이었다. 원 generated suite의
C-contiguous raw 생성만으로는 이 배치 문제를 검출하지 못했다.

MAT/F 추출 Q/Q2의 물리 순서는 person/class/block/interface/channel/feature였고,
논리 축은 person/interface/block/class/channel/feature다. 실제 재감사에서는 Q/Q2를
고정 permutation `(0,3,2,1,4,5)` → contiguous copy → 같은 permutation으로 복원한다.
이 복원은 값/클래스/사람/블록 순서를 바꾸지 않고 strides와 float32 집계 순서만 되돌린다.
`_load_arrays`의 scipy MAT 경로 및 `_wearable_data`의 축 보존 코드가 이 가설의 근거다.
실제 원본 strides가 영수증에 없으므로, 이것을 미리 확정된 실제 원인으로 단정하지 않는다.
한 번의 실제 감사에서 모든48개 저장 normalizer와 비교하고, 맞지 않아도 다른 배치를
시도하거나 허용오차를 늘리지 않는다.

Normalizer `rtol=atol=1e-10`, 저장 context `rtol=1e-6/atol=1e-5`, 독립 NumPy logits
최대절대차1e-4를 **변경하지 않는다**. Argmax 차이0을 명시적 필수 조건으로 강화한다.
48 source normalizer,24 final모델의 contexts/forward, source validation12개 선택,
저보정/조건별 정확도·개인별 차이·seed·bootstrap·비용·중단 판정을 검산한다.
Baseline은 저장 score에서 보고 통계를 재계산한다. Raw preprocessing/훈련/CCA forward
독립 재현은 이 검산 범위가 아니다. 새 판단에 맞춰 과학 threshold를 수정하지 않는다.

원 auditor 소스도 `docs/reports/joint_harmonic_audit_v1_source.txt`에 같은 SHA로 보존했다.
두 번째 생성 호출은 위 배치/normalizer 회귀와 기존 generated checkpoint의 재검산만
수행한다. 새 generated fitting도0이며, 통과해야 실제 저장 결과 재감사1회를 실행한다.
