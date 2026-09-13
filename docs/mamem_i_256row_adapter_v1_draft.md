# MAMEM I 공통 256행 입력 격리 관문 — 후속 초안

2026-09-13. DRAFT_NOT_EXECUTED. [scalar 관문](mamem_i_scalar_adapter_v1_results.md)
이후의 별도 실행 계획을 위한 명세다. 만료된 reader의 scope를 넓히지 않는다.

## 목적과 허용하는 해석

원본 row257의 물리적 정체를 해결하는 대신, 그것을 쓰지 않는 공통 입력 정책의
안전성을 검증한다. 처음 256행은 원래 순번만 유지한다. E1–E256은 명목상 이름이며
실제 전극 좌표·후두부 매핑·reference가 검증됐다는 주장이 아니다. 결과는 이 정책
아래의 단일 자극 frequency-decoding 개발 분석으로 제한한다. 동시 다중표적 BCI
및 online ITR 성능으로 확대하지 않는다.

## 다음 실행 계약에 고정할 것

1. Adapter 하나, 원래 rows[0:256]을 모든 QC/정규화/특징 전에 복사한다. Row257은
   label 생성에도 쓰지 않는다. DIN label 생성은 predictor와 별도 인터페이스다.
   무단 montage/re-reference/resampling/단위 변환은 하지 않는다.
2. 생성 자료로 row257의 label/NaN/큰 값 변경과 승인 시간 밖의 변경이 출력에
   전혀 영향을 주지 않는지 확인한다. Shape/order/time bounds/alias/input-key 검사를
   포함하고 실제 reader 연결 전에 독립 검토를 한다.
3. 실제 확인을 한다면 이미 개발용인 S001a의 고정된 최소 구간 1개만 지정한다.
   구간의 시작/끝·선정 이유·검사 수치·메모리/시간/출력/시도 상한을 값 읽기 전에
   별도 manifest로 고정한다. 무조건 파일 첫 구간을 SSVEP라고 부르거나 결과를 보고
   새 구간으로 교체하지 않는다. 초기 목적은 입력 integrity이지 효능 평가가 아니다.
4. SciPy가 `eeg` 전체를 decode하면 이를 명시한다. ‘제외 행/시간 bytes를 읽지 않음’
   대신 ‘선택 이후 계산·출력에서 제외’를 보장한다. 작업자원 상한/단회 영수증 적용.
5. 원본 unit/reference의 미확인은 기록하되, geometry-free·동일 reference 공통정책
   분석과 생리학적 위치/절대전위 해석을 구분한다. 이 선택의 강건성 검증은 후속이다.

## 학습 루프로 연결하는 조건

- 다음 실제 학습 후보는 source/pre-query support의 **기록된 event 규칙성**이
  같은 support EEG의 추정 불확실성을 Q 이상으로 설명하는지 묻는다. 물리적
  jitter를 복원했다는 가설은 아니다. 이번 250Hz 일치만으로 후보가 성립하지 않는다.
- 알고 있는 class/nominal schedule과 공통 deterministic correction은 모든 arm에
  제공한다. M에서 단순 주파수·고정순서·절대시간을 우회 복구하지 않게 설계한다.
  Query DIN은 label 작성에는 쓸 수 있어도 predictor/선택기에는 전달하지 않는다.
- k1에서도 실제 연산이 달라져야 한다. 예컨대 source 대비 support 추정의 축별
  shrinkage를 조건화하는 방향은 검토 가능하지만, 정규화 trial scalar weighting은
  한 trial에서 상쇄되므로 재사용하지 않는다. 구체 수식·학습 표적은 아직 미동결이다.
- Q/Q2/QM/SHAM에 같은 EEG행·support/query·class/common정보·획득비용을 제공한다.
  S001은 독립 효능에서 제외한다. 성능으로 후보를 수정하기 전에 fit 예산·효능/중단
  기준을 동결한다. 기존 gyro/source39 실패 구제나 held60 개봉은 이 초안에 없다.

의미가 불명확한 DIN descriptor/원본 257행을 더 열어 추측하는 것은 다음 할 일이
아니다. 안전한 입력 격리와 관측 M의 명확한 정의가 더 직접적인 다음 단계다.
