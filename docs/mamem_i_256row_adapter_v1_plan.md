# MAMEM I 공통 256행 입력 관문 v1 — 실행 계획

2026-09-13, base c62ea22/main. Active goal ‘계속 연구’의 다음 유한 입력 관문이다.
직전 scalar 시도는 완료됐으며 이번 별도 계약이 그 reader의 재실행은 아니다.
원 저보정 SSVEP 목표·기존 부정 결과·held60 별도 승인 경계를 유지한다.

## Signature와 이번에 줄일 불확실성

Representation: 원본256행 EEG의 고정2초 구간. Bottleneck: 모호한257행 및 선택
시간 밖 값이 predictor용 처리에 유입되는가. Operation: 먼저 copy/slice한 뒤
geometry-free 입력 integrity 검사. Objective: support-only acquisition M 학습의
공통 EEG 입력 기반을 실제 데이터에 연결. Feedback: schema·격리 불변성·finite/상수
채널·hash, 분류 성능 아님. Failure: 결과를 보고 구간/행/threshold를 바꾸거나 원본
row257 identity·SSVEP 존재·물리 latency·M 효능을 검증했다고 주장하는 것.

## 범위·자원·소유권

- Root만 코드/docs/SQLite 쓰기. 읽기전용 agent3개는 기존 source의 index 의미,
  후속 학습 기작, protocol/code를 검토한다. 입력계약/reader가 결합돼 있어 별도
  writing worktree 없이 순차 구현한다. 기존41worktrees/8untracked 보존.
- 새 network/PDF/archive/extraction/participant 0. 기존 source·probe JSON 및
  prior250Hz 영수증은 재검산 가능. 현재 가용314,712,840KiB, 약322GB다.
- Overall deadline **2026-09-13T12:10:00Z**, 실제 reader 시작/종료도 이 안에 둔다.
  새 직접 task artifacts <=1MiB (기존 SQLite/rendered-view 갱신 제외), free>=8GiB.
  Child AS2GiB/CPU30초/wall60초, single BLAS thread, stdout/stderr 각각32KiB.
  생성 검사/수리 뒤 실제 값 읽기는 **1시도, 실패 후 재시도/구간 교체 0**.
- 실제파일 scope 전에 reader/manifest/검사를 commit으로 고정하고 독립 검토한다.
  STARTED/worker claim/terminal, 고정 outputdir, producer·dependency hash 결합.

## 실제 고정 입력과 선택 규칙

파일 `/home/whwovy/data/mamem_i_v1_20260913/development_first.mat`,137357437bytes,
SHA256 `57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a`.
Role JSON SHA `dc26f85099ed667208ec7b992430d257d01f2a4aa727b7b98f8a5195f6d9af18`.
S001 전체 개발용 역할/원래S001a member를 검산한다. 독립 효능에는 사용하지 않는다.
저장250Hz는 직전 pinned child receipt SHA
`09e994853875fb406c73201e0d1f2983bc5d4790de5abef22aa9ef52176106e9`로 결합한다.

1. Header는 EEG double257×117917, DIN cell4×1966이어야 한다. 지난 DIN report가
   첫 group72events/complete라고 기록한 사실과 그 report SHA를 확인한다.
2. `loadmat(variable_names=['DIN_1'])`로 DIN 전체 decode를 명시한다. **row4의
   첫 event와72번째 event sample scalar 두 개만** 읽는다. Timestamp/descriptor/
   다른 event 값·class/frequency를 계산/출력하지 않는다. Sample은 양의 정수,
   증가하고 파일 내여야 한다. MATLAB index convention은 기존 author 근거를
   검토해 구현 전에 확정하며 물리적 onset 정확성을 주장하지 않는다.
   보존된 저자 Session.m lines463/465/482/499/509/513의 직접 MATLAB indexing에
   따라1-based를 채택한다. MOABB와 같은 정수에서1sample차이인 것은 구현 관례의
   차이이며 물리적 정답 판정이 아니다. Endpoint는 마지막DIN sample 포함 허용,
   즉 end0<=s72를 고정한다. 마지막DIN을 생리학적 자극 종료라고 부르지 않는다.
3. 첫 sample을s라 할 때, zero-based start=s−1+250, end=start+500이다. 즉 첫
   기록 marker 뒤 [1,3)초의2초 구간 하나. End−1이72번째 marker의 sample−1을
   넘으면 읽기를 중단한다. 다른 구간으로 대체하지 않는다. 처음 group을 고른 이유는
   기존 개발 probe의 고정 첫 group이고 결과 기반 고르기가 아니기 때문이다.
4. `loadmat(variable_names=['eeg'])`로 EEG 전체 decode를 명시한다. 함수는 shape/
   dtype만 확인한 후 **rows[0:256], samples[start:end]를 먼저 독립 copy**한다.
   이후에만 finite/상수채널/절댓값 최대/selectedtensor SHA를 계산한다. 선택 행/시간
   밖 값의 QC·정규화·필터링·특징·출력은0. 전체 decode는 ‘그 bytes 미열람’과 다르다.
5. 선택된256×500이 finite이고 적어도 한 채널이 비상수면 input-integrity PASS.
   상수채널 count는 보존하되 행 선택을 바꾸지 않는다. 알려지지 않은 voltage unit에
   임의 amplitude threshold를 적용하지 않는다. Waveform/spectrum/label을 저장하거나
   비교하지 않는다. Reref/resampling/montage/normalization/filter/fit0.

## 생성 검증·독립 검토

Row257을 label/NaN/Inf/큰 값으로 바꿔도 모든 선택 tensor/QC/hash가 동일해야 한다.
시간 밖 값도 동일하게 변화시킨다. Row order·copy isolation·bounds·shape·dtype·
unexpected keys·sample integer/one-based conversion·containment·S001 role·hash·
size·sampling receipt binding을 검사한다. 실패/timeout/마감/출력 cap/단회 영수증과
parent의 child 상태 검증은 실제파일 없이 검사한다. 제외된 행의 정체를 검증하는
검사가 아니라 사전 공통 제외 정책의 정보 격리 검사다.

## 후속 학습으로의 연결과 중단

성공하면 안전한 공통 input adapter를 유지하고, support-only recorded-event M의
명확한 표현과 k1에서 상쇄되지 않는 학습 연산·Q/Q2/QM/SHAM을 후속 후보로 구체화한다.
실패면 유형과 원문을 보존하고 실제 data 재시도 없이 다음 필요한 수리를 기록한다.
이번 결과로 M efficacy나 보정량 감소를 주장하지 않는다. Known class/schedule와
공통 보정·실제 획득prefix/setupcost·single-stimulus 한계를 계속 유지한다.
