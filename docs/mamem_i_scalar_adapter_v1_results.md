# MAMEM I scalar·입력 누출 관문 결과

2026-09-13. **용량 문제는 해소됐고, S001a의 저장된 samplingRate=250Hz를 확인했다.**
원래 저보정 SSVEP 목표는 그대로다. 이번 결과는 입력 설정 확인이며 metadata의
성능 이득·보정량 감소를 검증한 실험은 아니다.

[고정 계획](mamem_i_scalar_adapter_v1_plan.md),
[실행 영수증](reports/mamem_i_scalar_adapter_v1_run/terminal.json),
[상태](reports/mamem_i_scalar_adapter_v1_state.json).

## 실제로 진행한 것

- 공간 확보 뒤 10:28:33UTC에 가용 315,703,184KiB, 약 323GB/301GiB를 확인했다.
  기존 파일·41worktrees·8untracked 디렉터리를 삭제하지 않았다. 이번 유한 계약의
  작은 문서·scalar 범위는 유지했고 추가 대용량 압축 해제는 하지 않았다.
- 구현과 생성 검사를 58043ba로 먼저 고정했다. 기존 S001a 한 파일의 크기·SHA256·
  개발용 역할을 검산하고 `loadmat(variable_names=['samplingRate'])`로 **값 하나**를
  읽었다. 10:40:02.064–10:40:02.379UTC, 시도 1회, 결과 250.0Hz/MATCH다.
- EEG/DIN 배열은 요청하지 않았다. 파일 hash/header는 압축 바이트를 처리할 수
  있으므로 ‘EEG 관련 byte를 전혀 처리하지 않았다’는 주장은 하지 않는다.
- 이는 저장된 명목상 sampling rate가 배포 설명과 일치한다는 뜻이다. 실제 ADC
  clock 정밀도·자극 onset·광학 jitter의 측정은 아니다. 250Hz 자체를 새롭게 변하는
  acquisition M 특징으로 승격하지 않는다.
- 새 생성 34개+관련 기존 43개=77tests PASS. 잘못된 역할/해시/크기/변수/shape/value,
  timeout·출력 상한·실패 영수증·무재시도·deadline 전후를 검사했다. 전체 repo suite는
  이번에 실행하지 않았다. 실제 EEG fit/정확도/보정절감/held60은 0.
- 최종 독립 감사는 producer/config와 고정 커밋, STARTED→worker claim→child→terminal
  시각·해시·범위를 대조해 PASS했다. 기록된 전체 시도는 0.315117초이고 stdout/stderr는
  각각 0 bytes다. 감사자가 실제 MAT를 재열람하거나 테스트를 별도 재실행한 것은 아니다.

## 공개 구현에서 확인한 중요한 구분

[공식 MOABB 구현](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/ssvep_mamem.py)의
보존본에서 `mamem_event`는 DIN으로 만든 class marker를 마지막 행의 event 위치에
기록한다(lines57–100). Loader는 E1–E256 뒤에 `stim`을 붙이고 마지막 채널을
명시적으로 **stim 타입**으로 지정한다(lines157–170).

따라서 **MOABB가 정상적인 사용에서 정답을 EEG로 누출한다는 뜻은 아니다.** 우리가
변환된 마지막 행을 EEG predictor에 잘못 포함하면 정답이 유입된다는 경고다.
또 이 변환 관례만으로 원본 v1 MAT의 row257이 기준전극·빈 행·event행이었다고
판정할 수 없다. 코드의 montage/250Hz 지정도 실제 v1의 좌표·clock 검증은 아니다.
로컬 설치본과 최신 공식본의 전체 SHA는 다르며 관련 처리 구간만 양쪽에서 확인했다.
MOABB loader를 실제 데이터에 실행하지 않았다.

[Cedrus 원래 ST100–EGI 설정 안내](https://cedrus.com/support/stimtracker_1g/tn1468_egi.htm)는
NetStation4.5.5 예시, Pulse 설정, 소프트웨어 표시 trigger 번호를 설명한다.
페이지 자체가 2022년 timing/polarity 공지도 함께 보라고 명시한다. 이것은 실제
MAMEM의 eventedge/export 설정이나 DIN descriptor 셀의 정의를 주지 않는다.
따라서 **첫 이벤트의 미정의 descriptor 읽기는 생략**했다. 일반적인 센서 번호를
그 셀의 의미로 추정하지 않았다. 이 페이지의 이미지는 추가로 열지 않았다.

Source 예산은 3unique targets+locator 1회로 종료했다. EEGLAB의 특정 코드 경로는
404였고 재시도하지 않았다. 성공한 Cedrus/MOABB 응답 2건을 root가 한 번씩 보존해
읽었고 독립 agent가 크기·SHA·MOABB git blob 및 위 해석을 검산했다. 새 PDF 0.
원문·영수증은 research workspace `sources/mamem_cedrus_setup_20260913.*`,
`sources/mamem_moabb_adapter_20260913.*`에 있다.

## 불필요하게 멈추지 않도록 다음 관문 수정

원본 row257의 물리적 의미는 아직 모른다. 그러나 **그 행을 모든 arm의 모든
수치 처리 전에 제외하는 보수적 입력 정책**은 별도로 정의할 수 있다. 기존
‘임의 삭제 금지’는 의미를 확인한 척하거나 결과를 보고 유리한 행만 제거하지
말라는 경계다. 명시적인 공통 256행 분석 정책까지 영원히 막을 이유는 없다.

다음 [입력 격리 관문 초안](mamem_i_256row_adapter_v1_draft.md)은 원래 앞 256행의
순번만 사용하고 geometry/reference 주장은 하지 않는다. Row257에 정답·NaN·큰 값을
넣어도 선택 신호의 QC·특징이 같아야 한다. 이 관문을 통과한 뒤 support-only
recorded-event M의 정의와 k1에서 상쇄되지 않는 학습 연산을 고정한다.

Query DIN은 예측 입력에서 계속 격리한다. 공통 class/schedule·보정과 Q/Q2/QM/SHAM,
실제 획득 prefix/setup 비용, S001 전체 개발용, 기존 부정 결과를 유지한다.
새 학습 후보의 효능은 아직 미확립이고 held60·외부 요청·유료 사용은 하지 않았다.

Academic-research는 공식 근거·불확실성·측정 주장을 분리해 보존하는 데 사용했고,
coordinate-worktree-changes에 따라 읽기전용 병렬 검토와 root 단독 쓰기를 유지했다.
독립 검토의 deadline 지적을 실제 접근 전에 수정했다. 이번 원문·scalar 절차에서
알려진 예산/범위 이탈은 없으며 이전 PDF queue 이탈·실패 기록은 그대로 보존한다.
