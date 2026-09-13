# MAMEM I 공통 256행 입력 관문 결과

2026-09-13. **고정된 S001a 구간의 256행 입력 처리를 실제로 연결했다.**
입력 integrity PASS이며, SSVEP 존재·정확도·metadata 이득의 PASS는 아니다.
[계획](mamem_i_256row_adapter_v1_plan.md),
[실제 영수증](reports/mamem_i_256row_v1_run/child.json),
[상태](reports/mamem_i_256row_adapter_v1_state.json).

## 실제 결과

- 고정 계획 f053400, 구현/manifest/생성 검사 417ce74. 실제 단회 실행은
  11:35:25.122681–11:35:26.088762UTC, 0.966081초, retry0이다.
- 같은 S001a의 DIN row4 첫/72번째 sample 두 개만 검사했다. 1-based25239/26443에서
  zero-based `[25488:25988]`, 정확히500samples를 선택했다. 마지막 선택 sample이
  마지막 지정 marker를 넘지 않는다. Timestamp/descriptor/class는 검사하지 않았다.
- 원본 EEG257×117917과 DIN4×1966은 **변수 전체 decode**했다. 이후 EEG의
  rows[0:256]×위500samples를 독립 copy하고, 이 선택에만 수치 검사를 적용했다.
  제외된 행·시간의 수치 QC/정규화/필터링/특징 사용은0. 배열을 밖으로 저장하지 않았다.
- 선택값은 전부 finite, 상수 채널0개였다. 절댓값 최대는 저장 단위로69188.140625다.
  **단위를 확인하지 않았으므로 microvolt로 부르거나 정상 진폭이라고 판정하지 않는다.**
  이 값만으로 clipping/잡음/생리적 품질을 판단할 수도 없다. 후속 decoder에서 공통
  window-centering 등 전처리와 DC/단위 문제를 별도 명세해야 한다. 지금 채널을
  임의 삭제하거나 결과를 보고 구간을 교체하지 않았다.
- Selected tensor SHA256은
  `c1c017a85f2eb32b145ce877f9ec370051c205cb8aaaf0f80da469b4267ae4ea`다.
  Little-endian float64 C-order256×500의 hash이며 공개 waveform export가 아니다.

## 인덱스 근거와 해석 한계

[고정 저자 Session.m](https://github.com/MAMEM/eeg-processing-toolbox/blob/5a03abe2a6a874e9adaceea29a52c2fce35d8a03/%2Beegtoolkit/%2Butil/%40Session/Session.m)
lines463/465/482/499/509/513은 DIN4행 sample을 MATLAB 배열에 직접 사용한다.
따라서1-based→Python0-based 변환을 채택했다. 보존본
`sources/mamem_session_loader_20260912.json` SHA
`4cb9aa3267820d33671c8d8a6ea521c8f9104ab26d6d6efb337299c9dec7ba08`의 선택 code를
다시 읽었으며 새 network는0이다. MOABB의 같은 정수 사용과1sample 차이가 있지만
이 관례 차이를 물리적 timing 오류의 증거로 확대하지 않는다.

선택은 첫 기록 marker의 명목상1초 뒤2초다. 실제 광학 onset 또는 신경 latency를
측정·교정한 것이 아니다. 끝 marker도 실제 생리적 자극 종료라고 부르지 않는다.
원본 row257의 정체·실제 montage/reference는 미확정이며 공통 제외 정책만 검증했다.

## 검증과 다음 연구

새 생성54+이전 관련77=131tests PASS. Row257/시간 밖의 NaN·Inf·큰 값·가짜 label에
대해 선택 tensor/QC/hash 불변, 선택 안의 변경에는 민감함, 순서/copy 독립성,
두 DIN 셀만 접근, 해시/역할/경계/whitelist/마감/출력/단회 제한을 검사했다.
전체 repo suite는 이번에 실행하지 않았다. 독립 검토 뒤 deadline와 parent의
의미·단위 검사를 보강했으며, 최종 저장 영수증/커밋/hash/시각 감사도 PASS했다.
독립 감사는 실제 MAT/선택 tensor를 다시 읽거나131tests를 별도 실행한 것은 아니다.

이 관문에서 새 EEG 수치 접근은 S001의 위 구간1개, 새 참가자0, fit0,
분류/query/held60/외부 요청/유료0이다. 기존 실패/부정 결과를 보존한다.

다음 후보는 support의 **기록된 frame-grid 잔차 규칙성**으로 one-shot 공간
template를 source prior에 얼마나 섞을지 학습하는 것이다. 한 trial의 가중치가
정규화돼 사라지는 방식이 아니라 PSD template 자체가 변해야 한다. 먼저 별도
유한 생성 기작 계약으로 k1 작동과 null/schedule 대조를 검사하고, 실제 Q/Q2/QM/SHAM
및 source-only 반복 표적과 참가자 분리·획득비용 계약으로 연결한다. 생성 성공을
실제 metadata 효능으로 확대하지 않는다.
