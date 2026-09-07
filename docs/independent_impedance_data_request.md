# 독립 acquisition-context replication 자료 요청 준비

> 2026-09-08 정정: 이 요청은 외부 획득환경 재현을 위한 **선택적 보강**이다. 현재 자료의 좁은 개발·진단을 막는 필수조건이 아니다. [현재 자료 우선 계획](current_data_first_research_plan.md)을 따른다. 발송0/새 자료 수령0이며 아래 제공 가능성·권리·소속 확인 항목은 유지한다.

2026-09-07 기준. **아직 발송하지 않았고 새 자료를 받았다는 뜻이 아니다.**

현재 Wearable source39만으로 개발은 가능하다. 하지만 개발에 이미 쓰인 같은 참가자를
새 데이터처럼 세지 않는다. 추가 코퍼스가 있어도 장비명이나 문서의 임피던스 상한만 있으면
실측 block-level metadata 순증분의 독립 replication이 되지 않는다.

우선 요청 후보는 [Liu et al. (2022), Facilitating Applications of SSVEP-Based BCIs by
Within-Subject Information Transfer](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.863359/full)다.
공식 본문의 Data Availability Statement는 합리적인 요청을 통한 저자 제공 경로를 명시한다.
다일 wet/dry 설계라는 관련성은 있지만, 필요한 channelwise metadata 공개 여부와 기존
Wearable102 참가자와의 중복은 확인되지 않았다. 이전 session이 있는 같은 사람의 transfer는
새 사용자의 처음 calibration과 다른 권한이므로 primary로 혼동하지 않는다.

## 요청할 최소 자료

- 익명 EEG 원본 또는 epoch: channel names/order, sampling rate, reference, event timestamps,
  class frequency/phase, block/session/day/interface IDs, 전처리·결측/제외 내역.
- 각 block **전에** 측정된 channel별 실제 impedance, 단위, 측정 시각과 장비, missing code.
  기준 상한이나 전체 평균만 있는지 구별할 수 있는 data dictionary.
- Headset 순서와 session/block 시각: chronology-only 대조군을 만들 수 있어야 한다.
- Subject pseudonym과 일자간 연결키, 기존 Wearable102와의 overlap 여부(신원정보 불필요).
- Secondary-use license/사용조건, 재배포·파생특징·논문 aggregate 공개 허용 범위,
  필요한 data-use agreement 또는 기관별 검토 조건.

## 보낼 때 필요한 사용자 정보

사용자의 실제 이름·소속·연락처, 연구 책임자/기관이 필요한지, 해당 기관의 공개·비식별 자료
secondary-use 절차를 확인해야 한다. 이 항목을 AI가 만들어 쓰거나 윤리승인이 있다고
표시하지 않는다. 현재 연구 실행 승인만으로 외부 메일 발송 권한을 추정하지 않는다.

## 영문 요청 초안 — 발송 전 인적 정보와 조건 확인 필요

Subject: Request for de-identified multi-session SSVEP EEG and pre-block impedance metadata

Dear corresponding authors,

We are preparing a study of whether pre-query acquisition context can reduce labeled calibration
requirements in closed-set SSVEP decoding beyond an EEG-only comparator. Your 2022 study of
within-subject information transfer is particularly relevant because of its multi-session wet/dry design.

Would you be willing to share de-identified EEG and a data dictionary, including trial/block/session
identifiers, stimulus parameters, headset order, and, if recorded, channelwise impedance measured
before each block with units, timestamps, and missing-value definitions? We would also appreciate
confirmation of any participant overlap with the public 102-participant wearable SSVEP dataset.

We would use same-participant previous sessions only in an explicitly separated analysis; they would
not be treated as data available to a first-time user. Please let us know the applicable license,
secondary-use requirements, and any data-use agreement needed before access.

Sender identity and institutional details will be supplied by the actual researcher before sending.

## 수령 후 gate

License·schema·overlap 확인 → outcome을 보지 않은 acquisition metadata 적합성 감사 →
별도 고정 replication protocol과 split → decoder/parameter freeze → 단회 평가.
이번 source39의 결과를 보고 새 자료의 조건·participant를 성능으로 고르지 않는다.
