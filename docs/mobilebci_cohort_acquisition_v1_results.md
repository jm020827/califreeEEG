# 공개 EEG–IMU 16명 코호트 확보 완료

2026-09-11. **실제 원시 파일96개,16명×3조건=48회 기록,총2,204,445,401bytes를 확보했다.**
신규94개2,161,017,719bytes를525.936885초(약8분46초)에 받았고,
기존s01·standing2개43,427,682bytes를 재검증해 재사용했다.

## 어떤 자료인가

- [Figshare13604078/v1](https://api.figshare.com/v2/articles/13604078/versions/1)의 공개 CC BY4.0 배포본.
  s01,s03–s17의 standing0.0/slow0.8/fast1.6m/s 각각 scalpEEG와IMU다.
- S02는IMU0.8파일이 없어 사전에 제외했다. 결과를 보고 제외한 것이 아니다.
  목록의SSVEP ID는17명이며 설명의18명과 차이가 있다. 최종논문24명자료와 합치지 않는다.
- 96파일을96독립표본으로 세지 않는다. 독립 참가자는16명이고 각자의3조건이 반복 측정이다.
- 신규 파일 위치: `/home/whwovy/eeg-data/raw/mobilebci-cohort-v1-FYzkpM`.
  기존 한 쌍은 `mobilebci-pair-preflight-o14ykY`에 그대로 있다. 정확한 경로·SHA는
  [확보 관측](reports/mobilebci_cohort_acquisition_v1_observation.json)에 있다.

## 검증한 것

모든 신규 파일은 HTTP200/시도1/redirect1로 완료했고 크기·MD5·SHA256을 스트리밍 검증했다.
재시도·실패·남은.part파일0. 독립 검토는96개의공개목록/manifest/관측 ID·이름·크기·MD5 대응,
실제파일96개크기,94줄journal과관측일치를 확인했다. 독립검토가2GB를재해시했다고하지않는다.

이미 저장된 top-level schema에서 모든IMU는27raw채널/400×27×60preprocessed,
모든EEG는36raw채널/400×32×60preprocessed다. 원시sample수 범위는IMU68156–83808,
EEG266236–327375. 이 검사는 숫자 파형 해독이나 각run의시간정렬 검증이 아니다.

사전상한2.5GiB/30분/파일당90초를 지켰다. 최장20.870984초,
한정공개metadata조회1회와신규파일94초기GET/94redirect였다. private문의·유료·held60접근0.
전송코드는 이전에 검증한pair downloader를 그대로 재사용했다. Wrapper의생성검사2회4/5PASS,
관측pipe중단을치명오류로처리하지않고journal/report에flush+fsync를추가했다.

## 연구·구현도 진행한 부분

1. [한 쌍 시간 검사 복구](mobilebci_event_time_v2_results.md): 60event의class순서대응을 확인했다.
   ms해석의차이는최대9.3125ms이나marker의물리적onset의미까지확정한것은아니다.
2. [one-shot 기본 연산](mobilebci_reference_ridge_primitives_v1.md): k1공간회귀,
   phase-insensitive reference-projection점수, Q고정/trace보존/양의정부호 잔차를구현했다.
   CPUfloat64 인공9testsPASS, metadata모사입력이상대classmargin과기울기를바꾸는경로를확인했다.
   실제M 학습기·Q2/SHAM재학습·사람효능은아직아니다.

## 바로 다음 실행

**접근 가능성이나 샘플확보를 다시 반복하지 않는다.**

- 48run의작은metadata를한번씩검사해실제rate/채널명/class순서/시간grid·pair차이를대조한다.
  이때샘플한쌍관측을다른47run의검증으로확대하지않는다.
- 명시적인marker-relative가정아래raw support-only추출을구현한다. 제공preprocess_x의
  전체run처리를그대로prospective학습으로쓰지않고queryIMU/미래EEG로특징·변환을갱신하지않는다.
- 기본연산을실제Q→prior학습및Q고정QM/Q2/SHAM에연결한다. 참가자분리·같은support·같은query·
  공통speed/order·실제수집비용과중단기준을첫사람outcome전에고정한다.

이번에데이터부족/비공개접근의병목은줄었지만,**외부metadata가Q이상으로도움이되는지와
보정량을줄이는지는여전히미검증**이다. Choi보류와이전실패/부정결과는유지한다.
