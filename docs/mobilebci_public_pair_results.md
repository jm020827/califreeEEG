# MobileBCI 공개 EEG–IMU 한 쌍: 확보와 시간 연결 검증

2026-09-11 기록. 9월10일 수행된 파일 확보/채널 검사의 실제 출력을 보존한다.
**확보는 성공했지만, metadata의 정확도 이득·보정량 감소는 아직 평가하지 않았다.**

## 확인된 것

- Figshare13604078/v1의 s01·standing(0.0) SSVEP 파일 두 개를 정상 GET으로 받았다.
  각1시도/HTTP200/redirect1이며 크기·MD5가 공개 목록과 일치했다. 총43,427,682bytes.
  앞선 HEAD403 기록은 보존한다. HEAD 실패가 실제 GET 불가능을 뜻하지 않았다.
- `s01_IMU_SSVEP_0.0.mat`: 10,489,800bytes, raw128Hz/27채널.
  `HgyroX/Y/Z`를 포함한 H/L/R의 accel·gyro·mag 채널이 있다.
- `s01_scalp_SSVEP_0.0.mat`: 32,937,882bytes, raw500Hz/36채널(EEG32+EOG4).
  두 파일 모두 preprocess100Hz, 60epoch, epoch당400sample이다.
- 실제 숫자로 해독한 것은 sampling rate·채널명뿐이다. 다운로드/checksum의 전체 byte 읽기는
  EEG·IMU 파형을 숫자 배열로 분석한 것과 구분한다. 이 시점까지 event/t 값도 읽지 않았다.
- 원시 크기/MD5/SHA와 top-level shape는 [확보 관측](reports/mobilebci_public_pair_preflight_v1_observation.json),
  채널/주파수는 [선택 metadata 관측](reports/mobilebci_selected_metadata_v1_observation.json)에 보존했다.

## 아직 해결하지 않은 것

동일한 자극의 EEG와 IMU를 어느 raw sample부터 어느 sample까지 짝지을지 확정해야 한다.
채널명이나 sampling rate만 같거나 알려져 있다고 clock 정렬이 검증되는 것은 아니다.
이후 [명시적 event/t 범위](../configs/analysis/mobilebci_event_time_v1.json)에서 작은 시간 정보를
읽는다. 파형·새 참가자·성능은 이 검사에 포함하지 않는다.

공개 저자 코드/최종 논문의 **다른 배포 버전** 설명을 이 MAT에 그대로 적용하지 않는다.
현재 후보는 원래의 저보정 SSVEP 목표를 유지하며, 보정 구간의 머리 움직임만 학습 보조 입력으로
사용하는 제안이다. Q에도 속도·순서를 주고 query IMU는 쓰지 않는다.

Choi 문의 경로와 기존 부정 결과는 그대로 보존한다. held60 개봉·외부 발송·유료 사용은 없다.
