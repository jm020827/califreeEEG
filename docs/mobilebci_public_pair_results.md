# MobileBCI 공개 EEG–IMU 한 쌍: 확보와 시간 연결 검증

> 후속 완료: [v2시간파서복구](mobilebci_event_time_v2_results.md),
> [16명코호트96파일확보](mobilebci_cohort_acquisition_v1_results.md).
> 아래v1중단은보존된이력이며현재parser미수리상태를뜻하지않는다.

2026-09-11 기록. 9월10일 수행된 파일 확보/채널 검사와 이어진 시간 metadata 검사의 실제 출력을 보존한다.
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

## 시간 검사 결과: 읽기 코드 호환성에서 중단

[명시적 event/t 범위](../configs/analysis/mobilebci_event_time_v1.json)를 `af1c553`으로 고정하고,
출력 오류 기록 보완 `f565e8a` 후 실제 한 번 실행했다. 첫 IMU 파일의 event/t selective decode
후 `unsupported_metadata_dtype`에서 중단했다. [실제 출력](reports/mobilebci_event_time_v1_observation.json).

- `decode_calls=1`, 완성된 record0. EEG 파일의 event/t는 읽지 않았다. 실제 파형 해독/학습/평가0.
- `scalar_elements=3`은 출력 변환기가 방문한 원소 수이며 **실제로 해독한 전체 metadata 원소 수가 아니다**.
  `loadmat`은 실패 전 첫 파일의 event/t를 이미 요청·해독했다. 따라서 event값 접근0이라고 하지 않는다.
- 셀/객체 배열일 가능성은 있지만, 실패 경로와 dtype를 기록하지 않아 정확한 내부 형식은 미확정이다.
  데이터 손상, 시간 불일치, metadata 효능 부재로 해석하지 않는다.
- 생성 selftest는 예산2회 안에서8/10cases PASS. 기존 테스트에 실제 MAT의 내부 형식 대응이
  충분하지 않았다는 한계가 드러났다. 실제 파일 재시도0, 전체 repo test는 실행하지 않았다.

동일한 자극의 EEG와 IMU를 어느 raw sample부터 어느 sample까지 짝지을지는 아직 미확정이다.
채널명이나 sampling rate만으로 clock 정렬을 검증했다고 하지 않는다.

## 공개 문서에서 찾은 배포본 차이

[arXiv v1](https://arxiv.org/html/2112.04176)의 Methods/Data Records와
[저자 loader](https://github.com/youngeun1209/MobileBCI_Data/blob/main/a_load_all_data.m)는
BrainVision/BIDS의 다른 배포본(Figshare16669072),5초 자극/구간을 설명한다.
현재13604078/v1 MAT는400samples@100Hz이므로 그대로 같은 시간 설정을 적용할 수 없다.
논문의 동시 trigger 설명은 후보 대응 방식의 근거이지 이 MAT의 index 원점·단위·drift 검증이 아니다.
[읽기 범위·오류·출처 기록](reports/mobilebci_time_sources_v1.json)을 남겼다. 전체논문/PDF 정독 완료는 아니다.

## 다음에 할 최소 수리와 실험 연결

이번 v1 실제 검사·생성 테스트 예산은 닫는다. 새 generated-only 수리에서는 작은 합성 MAT로
중첩 셀/객체 컨테이너의 허용된 숫자·문자·dict만 처리하고, 실패 경로/dtype/shape를 남긴다.
깊이·원소·출력 제한과 waveform 미요청을 유지한다. 임의 객체 허용이나 무한 재시도는 하지 않는다.
그 수리를 통과한 뒤 **별도 유한 recovery 범위**를 기록하고 기존 두 파일의 시간 metadata를
다시 확인한다. 후속 수리와 recovery는 아직 미실행이다.

시간 대응이 확인되면 raw support 구간만 추출하는 경로와 Q/Q2/QM/SHAM 학습기를 연결한다.
사람 효능 실험 전에 참가자 분리·정확한 특징/점수·목표 정확도·실제 수집비용·상한을 고정한다.
이번43MB 한 쌍을 모집단 효능 검증이나 전체 데이터셋 확보로 과장하지 않는다.

공개 저자 코드/최종 논문의 **다른 배포 버전** 설명을 이 MAT에 그대로 적용하지 않는다.
현재 후보는 원래의 저보정 SSVEP 목표를 유지하며, 보정 구간의 머리 움직임만 학습 보조 입력으로
사용하는 제안이다. Q에도 속도·순서를 주고 query IMU는 쓰지 않는다.

Choi 문의 경로와 기존 부정 결과는 그대로 보존한다. held60 개봉·외부 발송·유료 사용은 없다.

`academic-research`가 실제 배포본 차이를 찾아 시간 설정의 잘못된 이식을 막았다.
`coordinate-worktree-changes`에 따라 root만 main/연구 DB를 작성했고 두 agent는 읽기전용이었다.
기존41worktrees와8개 미추적 출력 디렉터리를 보존했으며 새 worktree/설치/삭제/push는0이다.
