# 다음 후보 조사 결과: Eye-BCI 논리적 파일쌍 확인, 익명 파일 접근 보류

2026-09-11. 이전 gyro source-routing 후보는 종료 상태 그대로다.
새 사람 학습·성능 평가·held60 개봉은 없었다.

## 이번에 확인한 것

새 후보3종을 구분했다. 실제 stimulus timing, 실제 electrode/contact 변화,
독립 eye-tracker 수집 품질이다. 기존 연구장부를 읽은 결과, nominal/digitized
채널 좌표는 이미 공통 정보로 정의돼 있고 현재 원본에서 session별 추가
placement/contact 측정이 확인되지 않았다. Gyro와 event marker를 photodiode
지연 측정으로 바꿔 부르는 방법도 제외했다.

가장 구체적 단서는 Guttmann-Flury의 [Eye-BCI 공개 프로젝트](https://www.synapse.org/Synapse:syn64005218/wiki/630018)였다.
공식 wiki는 Tobii의 `ValidityLeft/Right`, `PupilLeft/Right`, `DistanceLeft/Right`
등을 설명하고 CC0를 표시한다. 등록 없이 프로젝트·wiki·참가자 목록을
조회하는 데 성공했다. **이것은 실제 CSV의 열이나 다운로드 권한 확인이 아니다.**

첫 S01 세션을 결정적으로 선택해 최대4회 목록 조회를 수행했다.

```text
syn64005218 / S01(syn64071512) / Sess01(syn64071513)
├─ Neuroscan(syn64072621) / SSVEP011.csv (syn64072665), version1
└─ Tobii(syn64086404)     / SSVEP011.csv (syn64086409), version1
```

4회 모두 HTTP200이며 목록 항목 수1/5/5/5, JSON 응답 합계4,821bytes다.
같은 참가자·세션·파일명의 **논리적 pairing**만 확인했다. 실제 행 수,40trials,
주파수,시간 정렬,CSV 크기와 전체 checksum은 검증하지 않았다. Upload 시각도
실제 취득 시각이나 calibration 비용으로 쓰지 않는다.

## 헤더-only 검사는 어디에서 멈췄나

[별도 고정 계약](eye_bci_header_probe_v1_contract.md) 아래 두 파일의 첫 줄만
읽는 reader를 구현했다. Generated4case/1회는 통과했고, 이후 정적 검토의
request cap·strictCSV·양쪽Validity 검사를 반영한 코드 `a6f9d92`로 실행했다.
이 추가 boundary 변경 뒤에는 정한 unit 예산을 늘려 재실행하지 않았다.

첫 Neuroscan FileEntity 조회는 성공했다. ID/parent/filename/version1이 모두
일치했고 `dataFileHandleId=149810668`을 확인했다. 하지만 이어진 익명
versioned file GET은 **HTTP403**이었다. 계약에 따라 즉시 종료했다.

- 논리/네트워크 요청2회, redirect0,1.703804초.
- CSV 헤더0개, numeric samples decoded0, 두 번째 Tobii 파일 요청0.
- 로그인·계정생성·사용조건 대신 승인·다른 endpoint 시도·저자 요청0.
- [원본 관측](reports/eye_bci_header_probe_v1_observation.json)과 attempt 보존.

**현재 경로에서 익명 파일 접근이 실패했다**고만 말할 수 있다. 403 응답의
상세 원인을 조사하는 추가 요청은 하지 않았으므로 “반드시 로그인하면 된다”,
“비공개 데이터다”, “저자가 승인해야 한다”는 결론은 내리지 않는다.
공개 metadata/CC0 표기와 현재 파일 접근 실패를 함께 기록한다.

이 경로는 **PARKED — anonymous file access unresolved**다. 권한 우회를 하지
않고, 사용자에게 당장 계정이나 저자 요청을 요구하지 않는다. 이미 적법한
접근 경로가 있거나 공개 배포 상태가 바뀌면 별도 작은 재검토를 할 수 있지만,
그 전까지 연구의 필수 경로에 두지 않는다.

## 이 자료를 쓰게 되더라도 필요한 연구 조건

제안한 M은 paid calibration support의 eye-tracker 수집 신뢰도였다. 이것은
아직 선택된 학습기나 유익성이 확인된 특징이 아니다. Validity는 tracker의
신뢰도이지 곧바로 EEG 품질·눈 감김·시각 입력의 정답이 아니다.

- Query gaze/validity, cue 방향, 화면 영상, whole-session 정규화는 제외한다.
  Validity도 gaze 위치와 함께 변할 수 있으므로 class shortcut을 따로 점검한다.
- `Blinks` 열의 생성 출처가 확인되기 전 externalM로 쓰지 않는다. Pupil은
  생리 상태와 gaze 의존성이 섞이므로 primary acquisitionM로 성급히 채택하지 않는다.
- 모든 방법에 동일 Q·구조·paid labels·channel 규칙을 제공한다. Wiki는 posterior
  채널 다수를 포함한 불량/bridging 채널의 제외를 권하므로 MobileBCI9채널을
  그대로 복사하지 않는다. 실제 어떤 채널이 usable인지는 아직 검증하지 않았다.
- 추가 eye-tracker 설치·calibration 비용은 별도다. EEG label 수 감소와 전체
  사용자 준비 부담 감소를 혼동하지 않는다.
- Source-only 선택과 Q/Q2/SHAM·M-only 개입·실제 acquisition prefix 비용 비교가
  가능해야 한다. Header 접근 성공만으로 사람 효능 실험을 개시하지 않는다.

## 다음 조사 우선순위

한 번의 좁은 scholarly search에서 [Online Compensation of Systematic Effects
in Stimuli Generation for XR-Based SSVEP BCIs](https://doi.org/10.3390/s26030766)를
찾았다. 현재는 **OpenAlex abstract만 본 후보**이며 원문 Methods/Data Availability
검증은 하지 않았다. 제목/abstract만으로 공개자료·저보정 우위·학습기 기여를
확정하지 않고 academic workspace의 targeted deep-read queue에 넣었다.

다음 단계는 이 논문에서 **실제 frame-time/fps 측정과 EEG가 공개돼 있는지**,
변동 timebase 보정이 어떤 연산인지, label/query 접근 조건과 보정량 평가가
무엇인지 확인하는 것이다. 단순 deterministic reference 보정을 새 metadata
학습 성과라고 부르면 안 되며, 학습 후보가 된다면 그 보정 자체도 공정한
대조군에 넣어야 한다. 요청 의존이면 역시 보류한다.

이번 finite feasibility/schema/header 예산은 종료했다. 실제 EEG/M를 새로
확보했다거나 연구 효능이 향상됐다는 결과는 없다. 다만 추상적인 “눈 정보가
있다”에서 공식 공개 목록·정확한 파일쌍·실제 접근 제한까지 확인한 진전이다.

## 실행·근거 기록

- [3-route 범위](post_gyro_candidate_feasibility_v1.md),
  [4-request schema 범위](eye_bci_schema_preflight_v1_contract.md),
  [헤더 범위](eye_bci_header_probe_v1_contract.md)를 단계별 사전 고정했다.
- Root3API JSON 성공:298+5122+9259bytes. Schema4응답4,821bytes. Header 단계의
  FileEntity387bytes. 이는 성공 API payload 합이며 오류응답/전송 전체 byte계측은 아니다.
- Root Nature 접근은 redirect에서 실패했고 우회하지 않았다. Scout는 공식
  논문/저자repo 성공,논문 dataset reference/Synapse web API 실패를 남겼다.
  이후 공식 익명 API 접근 성공은 별도의 공개 API 관측이며 인증 우회가 아니다.
- 첫 로컬 listing wrapper의 `Request.method` 오류와 session-id 저장 오류는
  HTTP 접근 전에 발생했다. 수정·보존했고 실제 실패 요청으로 세지 않았다.
  Header 실행 전 lint1건(TypeError 권고)을 고쳤으며 실제 header재시도0.
- Search1/2사용,8/8공식target slots사용,PDF0. Metadata-only schema4/4사용,
  header는첫실패2requests에서종료. 세 단계는 다른 질문이며 실패 후 같은
  효능 실험의 예산을 보충한 것이 아니다. 전체 repo suite 미실행.
- Root 단독 main/SQLite 작성,검토자 읽기전용. 기존41worktrees/8untracked 보존,
  새tree/설치/삭제/push0. 원목표/기존 부정/held60/Choi/source39경계 유지.
