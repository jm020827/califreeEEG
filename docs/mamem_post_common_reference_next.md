# 공통 baseline 종료 뒤: 새 공개 paired 정보원의 자격 확인

2026-09-14 KST / 2026-09-13 UTC. **DRAFT / 미실행**. 이 문서는 다음 유한 작업을
정의하기 위한 초안이며 이번 EEG run의 추가 실험·재시도 권한이 아니다.
[완료 결과](mamem_common_reference_v1_results.md)는 32.67%→58.00%이지만 두 arm 모두
준비 기준 미달이다. 고정 E126 비교, full잔차 직접전달, M2→PSD v2는 종료를 유지한다.

## 다음 질문을 바꾸는 이유

공통 reference를 바꾸는 것만으로 큰 차이가 났다. 그렇다면 다음 학습 M은
그 공통값의 다른 표현이 아니라, **query 전에 알 수 있고 해당 EEG의 측정 조건과
대응되는 추가 정보**여야 한다. 현재 자료의 오류를 보고 채널/창/계수나 참가자를
고르는 것으로 그 추가 정보를 입증할 수는 없다.

우선 새 모델을 만들지 않는다. 공개 데이터의 문서와 파일 목록으로 다음 질문을
판단하는 것이 먼저다. 통과하더라도 실제 파일 schema 확인과 학습 계약이 따로 필요하다.

1. 정확한 공개 release에서 동일 사람·session·block의 EEG와 외부 측정 파일이 연결되는가?
2. 외부 필드가 무엇을 측정하며, 변환·결측·유효성 코드가 독립적으로 정의돼 있는가?
3. 실제 cue label과 EEG/외부 stream clock의 대응 및 support→query 시간 경계가 있는가?
4. 추후 query나 정답을 보지 않고 해당 support 시점에 M을 얻을 수 있는가?
5. 그 정보가 EEG-derived Q 이상의 학습 이득을 낼 물리적/측정 기작은 무엇인가?
6. 추가 장치 calibration/recalibration, setup, 전체 획득 prefix 비용을 셀 수 있는가?

## 우선 확인할 정확한 공개 경로: MMV, 단 한 번의 문서 capture 자격 확인

이는 새 긍정 후보 선정이 아니라 **과거 로컬 응답 capture 실패를 해소할 수 있는지**
판단하는 작업이다. [기존 공식 provenance 확인 결과](public_provenance_resolution_v1_results.md)와
[기존 preflight 이력](mmv_public_metadata_preflight_v1_draft.md)을 보존한다.
과거 DataCite GET200은 DOI `10.57760/sciencedb.ai.00010`, V3/CC BY 4.0 선언과
배포 주소를 확인했지만 실제 파일 inventory나 익명 다운로드 가능성을 검증하지 않았다.
배포 페이지의 실패는 보존되지 않은 큰 도구 출력의 처리 문제였으며, HTTP 접근 거부라는
증거가 아니다. 이 문서 작성 중 새 인터넷 확인은 0이다.

이미 기록된 정확한 seed:
`https://www.scidb.cn/detail?dataSetId=270bcdeaab0c48adb8eee700479daafd`

별도 실행 계약에서 고정할 제안 예산:

- 위 페이지 1개와 **응답에 명시적으로 연결된** inventory/README/loader-schema 공개
  문서 최대 3개, 총 **4 targets / target당 1회 / target당 2MiB / 전체 8MiB**.
- 요청당 최대 30초, 전체 180초. 문서 본문을 먼저 bounded local artifact로 보존하고
  status/bytes/hash/최종 URL을 작은 영수증으로 출력한다. 원문 전체를 도구 JSON에
  집어넣는 이전 capture 경로를 반복하지 않는다. size cap 도달도 실패로 보존한다.
- 동적 페이지에 필요한 **명시된 공개 문서/asset 링크**가 없다면 추측 API·endpoint
  탐색으로 확장하지 않는다. redirect는 보존하고 공개 범위·예산에 포함한다.
- 기존 DataCite 성공 원문은 재사용하고 재요청하지 않는다. generic search/PDF/raw EEG/
  사람별 annotation/numeric tracker 값/로그인/계정/DUA/사람 요청/유료 0이다.
- 403/429/challenge/계정 요구는 우회·재시도하지 않는다. 부분 응답과 이유를 남기고 중단한다.
  이 루트를 반복해서 새 예산으로 열지 않는다. 막히면 사용자 지시대로 보류한다.

현재는 위 예산을 **실행하지 않았으며**, 정확한 downloader·응답 보존·누적 byte 경계를
구현/검증하고 실행 전 새 계약을 동결해야 한다. 공개 문서 확인은 기존 사용자 연구
범위 안에서 다음 단계로 진행할 수 있지만, 사람 요청·유료·held60은 여전히 별도 승인이다.

## 승격시키면 안 되는 것

- `LOST_DATA_EVENT`는 stream gap이지 그 자체로 eye-tracking 실패나 EEG noise가 아니다.
- PERCLOS·pupil·blink·gaze를 acquisition M이라고 이름만 바꾸지 않는다. 생리/행동 정보와
  장치 측정 유효성을 구분한다. 다른 연구목표인 졸림/시선 예측으로 바꾸지 않는다.
- Query의 주파수·fixation/target identity 또는 online decoded command를 M/정답의
  독립 측정으로 쓰지 않는다. 미래 smoothing/interpolation·stream별 시작점 빼기를
  정렬 근거로 삼지 않는다.
- MAMEM-I 광학 DIN의 저자보고를 Dataset-II 또는 MMV의 획득 조건으로 옮기지 않는다.

## 유한 판정과 그 다음

**DOCUMENTED_PAIR_CANDIDATE**: 공개 inventory와 위 문서 근거가 실질적으로 연결된다.
이는 문서 단계의 통과일 뿐, 실제 숫자나 학습 효능을 확인했다는 뜻은 아니다. 정확한
최소 파일쌍·byte/역할 제한을 고정한 별도 schema 진단을 먼저 설계한다.

**PARK_ACCESS_UNRESOLVED**: 공개 문서 접근/목록이 불충분하거나 계정·사람 요청이 필요하다.
자동 요청하지 않는다. 같은 루트를 반복하지 않고, 이후의 별도 제한된 후보 조사로 넘긴다.

**REJECT_M_OR_TASK_PROVENANCE**: 외부 정보가 정답 단서·생리치의 재명명·query 후처리일
뿐이거나 시간적 대응/저보정 SSVEP 연결이 성립하지 않는다. 원 연구목표를 바꾸어 구제하지 않는다.

새 학습 후보는 위 자격과 구체적 기작이 확인될 때 **하나만** 별도 예산으로 등록한다.
모든 arm의 공통 source reference/정보권한을 맞추고 nominal0/common0/directM 및
Q/Q2/QM/SHAM을 포함한다. 참가자 분리, 선택에 사용한 개발자료와 후속 독립 검증의 분리,
보정 trial 및 실제 획득·장치 비용, 악화와 중단 기준을 실행 전에 고정한다.
현재 새로운 학습 적격 후보·유망한 metadata 효과·보정 절감은 확보하지 못했다.
