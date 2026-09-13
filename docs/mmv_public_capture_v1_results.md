# MMV 공개 페이지는 확보했지만, 학습용 파일쌍·필드 정의는 미확인

2026-09-14 KST / 2026-09-13 UTC. 정확한 MMV 배포 페이지를 **익명 GET 1회로
HTTP200 / 785,384bytes** 받아 원문과 해시를 보존했다. 이전의 큰 응답 출력/capture
문제는 이번 저장 경로에서 해소됐다. 그러나 이는 EEG·외부 측정 파일을 확보했거나
metadata 학습 적격성을 확인했다는 뜻이 아니다.

문서 자격 판정은 **`PARK_ACCESS_UNRESOLVED`**다. 여기서 남은 문제는 **공개 페이지의
존재가 아니라 실제 inventory/schema**이며, 비공개 데이터라고 단정하지 않는다.
추가 요청·학습·원시자료 다운로드로 확장하지 않고 이번 MMV 경로를 종료한다.

## 새로 확인한 사실

[ScienceDB의 정확한 배포 페이지](https://www.scidb.cn/detail?dataSetId=270bcdeaab0c48adb8eee700479daafd)의
`script[type="application/ld+json"]`에 다음 선언이 있었다. 본문은 보존된
[body_1.bin](reports/mmv_public_capture_v1_run/body_1.bin)에 해당한다.

| JSON-LD 필드 | 확인한 값 | 의미의 한계 |
| --- | --- | --- |
| `name` / `@id` | MultiModal Vigilance dataset / DOI 10.57760/sciencedb.ai.00010 | 정확한 dataset landing page 확인 |
| `conditionsOfAccess` | `PUBLIC` | 페이지의 공개 선언; 개별 파일 익명 GET 성공은 아님 |
| `version` | `3.0.0` | `citeAs`에는 V3; 실제 파일 version 검증은 아님 |
| `license` | CC BY 4.0 링크 | 배포자의 라이선스 선언 |
| `size.value` | 265,453,813,779bytes | 선언된 전체 크기; 전체 파일 수·checksum 검산은 아님 |
| `description` | 두 BCI 과제에서 얻은 일곱 생리 신호라는 요약 | acquisition M의 필드/시간/기작 정의는 아님 |

이는 기존 [DataCite 기록 확인](public_provenance_resolution_v1_results.md)과 부합하지만,
기존 registry는 이번에 다시 요청하지 않았다. Raw 파일 접근이나 독립 현장 측정 검증을
추가한 것은 아니다. JSON-LD의 Croissant 준수 선언만으로 개별 record/file schema가
포함됐다고 보지 않는다. 실제 객체에는 파일 목록인 `distribution`/`recordSet`이 없었다.
받은 HTML의 script/style을 제외한 visible text는 dataset 제목뿐이었다. 브라우저 JS를
실행·렌더링하지 않았으므로 사용자 브라우저의 최종 화면을 확인했다고 주장하지 않는다.

## 두 번째 요청과 정확한 중단 이유

첫 HTML의 `link[rel="canonical"]`은
[영문 주소](https://www.scidb.cn/en/detail?dataSetId=270bcdeaab0c48adb8eee700479daafd)를
직접 가리켰다. 영문 문서에 inventory/README 설명이 포함될 수 있는지 확인하기 위해
이 링크를 두 번째 대상으로 선택했고, 실행 전에 이유를 알렸다.

두 번째 요청은 **302**로 이미 요청한 원래 주소를 `Location`에 돌려줬다.
`Content-Type`이 없어 수집기는 사전 규칙에 따라 **`STOPPED_NO_RETRY /
non_document_mime`**를 남겼다. 보존한 body는 0bytes다. 이는 서버가 보내려던 전체
body 크기가 0이라는 주장이 아니다. MIME 경계에서 body를 읽지 않았다.
실제 중단 분기는 MIME 검사이며, 그와 별개로 Location을 따라갔어도 이미 방문한 URL의
재시도가 된다. 자동 redirect·세 번째 요청·정적 asset 추적·추측 API·로그인은 모두 0이다.

| 요청 | UTC 시작–완료 | HTTP / 상태 | 보존 body |
| --- | --- | --- | ---: |
| 1: exact landing | 15:56:05.772149–15:56:06.252653 | 200 / CAPTURED | 785,384bytes |
| 2: 명시된 canonical | 15:57:18.743116–15:57:19.077389 | 302 / STOPPED_NO_RETRY | 0bytes |

총 **2/4 HTTP 시도**, 자동 redirect 0, 같은 URL 재시도 0. 요청 경과는 각각
0.480511초와 0.334276초이며, 외부 작업은 고정한 180초 창 안에서 끝났다.
네트워크 body 보존량 785,384bytes는 8MiB보다 작다. Header·TLS·OS socket buffer를
포함한 전체 wire bytes를 측정한 것은 아니다. 슬롯을 모두 소진한 것이 아니라
**사전 중단 조건에 도달해 종료**한 것이다.

## 학습 적격성은 어디까지 왔나

| 필요한 근거 | 이번 상태 |
| --- | --- |
| 정확한 공개 landing / 배포자 선언 | 확인 |
| 동일 release의 EEG–외부 파일 목록과 join | 미확인 |
| 실제 offline acquisition-validity 필드·단위·결측 의미 | 미확인 |
| EEG/외부 clock·cue label·support→query 시간 경계 | 미확인 |
| Query 전에 얻을 수 있는 추가 M | 미확인 |
| 그 M이 Q·공통 정보 이상의 EEG 학습 이득을 낼 기작 | 미확립 |
| Tracker calibration/setup·전체 획득 prefix 비용 | 미확인 |
| 실제 학습·EEG 정확도·보정량 감소 | 이번에는 실행하지 않음 |

`PUBLIC`이라는 단어는 이 표의 나머지 칸을 채우지 못한다. 반대로 문서에 없다는
이유로 실제 필드도 없다고 결론내리지 않는다. Pupil/PERCLOS/gaze를 acquisition M로
바꾸거나 `LOST_DATA_EVENT`를 EEG noise라고 해석하는 기존 금지 조건은 유지한다.

## 구현·검증과 보존

- [사전 계약](mmv_public_capture_v1_contract.md)과 [수집기](../scripts/analysis/capture_mmv_public_docs_v1.py),
  테스트를 **0a96034**로 커밋한 뒤 실제 HTTP를 시작했다. Root만 코드/실행/문서/연구 DB를
  쓰고, agent는 읽기 전용 경계·기작 검토를 맡았다. 새 worktree 0, 기존 46개와
  8개 untracked 디렉터리 보존. 가용공간 약 292GiB, 삭제·설치·push 0이다.
- 기존 PDF/MAT downloader는 known-size/다른 host 전용이며 partial 보존과 redirect
  집계가 맞지 않아 재사용하지 않았다. 새 경로는 독점 claim, code/contract pins,
  30초 절대 timer, 자동 redirect 금지, body cap, 일부 응답·HTTP 오류 보존을 구현했다.
- 합성 2suite 예산: **23PASS/0.09초**, 보강 후 **29PASS/0.10초**. 실패 0.
  그 뒤 timer 활성화 한 줄을 `try` 안으로 옮겨 일반 SIGALRM의 receipt 누락 경계를
  보완했다. 이 마지막 변경은 **정적 재검토·ruff PASS만 했고 pytest 재실행은 하지 않았다**.
  29개 검사가 마지막 1줄 변경 이후 실행됐다고 말하지 않는다. Whole-repository suite 0.
- 초기 broad-exception lint 및 import-order lint 각1건을 실제 전에 수정했다.
  동적 JS 문자열 조립 prefix 제외, 강한 challenge/auth HTML 표식, 일반 login 링크와의
  구분, 실제 GET 전 남은 시간, `IncompleteRead.partial`, timer 해제 후 실제 보존 body
  재해시, Location 선보존·fragment 제거도 실행 전에 반영했다. Challenge 검출은
  명시 패턴 범위이며 모든 접근 장벽을 완전 판별한다는 보장은 아니다.
- 실제 후 수집기·계약 변경/재실행 0. 기존 연구 보호 6개 hash는 불변이다.
  원시 EEG/metadata 숫자/annotation/PDF/fit/EEG prediction/held60/외부 사람 요청/유료/GPU 0.
- Run의 7개 파일은 manifest, 두 claim/receipt/body이며 원문을 포함해 모두 보존한다.
  파일 크기 합은 788,248bytes다.
  Body1 SHA `892f9c5c5051be32990e4b7662f43b04ed5dd3ae94801ec1f35e51a2901c7ff7`.
  [receipt1](reports/mmv_public_capture_v1_run/receipt_1.json),
  [receipt2](reports/mmv_public_capture_v1_run/receipt_2.json)에 상태·시각·핀을 남겼다.
- [독립 저장 문서 감사](reports/mmv_public_capture_v1_audit.json)는 PASS였다. Reviewer가
  7개 산출물의 5개 JSON 핀·두 body 해시/크기, literal canonical, JSON-LD, 기록된
  요청 범위를 직접 검산했다. 외부 창 73.307353초, 감사 계산 0.004307초다.
  Root가 reviewer 반환을 JSON으로 옮긴 감사 기록이며 별도 실행형 auditor를 만들었다는
  뜻은 아니다. Audit의 네트워크/원시 사람자료/이전 EEG outcome/테스트/쓰기 0.
  JSON-LD 시작은 body1 4행15753열, canonical은 4행1397열이다. `@context` 어휘에
  `field`/`recordSet`이 있다는 사실을 실제 파일 schema로 읽지 않았다.

Academic-research의 출처·직접 관찰·미확인 조건 분리 원칙에 따라 claim
`26df38f63b33c11e`(QUALIFIED, 근거3개)와 기존 gap `3f973164c88eabce`를 갱신했다.
누적98claims/260evidence/45gaps, workspace11views 재생성과 SQLite quick_check `ok`.
신규 scholarly search/PDF card는 0이며 기존 cutoff2026-09-04를 바꾸어 문헌 최신성을
확장했다고 주장하지 않는다. Coordinate-worktree-changes에 따라 root 단독 writer와
읽기 전용 검토를 사용했으며 신규 worktree나 통합 충돌은 없었다.

## 다음 방향

이번으로 MMV의 동일 접근 루트 반복은 멈춘다. 별도 읽기 전용 확인에서 Eye-BCI도 이미
정확한 파일쌍 뒤 익명 파일 GET403으로 보류한 경로임을 확인했다.
[그때의 결과](post_gyro_candidate_feasibility_v1_results.md)를 새 후보라고 다시 포장하거나
참가자를 바꿔 요청하지 않는다. 이번 확인만으로 DOI→Synapse 연결을 새로 검증한 것도 아니다.

다음은 [다른 공개 acquisition 정보원의 제한된 탐색 초안](public_acquisition_candidate_round_next.md)이다.
아직 접근 가능한 새로운 학습 후보를 확보하지 못했다. 문서 수집 성공을 metadata 효능으로
대체하지 않으며, 원 저보정 연구목표와 Q/Q2/QM/SHAM·공통 정보 대조, held60 보호를 유지한다.
