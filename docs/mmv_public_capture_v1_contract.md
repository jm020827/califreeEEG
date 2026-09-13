# MMV 공개 문서 capture v1 — 실행 전 계약

2026-09-14 KST / 2026-09-13 UTC, base36855a3. 직전 goal turn은 실제 공통 baseline
300예측과 감사 완료로 progress였다. 이번은 공개 문서 자격 확인이며 학습 실험이 아니다.
원 저보정 SSVEP 목표와 기존 모든 부정 결과, held60·사람 요청·유료 별도 승인 유지.

Problem signature: representation=release inventory/README/loader schema;
bottleneck=과거 ScienceDB 응답을 로컬에 보존하지 못해 access와 pairing이 미확정;
allowed=익명 공개 문서의 bounded GET; objective=정확한 pair/필드/clock 근거의 자격 판단;
resources=최대4 HTTP attempts·2MiB body/attempt·8MiB 전체 body;
feedback=보존 status/URL/hash/body와 문서의 정확한 필드 정의;
failure=도구 capture 성공을 학습 적격성으로 혼동, 숨은 API 추측, physiology/정답을 M로 대체.

## 고정 경로·예산

첫 URL은 `https://www.scidb.cn/detail?dataSetId=270bcdeaab0c48adb8eee700479daafd`.
이후 최대3개는 보존된 성공 문서에 **문자 그대로 연결된** 문서/asset/redirect만 선택한다.
Same official host family(scidb.cn/sciencedb.cn 및 www) HTTPS/443만 허용한다.
Fragments는 제거하되 URL path/query를 임의로 합성하지 않는다. DataCite 성공 원문은
기존 기록 재사용, 재요청0. PDF·raw EEG/annotation/numeric tracker 파일·계정·DUA·
발송·유료·브라우저 session·JS 실행·generic search·추측 endpoint·downloader install0.

HTTP redirect 자동 추적0; Location은 별도 요청 슬롯을 쓰고 다음 요청 전에 적격성 확인.
URL당 최대1회, 동일/순환 URL 거부. 모든 HTTP 실패도 슬롯에 포함한다. 403/429/
challenge/auth 요구·예외·size cap·미허용 MIME이면 보존 후 이 경로 종료, 우회/재시도0.
성공적인 페이지의 일반 login 링크만으로 접근 거부로 단정하지 않는다.

Request당30wall초; 첫 HTTP 시도 직전 시작되는 전체 외부 작업180wall초. 분석/다음링크
선택도 전체시간에 포함한다. 최종 시작 허용 deadline2026-09-13T16:20:00Z.
한번에64KiB 이하를 읽고2MiB에 도달하면 EOF 추정을 하지 않고 보수적으로 cap 종료한다.
identity encoding 요청, 비identity 응답은 decode하지 않고 종료. MIME은 HTML/plain/
JSON/JavaScript만; PDF/MAT/EDF/CNT/archive 거부. HTTP header/transport buffer는
이 body 측정과 별개이며 총 wire bytes를 보장했다고 주장하지 않는다.

본문은 독점 생성된 로컬 body 파일에 바로 기록하고 정상/부분 응답 모두 보존한다.
HTTP status, MIME, Content-Length, body bytes/SHA, 오류 종류, 시각, code/contract
핀과 parent 문서의 URL 근거를 작은 JSON receipt에 저장한다. Cookie/auth header/일반
환경변수/큰 본문은 도구 출력에 싣지 않는다. 출력 경로는
`docs/reports/mmv_public_capture_v1_run`. 첫 manifest와 각 claim은 exclusive/fsync;
receipt가 없는 미완료 claim, 이전 종료실패, pins 변경 또는 기존 URL은 재개/재시도하지 않는다.
일반 SIGALRM 예외까지 receipt로 남기되 kill/powerloss 뒤 자동 복구를 주장하지 않는다.

## 구현 검증·소유

Root만 계약/작은 수집기/test/실제수집/보고/연구SQLite를 쓴다. 기존 exact-size PDF/MAT
downloader는 URL권한과 oversize partial 처리 목적이 달라 그대로 재사용하지 않는다.
Read-only reviewer 둘이 경계/기작·전송코드를 검토한다. 새 writer/worktree0,
기존46worktrees/8untracked 보존, 현재free292GiB. 공유 DB·네트워크는 root 단독.

Synthetic HTTP stream/URL/attempt/pin/deadline/partial/redirect 검사 최대2suite 각60초,
실제HTTP0인 상태에서 검증하고 code/contract commit 뒤 실제수집한다. 실제후 수집기수정0.
문서 receipt/hash/해석의 독립 read-only 감사1회; 원시수치·학습·EEG예측·held600.

## 판정과 종료

Transport `CAPTURED`는 바이트 보존 성공일 뿐이다. 공개목록/정확한 release·pair/session/
block, 비정답 acquisition 필드 정의, cue와 clock 대응, support/pre-query 가용성,
tracker calibration/전체획득 비용 및 EEG 학습과의 기작을 표로 확인한다.

- DOCUMENTED_PAIR_CANDIDATE: 문서 근거가 연결됨. 실제 schema/learning은 미검증.
- PARK_ACCESS_UNRESOLVED: 목록/문서/공개 접근/정의가 부족함. private라고 추정하지 않음.
- REJECT_M_OR_TASK_PROVENANCE: 정보가 정답/생리 재명명뿐이거나 시간/과제 근거가 모순됨.

LOST_DATA_EVENT는 stream gap이지 EEG noise가 아니다. Gaze/pupil/PERCLOS를 획득 M로
바꾸지 않는다. Query 정답·주파수/미래 보간을 predictor에 넣지 않는다. 공통 reference는
차후 모든 Q/Q2/QM/SHAM/common0/directM arm에 동일하게 제공해야 한다.
이번 예산에서 학습 후보나 raw 다운로드로 승격하지 않는다. 막히면 현재 MMV 루트는
보류하고 반복예산을 열지 않는다. 이후 다른 공개 후보는 별도 이유·예산으로만 진행한다.
