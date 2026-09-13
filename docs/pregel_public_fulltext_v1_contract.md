# Pre-Gelled 단일 공개 원문 확인 v1 — 실행 전 계약

2026-09-14 KST / 2026-09-13 UTC. Base `dd8a1d8`.
사용자의 용량 확보 후 계속 지시에 따라 여유290GiB를 확인했다. 앞선 round4 예산은 종료됐으며,
본 작업은 그 예산을 늘리는 broad search가 아니라 이미 queued된 논문 한 편의 공개 원문 확인이다.

## 문제와 결정

- 대상: *A Pre-Gelled EEG Electrode and Its Application in SSVEP-Based BCI*,
  DOI `10.1109/tnsre.2022.3161989`, OpenAlex `W4220855852`.
- Representation: 실제 접촉 임피던스와 SSVEP EEG의 대응 관계. 현재는 색인 초록뿐이다.
- Bottleneck: raw paired release인지 aggregate hardware 비교인지 미확인.
- Allowed operation: 공개 문서 해석/다운로드와 targeted fulltext 독서, 로컬 기록.
- Objective/feedback: release/file/단위/clock/prequery/cost 근거로 후속 schema 진입 여부 결정.
- Failure: 원문 미확보, 공개 paired 데이터 근거 없음, common electrode label만 존재.
- 새로운 모델·Q/Q2/QM/SHAM outcome, 실제 EEG·임피던스 원자료 값, held60는 다루지 않는다.

## 경로와 예산

1. 기존 검색 manifest에 이미 있는 정확한 source-native publisher PDF URL 1회:
   `https://ieeexplore.ieee.org/ielx7/7333/9695946/09740692.pdf`.
   실패했던 DOI safe-open과 다른 구체적 공개 PDF 후보이며, PDF_URL은 접근 성공의 증거가 아니다.
2. 공개 문서를 못 얻으면 정확한 OpenAlex work metadata GET 1회로 공식 OA 저장소 위치를 확인할 수 있다:
   `https://api.openalex.org/works/W4220855852`.
   기존 IEEE endpoint를 재요청하지 않고 OA로 명시된 다른 공식 repository/author manuscript만 허용한다.
3. metadata에 실제로 명시된 공개 저장소 원문/landing을 최대2 distinct URL 확인할 수 있다.
   각 URL의 선택 이유를 실행 전에 기록한다. 새 제목/keyword 검색·DOI/publisher 재시도는0.
   알려진 접근 제한을 우회하는 mirror/proxy/인증 사용은 금지한다.

총 raw HTTP GET 최대4회(redirect도 별도 승인된 request로 센다), 자동redirect0/retry0,
각60초+connect15초, 각 body 최대20MiB, 전체 보존80MiB 이하. 첫 요청에서 마지막 요청까지
15분 이내에 종료하며 성공 PDF는1개만 확보하고 즉시 독서한다. 시작시각은 첫 실제 요청 직전에 기록한다.
여유 디스크20GiB 미만이면 시작하지 않는다. curlrc 무시, TLS검증 유지, cookie/auth/netrc0.
curl의 Content-Length cap 외에 subprocess file-size limit으로 크기 미상 응답도 제한한다.
HTTP/status/MIME/magic/PDF parser를 확인하고 부분파일/오류/headers/시간/hash를 보존한다.
403/401/429/challenge면 해당경로 종료, 계정/요청/유료로 넘어가지 않는다.
상한은 retained body/file size이며 전체 wire-byte 측정이나 packet audit라고 주장하지 않는다.

## 도구·독서·소유권

`resolve_full_text`의 기존 legacy fallback은 여러 provider/후보 자동download, retry와
TLS검증해제 경로가 있어 본 계약에 맞지 않는다. 이 함수를 실행하거나 외부 backend를 고치지 않고,
위 exact URL에 대한 bounded anonymous transport를 사용한다. 정적 코드 확인은 접근 실행이 아니다.

academic-research의 queued 질문에 따라 Methods, impedance 비교, SSVEP protocol,
Data Availability/limitations를 targeted review한다. PDF skill은 현재 세션에 없으므로
기존 pdfinfo/pdftotext/pdftoppm 및 image viewer로 대체하고 hash/revision/page/locator를 기록한다.
표/그림에 의존하는 페이지는 렌더링한다. 읽자마자 paper card를 갱신한다.

Root는 repo/연구SQLite의 단독 writer. Agent는 읽기 전용 도구 정적 검토 및 저장 산출물 검토.
새 worktree0, 기존46/기존8untracked 보존. 공통 정보와 Q를 넘는 M, support-only,
누출 대조군과 총 calibration/setup비용 조건 유지. 기존 negative 결과 불변.

결과: DOCUMENTED_SCHEMA_NEXT_CANDIDATE / NO_RELEASE_OR_RELEVANT_M_DOCUMENTED /
PARK_PUBLIC_FULLTEXT_UNRESOLVED. 마지막 두 결과는 모든 데이터/metadata의 부정을 뜻하지 않는다.
Raw download/fit/EEGprediction/held60/사람요청/계정/유료/설치/삭제/push는0이다.
