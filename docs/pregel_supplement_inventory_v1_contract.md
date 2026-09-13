# PreG 보충자료 inventory 확인 v1 — 실행 전 계약

2026-09-14 KST / 2026-09-13 UTC, base `1a4cda3`, root solewriter.
직전 goal turn은 공식 PDF 확보·표적정독·설계추가규칙 반영으로 PROGRESS다.
원목표인 학습M의 Q/common 이상 이득과 실제 보정절감은 여전히 미입증이다.

## 질문·근거

Representation: 실제 acquisition impedance와 SSVEP EEG의 연결. Bottleneck: 정확한 공개
파일 release/측정시점/join/cost 미확인. Allowed operation: 공식 보충자료 inventory와
필요한 설명 문서 확인. Objective/feedback: 후속 raw/schema 자격검사를 할 구체적 근거가
있는지 결정. Failure: 설명그림/시연영상뿐이거나 pairing 미확인, 또는 접근 제한.
이는 새 EEG outcome/새모델 실험이 아니다.

로컬 공식논문 PDF(p.1각주, p.4 II.G)는 supplementary material/Video1/Fig.S1–S4를
가리키지만 실제파일주소는 담지 않는다. pdfinfo -url 결과도 ORCID4개뿐이었다.
이전 성공 PDF URL의 accession `09740692`에서 **IEEE document번호9740692를 유도**하여
다음 공식 landing route를 첫 요청으로 고정한다:
`https://ieeexplore.ieee.org/document/9740692`.
이 주소는 원문에 literal링크가 있었다는 주장이 아니며, 정확한 논문인지 response에서 검증한다.
이전실패 DOI와 성공PDF를 재요청하지 않는다.

## 예산·중단

- 새검색0. 공식 landing1회, 여기서 literal로 확인한 metadata/보충자료 링크에만 후속GET.
- 총 HTTPS GET 최대4회, distinct URL만, 자동redirect0/retry0, 첫요청부터15분이내.
  수동redirect도1request로센다. 목적과 실제URL provenance를 다음요청 전에 기록한다.
- 각 connect15초/total60초, retainedbody20MiB이하, 총80MiB이하. curlrc무시,
  TLS검증유지, auth/cookie/netrc0, useragent고정. Content-Length cap과 subprocess
  file-size limit 사용. curl JSON·headers·body/hash·실패를 workspace에 보존한다.
- HTTP401/403/429/challenge이면 해당경로STOP, UA변경/계정/유료/저자요청/우회없음.
  파일이없다는판정으로확대하지 않는다. 의도하지않은MIME/다른논문이면 STOP.
- 다운로드 대상은 HTML/JSON/XML/설명PDF/TXT뿐. Raw EEG/임피던스 MAT/CSV/archive와
  동영상은 inventory만 보고 다운로드/실행하지 않는다. signed/auth URL은 사용하지 않는다.
- 보충PDF가 필요하면 먼저 decision-specific deep-read를 등록하고 PDF1개까지 읽는다.
  세션에 PDFskill없으므로 기존 extraction/render 도구로 대체, hash/pages/locators 기록.
  논문본체card와보충문서근거를구분하며fulltext_complete라고하지않는다.
- disk약290GiB; 신규보존최대80MiB, 여유20GiB미만이면실행중단. 새worktree/설치/삭제0.

## 자격 판정

EXACT_RELEASE_SCHEMA_NEXT는 공개 actualfiles/설명으로 사람·session·채널·측정단위·시점·
SSVEP pairing 후보가 확인될 때만 추천한다. 이는 raw학습 승인이 아니다.
그 외 DOCUMENT_ONLY_NO_PAIRED_RELEASE_IDENTIFIED 또는 PARK_ACCESS_UNRESOLVED.
문서/영상만 확인한 결과를 EEG나 M값 전체가 없다는 증명으로 쓰지 않는다.
기존실패·Q/Q2/QM/조건부SHAM/directM/전체비용 조건과held60봉인을유지한다.

Root는 repo/SQLite/수집의단독writer; read-only agent는 자격조건·저장근거를 검토한다.
새 학습·예측·held60·사람요청·계정·유료·push0. 예산 소진 또는 판정 가능 시 종료하고 기록한다.
