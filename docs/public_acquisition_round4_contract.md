# 다른 공개 acquisition 정보원 round4 — 실행 전 범위

2026-09-14 KST / 2026-09-13 UTC, base06625a2. 직전 turn은 MMV public landing
capture/독립 검산으로 progress였지만 inventory/schema 미확인으로 그 접근 루트는 종료됐다.
원 저보정 SSVEP의 외부 M 학습 이득/Q·공통 정보 대조/실제 보정 부담 감소 목표는 유지한다.

Signature: representation=공개 SSVEP와 대응되는 실측 acquisition context;
bottleneck=새로운 학습 적격 paired source의 미확보;
allowed=scholarly metadata와 primary 공식 문서;
objective=정확한 다음 schema 후보 최대1개 또는 NONE;
resources=2searches/4distinct official documents/0raw/0PDF/0fits;
feedback=새 release·field·pairing·clock·pre-query 가용성과 권리/비용의 직접 근거;
failure=접촉 임피던스 언급만으로 공개 측정값을 가정, datasetID/공통 설정을 추가 M로 재명명,
종료한 경로를 새 후보처럼 반복하거나 연구목표를 생리/시선 예측으로 바꾸기.

## 고정 검색과 source 범위

1. implementation: `SSVEP dry electrode dataset impedance`.
   이유: contact/interface 변동을 실측한 공개 SSVEP release/필드를 찾는다.
2. implementation: `SSVEP motion accelerometer dataset`.
   이유: 움직임의 실제 외부 측정과 EEG의 대응·시간/역할을 명시한 source를 찾는다.

각 query는 academic-research CLI space discover로1회. Sources=openalex,crossref,
max-results-per-source=8,merged limit=12,published-before=2026-09-14,
unknown publication dates 제외. 기존 workspace 기본 cutoff2026-09-04는 변경하지 않는다.
이번 두 query만 새 날짜범위를 명시한다. Provider metadata의 날짜오류/누락 및 실제
요청 fan-out은 한계로 기록한다. 이미429가 기록된 Semantic Scholar를 재시도하지 않는다.
CLI timeout180초/각,같은 query 재실행0. 출력은 검색 manifest 보존 후 compact 요약으로만
읽는다. SQLite root단독 writer라 두 discover는 순서대로 실행한다.

검색후 primary official paper/data/repository 문서 **최대4 distinct targets**를 직접
읽는다. 정확한 검색결과 URL 또는 그 문서의 명시된 링크만, 실패한 target 재시도0.
Browser/web tool의 underlying bytes·HTTP redirect 수를 계측했다고 주장하지 않는다.
성공한 도구 text representation은 원HTML/PDF인 척하지 않고 URL/time/locator와 보존한다.
추가 generic web search/다른query/PDF/raw EEG/annotations/fileheader/fit/outcome은0.
본문이 불충분하면 불충분한 것으로 종료하며 title/abstract를 fulltext로 승격하지 않는다.

MMV 현재shell/canonical,Eye-BCI403,Choi/기존 요청 의존 route,이전 MAMEM·mobile
종료경로의 동일 재시도는 제외한다. 새 공개 release/field 근거가 없으면 기존 후보를
개봉하지 않는다. 403/429/challenge/auth/계정/DUA/유료는 보존후해당route중단/우회0.
외부 작업 종료시각2026-09-13T17:00Z,학술검색2회나문서4개 이전이라도 적격후보1개 또는
증거포화이면 종료. 새 후보가 없으면 NONE을 보고하며 실패 뒤 예산을 보충하지 않는다.

## 판정·소유·후속

Classify: DOCUMENTED_SCHEMA_NEXT_CANDIDATE / PRELIMINARY_METADATA_ONLY /
PARK_ACCESS_UNRESOLVED / REJECT_SCOPE_OR_M / DUPLICATE_CLOSED_ROUTE.
At most1명확한 후보를 다음 최소파일/schema 확인으로 추천, learning eligible 판정0.
필수근거는 taskSSVEP,독립실측M의뜻,동일releaseEEG pairing,pre-query 시간/권한,
공통정보·정답shortcut구분,공개접근과장치/setup비용이다. 미확인을 부재로 단정하지 않는다.
실측값이 미래query를 포함하거나 class정답을 알려주면 허용된 M로 쓰지 않는다.

Root=검색/직접primary읽기/repo문서/SQLite solewriter. Local exclusion/skeptic agents=
읽기전용,network/raw/write0. Coordinate-worktree-changes에따라새worktree0,
기존46개·8untracked보존,free291GiB. Code/학습기수정이없으면 pytest0,
savedmetadata/claimhash/문서·판정교차검사만수행. 설치/삭제/push0.
원 목표 active; 신규raw/schema/학습은 여기서 자동실행하지 않는다. Held60·사람 요청·
유료는 별도 승인이다. 다음 후보의 기작과 예상차이는 실행전 별도 예산에 기록해야 한다.
