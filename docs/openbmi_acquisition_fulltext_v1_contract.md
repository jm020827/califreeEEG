# OpenBMI acquisition 필드 단일 원문 확인 — 실행 전 계약

2026-09-14 KST / 2026-09-13 UTC. Base `ff6f873`. PreG supplemental challenge경로는
중단했고 그 예산을 늘리지 않는다. 보존round4 metadata3파일의read-only비교로OpenBMI한편을
선정했다. 저자초록에다세션SSVEP가명시되지만실측acquisitionM공개는미확인이다.
기존 P3 representation/context분류를 바꾸거나설문을자동으로M로재정의하지않는다.

Representation은실측획득상태와SSVEP의paired정보, bottleneck은threshold/setup설명과
실제파일값의구분, allowedoperation은공개원문표적독서, objective/feedback은실제release/
필드/time근거유무, failure는추상적설정/설문뿐이거나원문접근실패다.

대상 DOI `10.1093/gigascience/giz002`, OpenAlex `W2912885887`.
새genericsearch0, exact source-native PDF1개:
`https://academic.oup.com/gigascience/article-pdf/8/5/giz002/28563177/giz002.pdf`.
이 URL은 기존search868060228aabfb30의paper record에있다. 이전실패DOI/silverchair주소는
재요청하지않는다. 실제response의DOI/title가일치해야 독서한다.

예산: 첫PDF GET1회, literalHTTPS redirect가반환된경우에만다른정확한공식문서주소1회까지
수동확인(총2GET). 동일실패주소redirect/401/403/429/challenge/다른논문이면즉시STOP.
자동redirect/retry0, TLS유지/auth·cookie·netrc0, fixedUA/curlrc무시.
각20MiB/60초/connect15초, 총40MiB, 첫request부터10분창. file-size subprocess상한과
Content-Length cap. body/headers/curlJSON/hash/partial/실패보존. PDF얻으면즉시추가접근종료.
새OAmetadata조회/우회/다른endpoint추측/저자요청/계정/유료0.

사전에 deep-read queue 등록했다. PDFskill없어pdfinfo/pdftotext/pdftoppm/view_image대체.
필요Methods/Dataavailability/기록schema설명을targeted읽고레이아웃의존page렌더.
원문확보전에는원문결론을주장하지않고,card/evidence갱신은실제로읽은직후한다.
실제값의공개파일/단위/시점/SSVEPjoin이있으면후속schema후보로만추천;threshold/설문/
EEG-derivedQ만명시되면본목표후보추천안함. 불명확한경우UNKNOWN이며부재증명으로안씀.

Rootsolewriter/SQLite단독, read-onlyagent는후보비교·독립본문검토. 여유290GiB/새worktree0,
기존46worktrees/8untracked유지. 새raw/수치EEG/fit/예측/held60/설치/삭제/push0.
Q/common/Q2/SHAM/directM·전체비용 및기존negative보존. 이계약은목표·실험후보변경이아니다.
