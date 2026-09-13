# OpenBMI 공식 OA 저장소 원문 확인 v1 — 실행 전 계약

2026-09-14 KST / 2026-09-13 UTC. Base `cb88af2`.
직전goalturn은실제access응답2건과개인1회실측M설계명확화로PROGRESS였지만원목표는미완료다.
이계약은차단publisher주소재시도가아니라독립적인공식공개저장소의새출처확인이다.

문제signature: representation=외부실측acquisition상태–SSVEP연결; bottleneck=공개필드와
timing미확인; allowedoperation=정확한OA위치조회/공개논문표적독서; objective=실제release/
schema후보여부결정; constraint=원목표/P3분류/기존실패보존; feedback=Methods/DataAvailability
의구체적값·파일·단위·시점근거; failure=nominal설정/설문뿐또는접근미해결.

## 경로와 총예산

1. 공식OpenAlex work metadata1GET: `https://api.openalex.org/works/W2912885887`.
   DOI10.1093/gigascience/giz002와제목일치,locations/best_oa_location/권리표기를검사한다.
2. 반환metadata에직접명시된독립적인공식OA repository landing/PDF만선택.
   PMC/EuropePMC/기관저장소등합법적저자또는공식사본이어야한다. Publisher OUP/silverchair
   실패주소,proxy/mirror나인증주소는선택하지않는다. metadata에없는URL을추측하지않는다.
3. Landing에서literal로확인한PDF/설명문서또는HTTPSredirect만추가한다.
   각선택의URL·provenance·이유를GET전에기록. Metadata의OA선언을실제원문접근과구분한다.

총HTTPS GET최대4회(1metadata+최대3문서/수동redirect),각URL1회,자동redirect/retry0.
각connect15초/total60초/retainedbody20MiB,총80MiB;첫GET부터15분이내.
curlrc무시/fixedUA/TLS검증유지,auth/cookie/netrc0. File-size subprocess상한과
Content-Length cap,headers/body/curlJSON/hash/부분파일보존. 원문PDF성공1개면추가접근종료.
유효한HTML/XML전문이면먼저Methods/Availability만확인하되PDF를읽었다고하지않는다.
403/401/429/challenge/논문불일치/잘못된MIME/예산소진이면해당경로중단,접근우회0.
이경로도실패하면publisher나동일metadata조회재시도/새키워드검색으로연장하지않는다.

## 읽기·결정·소유권

이미queued된deep-read의실측vsthreshold,공개file,시간·단위,SSVEPpairing/cost질문을사용.
PDFskill없어기존pdfinfo/pdftotext/pdftoppm/view_image로대체. 실제hash/revision/pages/locators,
표·그림렌더확인을기록. Targetedfulltext로한정하고원문card는읽자마자갱신한다.
HTML/XML만읽었다면그형식과locator를명시하고PDFprovenance를만들지않는다.

실측외부M공개근거가있으면NEXT_SCHEMA_CANDIDATE(학습승격아님),
설정/설문/EEG-derivedQ뿐이면NO_ACQUISITION_CANDIDATE_IDENTIFIED_IN_READ_SCOPE,
불명확하면PARK_UNRESOLVED. 파일부재전체증명이나P3에서학습후보로자동변경은아니다.
Rootsolewriter/SQLite단독,독립agent는read-only원문/저장artifact검토. 새worktree0,
기존46/8untracked보존. 여유290GiB,20GiB미만이면시작금지. 새raw/학습/EEG예측/
held60/외부사람요청/계정/유료/설치/삭제/push0. Q/common/Q2/SHAM/directM·총비용조건유지.
