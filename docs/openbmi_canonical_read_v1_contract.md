# OpenBMI canonical 본문·획득필드 판독 v1 — 실행 전 계약

2026-09-14 KST / 2026-09-13 UTC. Base `c5bc4a3`.
직전단계는공식OA위치·세redirect·정확한canonical확인으로PROGRESS였고4GET예산은종료했다.
이번에는미요청이었던canonical부터시작한다. OpenAlex/oldredirect/OUP/PreG는재요청하지않는다.

목표: 원저보정SSVEP질문에쓸실제acquisition측정값의공개여부와SSVEP연결을본문에서확인한다.
Representation=외부획득상태–EEGpair; bottleneck=threshold/설문과실측release의구분;
allowedoperation=공식본문·설명문서판독; feedback=실제file/단위/시점/공개release/cost근거;
failure=미확인/범위불일치/접근제한. Nominal/Q/설문을실측M로자동변경하지않는다.

첫URL: `https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/`.
이는전단계request4 Location의literal상대경로를같은PMC origin에해결한주소다.
그다음은본문에literal로제시된공식PDF·supplement설명·data-availability/readme만허용한다.
DOI/title/권리/버전확인,각추가URL선정이유를실행전에기록한다.

총HTTPS GET최대6회(redirect포함),각URL1회,auto-redirect/retry0,첫요청부터20분이내.
각connect15초/total60초/retainedbody20MiB,총120MiB. Curlrc무시/fixedUA/TLS유지,
cookie/auth/netrc0. Content-Length/file-size subprocess상한을함께사용하고
body/headers/curlJSON/hash/실패를연구workspace에보존한다.
401/403/429/challenge/논문불일치/잘못된MIME면해당경로중단,우회/URL추측/계정/유료0.
필요한본문근거를확보하면남은슬롯을채우지않고판독·기록으로넘어간다.

Queueddeep-read질문: 실측impedance/획득값인지상한설정인지; 공개file/변수명/단위;
사람sessionchannel과SSVEPjoin; support/pre-query시점; 총준비/측정비용.
PDFskill은세션에없어pdfinfo/pdftotext/pdftoppm/view_image로대체. 표/그림의존page렌더,
실제PDFhash·revision·page/section기록후즉시card갱신. HTML/XML만확보하면그형식과
sectionlocator만주장하고PDF를만들어낸척하지않는다. fulltext_complete로과장하지않는다.

결론: NEXT_SCHEMA_CANDIDATE / NO_ACQUISITION_CANDIDATE_IDENTIFIED_IN_READ_SCOPE /
PARK_UNRESOLVED. 본문밖원자료부재의증명으로쓰지않으며P3/context/원목표를자동변경하지않는다.
실제raw EEG/개인임피던스값file/ZIP/video/학습/예측/held60/사람요청/설치/삭제/push0.
Rootsolewriter/연구SQLite단독,agent는읽기전용원문·artifact감사. 새worktree0/기존46·8untracked
유지,여유약290GiB(20GiB미만시시작금지). 기존negative/Q/common/Q2/SHAM/directM·총비용조건유지.

## 2026-09-13 17:15 UTC — 요청5 전 공개 경로 규칙 보완

요청3의 공식 DOI가 `http://gigadb.org/dataset/100542`를 반환했다. HTTP는 요청하지 않는다.
동일 host/path의 `https://gigadb.org/dataset/100542`로 protocol만 강화한 주소를 요청5로
허용한다. 이는 literal HTTPS link가 아니라 명시적인 transport normalization이다.
기존 URL 추측 금지를 이 한 주소에 한해 보완하며 사후에 원계약 그대로였다고 주장하지 않는다.
과학적 판정·파일 범위·전체6GET·20분·TLS검증·challenge중단 조건은 그대로다.
공개 inventory의 나머지1요청만 허용하며 raw EEG/개인값/ZIP 다운로드는 계속0이다.
