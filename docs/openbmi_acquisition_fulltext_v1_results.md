# OpenBMI 원문 확인: PDF 대신 challenge 반환, acquisition 적격성 미확인

2026-09-14 KST / 2026-09-13 UTC. [계약](openbmi_acquisition_fulltext_v1_contract.md)
`0011f30` 이후 exact source-native PDF를1회 요청했으나 **HTTP403/challenge HTML**이
반환됐다. `PARK_PUBLIC_FULLTEXT_UNRESOLVED`, 새 원문·raw·학습 후보0이다.

## 왜 이 문헌을 골랐나

PreG 보충자료 route중단 후, read-only scout와root가 round4의 보존metadata를 대조했다.
PreG와기존Wearable/Mobile를 제외한 잔여항목 중 OpenBMI는 색인된저자초록에 다세션
SSVEP데이터가 명시돼있고 exact PDF URL도 있었다. 다른 제목-only접촉논문이나 자극설계/
decoder논문보다 **실제 측정값의 공개 여부를 문헌한편으로 가르는 효용**이 높다고 판단했다.
이는 abstract-level 독서추천이지 새로운 데이터셋·학습후보 추천이 아니다.

논문: *EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy*,
DOI `10.1093/gigascience/giz002`, OpenAlex `W2912885887`.
선정근거는 `searches/2026-09-13T161145-ssvep-dry-electrode-dataset-impedance.json`의
author abstract다. 원문의 Methods/Results/DataAvailability는 읽지 못했다.
초록의 설문·휴지상태·artifact·EMG를 실측 acquisition M이라고 단정하지 않는다.
기존 [P3 representation/context 분류](research_significance_data_strategy.md)는 유지한다.

## 실제 수집 결과

기존검색manifest의 literal PDF URL:
`https://academic.oup.com/gigascience/article-pdf/8/5/giz002/28563177/giz002.pdf`.
실제응답403, `cf-mitigated: challenge`, `text/html; charset=UTF-8`,5632bytes였다.
파일검사도 HTML이며 PDF가 아니다. 성공 download로 세거나 PDF parser/독서를 진행하지 않았다.
Curl0.043451초, redirect0/retry0, request전16:49:38UTC/HTTPDate16:49:39GMT.
Header2119bytes. [전송기록](reports/openbmi_acquisition_fulltext_v1_transport.json).
원응답은 workspace `source-probes/openbmi_acquisition_fulltext_v1`에 로컬보존했다.
Challenge본문/opaque token을 보고서나Git에 복사하지 않는다.

- bodySHA256 `a30859872012b3f9ef0a4bbefec570174c52530351a3eb0f2bd7357d881ab7ad`.
- headersSHA256 `02af7b17bbac3d4c8d64d9c18e734652f27e88889654d0a228cbe8f3a2b0893d`.
- 예산2GET중1회에서중단. 다른URL·UA·브라우저·계정·저자문의·유료·우회0.
- Deep-read는다운로드전등록했으며 **queued유지**. 완료로세지않고abstract-onlycard만저장했다.

## 이 결과가 뜻하는 것과 다음 조건

출판사주소의 현재 익명접근이 실패했다는 관측이다. 논문·데이터가 비공개, paired값이 없다,
metadata가무용하다는결론이 아니다. 이전 DOI/silverchair도구오류와이번HTTP403을구분한다.
같은publisherPDF주소는새접근변경없이재시도하지않는다.

후속 작업을 한다면 별도예산으로 정확한OpenAlex work의 **공식 OA repository location**
한번을확인해볼수있다. 이는이번계약에서실행하지않았고, challenge를proxy/로그인/TLS완화로
우회하는방식은허용하지않는다. 법적으로공개된사본을확인한뒤에만기존queued질문을읽는다.
공개복사본없음도하나의metadata결과만으로확정하지않되 반복검색으로늘리지않는다.

전체연구목표는유지되고, 현재신규acquisitionM적격후보는NONE이다. 실제값·파일/clock·
support권한·Q/common이상의정보·전체준비비용조건을만족하기전에는학습으로넘어가지않는다.
새raw/fit/EEG예측/held60/발송/유료0, 기존실패보존. 이번두접근결과는학습가설의부정결과가아니다.

[두응답의독립저장감사](reports/pregel_openbmi_access_review.json)는status/challengeheader/
bytes/hash/기록된시간·redirect를대조해PASS였다. 실제제한옵션/미기록요청부재의독립network감사는
아니다. Root가agent반환을옮겼으며새실행형auditor를구현하지않았다.
[최종상태](reports/openbmi_acquisition_fulltext_v1_state.json).

Academic-research에따라abstract-onlycard/queued독서/접근근거를분리했다.
Claimd5cdfabe65fae3d5 QUALIFIED/근거2개와기존gap갱신. Catalog1151/search92/card48/
technique18/claim102/evidence270/gap45/analogy6/deepread40(completed28/queued11/skipped1).
Render11views/SQLitequick_checkok. Coordinate-worktree-changes에따라rootsolewriter+
read-only검토로진행,새worktree0/기존46·8untracked보존,코드변경없어pytest0이다.
