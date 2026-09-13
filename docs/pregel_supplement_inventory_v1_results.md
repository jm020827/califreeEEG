# PreG 보충자료 inventory: 접근 challenge로 보류

2026-09-14 KST / 2026-09-13 UTC. [계약](pregel_supplement_inventory_v1_contract.md)
`5f3d7bc` 이후 **1GET에서 중단**했다. 판정 `PARK_ACCESS_UNRESOLVED`.
보충자료 목록·실제 paired 파일은 확인하지 못했다. 지난번 성공한 논문 PDF와 표적정독 결과는
그대로 유효하며, 이번 결과로 논문이나 모든 데이터가 비공개라고 바꾸지 않는다.

## 실제 접근

로컬 PDF URL의 accession09740692로부터 유도한 공식 document9740692 route를 사용했다.
논문에 literal 링크가 있었다고 주장하지 않는다. PDF내 URI annotation은 ORCID4개뿐이었다.
`https://ieeexplore.ieee.org/document/9740692`의 HTTP응답은202,
header `x-amzn-waf-action: challenge`, body0bytes였다. 출판사 논문내용은 얻지 못했으므로
페이지에서 DOI/제목 일치를 검증한 것은 아니다.

첫 요청전16:44:05UTC, curl0.230857초, HTTP Date16:44:06GMT. Curl기록에0redirect,
본 실행retry0/추가request0/새검색0. [전송기록](reports/pregel_supplement_inventory_v1_transport.json).
Body0은 반환본문이 비어있다는 뜻이지 보충자료 파일이0개라는 뜻이 아니다.
Header556bytes/SHA256 `f5bd515f7ad31c7eb3b745689d83e5337a8ae922f2d1dca6b61a8c6f21464f5c`.
EmptybodySHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
원문/headers는 연구workspace `source-probes/pregel_supplement_inventory_v1`에 보존했다.

계약상 challenge 중단조건이므로 남은3request를 쓰지 않았다. 브라우저/UA/주소변형/저자문의/
계정/유료/접근우회0. 성공PDF재다운로드·다른supplementfilename추측0. 새PDF/raw/fit/EEG예측0.

## 연구 판단

현재는 PreG를 새 공개 학습데이터로 추천할 근거가 없다. 이 접근 경로는 보류하며 새로운
공식 접근변경·직접 확인된 공개목록 없이 재시도하지 않는다. 학습M 질문이 부정됐다는 뜻은 아니다.

읽기전용 reviewer는 [측정 추가 규칙](acquisition_contact_measurement_addendum.md)의
ID암기 금지와 개인1회실측M 허용을 더 명확히 하도록 지적했고 문구를 수정했다.
시간미확인은 future/query 사용이 입증된 경우와 구분한다. Fig.S1–S4/Video1의 존재만으로
raw파일이나원자료부재를 추정하지 않는다. 새raw schema 진입에는 실제release/파일·시간·
개입상태·단위·cost근거가 필요하다.

다음은 기존round4 예비후보의 저장metadata를 대조해 새 원문 독서1개의 의사결정효용을
평가하는 일이다. 같은PreG/이미종료한Wearable/Mobile기작을 새 이름으로 재개하지 않는다.
후보선택과 별도 예산을 실행 전에 정하며, 저보정 목표·Q/Q2/QM/SHAM/directM와
held60·외부사람요청·유료 별도승인 조건은 유지한다.
