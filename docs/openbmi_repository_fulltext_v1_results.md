# OpenBMI 공식 OA 사본 위치 확인: 본문 수집은 아직 미완료

2026-09-14 KST / 2026-09-13 UTC. [계약](openbmi_repository_fulltext_v1_contract.md)
`374564b`의 4GET를 실행했다. **공식OA위치와최종canonical경로를확인했지만,원문본문은
아직확보하지못했다.** 같은차단publisher주소재시도나우회는하지않았다.

## 실제 경로

| 요청 | 근거·주소 | 관측 |
| --- | --- | --- |
| 1 | `https://api.openalex.org/works/W2912885887` | 200/application-json/30,377bytes; DOI·제목일치 |
| 2 | 반환`locations[2].landing_page_url`: `https://www.ncbi.nlm.nih.gov/pmc/articles/6501944` | 301, 새PMC도메인으로Location |
| 3 | request2의literalLocation: `https://pmc.ncbi.nlm.nih.gov/articles/6501944` | 301, relativeLocation `/articles/6501944/` |
| 4 | relativeLocation을같은origin으로해결: `https://pmc.ncbi.nlm.nih.gov/articles/6501944/` | 301, `/articles/PMC6501944/`로Location |

다음주소는 [공식PMC canonical](https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/)이며
**이번에는요청하지않았다.** URL을추측한것이아니라실제반환Location으로확인했다.
4회예산소진후5번째GET를추가하지않았다. Auto-redirect0/retry0/새genericsearch0.
본문보존합계30,670bytes, 기록된curl시간합계2.401983초. 이는요청사이판독/대기시간을
포함한총경과시간이나전체wirebytes가아니다. [전송기록](reports/openbmi_repository_fulltext_v1_transport.json).

## 확인한 것과 미확인인 것

OpenAlex는해당DOI의PubMedCentral location을 `is_oa=true`, `license=cc-by`,
`version=submittedVersion`으로반환했다. `best_oa_location`은여전히차단됐던OUP PDF라
선택하지않았다. OA표기·버전·license는**색인metadata의보고**이며,실제PMC본문의출판버전/
라이선스를확인한것은아니다. 다른대학repository항목도있지만추가요청하지않았다.

NCBI→PMC도메인, trailing slash, PMC식별자prefix를서버가순차정규화했다.
이응답들은접근challenge가아니라주소이동이다. 반대로정상redirect라는것만으로최종본문
접근성까지확인한것은아니다. 현재 `PARK_UNRESOLVED`는**원문미수집/예산종료**를뜻하며,
공식사본이없다는뜻이나원래OUP challenge를해결했다는뜻이아니다.

기존deep-read는queued/카드는abstract-only유지. Methods/DataAvailability/실측impedance/
SSVEP파일join/총비용을새로읽었다고하지않는다. 신규raw·PDF·학습후보·fit·EEG예측0.
저보정목표·Q/common/Q2/SHAM/directM·held60봉인·기존negative는그대로다.

## 다음 결정

후속은이미확인된canonical주소에서**한편의본문을얻고읽는것**이다. 독립적OA경로
탐색은여기서종료하며OpenAlex조회나세redirect를다시밟지않는다. 새실행전에문서GET/
실제PDF링크확인예산을고정하고,challenge나인증요구시멈춘다. 본문을얻기전새실험후보나
raw다운로드를시작하지않는다. 새로운본문근거가없으면현재P3/context분류를유지한다.

Academic-research의출처·읽기수준구분에따라원문미확보를명시했고,
coordinate-worktree-changes에따라rootsolewriter와read-only저장artifact검토를사용했다.
이번진전은exactOA경로확인이지metadata학습효능·보정량감소의진전으로세지않는다.

[독립body/header감사](reports/openbmi_repository_fulltext_v1_audit.json)PASS:
8artifact의hash/bytes/DOI·title/locations[2]/세Location연결을검산했다. Agent읽기시점에는
transportJSON이없었으므로curl시간·auto-redirect집계는독립대조범위에포함하지않는다.
Root의실행기록과검산은[상태](reports/openbmi_repository_fulltext_v1_state.json)에구분했다.
Record4선택의provenance문구를literal상대경로+origin해결로명확히했으며요청URL은변경없다.

Claim81f9eed5675c80db QUALIFIED/근거2개와기존gap갱신. Catalog1151/search92/card48/
technique18/claim103/evidence272/gap45/analogy6/deepread40(completed28/queued11/skipped1).
Render11/SQLitequick_checkok,코드변경없어pytest0. 기존46worktrees/8untracked보존.
