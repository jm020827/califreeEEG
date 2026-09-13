# 데이터 파일 중심 acquisition M 탐색 v1

2026-09-14 KST / 2026-09-13 UTC. Base6ef29a7. 직전 단계는 실제OpenBMI HTML 판독과
설계 경계 갱신으로PROGRESS였다. 기존 negative/종료후보/접근제한을 보존한다.

## 실행 전 문제와 예산

Representation=공개 SSVEP와 대응되는 실제 획득값; bottleneck=논문 관리기준과 배포필드의 차이;
operation=공개 데이터 설명/헤더schema/코드 정적판독; objective=Q/common 이상의 저보정
학습 후보 자격; feedback=실제필드·측정출처·시점·동일EEG대응; failure=nominal/Q/무관과제/접근제한.
측정된 전극좌표나 접촉상태가 support의 공간패턴 불확실성을 설명할 가능성은 가설이지 효능증거가 아니다.

학술 landscape와 기존 문서에서 EEG-BIDS의 nominal/digitized provenance 구분은 있었지만
새 독립/미종료 후보의 실제 SSVEP 파일 수준 적격 측정값 확보는 확인되지 않았다.
기존Wearable Impedance.mat 확보·실험이력까지없다고하지않는다. 이번에는data-native vocabulary로찾는다.

- 검색 최대4회: (1) `SSVEP "electrodes.tsv"`, (2) `SSVEP "impedance" "BrainVision"`,
  (3) `SSVEP dataset "digitized" electrodes`, (4) `SSVEP dataset "impedance" "vhdr"`.
  Branch=implementation, rationale=실제 저장필드/측정출처로 공개후보를 판별.
- 결과의 서로 다른 공식 문서/저장소 최대4targets. 재시도0, 기존 challenge URL 미사용.
  검색·페이지 도구의 숨은 네트워크 바이트/GET 수는 측정했다고 주장하지 않는다.
  직접curl 사용시 각URL1회/60초/20MiB/명시적redirect만budget에포함/TLS유지.
- HTML/공개README/목록/schema만 읽는다. 새numeric EEG/개인값 다운로드·PDF·fit·예측0.
  DOI/title 신원과측정임피던스/실측좌표여부·SSVEP대응·사전시점이 필요하다.
- 새후보 최대1개 추천. 없으면NONE으로 끝내고 같은query를 새예산으로 반복하지 않는다.
  raw/학습 승격은별도필드·역할·공정대조군·비용검증후에만 한다.
- 사람요청/계정/유료/held60/설치/삭제/push0. 출판cutoff기본2026-09-04는변경하지않고,
  웹검색은현재공개파일발견용이라날짜필터가강제된문헌검색이라고하지않는다.

## 소유권

Main6ef29a7/약290GiB/기존46worktrees/8untracked보존. Root단독문서·SQLite writer,
agent는기존문서읽기전용 coverage감사. 구현병렬lane없어새worktree0.
학습단계에서만QM/Q/Q2/조건부SHAM/directM과총보정비용을평가한다. 기존학습결과소급변경0.

## 실행 결과

완료: 검색4/4, 서로다른공식targets4/4. **PROVENANCE_CORRECTED_NO_NEW_ELIGIBLE_M**.
새학습후보는없지만,원본획득경로의잘못된식별을교정했고원본헤더검증의구체적이유를확보했다.

### 기존 공통 정보 계약 확인

독립 local review와 root 확인: `metadata_protocol_amendment.md` §5.1은 nominal뿐 아니라
session-digitized 좌표도 `m_struct`로 모든arm에 주도록 정했다. 따라서 이번검색에서실측좌표를
발견해도 QM만의새M처리로세지않는다. 별도geometry ablation과획득M가설을혼동하지않는다.
좌표가미검증이라는말은기존공통정보권한을바꿀수있다는뜻이아니다. 원계약을수정하지않았다.

### 공식 문서 target4 선택 — 실행 전

Targets1–3은MNE BrainVision reader문서, FieldTrip impedance FAQ, MNE SSVEP tutorial이다.
Target4는기존저자문헌이명시한별도최종배포Figshare16669072의공개metadata API
`https://api.figshare.com/v2/articles/16669072`다. 원문링크의article ID를공식API패턴에적용한
metadata주소이며literalpaperURL이라고하지않는다. 기존13604078 MAT획득계약을재사용하지않는다.
필드존재를가정하거나array를다운로드하지않고,명시적파일목록·버전·권리·헤더접근경로만검토한다.

### 결과와 출처

| 경로 | 직접 확인한 범위 | 판정 |
| --- | --- | --- |
| [MNE BrainVision reader](https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html), Notes | 원본header에값이있으면raw.impedances로읽지만save/reload후속성은남지않는다고명시 | 원본header확인필요. 특정SSVEPheader에값이있다는증거는아님 |
| [FieldTrip FAQ](https://www.fieldtriptoolbox.org/faq/preproc/dataformat/impedancecheck/), Read impedance values | PyCorder .vhdr의Impedance행을읽는예시 | 저장형식의구체적가능성. 실제측정파일배포/우리코호트효능아님 |
| [MNE SSVEP tutorial](https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html), Data and outline/Data preprocessing | 2명·12/15Hz·32wetEEG의header경로를제시하며standard easycap-M1 montage를설정 | engineering용후보단서일뿐,새실측좌표/impedance확보아님.2명을모집단검증자료로승격하지않음 |
| [Figshare16669072 API](https://api.figshare.com/v2/articles/16669072) | 제목Metadata record, v1, CC0, data.json4416bytes와metadata summary.csv293bytes만존재 | **raw EEG archive가아니다**. 과거‘16669072 BrainVision배포본’식별정정 |

API는1GET/HTTP200/application-json/4,802bytes/1.215908초/retry0/redirect0,
body SHA256 `cf7141c42a6e761a16045765e58f01dee349805c269c2a07a0fd1191ba9faa46`.
CC0는이metadata record의표시이고실제EEG의권리라고하지않는다.
원문body/header및MNE tool-rendered자료는연구workspace/source-probes/acquisition_file_native_v1에보존했다.
Web의바이트/내부GET수는미측정이다. MNE색인검색결과는1.12.1로표시됐으나실제열린stable문서는
1.13.2/updated2026-09-13이었다. 설치된로컬MNE버전이나실행검증으로대체하지않는다.

네검색의나머지결과는주로기존MobileBCI의명목threshold,비SSVEP VEP/orthosis,
디지털신호(digitized signal)를전극좌표로잘못매칭한결과였다. 실제과제/측정/공개필드확인없이는
후보로세지않았고검색snippet을원문정독으로기록하지않았다. 기존검색query재실행0이다.

### 기존 기록 수정과 새 경계

`mobilebci_public_pair_results.md`, `mobilebci_time_sources_v1.json`, round4보고서에
metadata record/raw distribution식별정정을추가했다. 과거source감사와MAT4초/다른문헌5초의
불일치관측은삭제하지않는다. 실제원자료위치는16669072번호만으로아직확정되지않았다.

Local read-only검토에따르면기존MobileBCI두다운로드폴더는96MAT이며BrainVisionheader는없다.
다른파일시스템전체를검사한것은아니다. 기존Wearable Impedance.mat 확보와부정결과는유효하다.
contact addendum의channelwise요구가모든session/block acquisition값을배제하지않도록명확화했다.
새필드가같은참가자의다른export에서나와도새독립코호트로세지않는다.

### 다음의 정확한 미해결 항목

공식record에나온 `data.json`(file31816823,4,416bytes,
MD5 `3412b6c38efa578f33e17ce51505b065`,
`https://ndownloader.figshare.com/files/31816823`)은아직받지않았다.
다음은이작은공개설명파일이원EEG저장소를정확히지시하는지확인하는것이다.
이는actualM파일후보승격이아니라raw배포본식별수정의후속작업이다. 이번4target예산은종료했다.
설명파일다음에임의로대용량archive나개인header를여는권한으로확대하지않는다.

학습진입에는실측값·단위·pre-query시점·조정전후·EEG대응·총비용이필요하다.
이번numeric EEG/개인좌표/impedance값/PDF/fit/예측/held60/외부사람요청/유료는모두0이다.
목표나실패판정은바꾸지않았으며,효능/보정절감을아직입증하지못했다.

## 후속: 실제 원자료 pointer 확인 — 실행 전

2026-09-14 KST / 2026-09-13 UTC, basebf5e148. 직전의4query/4target탐색은완료이며
재개하지않는다. 이번에는위에서확인된단일설명파일31816823부터출발한다.
질문은‘실제EEG배포경로·형식·권리가무엇인가’이며,측정값이나학습효과를추측하지않는다.

- 최대6HTTPS GET(redirect포함), 15분, 요청당60초/20MiB. 총120MiB상한이나
  최초파일은공식기재4416bytes/MD5와일치해야한다. AutoHTTPSredirect최대2/전송,
  누적요청수가6을넘지않게다음전송의redirect여유를줄인다. Retry0/TLS검증/curlrc무시.
- 이후에는받은설명파일의literal공개저장소/문서/목록pointer만확인한다.
  기록된공개저장소ID를공식metadata API에적용할때는유도방식을명시한다.
  새검색0/동일실패URL재시도0/ZIP·numericEEG·개인header·측정값·PDF·fit0.
- DOI/title/권리/파일역할을확인하고자료부재와metadata-only·접근실패를구분한다.
  401/403/429/challenge/계정요구시해당경로중단. 사람요청/유료/held60은별도승인원칙유지.
- Root단독writer,읽기전용독립review. Coordinate-worktree-changes에따라새worktree없이
  기존46개/8untracked/여유290GiB를보존한다. 과정의signed redirect query는보고서에노출하지않는다.

### 후속 결과 — 원자료 pointer는 기존 접근 실패 OSF와 동일

`data.json`을 실제 다운로드해 4,416bytes/MD5 일치를 확인했다. HTTP302→200,
2/6 GET, retry0, curl1.768052초다. 첫 응답 Date는 2026-09-13 17:34:15UTC,
최종 응답은17:34:17UTC다. body SHA256은
`7df70796066f7bfaf733c2bded73db2415d65fd8824d4988797a136707e0434b`다.

`/repositories/0/value`는 `https://doi.org/10.17605/OSF.IO/R7S9B`다. 하지만 기존
`reports/alternative_metadata_candidate_triage_v1_sources.json`의 `id=osf`에 같은
OSF 프로젝트의403 및 root API 접근 실패가 이미 기록돼 있다. 따라서 OSF/DOI/API를
재요청하지 않고 **POINTER_RESOLVED_TO_PREVIOUSLY_FAILED_ROUTE**로 종료한다.
남은4GET은 사용하지 않으며 다른 검색이나 raw 확보에 전용하지 않는다.

JSON에는 EEG/각속도/가속도/자기장이라는 측정 범주와 자극 주파수·참가자 속도라는
요인명이 있지만, 개인 측정값·파일 inventory·시간/채널/EEG 대응 schema는 없다.
`Data Descriptor License=CC-BY-4.0`도 설명 논문의 표시이지 OSF raw 권리 검증이 아니다.
새 dataset/독립 cohort/실측 M/학습 후보는0이며, 기존 Wearable/MobileBCI 실자료와
그 부정 결과는 그대로다. 공개되지 않았다고 단정하지 않고 **현재 접근 미검증**으로 남긴다.

독립 read-only 감사는 body/header 크기·hash·pointer·license 범위를 확인했다.
curl 실행 자체와 미기록 요청 부재까지 독립 검증한 것은 아니다. 상세 receipt는
[후속 상태](reports/mobilebci_original_pointer_v1_state.json)에 보존했다.
새 EEG/개인 header/PDF/fit/예측/held60/사람 요청/유료/설치/삭제/push는0이다.
