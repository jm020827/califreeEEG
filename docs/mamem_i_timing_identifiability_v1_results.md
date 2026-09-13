# MAMEM I 시간 metadata 식별 관문 결과

2026-09-13 KST. **배포본의 S011 이름은 설명서로 확인했다. 그러나 DIN 간격의 흔들림을
곧바로 물리적 자극 jitter라고 해석하는 후보는 보류한다.** Metadata의 예측 가치나
EEG+metadata 학습이 불가능하다는 결론은 아니다. 원 저보정 SSVEP 목표와 기존 부정
결과는 유지한다. 이번에는 새 EEG 기록/학습/정확도/보정절감 평가를 하지 않았다.

[실행 계획](mamem_i_timing_identifiability_v1_plan.md),
[생성 실험 영수증](reports/mamem_timing_identifiability_v1_generated.json),
[상태](reports/mamem_i_timing_identifiability_v1_state.json).

## 공개 원문에서 새로 확인한 것

### 배포본의 참가자 이름은 일부 해결

정확한 v1 첨부문서 [Data Acquisition – Detailed Description](https://ndownloader.figshare.com/files/3687771)
(29Jan2016), p1 Table1은 S001–S011을 나열한다. 따라서 실제 파일의S011을
이 배포본의 정상 ID로 유지한다. 저자 code의S013과 같은 사람이라는 연결은 없다.
자동 alias/rename하지 않으며 이 둘의 동일성 확인을 다른 참가자의 사용 조건으로
불필요하게 확대하지 않는다. 향후 adapter는 배포본의 filename namespace를 따른다.

같은 문서 p2는256channel HydroCel/EGI300,250Hz,60Hzdisplay와 ST100+light sensor를
보고한다. **257번째 EEG행의 의미·전체 channel 순서·reference는 정의하지 않는다.**
원자료를 새로 열거나 마지막 행을 임의로 삭제하지 않았다. Figure1은 acquisition
구성 예시이지 실제 모든 파일의 eventedge/export 옵션을 보증하는 설정 기록은 아니다.

읽은 공식 GitHub issue/comment 목록에서 참가자 alias나 row257 설명은 찾지 못했다.
[Maintainer의 행렬 방향 설명](https://github.com/MAMEM/eeg-processing-toolbox/issues/49#issuecomment-219976377)은
채널×시간 형식만 확인한다. [과거 누락파일 답변](https://github.com/MAMEM/eeg-processing-toolbox/issues/55#issuecomment-257247938)은
배포 누락 이력이며 S011/S013 연결 근거가 아니다. 댓글의 외부 파일 링크는 열지 않았다.
사용자의 ‘파일을 찾았다’는 댓글도 저자의 동일인 확인으로 취급하지 않았다.

### ST100–EGI의 eventedge는 실제 확인할 이유가 있다

Cedrus의 [An Important Update for EGI Users](https://cedrus.com/blog/update-for-egi-users.htm)
(게시13Jun2022, 현재 응답의 선택 본문) `What Happened?`, `The Fix`, `Second Issue`,
`How You Can Correct Your Data`는 다음을 설명한다.

- 원래ST100은 positive logic에 고정되어 있고 EGI는 negative logic을 사용한다.
- 이 불일치는 NetStation이 시작 대신 끝의 marker를 기록하는 문제를 만들 수 있다.
- NetStation4.5의 edge 선택과5의 기록 옵션은 다르며, 시청각 자극 보정에는 실제
  시작–끝 대응을 확인해야 한다. 현행 Duo/Quad의 firmware/성능 수치를 ST100에 전용하지 않는다.

이는 **MAMEM에서 오류가 실제 발생했다는 증거가 아니다.** 사용한 NetStation
설정·edge·배선·pulse 폭·export clock은 아직 확인하지 못했다. Dataset 문서의
‘onset’ 설명은 저자 보고로 보존하고, vendor 공지를 이유로 자동 offset 정정하지 않는다.
또 이 공지는 아래 인공 실험의 수ms 가변 지연이 실제 장비에 있었다는 증거도 아니다.

추가로 p3의 adaptation 본문100초와 Figure3의80초가 다르다. p4의 제외 세션/1104trial
설명만으로 파일 letter별 제외 이력을 완전히 복원할 수 없다. 이 설명과 p5 artifact
관찰을 새 학습 입력·대상 선택에 쓰지 않았다. 보정비용은 실제 acquired-prefix와
장치 설정 비용으로 계산하고, 편리한 nominal 시간만 택하지 않는다.

## 인공 실험은 무엇을 확인했나

3개 상황·각120이벤트·고정 숫자열·random draw0·fit0이다. 정해진 frame count
[4,4,5,4,3],float64 시간에 NumPy nearest rounding을 적용한1ms timestamp/4ms sample은
**예시 가정**이지 실제 archive의
분해능·자극열 추정이 아니다. 관측 시간은 `자극시간+marker 경로 지연`에서 두 필드로
기록되는 모형이다. 모든 지연은 양수이며 공통 baseline4ms를 고정했다.

| 생성 상황 | 실제 자극의 예정 대비 변화 | 기록 경로 | 확인한 사실 |
|---|---|---|---|
| 예정된 frame 간격+양자화 | 없음 | 고정 | 간격 표준편차10.48ms가 생겨도 자극 jitter는 아님 |
| 실제 자극 시점 변화 | 고정된 ±편차 | 고정 | 아래 상황과 timestamp/sample이 완전히 같음 |
| 공유 marker 경로만 변화 | 없음 | 같은 편차 | 위와 DIN은 같지만 실제 자극과 toy response는 다름 |

뒤 두 상황의 관측배열SHA256은 같다. 따라서 **허용한 공유 경로 오차모형 아래
DIN 두 필드만으로 두 원인을 구분할 수 없다.** EEG까지 동일하다는 반례가 아니며,
추가 EEG/외부 센서/설정 근거가 원인을 구별할 가능성은 열려 있다. Timestamp만
sample 할당 뒤에 바꾸는 오차는 별개이며, 이 정의도 별도 단위검사로 구분했다.

실제 자극 편차를 정답으로 아는 oracle 좌표 보정은 toy complex representation의
최대오차0.55798을1.57e-16으로 줄였다. 같은 보정을 실제 자극이 변하지 않은 상황에
적용하면0.55798의 오차를 만든다. **Quantized DIN에서 정답 편차를 복원한 실험도,
실제 EEG 보정 성능도 아니다.** Q/Q2/QM/SHAM 이름의 복사본에 동일 변환을 적용한
대수 검사일 뿐, 네 학습기를 학습했다는 뜻이 아니다. 생성 반올림의5ms 오차상한은
직전 실제 probe의4.001ms 기준을 바꾸지 않았다.

독립 생성 재계산은3개 observation hash와 모든 수치·producerSHA의 일치를 확인했다.
정확한 유리수60Hz와 float64 경계를 비교하면 normal상황의 sample index2/120개가
달라지는 정밀도 한계도 기록했다. 뒤 두 세계의 동일성은 영향이 없으며 관측 결과를
고치기 위한 재실행은 하지 않았다. 생성 단위검사6개와 이전 입력관문37개는 별도다.

## 유지·수정·보류 결정

- **유지:** 확보한 MAMEMI 자료와 원 저보정 목표. S001 전체 개발용, 나머지 기록은
  이번에 새로 열지 않았다. S011은 배포본 namespace로 사용 가능하다.
- **보류:** ‘DIN interval spread=물리적 자극 jitter’라는 직접 해석/특징 이름.
  Polarity 불명확성을 이유로 자동 시간 이동·새 optimizer 반복도 하지 않는다.
- **열린 가설:** `기록된 이벤트의 규칙성/불확실성`이라는 관측량이 support EEG
  추정의 신뢰도를 Q 이상으로 설명할 가능성. 물리적 원인이 미식별이라고 해서 이
  예측 가치를 부정하지 않는다. 다만 새로 정의한 관측량·작동 연산·공정 대조군이
  없이 기존 learner를 이 자료에 돌리는 것은 다음 실험이 아니다.
- **학습 전 남은 필수 확인:** 비EEG/label shortcut 행을 포함하지 않도록 row257과
  channel 의미를 확인하고, 실제 samplingRate/export event semantics를 명시한다.
  QueryDIN은 label 생성에 사용되더라도 predictor에서는 격리한다. 공통 schedule/
  deterministic correction은 모든 arm에 제공하고 잔여 정보만 비교한다.

[후속 bounded adapter/metadata 계약 초안](mamem_i_recorded_event_metadata_v1_draft.md)을
남겼다. 이번 source4targets/3generated scenarios 예산은 종료했으며 새 사람 fit0이다.
유망한 learned-M 후보를 얻었다고 보고하지 않는다.

## 실행·검증·절차 한계

Base9cafdce/main, 계획c8759b0→구현/사전 대수 보완306da05. 생성 실행02:45:53UTC.
Root는 PDF1개와 성공 scout source2개를 보존했고 scouts는 GitHub2target/Cedrus1target+
locator1회를 썼다. Unique source4/4, 성공 source 재보존2/2, 새 scholarly discovery0.
마지막 source 응답02:43:34.555UTC로03:08경계 안이다. 추가 raw archive/MAT0.

PDF276431bytes/MD5/SHA256 검산,5페이지 text 검토·1–4페이지 렌더 확인과 즉시
targeted_fulltext 카드 저장을 마쳤다. PDFskill 부재로 pdftotext/pdftoppm fallback을
사용했다. 단, persistent deep-read queue 삽입이2회 실패한 배치에서 다운로드가
먼저 시작된 **절차 이탈**이 있었다. 질문/예산은 GET전 계획에 있었고 PDF내용
읽기 전 queue를 수리했지만, 엄격한 queue-before-download PASS라고 하지 않는다.
첫 실패는 미등록paper, 다음은 backend upsert를 commit하지 않은 실수였다.

Dataset attachment1건이 workspace paper registry에 추가됐지만 새 동료심사 논문으로
세지 않는다. Claim 근거와 source body hash를 연결했다. GitHub issues 목록 전체
body는 scout hash만, comments와 vendor HTML/PDF는 root가 원본을 보존·재검산했다.
문헌의 획득 설명·vendor 일반 경계·인공 반례를 서로 다른 증거 수준으로 유지한다.

Root 단독쓰기·read-only3agent로 동시 worktree 쓰기 충돌이 없었다. Disk가 촉박해
새 worktree/환경을 만들지 않았다. 기존41worktrees/8untracked directories 보존,
설치/삭제/push0. Held60/외부발송/계정·DUA/유료0, 기존 부정 결과 보호 유지.

최종 관련 테스트43개 PASS(새 생성6+직전입력관문37). 전체repo suite는미실행이다.
보호문서4개hash불변, SQLitequick_checkok, research views11개render완료.
Academic source claim b2f0388f61e8e0b4는QUALIFIED, 조건부생성claim6296f5e268190608은
명시한모형에한정해VERIFIED다. 누적1134registry records(이번dataset첨부1포함)/90searches/
46cards/18techniques/83claims/221evidence/45gaps/6analogies/38deepreads
(27completed/10queued/1skipped),cutoff2026-09-04불변이다.

최종read-only검토는문서범위/해석/state와4개보존artifacthash의정합성에PASS였다.
Queue로그/DB전체감사나전체절차PASS를대신하지않는다. 주요보존source10파일은
합계1,123,197bytes이며마지막free8,627,294,208bytes로8GiBreserve이상이다.
공유disk의변화를이작업만의사용량으로귀속하지않았다.
