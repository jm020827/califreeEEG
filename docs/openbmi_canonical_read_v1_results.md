# OpenBMI 본문 정독: 임피던스 관리 기준과 공개 측정값은 다르다

2026-09-14 KST / 2026-09-13 UTC. 여유 공간 약290GiB를 확인하고
[계약](openbmi_canonical_read_v1_contract.md) `68642dd`의 공개 문서 판독을 진행했다.
**본문은 확보·표적 정독했지만, 학습에 넣을 실제 acquisition M의 공개 파일은 확인하지 못했다.**
판정은 `NO_ACQUISITION_CANDIDATE_IDENTIFIED_IN_READ_SCOPE`이다.
이는 모든 보조파일의 부재나 metadata 일반의 무용성을 뜻하지 않는다.

## 확보·판독 범위

- 원문: [Lee 등, GigaScience 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/),
  DOI `10.1093/gigascience/giz002`. 제목·DOI가 일치하는 실제 article HTML이다.
- 판독: `#sec1-2` Experimental procedure와 하위 EEG recording/SSVEP 절,
  `#tbl1`–`#tbl3`, `#sec2-5` Source code scripts / `#sec2-5-1` Data structure,
  `#sec4` Source code availability, `#sec5` Supporting data, `#bib51`, `#bib71`.
- 형식은 **primary HTML / targeted_fulltext**. 전체 Results/Discussion 정독이나
  `fulltext_complete`가 아니다. PDF 페이지를 지어내거나 웹 본문을 PDF라고 세지 않았다.
  표는 HTML 셀·행 구조를 읽었다. 그림의 시각적 형태나 성능 표 `#tbl4`를 분석하지 않았다.
- 본문에 CC BY4.0 표기가 있다. 이를 원자료의 미확인 라이선스에 전용하지 않는다.
  MATLAB 코드의 GPL3도 데이터 라이선스와 별개다.
- 실제 HTML: 연구 workspace의 `source-probes/openbmi_canonical_read_v1/request1.body`,
  SHA256 `11f0a13b8b10dbb7d1ad03c87400ee0dd8bfe7001bc79bbfee4a54b8acb44b14`.
  HTTP Date `2026-09-13 17:05:25 GMT`. 원문 전체는 Git에 복제하지 않았다.

## 원문이 말하는 것과 우리 해석

| 직접 근거 | 저자 보고 | 연구 설계상 해석 — 우리 추론 |
| --- | --- | --- |
| `#sec1-2-2` EEG recording | 62 EEG 전극, nasion reference, AFz ground, 1kHz; impedance를10kΩ 미만 유지 | 공통 장치/설정과 관리 상한이다. 개인별 실제 측정값으로 채워 넣을 수 없다 |
| `#sec1-2-1`, `#tbl3` | paradigm 사이 impedance 점검, MI/SSVEP 진입 전 각5분 일정; 전극 설치25분 | 측정·조정 시점은 중요하지만 표의 예정 시간을 개인별 실측 calibration 비용으로 쓸 수 없다 |
| `#sec2-5-1` | `x,t,fs,y_dec,y_logic,y_class,chan`의7개 필드 설명 | 이 목록에 impedance 필드는 없다. 미확인 보조파일까지 없다는 증명은 아니다 |
| `#sec1-2-2`, `#fig1` 설명 | EEG와 팔 EMG 전극 구성 및 별도 유발 artifact 기록 | SSVEP 파일에 동시 EMG가 배포되는지, 시계·채널 대응이 무엇인지는 미확인. EMG 자체를 acquisition M으로 자동 재정의하지 않는다 |
| `#tbl1`, `#tbl2` | 실험 전 Questionnaire I, 각 run 후 Questionnaire II; 후자는 놓친 시도·예상 정확도 등을 포함 | 해당 run 이후의 정보는 사전 입력이 아니다. 설문/개인 상태는 현재 acquisition-M 범위와도 다르다 |
| `#sec1-2-1`, `#tbl3` | ERP→MI→SSVEP 고정 순서, 유연한 휴식 | SSVEP가 항상 마지막이므로 시간·피로·이전 과제 경험 효과를 분리할 수 없을 수 있다 |
| `#sec5`, `#bib71` | GigaDB supporting data DOI100542 | 공개 경로의 보고이지 이번에 실제 파일 schema·사용권·EEG–M pairing을 검증했다는 뜻은 아니다 |

저자는 네 MAT 유형(`EEG_Artifact`, `EEG_ERP`, `EEG_MI`, `EEG_SSVEP`), train/test 구조,
설문 Excel 등을 설명한다. **EEG 양이 많다는 사실만으로 외부 M 가설을 검정할 수는 없다.**
휴지·artifact EEG에서 계산한 값은 Q이며 장치/명목 자극 조건은 공통 정보이다.
신규 사용자에게 사전에 한 번 실제 측정한 M은 여전히 허용된다. 반복 측정 부재 자체가 탈락 이유는 아니다.

## 접근 결과와 남은 불확실성

[전송 기록](reports/openbmi_canonical_read_v1_transport.json): 총5/6GET,
retained body608,263bytes, curl 시간합2.616226초. 최초/최종 HTTP Date는
17:05:25/17:15:25UTC로20분 창 안이다. 재시도·자동 redirect·인증·우회0.

1. PMC article200/HTML293,451bytes: 본문 확보 성공.
2. literal PDF 링크200/HTML1,817bytes: 다운로드 준비 페이지, **PDF 확보 실패**, 그 경로 종료.
3. dataset DOI302: `http://gigadb.org/dataset/100542`를 반환, HTTP는 요청하지 않았다.
4. [저자 GigaScience 저장소](https://github.com/PatternRecognition/OpenBMI/tree/master/GigaScience/)
   200/HTML311,564bytes: revision `c0e56825366681ae576801af08eb5fe326a5afd8`.
   해당 디렉터리 README는 GPL 문서이며 acquisition schema 설명이 아니다. 전체 코드 감사는 하지 않았다.
5. 요청 전에 규칙을 명시적으로 보완하여 같은 host/path의 HTTPS 주소만 확인했다.
   200/HTML1,282bytes의 일반 GigaDB 앱 shell이며 dataset-specific 내용은 없다.
   JS 실행·추정 API 호출·추가 raw 다운로드는 하지 않았다. 이전 계약을 그대로 지켰다고 소급 주장하지 않는다.

실제 SSVEP 채널 목록·동시 EMG·별도 impedance 파일은 **미확인**이다.
이 판독만으로 전체 약209GB(논문 보고)의 다운로드를 우선하지 않는다. 용량 확보는 접근과
학습 적격성의 증거가 아니며, raw 다운로드가 이번에 완료됐다고 보고하지 않는다.

## 반영한 결정과 다음 단계

OpenBMI는 기존 P3/context 용도로 남기고 **현재 acquisition learner 후보로 승격하지 않는다**.
같은 PDF/landing URL을 반복 요청하거나 사후 설문을 M으로 바꾸며 이 경로를 연장하지 않는다.
재진입은 공개 파일 목록/설명에서 실제 측정값·단위·시점·SSVEP 대응을 확인하는 새 근거가 있을 때만 한다.
그 전에는 다른 공개 acquisition 후보를 우선하며 EEG 대량 다운로드/새 fit은 시작하지 않는다.
읽기전용 scout가 기존 round4 결과·상태·triage·다음 초안4개를 재검토했으나,
그 기록에서 지정할 수 있는 새 적격 후보나 구체적 public-schema 대상은 **NONE**이었다.
따라서 다음 후보를 확보했다거나 다음 학습이 이미 예약됐다고 보고하지 않는다.
후속 탐색은 실제 측정 필드/공개 파일의 새 근거가 필요하며, 이 결과가 공개자료 전체의 부재 증명은 아니다.

[추가 측정 규칙](acquisition_contact_measurement_addendum.md)에 사후 설문·고정 과제 순서·
예정 비용/실측 비용·보조센서 저장의 경계를 반영했다. 기존 실험 계약/부정 결과는 소급 수정하지 않는다.
실제 M이 확보되면 Q/Q2/QM/조건부 SHAM/direct-M과 총 준비비용을 동일 조건에서 비교한다.
이번에는 raw0/fit0/예측0이며 M 효능·보정량 감소를 증명하지 않았다. held60은 열지 않았다.

## 기록·독립 검토

Root와 읽기전용 protocol reviewer가 위 본문 구간을 각각 확인했다. Table4는 성능 표이며
data structure 근거가 아니라는 locator 교정을 반영했다. 별도 artifact 감사 범위는 상태 JSON에 기록한다.
`academic-research`의 근거/추론 분리를 사용했고 `coordinate-worktree-changes`에 따라
root만 문서·SQLite를 수정했다. 새 worktree 없이 기존46개와8개 untracked 시험 폴더를 보존했다.

연구 backend의 `targeted_fulltext` paper-card는 PDF만 받는다. 이를 우회하려고 가짜 PDF를
생성하거나 backend를 바꾸지 않았다. 기존 abstract card는 그 형식 제한을 명시한 pointer로 갱신하고,
**실제 HTML 정독 근거는 이 보고서와 fulltext-scope claim에 저장**, 표적 독서 항목은 완료로 기록한다.
따라서 legacy card의 `abstract` 표시는 현재 원문 독서가 없다는 뜻이 아니다. PDF 확보는0이다.
