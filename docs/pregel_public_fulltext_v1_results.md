# Pre-Gelled 원문 확보·표적 정독: 측정 기작은 확인, 공개 paired 자료는 미확인

2026-09-14 KST / 2026-09-13 UTC. **원래 저보정 SSVEP 목표를 유지한다.**
확보한 것은 새 학습 데이터가 아니라 공식 논문 PDF1개다. 판정은
`NO_RELEASE_OR_RELEVANT_M_DOCUMENTED`이며, 정확한 뜻은 **이번에 읽은 본문에서 공개
paired release가 문서화된 것을 확인하지 못했다**는 것이다. 임피던스 측정 자체가 없다는 뜻은 아니다.
보충자료를 읽지 않았으므로 “공개 데이터가 없다”로 단정하지 않는다.

## 실제 확보와 독서 범위

[사전 계약](pregel_public_fulltext_v1_contract.md) `c174fa6`을 먼저 커밋했다.
디스크 여유290GiB를 재확인하고, 기존 OpenAlex 검색 manifest에 있던
[IEEE 공식 PDF](https://ieeexplore.ieee.org/ielx7/7333/9695946/09740692.pdf)를 직접 GET했다.
첫1회에 HTTP200, application/pdf, 3,610,535bytes,8pages를 확보했다.
Curl 전송11.106406초, retry0/redirect0/추가metadata조회0/새검색0이다.
첫 요청직전16:31:30UTC, 최종 상태 확인16:32:21UTC는 실제 HTTP 종료시각의 상한이다.
PDF의 DOI/제목이 대상과 일치하며 p.1에 CC BY-NC-ND4.0 표시가 있다.
논문 라이선스를 미확인 EEG 데이터의 라이선스로 전용하지 않는다.

- SHA256: `5f13504edb12d14985f1105ad5bded92e9930c84aecb6f598f85e0a79ab93455`.
- 로컬 PDF: `/home/whwovy/research-spaces/califree-eeg-experiment-design/pdfs/pregel_public_fulltext_v1_run/paper.pdf`.
- [전송 기록](reports/pregel_public_fulltext_v1_transport.json). PDF/headers/text/렌더는 연구 workspace에
  보존하고 Git에 원문 전체를 추가하지 않는다.
- 읽기: PDFpp.3–7의 impedance/BCI Methods·Results, pp.7–8 Conclusion,
  p.1 DOI/라이선스/보충자료 각주. PDFpp.3–7을 렌더링해 시각 확인했다.
  Targeted fulltext이지 전체 재료제작법·보충자료를 완독한 것은 아니다.
- PDF skill이 세션에 없어 기존 pdfinfo/pdftotext/pdftoppm/view_image를 사용했다.

이전 DOI safe-open 오류와 달리 **정확한 PDF 경로는 실제로 성공**했다. 용량 확보 자체가
서버 접근 권한을 바꿨다고 주장하지 않는다. 자동 resolver는 여러 provider retry/TLS완화
경로가 있어 호출하지 않았으며 backend도 수정하지 않았다.

## 논문이 실제로 한 실험

출처는 모두 [Pei et al., IEEE TNSRE30:843–850 (2022)](https://doi.org/10.1109/TNSRE.2022.3161989)다.
아래 PDF페이지는 인쇄페이지보다842작다.

| 확인한 내용 | 본문 근거 | 우리 질문과의 차이 |
| --- | --- | --- |
| 10명, 후두부8채널, PreG/wet 전극 비교. 임피던스 시험에서 초기값과6분 간격 값을 측정 | PDFp.3/printed845, II.C | BCI 각 block/trial과 정렬된 공개 M 파일을 뜻하지 않음 |
| 7.8Hz·48nA peak-to-peak 시험전류, working/reference 사이 전압, impedance와 EEG 모드 전환 | p.3, II.C | reference·측정 모드·주파수·시각을 알아야 값의 의미가 정해짐 |
| 6시간 장기 시험은 대표1명 | p.3, II.C; p.5 Fig.4 | 10명 장기재현 실험으로 세면 안 됨 |
| BCI 전150kΩ 미만으로 조정하고 EEG를 육안 점검, 약3–5분 기술 | p.4/846, II.G | 최초 접촉 M과 조정 후 EEG를 혼용하면 안 됨; 실제 총비용에는 조정 포함 |
| 40target, 전극별6block×40trial, FBCCA 비교 | p.4, II.G | metadata를 학습에 넣은 few-shot 대조 실험이 아님 |
| 4초 정확도 PreG94.6%/wet96.1%; 정확도 p=.343, ITR p=.346 | p.6 III.D; p.7 Fig.8(b), 렌더 확인 | 비유의성은 동등성·비열등성·M무용성 증명이 아님 |
| optical sensor/trigger box와 보충자료/video가 언급됨 | p.1각주; p.4 II.G | raw photodiode/impedance 파일 공개의 증거는 아님 |

본문의 평균 임피던스 차이와 하드웨어 성능 비교는 확인했지만, 개인별 실제 M이 Q/common
이상의 예측력을 주는지는 시험하지 않았다. Fig.3의 처리 흐름은 amplitude/SNR 분석용이며
완전한 FBCCA 분류기 재현 설정으로 쓰지 않는다. Fig.7은 대표1인의6block 평균으로,
one-shot 결과가 아니다. Ref.[37]의 상관관계 설명을 이 논문의 새 분석으로 세지 않는다.

## 이번에 설계에 반영한 내용

[접촉 측정 추가 규칙](acquisition_contact_measurement_addendum.md)을 작성했다. 핵심은 다음이다.

1. **값과 측정 맥락을 분리:** nominal 장치·전극·reference 조건은 모든 arm의 공통정보,
   실제 pre-query 측정값·측정 이후 경과시간·조정 이력은 별도 실측 M 후보다.
2. **조정 전후를 분리:** 하드웨어를 고치는 효과와 같은 상태에서 M을 모델에 넣는 효과를
   한 비교에 섞지 않는다. 측정전류가 EEG와 동시에 기록된다고 추정하지 않는다.
3. **단순 단조 가중치 가정 금지:** 낮은 임피던스→높은 정확도라는 고정식을 이 논문으로
   정당화하지 않는다. 실제 적격 데이터가 생기면 Q/Q2/QM/조건부SHAM/직접M 기준으로 반증한다.
4. **보정비용 분리:** label trial 수, metadata측정, 재장착·재조정, 대기·버린support,
   최초 query-ready까지 시간을 각각 기록한다. 논문의3–5분은 총시간절감 측정값이 아니다.

긴머리6명은 모두 여성, 짧은머리4명은 모두 남성이므로 머리길이만의 인과효과를 추정하거나
성별 기반 품질계수를 만들지 않는다. 위 추가 규칙은 **향후 후보용**이며 종료한 기존 후보를
재개하거나 held60/기존 outcome으로 튜닝할 권한이 아니다.

## 남은 질문·중단선

이 논문의 본문 정독은 종료했다. 원문 접근은 해결됐고, **실제 공개 release·파일명·시각·
사람/session/channel join·단위·측정비용 기록**은 아직 없다. 보충자료의 구체적 파일 inventory를
별도 제한된 문서확인으로 검사할 수 있지만, 자동 raw다운로드/학습으로 승격하지 않는다.
계정·유료·저자 요청이 필요하면 이 후보를 보류하고 별도 승인 없이 문의하지 않는다.
현재 새 learning-eligible 후보0, raw/새fit/EEG예측/held60/발송/유료0.

독립 읽기 전용 reviewer가 같은 PDF구간과 렌더5개를 확인해 위 해석 경계를 지지했다.
새 코드/모델 변경이 없어 pytest는 실행하지 않았다. 로컬 문서관리는 잘못된 CLI명1회와
falsification 누락 validation1회로 실패한 뒤 수정·저장 성공했다. 이 기록 오류를 원문 접근
실패나 연구 negative로 섞지 않는다. 기존 abstract-only 입력/앞선 실패 보고는 보존했다.

별도 [저장 산출물 감사](reports/pregel_public_fulltext_v1_review.json)도 size/hash/8pages/DOI·제목/
HTTP200·MIME·기록된1GET/0redirect/11.106406초를 대조해 PASS였다. 실제 curl flags와
미기록 요청 부재를 이 산출물만으로 독립 입증한 것은 아니다. Root가 반환을 기록했으며
새 실행형 auditor를 만들었다고 주장하지 않는다. [최종 상태](reports/pregel_public_fulltext_v1_state.json).

Workspace card를 targeted_fulltext로 갱신하고 claim `c3b6e30fa6123b98`(QUALIFIED,
근거3개)와 기존 gap을 기록했다. Catalog1151/search92/card47/claim100/evidence266,
deep-read39(completed28/queued10/skipped1), render11/SQLitequick_checkok.
Root solewriter/읽기전용 병렬 검토로 새worktree없이 처리했고 기존46worktrees/8untracked,
보호6hash를 유지했다. 원 목표는 미완료이며 현재 공개 paired자료 공백을 연구 성공으로 세지 않는다.
