# MAMEM baseline 관찰의 기여 자격 검토

2026-09-14 KST / 2026-09-13 UTC. 원래 저보정 SSVEP 및 외부 acquisition M의
추가 이득이라는 목표는 유지한다. 사용자의 baseline 연구가치 질문은 종료한 실제
실험을 재개하거나 metadata 성공으로 재명명하는 승인이 아니다.

## 실행 전 질문·범위

이미 관찰한 공통 reference +25.33%p는 개발 진단이다. 아직 확인하지 못한
MAMEM-I 원 연구의 정확한 reference 설정이 이 관찰의 신규성 판단을 바꿀 수 있다.
Signature: representation=source-only common frequency bank; bottleneck=저자 baseline과의
차이; operation=기존 코드/보고서 및 정확한 원논문 표적 판독; objective=재현 진단과
방법 기여 구분; feedback=저자 설정·정보권한·검증자료; failure=다른 dataset 예제의
이식, query label reference, 미확인 설정을 새 기여로 포장.

기존 저장 tree152개와 README에서 Dataset-I default/optimal은 PWelch계열,
CCA-SVM 예제는 Dataset-III, CombiCCA/ITCCA/L1MCCA 예제는 별도 UCSD자료임을
확인했다. 일반 CCA.m 재독만으로 미확정 Dataset-I caller를 복원하지 않는다.
README가 직접 지목한 저자 보고서 arXiv1602.00904v2는 과거 획득장치·자료위치만
표적 확인했다. **이번 질문은 baseline의 reference/전처리/평가 설정**이다.

예산: 기존 로컬 출처 재사용을 우선하고, 이미 성공했던 정확한 공개
`https://arxiv.org/html/1602.00904v2` HTML 최대1GET/30초/8MiB/redirect0/retry0.
401/403/429/challenge면 중단. 새검색/PDF/개인값/EEG/fit/예측0, 결과표 수치의
재계산이나 우리 실험과의 직접 수치 순위 비교0. 원문 확인은 최대20분.
Root 단독 문서·SQLite writer, agent는 독립 읽기 전용. 새worktree/설치/삭제0.

결과는 (a) 저자 설정이 직접 확인됨, (b) 부분 확인, (c) 미확인의 하나로 남긴다.
기존 80% 준비 기준은 변경하지 않으며, 이를 출판 가능성의 보편 기준으로 쓰지 않는다.
같은10명에 새채널/filter/window를 적용하는 후속 실행 권한은 이 검토에 없다.
Held60·사람 요청·유료 사용은 별도 승인 원칙을 유지한다.

## 표적 판독 결과 — 찾던 비교 기준의 종류를 바로잡음

정확한 HTML 1GET은 HTTP200,724,536bytes,0.174655초,redirect/retry0이었다.
응답 Date=2026-09-13T17:46:09Z. SHA256
`d764662a2b2c6f8e75633c1afc843948d3e352ab7a4870df5189ecda529edef3`는
이전에 획득설정 판독 때 관측한 전체 응답 hash와도 같다. 이번에는 다른 결정 질문에
필요한 본문을 읽었다. 새 개정판·새 논문 발견으로 세지 않는다.

**직접 확인한 원문은 PWelch/SVM baseline을 설명한다. Dataset-I CCA reference
설정을 이 보고서의 기본 pipeline에서 계속 찾는 것은 맞지 않는다.** CCA는 관련연구에서
논의되지만, 이번에 읽은 평가 방법·기본/최적 설정에는 그 대신 spectral feature와
학습 classifier가 있다. 일반 CCA 코드가 toolbox에 있다는 사실과 논문의 실험을 구분한다.

| 질문 | 원 저자 보고 | 우리 진단과의 관계 |
| --- | --- | --- |
| 새 참가자의 보정 필요? | S2.p6.1, S3.p3.1, S6.SS2.p2.1: 다른 참가자의 labeled EEG로 학습하는 LOSO, subject-specific training 없이 운용 | 최초 무보정이라는 차별화 불가. Source 학습 비용도0이 아님 |
| 기본 decoder? | Table III(S6.T3), S6.SS7.p1.1: PWelch, linear SVM C=1, one-vs-all | E126 projection-energy argmax는 이 pipeline의 재현이 아님 |
| 창·전처리? | Table III: 5초,5–48Hz IIR-Chebyshev I, PWelch nfft512, artifact removal/feature selection 없음 | 우리 2초·무필터·기본파/2차고조파 진단과 직접 정확도 비교 금지 |
| 채널? | Table III는 Oz라고 부르고 S6.SS9.p1.1은 default126→optimal138을 기술 | 저자 코드 E126 선택과 연결되는 문헌 근거. 실제 원 acquisition 위치 독립 검증은 아님 |
| 명목 주파수표? | S5.SS3.p2.1은6.66/7.5/8.57/10/12Hz의 설계를 기술 | 실제 flash 주파수/clock 보정이나 우리 source-only 추정표의 동일성을 입증하지 않음 |
| 검증의 독립성? | S6.SS1/S6.SS2/S6.SS5는 비교·trial-and-error/grid search·LOSO로 설정 선택을 설명 | 검토 범위에 별도 nested selection/독립 최종 cohort 명시 없음. 그 선택 절차를 우리 확증으로 복제하지 않음 |

최적설정은 Table XV/S6.SS9에서 channel138,elliptic filter,AMUSE,SVD와
Spearman-kernel SVM으로 바뀐다. 여기서 여러 구성요소를 동시에 바꾸는 것을
‘reference만의 효과’라고 해석하지 않는다. 본문/표 추출 중 원 논문의 성능 수치도
노출됐지만 새 효능 계산·우리58%와의 순위 비교·사람 결과 선택에는 사용하지 않았다.
이 문서는 technical preprint의 표적 판독이지 peer-reviewed status 확인이나 전체 정독이 아니다.

## 현재 기여 판정

1. **보존할 결과:** 같은 개발10명/150창에서 공통 reference 변경의 +25.33%p.
   80% 운영 gate 미달은 그대로이며 연구 관찰 자체를 무효로 만들지는 않는다.
2. **아직 주장 불가:** 기존 저자 baseline 대비 우위, 일반적인 새 분류법,
   독립 재현, 학습된 acquisition M 이득, 실제 보정량/총시간 감소.
3. **이번에 바뀐 다음 행동:** 미확인 ‘저자 Dataset-I CCA recipe’ 탐색은 이 보고서
   경로에서 닫는다. 향후 비교를 설계한다면 source-trained PWelch/SVM을 직접 관련
   baseline으로 취급하고, 우리 reference 진단과 정보권한/창/전처리가 다른 점을 명시한다.
   이것만으로 현대의 강한 baseline을 모두 갖췄다는 뜻은 아니다.
4. **설계상 주의:** PWelch/SVM은 명시적 sine-reference bank를 받지 않는다.
   따라서 거기에 NOMINAL/COMMON 두 이름만 붙여 reference ablation인 척하면 안 된다.
   reference 효과는 동일한 reference 기반 decoder 안에서, 방법 전체 효과는 같은
   source/target 정보·보정량·시간창을 맞춘 서로 다른 decoder 간에 평가해야 한다.
5. **새 실험 진입 조건:** 명확한 개선 원리와 사전 비교/예산, source-only 선택,
   참가자 단위 불확실성, 미사용 검증자료의 역할을 먼저 정한다. 이미 쓴10명은 개발
   자료이며 같은 사람의 다른 세션도 독립 참가자가 아니다. 이 검토는 재학습 허가가 아니다.

## 검증·절차 기록

기본 system python에 bs4가 없어 첫 로컬 파싱은 실패했다. 설치하지 않고 기존
paper-search-mcp 환경의 bs4로 읽었다. 소스 GET의 재시도나 사람 실험 실패가 아니다.
표는 HTML의 caption과 순서가 보존된 행을 직접 파싱했으며, 새 PDF/그림은 읽지 않았다.
읽은 범위: S2.p6, S3, S5.SS3, S6.SS1–2, S6.SS3.p1.1, S6.SS5, S6.SS7–9,
Table III와XV. Keyword에서 노출된 다른 문단은 전체 정독으로 세지 않는다.

독립 read-only reviewer가 LOSO/target0,5초,linear/Spearman SVM,설정선택의 독립성
한계를 확인했다. Root는 reference/신호처리와 기존 code tree/README 연결을 확인했다.
새 raw/개인값/fit/예측/held60/사람 요청/유료/설치/삭제/push0,새worktree0.
academic-research가 원문 근거와 신규성 추론을 분리하는 데 영향을 주었고,
coordinate-worktree-changes에 따라 공유 작업공간에는 root만 썼다.

공개 원문: [arXiv1602.00904v2](https://arxiv.org/html/1602.00904v2#S6.T3).
실행·보존정보: [state](reports/mamem_baseline_contribution_v1_state.json).
