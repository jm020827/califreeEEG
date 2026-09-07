# 기존 판단 재검토와 유한 연구 루프

2026-09-07. 사용자 승인: “응 그렇게하자. 기존 근데 판단들도 다시보고”.
이 문서는 **향후 판단을 수정**한다. 기존 실험의 설정·결과·종료 판정을 바꾸지 않는다.
현재 방향에 관해 이전 결과 문서의 ‘다음 단계’보다 이 문서를 우선한다.

## 결론부터

연구목표는 유지한다. **외부 acquisition context가 EEG만으로 판단하는 방법보다 적은
정답 예제로 쓸 만한 SSVEP 성능에 도달하게 하는가?** 아직 입증하지 못했다.

제가 잘못 강제한 것은 ‘EEG-only 보정이 먼저 평균적으로 무보정보다 좋아야 metadata를
시험한다’는 순서다. 비교법의 구현·경쟁력을 확인하는 일은 필요하지만, 그 평균 우위를
metadata 연구의 논리적 입장권으로 삼을 필요는 없다. 이 전제가 최근 실험을 계속 새 EEG
분류기 개발로 이어지게 했다. 다음 후보는 자동으로 새 ECCA 결합식이 되지 않는다.

예를 들어 보정 예제가 상황 A에서는 +10pp, B에서는 −10pp를 준다고 가정하자.
두 상황이 절반씩이면 평균 보정 이득은 0이다. Metadata가 A를 실제로 구별해 A에서만
보정을 쓰면 +5pp가 가능하다. **이것은 논리적 반례이지 관측 결과가 아니다.** 실제로는
EEG Q만으로도 같은 구별이 가능한지, M 측정 비용까지 낼 가치가 있는지를 비교해야 한다.

## 판단 장부

아래에는 실제로 바꿀 전제와, 과거 문서에서 이미 올바르게 제한했던 해석을 함께 점검한다.
‘metadata 일반의 무용성’이나 ‘metadata를 한 번도 시험하지 않음’을 기존 공식 결론이었다고
소급해 만들지 않는다. 기존 결과 문서는 이미 그 일반화를 경계하고 있었다.

| 점검한 주장·전제 | 재판정 | 근거와 앞으로의 처리 |
| --- | --- | --- |
| 처음으로 few/one-shot SSVEP를 만든다 | 기각 유지 | 기존 저보정·transfer 선례가 있다. 신규성은 외부 M의 Q 너머 순증분과 실제 비용 절감에 한정한다. |
| Q>A0의 평균 이득을 먼저 통과해야 M을 시험할 수 있다 | 향후 필수조건 철회 | 위 조건부 효용 반례. 대신 검증된 Q, matched capacity, M>A0 실용성·harm·cost를 요구한다. 과거 AQ_NOT_ESTABLISHED는 그대로다. |
| 테스트·독립 산술 검산 PASS면 강한 논문 baseline을 확보했다 | 분리·수정 | 검산은 구현 정의·저장 통계 이후 산술의 근거다. 저자 코드 실행, 원 프로토콜 성능 재현, 우리 조건에서의 경쟁력은 서로 다른 검증이다. |
| 계속 실패하니 이 EEG에는 학습할 신호가 없다 | 일반화 금지 유지 | context 실험 wet0.5초 eTRCA의 k3/k5=38.12/47.78%, A0=27.61%; reference의 real>wrong도 반례다. 전체 조건에서 유용한 방법을 확보했다는 뜻은 아니다. |
| Metadata는 아직 한 번도 시험하지 않았다 | 부정확하므로 사용 금지 | V1 및 context 실험에서 실제 M을 시험했다. context는780packets, AQM−AQ eAUC=−0.000047483, 예측 변화5/14040이었다. 최근 spatial/reference만 M-free다. |
| 현재 실패는 metadata 일반의 무용성을 뜻한다 | 근거 없음 | 작동한 M 삽입 위치의 낮은 효과, 약한 support 방법, 부적절한 합성 난이도를 구분해야 한다. 임의의 M 방법 전체를 배제할 증거가 아니다. |
| V2·V4·context·reference는 같은 종류의 실패다 | 분리·수정 | V2는 인프라 inconclusive; V4 pilot은 k0부터 전원80%인 cost ceiling; AQ001은 primary 집합이 빈 설계 실패; context/reference는 정의된 개발 비교에서 실패했다. |
| Impedance가 낮으면 더 좋은 EEG이고 좋은 보정이다 | 조건부 가설로 제한 | 원 논문은 participant별10blocks×8channels 평균의 상관이다. block별 상대적 support 효용이나 인과효과를 검정하지 않았다. |
| Notch가 좋아졌으므로50Hz가 원인이며 해결됐다 | 근거 없음 | reference 이득은 여전히 음수. prefix history·필터 dynamics도 함께 다르다. 물리적 단일 원인이나 저자 전처리 재현이 아니다. |
| 같은39명에 새 버전·fold·실행 전 고정을 쓰면 새 검증이다 | 기각 유지 | 모두 적응적으로 재사용한 개발 자료다. 성공하더라도 독립 confirmation이 아니다. |
| 개발 실험도 매번 새 미관측 자료·복잡한 봉인이 필요하다 | 향후 경량화 | 개발에서는 실패를 기록하고 제한된 수정·재검산을 허용한다. 최종 확인의 미개봉 보호·사전 고정과 구분한다. 기존 봉인은 완화하지 않는다. |
| 더 많은 EEG dataset이면 이 M 가설의 반복 검증이다 | 용도 분리 | EEG-only 자료는 baseline/일반화 검증에 유용하지만 blockwise authentic M을 대신하지 못한다. 독립 paired-M 자료는 여전히 없다. |
| 두 낮은 정확도가 비슷하면 보정을 줄였다 | 기각 유지 | 유용한 절대 정확도와 같은 관측시간이 필요하다. k0부터 성공한 사람, 미달자, 비단조 곡선을 별도로 표시한다. |
| 원문을 읽고 수치를 확인한 것은 저자 결과를 로컬 재현한 것이다 | 증거 등급 수정 | 문헌 수치·기작은 author-reported, 구현 정의는 repository evidence, 로컬 측정은 measurement로 구분한다. 기존 일부 claim의 verified 표기는 과강했다. |

직접 근거: [context 결과](context_template_source39_v1_results.md)의 ‘고정 판정’·‘사후 원인 진단’,
[reference 결과](reference_calibration_source39_v1_results.md)의 모든 조건·검산 범위,
[V4 pilot](metadata_calibration_efficiency_v4_pilot_results.md)의 cost ceiling,
[AQ001](metadata_calibration_v4_aq_study_results.md)의 빈 primary,
[V2 종료](metadata_calibration_efficiency_v2_results.md)의 infrastructure inconclusive.
이 문서가 인용한 실제 수치를 재실험으로 얻은 것은 아니다.

## 원 논문을 다시 읽고 달라진 해석

[Wearable 원 논문](https://doi.org/10.3390/s21041256) pp.11–12 Section3.5/Figure9의
상관분석 단위는 사람별 평균 impedance다. ‘사람 평균 impedance와 정확도의 단순 상관’과
‘현재 Q를 알고도 특정 support의 상대적 효용을 M이 예측하는가’는 다른 estimand다.
전자에서 유의하지 않다는 사실은 후자의 긍정·부정 증거로 바로 바뀌지 않는다.
Wet−dry 차이의 관찰 상관도 impedance를 낮추면 성능이 오른다는 인과효과를 단독 식별하지 못한다.

같은 논문 p.8 Section3.3은 offline supervised TRCA에 leave-one-out 평가를 사용한다.
우리 첫5block support/뒤5block query와 적은 k, 짧은 crop은 다른 조건이다.
[기존 reference 설계](reference_calibration_source39_v1_design.md)도 DAN의2초·3band·query6–9와
우리0.5–1초·7band·query5–9 및 필터 차이를 이미 명시했다. 그런데 실행 전략에서는 이
‘정확한 저자 프로토콜 재현 아님’을 충분히 반영하지 않고 새 방법 구현을 계속했다.

개발 자료의 반복 선택에 따른 과적합 경고는
[Cawley & Talbot2010](https://www.jmlr.org/papers/v11/cawley10a.html)의 공식 abstract로
확인했다. 이것이 실험 횟수를 정확히1회로 정해주거나 아래 예산의 통계적 정당성을 주는 것은 아니다.

## 새 실행 순서 — 목표는 유지, 자동 승격은 없음

### A. 기존 저자 구현 하나의 호환성 확인

새 분류기를 만들지 않고 라이선스·revision·실행 API가 확인된 구현 하나를 사용한다.
로컬 합성 fixture/API 확인, 원 프로토콜에 가까운 source39-only 호환성 확인, 우리 제약으로의
protocol bridge를 구분한다. 원 논문의 전체102명 수치를 맞추려고 held60이나 retired1–3을 열지 않는다.
동일 코호트·분할·전처리가 아니면 **부분 호환성 확인**, 정확한 논문 성능 재현은 ‘미완료’라고 쓴다.

차이는 표로 고정한다: sampling/latency, full-epoch 대 crop/prefix filtering, filterbank,
support수·시간순 분할, channel/label/axis, score 정의. ‘정확도가 높아졌으니 맞는 구현’으로 판정하지 않는다.
저자 원형이 미래 sample을 쓰면 offline diagnostic으로만 허용하고 짧은 online 성능으로 보고하지 않는다.
Native와 bridge의 같은 사람 반복 측정은 독립 replication이 아니다.

예산은 **한 baseline 계열, 호환성 확인1개와 최대2개의 사전 명시 protocol bridge**다.
Bridge가 여러 요소를 바꾸면 bundle 차이로 보고하고 하나의 원인으로 돌리지 않는다.
명백한 입력/API 결함 수정은 원인·변경·재검사를 남긴다. 성능을 보고 새 계열을 자동 추가하지 않는다.
호환성 실패가 계속되면 이 단계의 구현/자료 병목을 보고한다. M의 무효 판정은 하지 않는다.

선정한 구현과 현재 fixture 상태는 [eTRCA 호환성 준비 기록](author_etrca_compatibility.md)에 둔다.
이는 toolbox 저자의 Python 구현이지2018 TRCA 원저자의 MATLAB 실행이 아니다.
공개 interface별 weight는 source39-only로 추정한 값이 아니다. 전체 Wearable 코호트에서
조정된 설정을 사용하는 native 호환성 진단과, held60에 완전히 독립적인 확인을 구분한다.
향후 확인에는 해당 코호트와 무관한 고정값 또는 source39-only 동결값을 우선 검토하고,
공개 설정을 유지하면 상속된 집계 tuning 노출을 명시한다. Held 파일 직접 접근0과 별개다.

### B. Metadata의 직접 가설 하나

> **Q만으로는 충분히 구별되지 않는 support의 상대적 유용성을 pre-query acquisition
> context가 알려주어, 검증된 동일 분류기의 보정 사용 결정을 개선하는가?**

EEG Q는 현재 query와 이미 허용된 labeled support의 신호·일관성 정보다. M은 정답을
모르는 상태에서 얻는 실제 block/channel impedance와 측정조건이다. Dataset/subject ID,
query 정답, 미래 block 또는 query batch의 통계를 넣지 않는다. ‘M은 noise와 단조 관계’라는
가정을 강제하지 않는다. Source 개발에서 확인할 예측 가설이다.

첫 구현은 M이 **support 기여도를 조절하는 한 위치**만 갖도록 한다. Q-only도 같은 위치에서
조절할 수 있는 강한 비교군이어야 한다. 고정 Q와 M 모델만 비교하면 약한 Q를 구제한 효과일 수 있다.
같은 backbone/support/prefix와 같은 fitting·regularization 예산을 사용하고, M 정보 증분과
파라미터 수 증가를 구분한다. 실제 source39 outcome을 더 읽기 전에 입력 feature, 결합식,
유한 fit grid, fold별 전처리/정규화와 endpoint를 짧은 실행 명세로 고정한다.
이 단계의 정확한 연산자·수치 기준은 **아직 실행 가능한 상태로 고정되지 않았다**.

필수 비교는 matched Q, 같은 learner의 M+Q, A0, missing M, correct-v-shuffled M,
stale M와 chronology/order 대조군이다. Impedance 고유 효과를 주장하려면 interface-only도
분리한다. Shuffle은 interface/missingness/허용 prefix를 보존하면서 EEG–M 연결을 실제로 깨야 하며,
두 방법의 입력·가중치·예측이 어디서 달라졌는지 검사한다. No-op shuffle은 기작 반증 실험이 아니다.

잘된 Q의 평균>A0를 입장 조건으로 요구하지 않지만 **M+Q가 강한 Q를 넘고, A0보다 유용하며,
안전성과 실제 비용 기준을 만족해야** 다음 확인 검토로 간다. Diagnostic oracle로 query 정답을
이용해 가능한 이득을 계산하더라도 이는 상한·분석에만 쓰고 배포 정책·selection에는 넣지 않는다.
현재 계획에 oracle 실행은 포함하지 않는다.

### C. 비용과 중단을 미리 구분

보정 횟수는 class당 k,12class에서는12k labeled trials다. Metadata가 k1에서 exact Q인
설계를 택하면 one-shot M 기여를 주장하지 않는다. 5→3-shot을 시험하면60→36trials의
비교이고, 동일한 window·같은 고정 query에서 유용한 성능을 유지해야 한다.
긴 window로 성능을 회복한 것은 별도 시간 비용이다. Impedance 측정 시간·준비·intertrial
overhead가 없으면 총 사용자 시간 절감을 주장하지 않는다.

모든 개발 반복은 간결한 ledger에 남긴다. **M 가설1개 + 명확한 실패기작에 근거한 수정 최대1회**
뒤에는 결과 부호와 무관하게 프로그램 점검을 한다. 수정은 변경될 관측 예측을 먼저 써야 하며
threshold/window/subgroup을 성공하도록 고르는 수정은 허용하지 않는다. 이는 운영 예산이지
반복선택의 오류율 보장이나 새로운 독립 표본을 만드는 장치가 아니다.

| 관측 상태 | 결정 |
| --- | --- |
| 코드/프로토콜 호환성 미해결 | 입력·구현 병목 해결 또는 단계 보류. M negative로 해석하지 않는다. |
| 비교가 포화·no-op·빈 집합 | 무엇을 측정하지 못했는지 명시. 남은 수정 예산 안에서만 assay 변경을 검토한다. |
| M 이득 추정이 작고 구간이 넓음 | 불확실. 같은39명 retuning보다 정보가 있는 독립 M 자료·정밀도 계획을 우선한다. |
| 유용한 효과를 배제할 정밀도와 controls가 있음 | 해당 mechanism/조건에 한정해 종료한다. M 일반 불가능성은 아니다. |
| 개발에서 실용성·M 순증분·cost·controls가 유망 | 최종 방법 고정, power/다중검정/held 접근권한을 별도 검토한다. 자동 held 실행은 없다. |

## 실행·자료·기록 경계

이미 노출된 source39는 앞으로도 명시된 범위의 개발용이다. 기존의 특정 closed attempt를
재개하거나 그 결과 파일을 덮어쓰지 않는다. Held60과 retired1–3, 폐쇄 runner/seal은 보존한다.
저자 구현 확인을 위해 기존 BETA35/20·V2 external 계획을 재활성화하지 않는다.

독립 paired-M 자료 확보와 [저자 요청 초안](independent_impedance_data_request.md)은 별도 과제다.
현재 추가 확보0, 발송0. 사용자 실제 소속/연락처·필요한 이용조건 없이 외부 연락하거나
윤리승인을 꾸며내지 않는다. EEG-only dataset을 이 빈칸의 대체 자료로 세지 않는다.

`academic-research`는 기존 landscape 재사용과 원문 pp.5–8/11–12 targeted 재독·Figure9
시각 확인, 추론/문헌 보고/실측 증거의 분리를 강제했다. 새 광범위 문헌 수집은 하지 않았다.
PMC CAPTCHA는 우회하지 않고 이미 보존된 원문 PDF(SHA256
`53a9f8548c579e915284c6139b56a3836f7e02db6a68c04b078be5c6fe11eb0f`)를 사용했다.
기존 카드와 새 재독 카드를 분리 보존한다. 자료 수집 규모가 문헌 검토의 완결성을 뜻하지 않는다.
추가 검색은 toolbox 논문 exact title 한 건(Crossref/OpenAlex, cutoff2026-09-07,
search:ea3ee2afe9600d55,3records/noerrors)에 한정했다. 원문 재독은 Wearable만,
toolbox는 code reading이며 새로운 broad systematic review가 아니다. 다음 불확실성은
문헌 개수보다 실제 baseline 호환성과 조건부 M 비교가 더 직접적으로 줄인다.
기존 impedance-proxy/conditioning claim2개의 논문 evidence grade를 verified에서 reported로
정정했다. 수치 재현이나 모든 연관 논문 재독을 주장하지 않고 이전 input을 그대로 보존했다.

`coordinate-worktree-changes`: base main `beb4ebc668f7798f317d55bc7b76f1b82a1beb38`, clean.
기존22worktrees를 읽기 전용으로 보존, free약298GiB. 이번에는 main 단독 순차 수정,
두 에이전트는 판단/구현 feasibility 읽기 전용이다. 같은 계약에 의존하는 코드·문서를 병렬
작성하지 않아 새 worktree가 필요 없다. 기존 환경·데이터는 read-only, SQLite도 main 단독 작성이다.
