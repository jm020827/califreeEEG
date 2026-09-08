# 연구목표에 도달하기 위한 실행 루프

> 2026-09-08 후속 설계 완료: [분류 목표에 직접 맞춘 단일 Q+M 후보](task_aligned_trca_shape_v1_design.md). Proxy 예측 대신 source 분류 CE로 학습하며 고정 Q에 작은 M 잔차를 추가한다. 3개 사전 λ/Q-only 선택, 총120head fits 이하, 모든8조건·cost/harm·matched controls 및 음성 종료를 명세했다. **설계 완료이지 실행/효능 검증 완료가 아니다.** 다음은 순수 인공 구현 검증이며 새 eta/window/seed를 성공할 때까지 추가하지 않는다. 이전 후보 결과와 held60 경계는 유지한다.

> 2026-09-08 후속 기하 진단 완료: [규제가 실제 support 필터를 얼마나 바꿨나](trca_support_geometry_v1_results.md). Metadata를 넣기 전 고정γ=.1만으로 필터 방향이 중앙값 약44.4°/42.3° 바뀌었다. 큰 연산자 변화는 확인했지만 정확도 손실의 인과 원인이나 새 metadata 이득은 입증하지 않았다. 원 후보 종료·held60 보호는 유지하며 새 gamma 탐색/분류 실행은 없다. 아래는 기존 효능 결과와 이전 단계 기록이다.

> 2026-09-08 실제 실험 종료: [Source39 metadata-prior 결과](metadata_prior_source39_v1_results.md). 39명×8조건의 단일 후보 학습·평가와 독립 검산을 완료했다. QM3−Q3 −0.006677%p, QM5−Q5 0%p이고312개 사람×조건의80% 최초 도달 단계가 모두 같아 보정량 절감은0이다. **이 구현의 metadata 이득 미확립으로 종료**한다. 공통 규제부터 native FULL보다 낮았다는 한계, 실행 복구와 정수 count 보고 정정은 결과 문서에 보존한다. 연구목표는 유지하되 새 후보 튜닝·held60 자동 개봉은 없다. 아래는 이전 단계 기록이다.

> 2026-09-08 검증 설계 수리 완료: [M-blind Q 학습·필터 전달성 검사](metadata_prior_validation_repair_results.md). 참가자 분리 nested Q 선택, proxy 수준/채널 모양 분리, 식으로 만든 배열의 필터→점수→선택 경로를 구현·검증했다. 새 M 효능 실험은 아니며 이전 합성 v1의 효과 미확립 판정은 유지한다. 다음은 M을 보지 않는 별도 난이도 검증 설계이며 실제 EEG/held60은 이번에도 접근하지 않았다. 아래는 이전 단계 기록이다.

> 2026-09-08 합성 단계 완료: [M-prior v1 실제 합성 결과](metadata_trca_prior_synthetic_v1_results.md). Native-compatible operator와 Q/QM·대조군 구현/검산은 완료했지만, 고정 screen은 **효과 미확립**이다. 주요3시나리오 Q 정확도100%로 분류 ceiling이 있었고, 반복 불일치 예측도 Q만 추가 적합한 Q2가 QM보다 좋았다. 실패 start와 같은 suite의 복구 결과를 모두 보존했다. 새 사람 데이터/held60 접근0이며 난이도·Q 적합의 검증 설계가 다음 검토 대상이다. 결과를 보고 설정을 바꾸거나 사람 실험으로 자동 승격하지 않는다. 아래는 이전 단계 기록이다.

> 2026-09-08 설계 검토 완료: [Metadata의 보정학습 삽입 재검토](metadata_learning_covariance_design_review.md). 목표는 그대로이며, native TRCA에 동일 총량의 Q/Q+M 채널 규제를 주는 후보 하나를 제안했다. V1·context-template도 이미 학습단계 M을 사용했으므로 ‘최초 metadata 학습’이 아니다. 관련 공개 PDF2편 선택 정독·인공 대수 검산 완료, 새 사람 데이터 접근·효능 결과는 0이다. 다음은 proxy·대조군 명세와 순수 합성 구현 검증이며, source-only 입력 준비·실제 평가는 별도 단계다. 아래 완료 결과와 종료 경계는 그대로 보존한다.

> 2026-09-08 현재 완료: [Headroom cold-r1 진단](native_subset_headroom_cold_r1_results.md)과 독립 수치 검산을 마쳤다. 실제 Q k3/k5=36.99/44.21%, 정답을 아는 사후 상한=46.67/56.06%이며, 이상적 선택도312개 사람×조건 중217개는 관측 grid에서80% 미도달이다. **선택 개선 여지는 있으나 M 효과·실제 보정량 절감을 입증한 것은 아니다.** 전체1561tests PASS, 실패한 이전 start 보존·held60 미개봉. [현재 자료 우선 계획](current_data_first_research_plan.md)의 외부 paired-M 선택적 보강 원칙을 유지한다. 새 learner/추가 조건 탐색은 자동 실행하지 않는다. 아래는 과거 단계 이력이다.

> 2026-09-08 현재 완료: [Known-zero 단일 수정 실제 결과](native_subset_known_zero_source39_results.md)를 개발39명에서 단회 평가하고 독립 전체 검산을 통과했다. 수정 QM−Q는 k3 **0pp**, k5 **+0.01068pp**이며 80% 최초 도달 보정량은312개 사람×조건 모두 같았다. 수정 자체도 k5 정답을 Q2개/QM1개 줄였다. **추가 M 보정비용 절감 미확립으로 이 구현 탐색을 종료한다.** 직접 M 평가1회+기작수정 평가1회 예산을 사용했으며 새 후보/재학습/held60 자동 개봉은 없다. 연구목표는 유지하고 다음 공백은 독립 paired-acquisition 자료와 사전 가설이다. 전체1520tests PASS, 새 독립M자료·외부요청0. 아래는 보존한 이전 단계의 이력이다.

> 현재: [envelope-r1 결과](native_subset_m_envelope_r1_results.md)로 같은 고정 M 방식의 개발39명 효능평가를 처음 완료했다. 입력복구 attempt까지 총시도2회/완료효능평가1회/기작수정0회다. Primary M 증분0pp, k5는 정답1개 증가에 그쳤고 추가 보정비용 절감은 없다. 다음은 동일답 후보 중복/정확히0인 gain 처리의 기작적 타당성을 먼저 명세하는 것뿐이며, 효과가 날 때까지 변형하지 않는다. 기작근거 수정 최대1회 뒤 프로그램 점검이라는 한도와 held60 미개봉을 유지한다. 아래는 이전 단계의 ‘현재/다음’ 이력이고 자동 실행 권한이 아니다.

> 최신 상태: [직접 임피던스 입력 검증 실패](native_subset_m_source39_v1_results.md). 새 실행은 시작했지만 metadata 저장형식11개 key를6개로 가정한 오류로 EEG/fit/평가 전에 중단했다. 이 사건은 **시도1회/완료효능평가0회/M기작수정0회**다. 기존 metadata는 읽었으며 start-only 기록을 보존한다. 과학적 가설·study_id/SHAM배정은 유지하고 입력-adapter만 고친 별도 attempt authority가 필요하다. 긍정 결과까지 변형하는 루프로 바꾸지 않는다.

> **현재 완료:** [native eTRCA source39·두 bridge 결과](author_etrca_source39_v1_results.md) 단회 실행과 독립 검산을 마쳤다. Wet0.5초에서는 ETRCA5−A0+18.08pp이지만 전8조건+2.24pp CI는0을 포함하고17명 평균 악화다. 원본 baseline 호환성 확인을 종료하고 직접 M 가설1개로 돌아간다. 알려진 interface는 native preset에 이미 사용됐으므로 다음 두 비교군에도 동일 제공하고 추가 M의 순증분을 검정한다. 아직 새 M 실행·독립 paired-M 추가·held60 개봉은 없다. 아래 최신 human은 이전 상태다.

기준: 2026-09-07. 상위 목표는 **처음 보는 사용자의 labeled SSVEP calibration 부담 감소**다.
검정할 수단은 **외부 pre-query acquisition metadata가 EEG-only Q를 넘어 주는 순증분**이다.
긍정 결과를 반드시 얻는 것이 목표는 아니다. 유효한 검증에서 해당 방법이 실패하면 그
후보를 종료하고, 확인한 범위와 미해결 질문을 남긴다. 손상 복원·새 class discovery·LLM/OOD로
primary를 바꾸지 않는다.

> **현재 후속 순서 수정:** [기존 판단 재검토·유한 루프](research_decision_reaudit_20260907.md)가
> 아래 historical 설계의 ‘Q utility 먼저, 그 다음에만 M’이라는 일반 필수조건을 대체한다.
> 저자 구현 하나의 호환성을 확인한 뒤 동일한 강한 Q와 M+Q를 직접 비교한다. 새 분류기
> 계열을 계속 추가하지 않으며, M 방식1개와 기작 기반 수정 최대1회 뒤 프로그램을 점검한다.
> 기존 gate·결과·closed runner는 수정하지 않는다. Source39는 계속 개발용, held60은 미개봉이다.

> 최신 human 결과: [reference-calibration-source39-v1 결과](reference_calibration_source39_v1_results.md)는 두 전처리 모두 AQ_NOT_ESTABLISHED, 독립 검산 PASS다. 올바른 labels의 정보는 있지만 ECCA가 reference 기준을 크게 악화시켰다. 모든 band 진단과 사후 fixed-decision 점수분해까지 완료했다. Reference-preserving support 결합은 당시 제안이며 현재 후속은 위 재검토 계획이다. 해당 실험 Metadata/held 접근0, 독립 M자료 추가0. [새 eTRCA 합성 API 확인](author_etrca_compatibility.md)은 PASS했지만 새 human 실험은 아니다. 아래는 직전 종료 및 historical context 설계다.

> 직전 종료: [spatial-calibration-source39-v1 결과](spatial_calibration_source39_v1_results.md)는 AQ_NOT_ESTABLISHED, 독립 검산 PASS다. Metadata-free IT-CCA도 유용한 보정 학습을 만들지 못했다. 이때 제안한 대역별/reference-guided 진단은 위 별도 실험으로 완료했다. 기존 결과와 Metadata/manifest/held 접근0 기록은 그대로다.
>
> 직전 실행 결과: `context-template-source39-v1`은39명/6조건 단회 완료하고
> `AQ_NOT_ESTABLISHED`로 종료했다. [결과와 사후 learner 진단](context_template_source39_v1_results.md)을
> 우선한다. 아래는 변경하지 않은 당시 설계와 목표 경계다. 다음 단계는 metadata를 강하게
> 만드는 것이 아니라 실제 support 학습의 유용성을 먼저 세우는 새 development 설계다.

## 현재 병목을 해결하는 순서

```text
이미 확보된 실제 source39와 authentic block impedance
  → 새 raw-window 처리 + 강한 EEG-only 후보와 eTRCA
  → 같은 Q에 실제 M / missing / shuffled / stale / chronology control
  → 보정 비용·개인별 harm·negative controls까지 개발 검산
      ├─ no-go: 해당 후보 종료, held60 그대로 보호
      └─ ready: 방법 고정 + power/확인 프로토콜/접근권한 점검
                   → 미개봉 자료의 단회 확인 → 외부 독립 replication → 논문 claim
```

V3 development-v5, V4 pilot001, AQ study001은 모두 보존한다. AQ001의 빈 A0 적격 집합을
소급 수정하지 않는다. 그 결과는 실제 source 개발을 준비하는 데만 사용한다.
현 단계는 새 `context-template-source39-v1`이며 [고정 계약](../configs/analysis/context_template_source39_v1.json)이
과학·데이터 접근 기준원이다. 합성 조건을 성공할 때까지 바꾸는 루프는 더하지 않는다.

## 지금 실행할 실제 데이터 연구

이미 outcome에 노출된 Wearable source39만 허용한다. 새 사용자의 계속 진행 지시는 이
새로운 source-development 분석의 근거로 기록하며, 기존 후보의 seal/runner/private signature를
재사용하거나 완화하지 않는다. Raw `S###.mat`은39개 이름의 명시적 allowlist로만 열고
S001–S003와 held60은 금지한다. Shared parquet에서는 source39 predicate와 acquisition-column
projection을 적용한 행만 반환한다. 기관별 secondary-use 요건을 이 기록으로 승인했다는
뜻은 아니다. 새 참가자 모집·연락·동의 대행은 하지 않는다.

기존 processed2초 EEG를 다시 짧게 자르지 않는다. 원래 pipeline은 전체 epoch를
filter/resample한 뒤 crop·zscore하므로 짧은 구간 밖 정보가 섞일 수 있다. Raw250Hz에서
sample160부터 **125/188/250 samples =0.5/0.752/1초를 먼저 crop**한 뒤 같은7-band filter를
사용한다. 현재 window 밖의 QC, query batch, 과거/미래 query는 쓰지 않는다.
배포 자료가 이미1000→250Hz downsample된 upstream 처리는 재구성하지 않으므로 완전한
online causal acquisition을 검증했다는 주장도 하지 않는다.

Wet/dry별 첫5개 block이 nested support, 나머지5개가 모든 k에 동일한 query다.
각 block은12 classes이고 k1/3/5는 interface당12/36/60 labeled trials다.
주파수·phase·native channel 순서와 검증된 `[dry,wet]` impedance 축을 유지한다.

EEG-only 후보는 평균 파형, support reliability, diagonal 또는 full shrinkage log-covariance
query-support matching이다. 후보·온도는 fit26명에서만 실제 fused NLL로 결정한다.
각 outer13명을 평가하는3-fold cross-fitting이며39명 전체가 과거에 노출된 개발집합이라는
한계는 사라지지 않는다. Inner-validation을 했다고 부르지 않고, 작은 유한 파라미터의
supervised NLL fit과 outer evaluation을 구분한다. A0 적격성으로 주평가 조건을 비우지 않는다.

M은 실제 pre-block impedance의 support-query 차이를 이용해 **같은 Q의 block weights만**
조정한다. Source fit의 unique block packet으로 scale을 정하고 β=0을 허용한다.
M 전용 temperature나 backbone은 없다. k0/k1과 M-missing은 exact Q다. 따라서 이번
보정 절감 가설은 **5-shot→3-shot**이며 **3-shot→1-shot이 아니다**. 단순 block 순서와의
구분도 확인한다. eTRCA는 동일한 데이터의 k3/k5에서 실행하고, 불가능한 k1은 만들지 않는다.

Primary development contrast는 모든6개 window/interface의 participant별 M−Q early AUC다.
1초의 M3−Q5, Q5−Q3,80% 도달, eTRCA3와의 비교를 함께 요구한다. 이 개발 단계의 CI와
screen은 독립 확인 결과나 모든 최신 방법 대비 superiority가 아니다. β=0이나 no-go도
유효한 종료 결과이며 다시 고르지 않는다.
80%는 wet/dry와 참가자를 평균한 목표이며 각 interface나 개인 모두의 보장이 아니다.
Interface별 BA·도달 여부는 별도로 보고한다. M은 shuffle뿐 아니라 독립 fit한 강한
order-control과 stale-query packet보다도 우위여야 READY가 가능하다. 세부 status 우선순위와
기술적 CI 공식은 JSON에 명시하며, fold 공유 의존성을 보정한 confirmatory coverage는 주장하지 않는다.

해석 경계도 결과 전에 고정한다. k0/k1에서 M이 exactQ이므로 M−Q eAUC는 k3 BA 차이의
1/3이다. 따라서 primary mean0.005는 k3 평균 **1.5 percentage points 이상**의 개선을
뜻하며 positive LCB도 별도로 필요하다. 5→3-shot은 같은 AQ5에 비해 interface당 labeled
trials60→36, 즉24개 감소라는 비교다. eTRCA3도 이미80%를 달성한다면 eTRCA 대비
label 절감은 보인 것이 아니며, eTRCA와 같은36 trials에서의 성능 비교만 가능하다.
READY도 이 좁은 개발 단계 의미이지 최선의 기존 방법 대비 calibration 최소화를 입증한
것은 아니다. 총 준비시간·개인별 보장·임피던스 측정 자체의 시간 절감을 주장하지 않는다.

## 문헌과 데이터가 정한 경계

[Kalunga et al.의 SSVEP Riemannian 연구](https://arxiv.org/abs/1501.03227)와
[Khazem et al.의 저보정 Riemannian transfer 연구](https://arxiv.org/abs/2111.12071)는
covariance 기반 표현과 저보정 연구의 선례다. 원래 온라인·transfer 방법의 target 접근권한을
그대로 가져오지 않고 현재 query 하나와 허용 support로 제한한다. 이번 source 확인은
official author abstract 수준이며 두 논문의 정확한 재현이나 효과 크기를 주장하지 않는다.
Matrix-log Frobenius distance는 [공식 pyRiemann 정의](https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.geometry.distance.distance_logeuclid.html)를
확인했다. 우리 block-weighted waveform으로의 전이는 별도 검정할 가설이지 선행 논문이
입증한 metadata 효능이 아니다. 새로운 pyRiemann 의존성을 설치하지 않고 NumPy/SciPy로 계산한다.

`academic-research`의 기존 landscape를 재사용했고, covariance 구현 질문만 좁게 검색했다.
Semantic Scholar429와 옛 pyRiemann utils URL404를 기록한다. 공식 geometry 경로로 정의는
확인했다. 논문 수를 늘리는 것보다 실제 M와 강한 Q의 비교가 다음 불확실성을 더 직접적으로
해소한다. Data scout가 확인한 Wearable source39는 기존 공개 배포의 CC BY4.0 자료다.
별도의 최신 법률/윤리 적합성 판단은 하지 않는다.

독립 M replication은 아직 미확보다. Wang/BETA/Dong의 장비·reference·electrode 상수는
실측 block impedance가 아니므로 개수를 늘려 같은 가설의 반복 검증이라고 부르지 않는다.
[Liu2022](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.863359/full)의
다일 wet/dry·block impedance 자료는 저자 요청 경로가 남아 있다. 요청할 내용은 raw EEG,
channel별 block timestamp/impedance/단위, condition order, license, Wearable102와 participant
overlap 여부다. 연락하거나 사용자의 소속·윤리승인 정보를 만들어 보내지 않는다.

## 완료 기준과 중단 조건

- 코드 완료: 입력 범위, current-query locality, crop 경계, Q 선택의 M 독립성, missing/k0/k1,
  exact prefix derangements, Gram와 직접 파형 계산, public FBCCA/eTRCA 일치 시험.
- 개발 완료: 고정 source39 전원·모든 조건, calibration curve·harm·negative controls·cost,
  score 기반 독립 검산, 결과와 한계 및 후보 종료/진행 결정. 기술 오류는 efficacy failure와 구분.
- 확인 준비: 방법·hyperparameters·대상 조건 고정, 기존 held60/독립성·power·다중검정·접근권한
  확인. 현재 source39 실행 자체는 held를 열지 않는다.
- 논문 수준 완료: M의 순증분과 보정 비용 절감이 독립 확인됐거나, 충분한 검정력과 controls에서
  제한된 null/negative 결론을 제시할 수 있어야 한다. 외부 acquisition 일반화는 독립 자료가 있어야 한다.

## Worktree와 실행 소유권

Base main `c926697b103a6e71a2c707d82817a5e39e7477e5`, 시작 clean/ahead71.
새 checkout≈9MB+artifacts/test temp를1GB로 제한하며 free≈320GB로 충분하다.
기존 환경/공유 raw data는 read-only; 별도 dependency 설치, GPU 변경, 기존 worktree 삭제는 없다.

| Lane | 단독 소유 | 자원·검증·통합 |
| --- | --- | --- |
| Main | 새 plan/roadmap/authority, 상태·연구일지, independent audit와 tests | 유일 literature SQLite writer, source contract 먼저, 최종 통합·실행 |
| Implementation | 새 isolated context-source worktree의 새 module/runner/tests3개 | fixture만; main interpreter read-only, worktree별 pytest temp; commit 반환 |
| Reviewer | main/worktree 읽기 전용 | 데이터·수식·정보권한·결과 검토, EEG나 study 실행 금지 |

실제 human source 실행은 main 한 명만 수행한다. 입력 계약이 정해지기 전에는 구현을 병렬로
시작하지 않았고, 문헌·데이터 검토만 병렬화했다. Result가 좋다는 이유로 held 실행을 암묵적으로
추가하는 통합은 허용하지 않는다.

Implementation 경로는 `/home/whwovy/califreeEEG-wt-context-source`, branch는
`codex/context-template-source39-v1`이다. 단독 파일은
`src/cfeg/analysis/context_template_source.py`, `scripts/run_context_template_source.py`,
`tests/test_context_template_source.py` 세 개이며, main의 독립 audit 두 파일과 겹치지 않는다.
