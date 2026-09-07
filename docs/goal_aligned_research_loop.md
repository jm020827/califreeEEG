# 연구목표에 도달하기 위한 실행 루프

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
