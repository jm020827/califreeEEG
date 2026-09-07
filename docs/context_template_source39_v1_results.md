# 실제 source39 context-template 개발실험 결과

2026-09-07. **단회 실행 완료, `AQ_NOT_ESTABLISHED`, 별도 구현의 독립 검산 PASS.**
이 후보는 종료한다. 목표는 metadata-assisted low-calibration SSVEP 그대로이며,
이 결과는 metadata 전체의 무용성이나 독립 확인 결과를 뜻하지 않는다.

## 쉽게 말하면

같은 자극의 보정 EEG를 평균해 “이 사람의 정답 파형”을 만들고, 현재 EEG와 비슷한
측정 상태의 보정 block을 더 많이 반영하도록 했다. EEG만으로 고른 weights가 AQ,
실제 pre-block impedance까지 더한 weights가 AQM이다.

문제는 **정답 파형 자체가 실제 데이터에서 약한 분류기**였다는 점이다. 보정 예제를
추가해도 기존 무보정 FBCCA 결과가 거의 바뀌지 않았다. Impedance가 weights를 바꾸기는
했지만 유용한 분류 개선이나 보정 비용 감소로 이어지지 않았다. 합성 pilot의 포화 문제와
달리 이번 실제 데이터에는 성능 개선 여지가 충분했으며, 현재 학습 방법이 그 여지를
활용하지 못했다. 이를 threshold 변경이나 다른 seed 재시도로 구제하지 않는다.

## 평가 범위와 실제 결과

과거에 노출된 Wearable source39, 3-fold fit26/evaluate13 development다. Native250Hz에서
먼저 crop한125/188/250 samples(0.5/0.752/1초), wet/dry를 모두 평가했다. 각 interface의
첫5 blocks 중 k개를 support로 쓰고 마지막5 blocks의60 queries는 모든 k에 동일하다.
12class이므로 k1/3/5는12/36/60 labeled trials다. Random chance는8.33%다.

아래는39명 평균 balanced accuracy(%). 모든 사전 조건을 표시한다.

| 조건 | A0, k0 | AQ, k1 | AQ, k3 | AQ, k5 | AQM, k3 | eTRCA, k3 | eTRCA, k5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dry,0.5초 | 12.52 | 12.52 | 12.48 | 12.52 | 12.48 | 11.24 | 13.59 |
| wet,0.5초 | 27.61 | 27.61 | 27.65 | 27.65 | 27.61 | 38.12 | 47.78 |
| dry,0.752초 | 29.49 | 29.70 | 29.70 | 29.83 | 29.70 | 11.50 | 13.80 |
| wet,0.752초 | 48.59 | 48.25 | 48.42 | 48.21 | 48.38 | 39.91 | 50.38 |
| dry,1초 | 37.52 | 37.78 | 37.91 | 37.56 | 37.91 | 11.32 | 13.89 |
| wet,1초 | 57.99 | 57.99 | 57.91 | 57.95 | 57.91 | 41.37 | 50.77 |

eTRCA는 같은 crop/filterbank와 기존 public helper의 고정 설정으로 평가한 comparator다.
원 논문 전체 프로토콜 재현, 최신 SOTA 대결, 모든 조건에서의 열등성을 주장하지 않는다.
특히 wet0.5초에서는 실제 calibration learning이 보이므로 “이 데이터에는 학습 신호가 없다”는
결론도 틀리다. k1 eTRCA를 임의 생성하지 않았다.

## 고정 판정

각 참가자 내부에서6조건을 동일 가중 평균했다. CI는 과거 노출 및 fold 공유 의존성이 있는
개발용 descriptive t interval이며 confirmatory coverage나 독립39명 iid 검정을 주장하지 않는다.

| 비교 | Mean | One-sided95 LCB | 판정 |
| --- | ---: | ---: | --- |
| AQ−A0 early AUC | +0.000296771 | −0.000514489 | ≥0.01 및 LCB>0 실패 |
| AQM−AQ early AUC | −0.000047483 | −0.000103341 | ≥0.005 및 LCB>0 실패 |
| M3−shuffle3 BA | +0.000035613 | −0.000199764 | 실패 |
| M3−order3 BA | +0.000071225 | −0.000331778 | 실패 |
| M3−stale3 BA | +0.000071225 | −0.000048857 | 실패 |
| 1초 Q5−Q3 BA | −0.001495726 | −0.003796501 | 학습 증가 실패 |
| 1초 M3−Q5 BA | +0.001495726 | −0.000805048 | 비열등 기준만 통과 |
| 1초 M3−eTRCA3 BA | +0.215598291 | +0.171952303 | classical 비교 통과 |

1초에서 wet/dry 평균 M3=47.906%, Q5=47.756%로80% 목표에 크게 못 미쳤다.
둘 다 낮은데 서로 비슷하다는 사실은 **60→36 trials로 유용한 성능을 유지했다는 증거가 아니다**.
AQ utility부터 실패하므로 공식 status는 AQ_NOT_ESTABLISHED다. M increment, 세 specificity,
Q learning, cost target도 실패했다. Cost noninferiority와 classical 비교 통과만으로 승격하지 않는다.

M0/M1은 exactQ이고 primary M increment는 k3 BA 차이의1/3이다. 실제 k3 M−Q는
−0.014245 percentage points로, 전체14040 query decisions에서 net2개 정답 감소에 해당한다.
AQ 대비 최종 예측 변화는5개뿐이다. 참가자별6조건 평균은 악화2명/개선0명/동률37명이다.
Metadata weights turnover는 k3 dry약0.0068–0.0069, wet약0.0166–0.0169이며 effective blocks는
약2.97–2.99다. “실행은 됨”과 “유용한 기여가 있음”을 구별한다.

80% 최초 도달도 AQ/AQM은 각 조건에서 A0의 도달자를 한 명도 늘리지 못했다.
1초 dry는4/39명이 이미k0에 도달하고35명은>5, wet는11/39명이k0에 도달하고28명은>5다.
eTRCA wet0.5초는 k3에5명,k5에추가5명,29명미달이다. 전체66개 method×cell attainment과
1638개 participant harm record는 canonical result에 남겼다. 미달자의 비용을 임의 보간하지 않는다.

## 사후 원인 진단 — 새로운 효능 실험이 아님

이 절은 결과 종료 후 **저장된 Gram/score만** 읽은 diagnostic이다. Raw EEG 재접근,
temperature 재적합, 새 조건 선택, original result 변경은 없다.
[재현 script](../scripts/diagnose_context_template_source.py)는 출력을 stdout에만 내고 파일을 변경하지 않는다.

Fold별 선택 Q는 reliability/uniform/reliability, M β는1/0/1, order β는0/0/4였다.
Full/diagonal covariance 후보가 선택되지 않았으며, source M NLL 개선 자체도 매우 작았다.
고정된 Q의 support temperature는 k1 모두10, k3는6.813/10/10, k5는6.813/6.813/4.642였다.
그 결과 support posterior의 평균 최댓값은 약8.56–9.21%로12class 균등확률8.33%에 가깝다.

| 조건 | 평균 파형 단독 k1 BA | k3 BA | k5 BA |
| --- | ---: | ---: | ---: |
| dry,0.5초 | 8.68% | 9.36% | 9.62% |
| wet,0.5초 | 10.64% | 12.91% | 13.59% |
| dry,0.752초 | 8.85% | 9.40% | 9.74% |
| wet,0.752초 | 11.71% | 13.12% | 15.64% |
| dry,1초 | 9.10% | 9.79% | 9.74% |
| wet,1초 | 12.09% | 15.04% | 16.41% |

Standalone argmax는 temperature와 무관하므로 이것은 단지 확률 보정만의 문제가 아니다.
실제 파형 template의 class 구별이 약하고, 강한 flattening은 그 문제를 드러내는 현상이다.
순수 uniform posterior를 A0와 절반 섞는 **support-free diagnostic**도 대부분 같은 NLL을
냈다. 예를 들어 wet1초 k3 AQ NLL1.856084 대 uniform mixture1.855967; A0 자체는1.576174다.
반대로 dry0.5초는 A0 NLL2.906422 대 AQ2.499150/uniform mixture2.499015로, NLL 개선만
보고 support 학습 성공으로 오해할 수 있다. 이 uniform 비교는 사후 설명이며 새 gate가 아니다.

왜 파형 구별이 약한지의 완전한 인과 원인은 미확정이다. eTRCA와 단순 평균의 차이는
공간필터 등 여러 요소를 바꾸므로 공간필터 하나가 원인이라고 입증한 ablation은 아니다.
다만 다음 개발의 우선순위를 정하기에는 충분하다: phase/공간 구조를 다루는 강한 supervised
support learner를 확보하고, 실제로 도움이 되는 learner에 metadata를 연결해야 한다.

## 다음 연구 루프와 종료 경계

1. 이 후보와 plan/result를 종료·보존한다. M β/grid/threshold를 바꿔 이 평가를 재사용하지 않는다.
2. 다음 후보는 **metadata-free calibration learner 검증부터** 한다. Source-only에서
   spatial-filter/template 또는 correlation-alignment baseline을 정확히 구현·대조하고,
   A0와 uniform-shrinkage를 넘어 실제 support 사용 이득·개인별 harm·window별 학습곡선을
   먼저 확인한다. 이번 wet0.5초만 사후 선택해 확인 성공으로 부르지 않는다.
3. 그 다음에만 같은 learner의 block weighting 또는 acquisition-noise prior 한 지점에
   M을 넣는 새 유한 설계를 고정한다. 구체적인 새 operator는 아직 확정·실행하지 않았다.
   현재 약한 template에 더 큰 metadata 계수를 붙이는 재튜닝을 다음 연구로 삼지 않는다.
4. 실제 user-time·80% target에 관한 window/label trade-off가 필요하면 observation 길이를
   별도 축으로 사전 고정한다. 긴 EEG를 썼다는 사실을 labeled calibration 감소로 숨기지 않는다.
5. 유용한 Q, 실제 M 순증분, specificity, cost가 새 source development에서 확인된 뒤에만
   별도 confirmatory protocol/power/access review를 한다. 이번 실행으로 held60을 열지 않는다.

독립 replication용 authentic channelwise impedance 자료는 아직 추가 확보하지 못했다.
[자료 요청안](independent_impedance_data_request.md)에 필요한 항목·이용조건·사용자 정보와
미발송 영문 초안을 남겼다. 논문 수준 전체 연구가 끝났거나 metadata 효능이 입증됐다고 말하지 않는다.

## 재현과 보존

Start UTC2026-09-07T07:12:13.934720+00:00. Clean source
`46b87b7a6c2bae2ca6ac35cdc079c7c8a50ebb0d`, tree`8ead7277a44c8f710de009ed4f47fdf8d8e34c9a`.
Plan SHA`a76f1df0e0d2989f6217013e610d074e50d9efb52b0cf636bbcdf38a905898d3`.
CPU4/BLAS1,26.01s, reported maxRSS513648KB. Start 후780metadata packets와 raw39만 접근했고
모든 fit freeze 후 outer evaluation했다. 9828metric rows, no study RNG, exit0.
Root는 `/home/whwovy/context-template-artifacts/source39-v1`, 정확히5files 모두0400/nlink1이다.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| start.json | 2033 | baa7104b659738b79a7cae78b958330195ab17d1adc591c097541c203f4c0125 |
| source-projection.json | 314455 | 1082a81d23ebffe26e8451a9e99f1a7d4c84d0ba2d9d00c6695b3a5c6a5114d4 |
| features.npz | 58684358 | 75f84e5f643691c33c25bd468085c519c5934825b917068b1615ddb65a8e600e |
| fold-freezes.json | 109107 | 540d91f63e599cca996f831508060770218aad4d8b11adc3267f6be9da052eac |
| result.json | 11054350 | 65e6f570a4a50281f23bfde805cdc569e96b8c55a5185ae58aab286724f31a61 |

[독립 auditor](../scripts/audit_context_template_source.py)는2532fit objectives,9828rows,
252aggregates,1638harm,66attainment 및 provenance를14.94s에 재계산해 PASS했다.
새 human EEG 재생성/전처리 replay가 아니라 저장된 sufficient statistics 이후의 별도 산술 검산이다.
실행 전 전체983tests PASS, 이후 read-only diagnostic까지 focused42tests PASS.
이전 V3/V4 결과·helper·closed runners·held/retired 자료는 변경하지 않았다.

`academic-research`는 covariance 선례와 우리 transfer 가설을 분리하고 측정 결과를 evidence로
기록하게 했으며, `coordinate-worktree-changes`는 구현3files/독립 audit/main docs를 분리해
통합했다. 문헌 수를 늘린 것을 효능 근거로 세지 않는다.
