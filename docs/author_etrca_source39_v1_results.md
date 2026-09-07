# Author eTRCA source39 v1 — 실제 결과와 다음 판단

2026-09-07. [고정 설계](author_etrca_source39_v1_design.md)·[JSON](../configs/analysis/author_etrca_source39_v1.json).
**단회 39명 완료, 독립 저장통계 검산 PASS.** 완료 상태는 COMPATIBILITY_ASSESSMENT_COMPLETE이며 효능 PASS가 아니다.

## 쉽게 설명하면

정답 예시를 주면 학습되는 신호는 있다. 하지만 예시를 주는 것이 항상 무보정보다 낫지는 않다.
Wet 0.5초에서 무보정33.42% → 클래스당3예시43.72% →5예시51.50%였다.
반면 dry2초에서는41.54% →22.52% →29.83%였다. **보정량5가3보다 나은 것**과
**보정이 무보정보다 유용한 것**은 다른 질문이다.

8조건 모두에서 ETRCA5의 평균 BA는 ETRCA3보다 높았다(+5.04–8.46pp).
그러나 동일8조건을 동등 가중한 ETRCA3−A0는−4.65pp [−7.58,−1.73],
ETRCA5−A0는+2.24pp [−0.59,+5.07]였다. 대응 개인별 평균에서3예시는12명 도움/27명 악화,
5예시는21명 도움/1명 동률/17명 악화다. “보정하면 전반적으로 해결됐다”는 결론은 안 된다.

현재 허용되는 결론은 **한 공개 구현의 source39 호환성 확인을 완료했고 조건별 학습 이득·손실을
확인했다**는 것이다. 적은 보정의 신규성, 원 논문 성능의 정확 재현, unseen cohort 일반화,
임피던스 추가 효과 또는 유용 정확도까지의 보정 절감은 입증하지 않았다.
정확도가 높아야 구현이 맞고 낮으면 틀렸다는 판정을 하지 않는다. 수식·API·분할 검사는 별도다.

## 무엇을 실행했나

- Toolbox revision3344bd199daf78888e364d9db00ae7d8128d2b5f의 ETRCA와 SCCA_canoncorr를 수정 없이 실행했다.
  A0_author는 native 선형 weighted rho이며 기존 squared-rho FBCCA와 같지 않다.
- Native-style2s:9support/1query LOBO10회. All10은120queries/명/전극, last5는60queries다.
- Bridge1:2s에서 앞3/5blocks 학습, 뒤5blocks 고정평가. Bridge2:동일 분할의0.5/.752/1s.
  매 길이에서 query-end까지만 native 전처리를 다시 계산했다.
- 전39명 stored dtype은float64, data shape[8,710,2,10,12]. 변수data만 파싱했고 축 추측·fallback·float32 중간 변환은 없다.
- Impedance.mat/manifest/M packet/held60/retired1–3/oldrunner 접근은 없었다.
  다만 **알려진 dry/wet interface는 층화와 공개 preset 선택에 사용**됐다.
  따라서 metadata_access:false를 ‘모든 acquisition context를 모르는 순수 EEG-only’라는 뜻으로 확대하면 안 된다.
  이 연구에는 새 M 연산자도 M 유무의 matched contrast도 없다.
- Source39는 반복 노출된 개발자료다. 모든 CI는39명 단위 기술적 paired/mean t95이며
  adaptivity·다중비교를 보정한 확인 검정이나 미관측 코호트 증거가 아니다.

## Native 조건: baseline가 어디까지 작동했나

모든 수치는 BA%, 대괄호는 기술적95% CI다. 아래 두 표가52개 사전 summary를 전부 포함한다.

| Query view | 전극 | A0_author k0 | ETRCA k9 | 순환 diagnostic k9 |
| --- | --- | --- | --- | --- |
| all10 | dry | 41.86 [32.54, 51.18] | 33.76 [25.29, 42.23] | 6.02 [5.25, 6.79] |
| all10 | wet | 76.03 [68.18, 83.87] | 76.13 [67.22, 85.04] | 2.17 [1.36, 2.98] |
| last5 | dry | 41.54 [32.22, 50.86] | 33.12 [24.67, 41.57] | 6.08 [5.31, 6.85] |
| last5 | wet | 74.70 [66.51, 82.89] | 75.77 [66.59, 84.95] | 2.20 [1.37, 3.04] |

Dry all10에서 ETRCA9−A0는−8.10pp, wet은+0.11pp [−5.00,+5.22]다.
Native 조건에서도 uniform superiority는 없다. 논문 전체102명과 동일 cohort·원 결과 파일 대조가
없으므로 ‘저자 숫자 재현 완료’라고 쓰지 않는다.

Last5 동일 query에서9support LOBO→chronological3/5의 변화는 dry−10.60/−3.29pp,
wet−13.38/−5.43pp다. **예시 수 감소와 학습 시간순서가 함께 달라졌다**.
전체 성능 하락을 시간 변화, sample 수, drift 중 하나의 원인으로 단정할 수 없다.
LOBO의 미래 support를 실제 온라인 초기 보정 비용으로 세지 않는다.

## 실제 시간순 저보정: 모든 조건

| 전극 | EEG초 | A0_author k0 | ETRCA k3 | ETRCA k5 | 순환 k3 | 순환 k5 |
| --- | --- | --- | --- | --- | --- | --- |
| dry | 0.5 | 15.13 [11.08, 19.18] | 14.36 [10.17, 18.55] | 19.70 [14.19, 25.22] | 7.79 [7.40, 8.17] | 7.30 [6.80, 7.80] |
| dry | 0.752 | 21.20 [15.40, 27.00] | 16.71 [11.01, 22.41] | 21.75 [15.19, 28.31] | 7.57 [7.05, 8.09] | 7.11 [6.52, 7.71] |
| dry | 1 | 25.43 [18.64, 32.21] | 17.74 [11.86, 23.61] | 22.78 [16.07, 29.48] | 7.48 [6.94, 8.01] | 7.02 [6.41, 7.63] |
| dry | 2 | 41.54 [32.22, 50.86] | 22.52 [15.13, 29.91] | 29.83 [21.81, 37.85] | 7.04 [6.37, 7.72] | 6.38 [5.65, 7.11] |
| wet | 0.5 | 33.42 [25.42, 41.42] | 43.72 [33.74, 53.69] | 51.50 [41.24, 61.75] | 5.12 [4.21, 6.02] | 4.41 [3.48, 5.34] |
| wet | 0.752 | 46.71 [37.52, 55.90] | 47.52 [37.05, 57.99] | 55.77 [45.16, 66.37] | 4.77 [3.82, 5.72] | 4.02 [3.06, 4.99] |
| wet | 1 | 56.24 [46.90, 65.58] | 52.18 [41.04, 63.32] | 60.64 [50.13, 71.15] | 4.35 [3.33, 5.36] | 3.58 [2.62, 4.53] |
| wet | 2 | 74.70 [66.51, 82.89] | 62.39 [51.61, 73.18] | 70.34 [60.25, 80.44] | 3.42 [2.44, 4.40] | 2.70 [1.78, 3.61] |

짧은 구간은 세 방법 모두2초보다 평균 정확도가 낮았다. 특히 A0의 관측시간 의존도도 크므로
서로 다른 EEG 길이의 정확도를 맞춰 ‘보정 절감’이라고 할 수 없다.

전8조건 평균은 A0=39.29%, ETRCA3=34.64%, ETRCA5=41.54%다.
Wet0.5초의 ETRCA5−A0=+18.08pp [12.12,24.03]는 개발 측정이지만,
평균51.50%는 기존 useful target80%와 거리가 있다. 좋은 한 cell을 사후 primary로 채택하지 않는다.
Dry2초의 ETRCA5−A0=−11.71pp [−16.99,−6.43] 등 불리한 결과도 같은 수준으로 보고한다.

순환 열은11개 candidate-score rotation의 BA 평균이다. Unique maximum에서(1−real BA)/11이므로
별도 잘못된 정답으로 학습한 실험의 독립 증거가 아니다. Human wrong-label refit이나 독립 specificity PASS로 쓰지 않는다.

## 개인별 변화와 보정 비용

각 행의 참가자는 같은39명이며 행들을 독립312명으로 세지 않는다. 도움/동률/악화는 같은60query의 A0 대비다.

| 전극 | EEG초 | k3 도움/동률/악화 | k5 도움/동률/악화 | 최초80%: k0 / k3 / k5 / 미도달 |
| --- | --- | --- | --- | --- |
| dry | 0.5 | 17/3/19 | 26/6/7 | 0 / 1 / 0 / 38 |
| dry | 0.752 | 12/3/24 | 15/7/17 | 1 / 1 / 0 / 37 |
| dry | 1 | 7/0/32 | 12/4/23 | 2 / 0 / 1 / 36 |
| dry | 2 | 6/1/32 | 11/2/26 | 7 / 0 / 0 / 32 |
| wet | 0.5 | 31/1/7 | 31/3/5 | 2 / 7 / 2 / 28 |
| wet | 0.752 | 19/2/18 | 25/4/10 | 8 / 3 / 4 / 24 |
| wet | 1 | 16/1/22 | 24/3/12 | 11 / 3 / 3 / 22 |
| wet | 2 | 10/5/24 | 16/6/17 | 22 / 1 / 1 / 15 |

Wet0.5초에서는 관측한0/3/5 중 한 번이라도80%에 도달한 사람이11/39명이다.
그중2명은 처음부터k0 성공,7명은 최초k3,2명은 최초k5이고28명은 관측 grid에서 미달했다.
이 값은 독립 확인이나 전사용자 보정 절감을 뜻하지 않는다.

최초 관측k는0→3→5 순서이며 나중 성능 하락을 감추지 않는다. 미측정k1/2/4는 보간하지 않는다.
최초 도달이 이후 보정에서도 유지된다는 뜻은 아니다. 예를 들어 wet2초에서 k5는
A0로80%에 도달했던3명을 다시 그 기준 아래로 낮췄다(개인별 곡선의 사후 설명, 새 gate 아님).
censored_above_5는 이 grid에서 미달이라는 뜻이고 진정한 최소 필요k>5라는 하한이 아니다.
Class당3예시는36labels,5예시는60labels다. 예컨대0.5초에서 retained EEG는18/30초,
추가 처리 history는5.04/8.40초다. 실제 raw acquisition의 stimulus duration·휴식·준비 및
미래 impedance 측정 비용을 포함한 사용자 wall-clock 절감은 아직 계산하지 않았다.

## 다음 판단: baseline 계열 탐색은 여기서 종료

[유한 연구 루프](research_decision_reaudit_20260907.md)의 호환성 sprint1개와 bridge2개를 사용했다.
새 custom decoder나 유리한 조건 검색을 추가하지 않는다. Q>A0 평균 성공을 M 검정 입장권으로 복구하지 않는다.

다음은 **같은 검증된 learner에서 EEG Q만으로 정한 support 기여도와 Q+실제 사전 acquisition 정보로
정한 기여도를 직접 비교**하는 하나의 가설이다. 아직 그 새 연산자를 실행한 것은 아니다.

1. 알려진 interface·기본 band preset을 두 비교군에 똑같이 제공한다. 그렇다면 impedance의 추가 효과는
   ‘EEG Q와 알려진 interface를 넘어서는 추가 정보’로 명시한다. 공개 preset이 상속한 전체 코호트 tuning도 별도로 남긴다.
2. 현재 관측한 wet/dry·window별 효과 차이는 **후속 가설을 세울 근거**일 뿐,
   impedance가 Q 이상으로 예측한다거나 전극 선택의 인과효과라는 증거가 아니다.
3. M의 입력과 단일 삽입 위치, 같은 capacity/regularization/학습예산의 Q 대조군,
   query-blind 정규화, source participant-disjoint fit, shuffle/stale/missing controls와 실제 비용을 먼저 고정한다.
4. 저보정 유용 정확도·개인 harm를 함께 평가한다. Source39는 개발용 그대로이고,
   M 방식1개+기작 근거 수정 최대1회 뒤 프로그램 점검 예산을 유지한다.
   새 독립 paired-M 자료는 확보되지 않았으며 held60 자동 실행 권한도 없다.

## 실행·증거·검증 범위

- Plan SHA d36886d4dfd80961d1d8b7ceac685d468ca12211ff6149ecbf993cc0b4502462.
- Clean source dfe18187f6b341c4ffc8edfb0253b14ecb7a75fe; tree 1f193f16138a74157c2ceb08bfb233f3b07054c3.
- UTC 2026-09-07T11:40:46.952423+00:00 → 2026-09-07T11:43:25.387596+00:00. Process wall159.52초, userCPU610.63초,
  system1.69초,383%CPU, exit0. CPU4/BLAS1, CUDA·기존 환경 변경0.
- GNU time maxRSS197952KiB는 process-group 동시 합계 peak가 아니다.
  실행 약94초 시점의 parent+4workers RSS 스냅샷합은670968KiB였으며 peak 보장이 아니다.
- 실제 최종 source에서전체1196tests PASS117.15초/기존Torchwarnings68.
  독립 native eigen/CCA·full artificial adapter 검사, root saved-correlation 산술 검산을 구분했다.
- 독립audit PASS:2028rows/52summary/50contrasts/624harm/312attainment와 start/source/plan/upstream/cache binding.
  Raw EEG 전처리 재실행을 독립 재현한 audit은 아니다.
- 출력3개 모두0400/nlink1, 합계38885472bytes. 기존 source 결과·봉인·worktree와 temp는 보존, push/cleanup없음.

| Artifact | SHA-256 |
| --- | --- |
| correlations.npz | 5e4197d2de7b05130a36a2bd980d395db1adc62dcec9beca2c1dc7f7cca8b21f |
| result.json | a03c258925d304454d71908de70b8c32020a62e1f957afc46e512191230bdf9e |
| start.json | e13f43ac109d52db673aa9696418018180e1f2706101e5c604b7bea4ed0c0668 |

재검산: main .venv로 scripts/audit_author_etrca_source.py에 --plan configs/analysis/author_etrca_source39_v1.json
--output /home/whwovy/author-etrca-artifacts/source39-v1를 전달한다. --details는 모든 개인 진단까지 출력한다.
[원 결과](/home/whwovy/author-etrca-artifacts/source39-v1/result.json)·
[독립 검증기](../scripts/audit_author_etrca_source.py).

academic-research는 논문 주장·원본 코드 정의·우리 측정을 분리하고, 독립 확인으로 과장하지 않도록
기록 구조에 반영했다. coordinate-worktree-changes는 producer2files의 격리 구현과 main 단독 통합에 적용했다.

## 사전 지정50개 대비 전체

평균차는 pp, CI는 앞서 명시한 기술적 구간이다. 결과에 따라 대비를 선택하거나 게이트를 만들지 않았다.

| 대비 | 전극 | N | 평균차 pp | 기술적95% CI pp |
| --- | --- | --- | --- | --- |
| native_ETRCA9_minus_A0_all10 | dry | 500 | -8.10 | [-12.18, -4.02] |
| native_ETRCA9_minus_A0_last5 | dry | 500 | -8.42 | [-13.16, -3.67] |
| chrono_ETRCA3_minus_native9_last5 | dry | 500 | -10.60 | [-13.96, -7.24] |
| chrono_ETRCA5_minus_native9_last5 | dry | 500 | -3.29 | [-5.87, -0.71] |
| chrono_ETRCA3_minus_A0 | dry | 125 | -0.77 | [-2.57, 1.03] |
| chrono_ETRCA5_minus_A0 | dry | 125 | 4.57 | [1.43, 7.72] |
| chrono_ETRCA5_minus_ETRCA3 | dry | 125 | 5.34 | [2.41, 8.28] |
| A0_author0_short_minus_2s | dry | 125 | -26.41 | [-33.65, -19.17] |
| ETRCA3_short_minus_2s | dry | 125 | -8.16 | [-12.63, -3.70] |
| ETRCA5_short_minus_2s | dry | 125 | -10.13 | [-14.10, -6.16] |
| chrono_ETRCA3_minus_A0 | dry | 188 | -4.49 | [-8.21, -0.76] |
| chrono_ETRCA5_minus_A0 | dry | 188 | 0.56 | [-2.84, 3.95] |
| chrono_ETRCA5_minus_ETRCA3 | dry | 188 | 5.04 | [2.26, 7.83] |
| A0_author0_short_minus_2s | dry | 188 | -20.34 | [-26.04, -14.64] |
| ETRCA3_short_minus_2s | dry | 188 | -5.81 | [-9.70, -1.92] |
| ETRCA5_short_minus_2s | dry | 188 | -8.08 | [-11.39, -4.76] |
| chrono_ETRCA3_minus_A0 | dry | 250 | -7.69 | [-10.50, -4.88] |
| chrono_ETRCA5_minus_A0 | dry | 250 | -2.65 | [-5.37, 0.07] |
| chrono_ETRCA5_minus_ETRCA3 | dry | 250 | 5.04 | [2.37, 7.71] |
| A0_author0_short_minus_2s | dry | 250 | -16.11 | [-20.53, -11.69] |
| ETRCA3_short_minus_2s | dry | 250 | -4.79 | [-7.22, -2.35] |
| ETRCA5_short_minus_2s | dry | 250 | -7.05 | [-9.63, -4.47] |
| chrono_ETRCA3_minus_A0 | dry | 500 | -19.02 | [-24.51, -13.53] |
| chrono_ETRCA5_minus_A0 | dry | 500 | -11.71 | [-16.99, -6.43] |
| chrono_ETRCA5_minus_ETRCA3 | dry | 500 | 7.31 | [4.35, 10.26] |
| native_ETRCA9_minus_A0_all10 | wet | 500 | 0.11 | [-5.00, 5.22] |
| native_ETRCA9_minus_A0_last5 | wet | 500 | 1.07 | [-4.38, 6.52] |
| chrono_ETRCA3_minus_native9_last5 | wet | 500 | -13.38 | [-19.06, -7.69] |
| chrono_ETRCA5_minus_native9_last5 | wet | 500 | -5.43 | [-8.28, -2.57] |
| chrono_ETRCA3_minus_A0 | wet | 125 | 10.30 | [4.27, 16.33] |
| chrono_ETRCA5_minus_A0 | wet | 125 | 18.08 | [12.12, 24.03] |
| chrono_ETRCA5_minus_ETRCA3 | wet | 125 | 7.78 | [4.29, 11.26] |
| A0_author0_short_minus_2s | wet | 125 | -41.28 | [-47.24, -35.32] |
| ETRCA3_short_minus_2s | wet | 125 | -18.68 | [-23.90, -13.45] |
| ETRCA5_short_minus_2s | wet | 125 | -18.85 | [-23.86, -13.83] |
| chrono_ETRCA3_minus_A0 | wet | 188 | 0.81 | [-5.42, 7.05] |
| chrono_ETRCA5_minus_A0 | wet | 188 | 9.06 | [2.62, 15.50] |
| chrono_ETRCA5_minus_ETRCA3 | wet | 188 | 8.25 | [5.34, 11.15] |
| A0_author0_short_minus_2s | wet | 188 | -27.99 | [-33.01, -22.98] |
| ETRCA3_short_minus_2s | wet | 188 | -14.87 | [-19.41, -10.33] |
| ETRCA5_short_minus_2s | wet | 188 | -14.57 | [-19.42, -9.72] |
| chrono_ETRCA3_minus_A0 | wet | 250 | -4.06 | [-10.06, 1.94] |
| chrono_ETRCA5_minus_A0 | wet | 250 | 4.40 | [-0.52, 9.32] |
| chrono_ETRCA5_minus_ETRCA3 | wet | 250 | 8.46 | [5.00, 11.92] |
| A0_author0_short_minus_2s | wet | 250 | -18.46 | [-22.20, -14.72] |
| ETRCA3_short_minus_2s | wet | 250 | -10.21 | [-13.90, -6.53] |
| ETRCA5_short_minus_2s | wet | 250 | -9.70 | [-13.03, -6.37] |
| chrono_ETRCA3_minus_A0 | wet | 500 | -12.31 | [-19.14, -5.48] |
| chrono_ETRCA5_minus_A0 | wet | 500 | -4.36 | [-9.93, 1.21] |
| chrono_ETRCA5_minus_ETRCA3 | wet | 500 | 7.95 | [4.21, 11.68] |
