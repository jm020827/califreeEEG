# Author eTRCA source39 v1 — native 호환성과 두 protocol bridge

2026-09-07, 사용자 “응 시작”에 따른 새 development-only 단계.
권위는 [고정 JSON](../configs/analysis/author_etrca_source39_v1.json)이며, 이전 후보는 재개하지 않는다.
이 단계는 [수정한 연구 루프](research_decision_reaudit_20260907.md)의 baseline 확인이다.

## 무엇을 확인하나

새 EEG 분류기를 만들지 않는다. 이미 합성 API를 확인한 toolbox 저자의 ETRCA와
무보정 SCCA를 수정 없이 실제 source39에서 실행한다. Metadata는 사용하지 않는다.

| 단계 | EEG 길이 | 보정과 평가 | 해석 |
| --- | --- | --- | --- |
| Native-style | 2초 |10개block 중 하나 전체를 query로, 나머지9개로 학습;10회 | 저자 방식에 가까운 offline LOBO,108labels/fit |
| Bridge1 | 2초 | 앞3개/5개block 학습, 뒤5개block 고정query | 보정량과 시간순서가 함께 달라지는 비교 |
| Bridge2 | 0.5/0.752/1초 | Bridge1과 같은 support/query | 시간창 단축; 각 길이에서 전처리를 다시 계산 |

Native LOBO는 전체10개query와 마지막5개query의 결과를 각각 낸다. Bridge1과 비교할 때는
후자의 동일 trial ID만 사용한다. LOBO는 미래 block도 학습에 쓰므로 online 또는 처음 사용자
보정 비용으로 해석하지 않는다. 모든 participant/interface/window를 보고하며 winner를 고르지 않는다.

Source39는 이미 반복 노출된 개발자료다. 단순 paired descriptive t95 구간은 보고할 수 있지만
반복 탐색을 보정한 확인 검정이나 미관측39명의 증거는 아니다. 전체102명 저자 수치와 동일한
cohort/분할이 아니며, 기계 판독 가능한 원 결과도 확보하지 못해 **정확한 논문 성능 재현은 아니다**.

## 입력과 구현

고정39개 raw 파일만 허용하고 S001–S003/held60/Impedance.mat/manifest/oldrunner는 금지한다.
새 start 기록 뒤 `loadmat(variable_names=['data'])`로만 읽는다. Shape `[8,710,2,10,12]`,
real numeric/finite를 검증하고 원 dtype을 기록한 뒤 직접float64로 처리한다.
이전 `_wearable_data`의 float32 중간 변환, 자동 mat73 fallback, 축 추측, 파일 검색은 하지 않는다.
Variable 제한은 MAT header 탐색까지 없다는 뜻이 아니다. 읽기 실패는 명시적으로 남긴다.

Native250Hz, dry0/wet1,8posterior channels,12frequency/phase 순서는 JSON과 기존 검증된
stimulus mapping을 따른다. Interface를 합쳐 학습하지 않는다. 각 N마다 **raw125:160+N**을
native47–53Hz bandstop→5band/detrend/ddof1 처리하고 앞35samples를 버린다.
추가 처리 history는0.14초다. 2초 처리 결과를 짧게 자르지 않으며 query 종료 뒤 sample은 쓰지 않는다.
구간 내 filtfilt와 미검증 upstream downsampling 때문에 완전한 acquisition causality는 주장하지 않는다.

- ETRCA: native ensemble filter의 Q-norm과 선형 weighted Pearson을 그대로 유지한다.
  Ridge, Euclidean 재정규화, temperature, fusion, 추가 centering을 넣지 않는다.
- A0_author: native SCCA_canoncorr의 `update_UV=True,force_output_UV=False,n_component=1`.
  `fit(ref_sig)`만 수행하고 query별 U/V를 저장·재사용하지 않는다. 역시 선형 weighted rho이며
  **기존 우리 FBCCA의 rho² score와 다르다**. Native gen_ref_sin의5harmonics와 phase를 사용한다.
- 가중치는 각 방법의 공개 preset이다. 따라서 A0와 ETRCA가 같은 band weights를 쓴다고
  주장하지 않는다. 이는 native pipeline의 실용 성능 비교이지 calibration만의 단일 인과효과가 아니다.
  두 가지 A0나 새로운 squared-score 후보를 추가하지 않는다.
- 공개 preset은 source39-only 추정값이 아니며 전체 코호트 tuning을 상속할 수 있다.
  Held 파일 미개봉과 완전히 독립적인 설정을 구분한다. 향후 M 비교는 같은 learner의
  weights·학습예산을 공유해야 하며, 독립 확인 설정의 출처는 다시 검토한다.

11개 candidate-score 순환은 score-permutation diagnostic으로만 낸다. ETRCA의 ensemble
label 순환 대응은 fixture에서 확인했으나 실제 human wrong-label refit을 한 것으로 쓰지 않는다.
각 순환의 BA를 평균하며 probability를 합치거나 가장 유리한 순환을 고르지 않는다.

## 평가·중단

기본 행은 participant/stage/view/interface/N/method/k/BA/query수/label수다.39명×52행=2028행이다.
Native all10/last5 각각 A0,ETRCA9,rotated9; chronological4windows×2interfaces 각각
A0,ETRCA3/5,rotated3/5를 보고한다. 원 band correlations도 보존해 별도 구현이 score와
정답 수를 재계산한다. 저장 통계 검산과 raw EEG 전처리 재현을 혼동하지 않는다.

별도 감사에서 percell ETRCA3/5−A0,5−3,LOBO9와의 동일query 비교,각 짧은window−2초,
개인별 harm, 최초 관측0/3/5의80% 도달 또는>5를 낸다. k1과 기존0/1/3 eAUC는 만들지 않는다.
k0는A0 anchor이며 기존k0성공을 새 calibration 이득으로 세지 않는다. 비단조성을 숨기거나
미달자의 비용을 보간하지 않는다. Labels=12k, retainedseconds=12k*N/250,
history=12k*.14초로 분리하고 전체 사용자 wall-clock 절감은 주장하지 않는다.

완료 상태 `COMPATIBILITY_ASSESSMENT_COMPLETE`는 모든 입력·split·finite·score 계산을
완주했다는 뜻일 뿐 efficacy PASS가 아니다. 낮은 정확도 때문에 새 방법/가중치/조건을
자동 추가하지 않는다. Numerical error이면 partial start를 보존하고 인프라/자료 inconclusive로
보고한다. Q>A0 평균 이득을 metadata 연구의 논리적 입장 조건으로 되살리지 않는다.
이 단계 뒤 M 연산자와 controls/cost를 명시한 별도 직접 가설 설계로 돌아간다. Held 자동 실행은 없다.

## 실행·소유권·검증

Base main833df7c029157e514a6534506d2186d62845e522, clean/ahead91.
기존22worktrees 보존. Free약297GiB, 새checkout<=12MiB+출력1GiB+통합temp1GiB 예산으로 충분하다.
기존 외부 Python3.9 환경과 upstream checkout은 read-only이며 dependency/CUDA 변경은 없다.

| Lane | 단독 소유 | 검증/자원/통합 |
| --- | --- | --- |
| Main | JSON·설계·일지·독립audit/tests·통합·실제실행 | 유일SQLite writer, 최종API/산술/전체pytest 확인 |
| Producer | 격리worktree의 run_author_etrca_source.py와 test_author_etrca_source.py | fixture만, raw/M접근금지; commit반환 후main통합 |
| Reviewer | 읽기 전용 | native API·경계·독립eigen/CCA fixture 검토 |

Producer 경로 `/home/whwovy/califreeEEG-wt-author-source`, branch`codex/author-etrca-source39-v1`.
계약 commit 이후 생성한다. CPU4workers/각BLAS1,예상<=2GiB메모리·30분,
output `/home/whwovy/author-etrca-artifacts/source39-v1`의 start.json/correlations.npz/result.json3개다.
새 결과는 기존 디렉터리를 덮어쓰지 않고 source/plan/upstream/hash에 연결해 보존한다.

원본 provenance는 [기존 API 준비 기록](author_etrca_compatibility.md),
[CCA public API](https://github.com/pikipity/SSVEP-Analysis-Toolbox/blob/3344bd199daf78888e364d9db00ae7d8128d2b5f/SSVEPAnalysisToolbox/algorithms/cca.py#L831),
[native preprocessing](https://github.com/pikipity/SSVEP-Analysis-Toolbox/blob/3344bd199daf78888e364d9db00ae7d8128d2b5f/SSVEPAnalysisToolbox/utils/wearablepreprocess.py)를 따른다.
academic-research는 기존landscape와source code 경계를 확인하게 했고, 추가 broad search/PDF는
필요하지 않았다. coordinate-worktree-changes에 따라 독립producer만격리하고main이통합한다.
