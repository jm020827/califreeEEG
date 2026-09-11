# SSVEP의 위상·동역학 관점을 원래 metadata 연구에 어떻게 반영할까

2026-09-11. **목표는 유지한다. 복소 표현·측정모형 관점은 채택하고 Neural ODE는 보류한다.**
이 문서는 개념 검토와 후속 후보의 순서를 정한다. 새 사람 효능 실험·STFT 추출·ODE 학습을
실행한 기록이 아니다. 기존 336/144-fit 부정 결과는 그대로다.

## 1. 쉬운 설명: metadata가 맡을 일

우리는 “어떤 주파수로 깜빡이는지”를 새로 알아내려는 것이 아니다. 자극 후보는 이미 안다.
문제는 다른 사람·측정 조건에서 그 반응이 EEG에 어떻게 나타나는지를 적은 보정으로 추정하는
것이다. 따라서 metadata는 정답을 알려주는 힌트가 아니라 **EEG를 어떻게 측정했는지에 대한
추가 단서**여야 한다. 이 단서가 EEG 자체를 보고 추정하는 것보다 도움이 되는지가 질문이다.

예를 들어, 실제 화면 출력이 기록 marker보다 늦었다는 독립 측정이 있으면 반응의 시간 기준을
맞추는 데 쓸 수 있다. 머리 움직임이 측정됐다면 보정 trial의 관측 신뢰도를 예측하는 가설을
세울 수 있다. 그러나 gyro가 신경 반응 지연이나 진동 주파수를 직접 알려준다고 가정하지 않는다.
현재 확보된 gyro M2는 photodiode·화면 지연 측정이 아니다.

| 정보 | 연구에서의 위치 | 공정 비교 |
|---|---|---|
| 모든 후보의 자극 주파수·고조파·알려진 phase codebook | 공통 정보 | 모든 arm에 동일 제공 |
| EEG로 추정한 SNR·위상·coherence·지연 | EEG-derived Q | 강한 EEG-only 대조군에도 제공 |
| 독립 측정한 화면/trigger 지연·센서 상태·support IMU | 외부 M 후보 | 실제 가용성·정확도·획득 시점 확인 후 QM에 추가 |

화면 지연이 전 실험에서 동일한 상수라면 그 보정은 공통 전처리다. 새로운 M 학습 효과로
세지 않는다. M가 recording/device ID만 대리하면 조건을 바꾼 검증 없이는 일반화 주장도 못 한다.

## 2. 인용문에서 가져올 부분과 고칠 부분

**가져올 부분:** 주파수뿐 아니라 phase·공간 패턴·고조파 구조를 고려하고, calibration을
개인/측정 조건의 반응 모양을 추정하는 과정으로 보는 관점이다. SSVEPformer의 공개 preprint
§2.2 식1–4는 실수부/허수부 연결을 사용한다. §2.6은 LOSO, §4.2는 target 한 block을 포함한
추가 학습을 다룬다. 한 block은 전체 trial 한 개가 아니다. 이는 표현·적응의 선행근거이지
외부 metadata 효과를 평가한 실험은 아니다. [SSVEPformer v1 원문](https://arxiv.org/html/2210.04172v1)

고쳐야 할 설명은 다음과 같다.

1. **완전한 complex FFT는 시간 정보를 본질적으로 버리지 않는다.** 역변환할 수 있다.
   Magnitude-only, 일부 bin 선택, pooling, 비가역 encoder가 정보 손실의 위치다. STFT의
   장점은 국소 시간 변화를 표현하기 편하다는 것이지 없어진 정보를 생성하는 것이 아니다.
   [NumPy DFT 정·역변환 정의](https://numpy.org/doc/stable/reference/routines.fft.html)
2. **1/T를 절대적인 분류 불가능 경계로 쓰지 않는다.** 통상 DFT 격자/분해능 척도이며,
   알려진 후보에서 직접 sinusoid 적합을 할 수 있다. 짧은 구간·잡음 때문에 어려운 것은
   사실이지만 1초이므로 0.2 Hz 간격은 무조건 불가능하다는 뜻은 아니다. Zero-padding은
   관측 정보를 늘리지 않는다. [후보 주파수별 sinusoid 적합의 공식 정의](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html)
3. **ODE의 t가 뇌의 실제 시간이라는 보장은 없다.** 원 Neural ODE §1·§3은 연속 network
   depth를, §5는 관측 timestamp를 갖는 별도 latent time-series 모형을 설명한다. 후자를
   연구하려면 초 단위 시간·관측식·window/hop을 정의해야 한다. 일반 ODE network의 latent
   계수에 생리적 의미를 바로 붙이지 않는다. [Neural ODE v5 §1·3·5](https://arxiv.org/html/1806.07366v5)
4. **Phase-locked는 모든 trial/person의 phase가 완전히 같다는 뜻이 아니다.** 일정한 입력
   지연과 변동 jitter도 구분해야 한다. 측정 phase를 곧바로 neural latency라고 부르지 않는다.
   Decoder weight를 cortical 활성지도로 읽거나 고조파만으로 특정 생성 기작을 확증하지 않는다.

Norcia review·원 TRCA의 제공 PMC 링크는 이번에 CAPTCHA로 본문 접근 실패했다. 우회하지
않았고 그 두 원문을 이번에 정독했다고 표시하지 않는다. 위 신경생리 전반을 새로 검증한
체계적 review는 아니다. SSVEPformer 최종 저널판과 preprint의 일치도 이번에 확인하지 않았다.

## 3. 현재 구현에서 특히 중요한 위상 불변성

현재 classifier는 단일 FFT peak detector가 아니다. 9채널·후보 주파수 5.45/8.57/12 Hz와
3개 고조파의 sin/cos reference, support ridge 필터, Gram 보정 projection energy를 쓴다.
따라서 “고조파·공간 정보를 처음 도입한다”는 식의 신규성은 틀리다.

Reference를 Y, 가역적인 상수 위상 회전을 R이라 하면 Y'=RY이다. Reference 부분공간의
projection P=Yᵀ(YYᵀ)⁻¹Y는 Y'를 써도 동일하다. 이는 수학적 도출이며, **sin/cos 쌍 전체를
사용하는 reference 원점 회전**에 관한 주장이다. 실제 EEG 구간 이동, transient, drift,
시간에 따라 바뀌는 jitter까지 무관하다는 주장은 아니다.

현재 W에도 W→WR의 오른쪽 직교 회전을 적용하면 공간 metric WWᵀ가 같다. 그러므로
그대로인 energy scorer 앞에서 위상만 회전시키는 M 경로는 최종 점수를 바꾸지 못할 수 있다.
학습 전에 이 대칭성과 M-only score 개입을 검사해야 한다.
[현재 scorer](../src/cfeg/analysis/source_expert_borrowing.py),
[기본 projection](../src/cfeg/analysis/mobilebci_reference_ridge.py).

현재 cache는 1초 구간의 covariance/cross-products와 요약량이다. Cross-products에는
reference-relative 정보가 있지만 일반적인 sliding amplitude/phase trajectory를 재구성할
수는 없다. STFT 동역학을 실제로 시험하려면 별도 raw 재추출 설계가 필요하다. 현재 실행은0이다.

## 4. 원 목표에 맞는 구체적 후보: 측정조건을 반영한 작은 적응모형

개념적인 모델은 다음과 같다. 이는 **아직 검증하지 않은 설계 가설**이다.

    공통 stimulus bank → source에서 학습한 harmonic/공간 반응 모형
                                 ↓
    EEG-derived Q + 외부 M → 작은 정렬·관측 신뢰도·적응 prior → 후보별 EEG 적합도

처음부터 M를 모든 neural weight에 넣지 않는다. 실제 측정과 관계를 설명할 수 있는 작은
경로를 하나 고른다. 현재 gyro라면 support 관측의 신뢰도/공간 잡음 또는 source 적응 prior가
후보이지 “gyro로 cortical damping을 정한다”가 아니다. 실제 timing M가 없다면 timing 후보는
대기/종료하며 private 자료 요청으로 돌아가지 않는다.

만약 위상을 다룬다면, 단순 phase-invariant energy와 구별되는 **stimulus-relative complex
template/저차원 phase-aware adaptation**부터 검토한다. 파형 전체를 사람 간 그대로 평균해
phase 차이로 상쇄시키지 않는다. EEG-only가 support에서 정렬하는 강한 대조군을 반드시 둔다.
측정 지연을 정렬한 후 남은 개인 phase를 support로 추정할 수는 있으나 둘을 같은 latency로
해석하지 않는다.

후속 순서는 다음으로 고정한다.

1. 이번 known-optimum 수치 진단: 완료. Scale/최적화 실패를 실제 M 무용론과 구분한다.
2. **다음 우선 후보는 작고 block-scaled인 source router/적응 경로의 통합 검증.** 인공 정보
   양성·M 독립 음성·source 순서·대조군·loss/최종 score 작동을 새 유한 계약으로 검사한다.
   이번 단순 평균화 결과만으로 사람 router를 바꾸어 즉시 재학습하지 않는다.
3. Phase-aware 관측모형은 두 번째 설계 후보로 둔다. 측정 가능한 M와 의미 있는 score 경로가
   먼저이고, 필요한 시간정보가 현재 cache에 있는지 먼저 판단한다. 미측정 M를 만들어 넣지 않는다.
4. Neural ODE는 단순 complex/harmonic 모델 또는 discrete state-space보다 필요한 이유가
   생길 때만 후보로 올린다. 새 raw 추출/훈련/solver 예산과 별도 대조군 없이는 실행하지 않는다.

## 5. 어느 모델을 쓰든 바뀌지 않는 판정

- 동일 representation·source data·support prefix·query·학습 예산으로 Q/Q2/QM/SHAM 비교.
  더 큰 network/고조파 추가 효과와 M 효과를 분리한다. Source-only 선택과 participant 분리 유지.
- 추론 때 정답 frequency 하나만 forcing으로 주지 않는다. **모든 후보 자극을 동등하게
  평가**하고 관측 EEG의 적합도로 고른다. 알려진 전체 stimulus bank는 공통 정보다.
- Reconstruction/harmonic/consistency loss에도 query 정답을 쓰지 않는다. Target 적응은
  비용을 지불한 support만; support-only M라는 현 설계를 query IMU 사용으로 확장하지 않는다.
- 빠른 solver·짧은 EEG window·적은 calibration trial은 서로 다른 결과다. 실제 보정 절감은
  같은 정확도 목표까지 필요한 획득 prefix trial로 측정한다. Window 겹침을 독립 표본으로 세지 않는다.
- M가 latent/logit을 바꾸는 것과 유효한 최종 확률/판정 변화는 구분한다. 최종 score가 불변이면
  사람 효능 실험 전에 멈춘다. 유망 후보도 exposed 16명 결과만으로 독립 검증 완료가 아니다.

## 근거 관리

`academic-research`의 기존 workspace와 세 읽기 전용 검토를 사용했다. Exact primary HTML
targeted review이며 broad literature update나 PDF 완독이 아니다. 두 HTML 원문을 hash와
함께 저장하고 fulltext claim evidence로 등록한다. Workspace의 paper-card fulltext는 PDF
provenance만 허용하므로 기존 abstract card를 허위 PDF 정독으로 승격하지 않는다.
전체 landscape cutoff 2026-09-04 유지; 이번 확인일만 2026-09-11이다.
