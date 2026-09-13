# E126 reference-sensitivity probe — 합성 전용 구현 계약 v1

2026-09-13. 부모 source 계획77857b4는 MAMEM-I 완전 CCA recipe qualification 실패로
닫는다. **이 새 단계는 합성 수학/API 검증만 허용한다. 실제 S001/다른 참가자 EEG,
DIN,NPZ,feature/outcome 읽기 및 실제 예측/새학습0이다.** E126 실제 probe는 미실행이다.

## 기작 가설과 정직한 이름

외부 기록에서 얻은 support-only frequency bank는 nominal bank와 다른 reference
subspace를 정의할 수 있다. 한 EEG 채널에서는 score가 centered signal의 해당
4차원 sine/cos subspace 투영 에너지 비율이므로, 256채널 공간필터의 자유도 없이
reference 변화의 전달성을 점검할 수 있다. 이것은 learned metadata 후보가 아니다.

나중에 사용할 채널 후보는 E126(0-based125),author Default/SampleSelection에서
가져온 결과 독립 선택이다. Oz라는 이름/해부학 위치/최적성은 주장하지 않는다.
우리 operator의 고정값: sr250,N500,2초,h1/h2,무filter·무CAR·무ridge·무학습,
float64,channel별 temporal centering,whole five-class bank,max squaredCCA argmax.
원 저자 PWelch/filter/SVM/5초도 DatasetIII CCA4harmonics/SVM도 재현하지 않는다.

## 순수 API

1. `reference_bank(frequencies)`는 정확히5개 finite Hz를 받아각(500,4) centered
   sine/cos의 SciPy `orth` 기저를 만든다. 시간은arange(1,501)/250,고정h1/h2.
   0<f<fs/4,ascending distinct,5개조건 위반 또는 수치rank4미달이면 STOP.
   eps×max(shape) cutoff는 수치rank이며 EEG통계적 denoising이 아니다.
2. `sample_clock_frequencies(samples_by_class)`는 class order로 넘긴 정확히5개
   support trial의 선택2초창 내 one-based integer event-index 배열을받는다.
   각≥4events,strictly increasing,span<500; 기록 한 주기당2events 가정으로
   f=250*(n−1)/(2*(last−first)). 레이블/EEG/timestamp를 predictor로 넘기지 않는다.
   이 API 자체는 입력이 support라는 provenance를 증명하지 못하므로 실제 reader
   계약이 별도로 필요하다. 반환값은 photon Hz/독립clock/뇌latency가 아니다.
3. `scores(query,bank)`는 단일finite500-vector와(5,500,4)orthonormal/centered bank만
   받는다. Query DIN/label/frequency 인자는 없다. Stable centering 후
   s_j=||U_j^T x_c||²/||x_c||². 분모0/invalidbank/nonfinite이면STOP. Score outside
   [−1e−12,1+1e−12]면STOP,허용 roundoff범위에서만[0,1]clip. Scorer는bank를변경하지않는다.
   `predict`는 전체5scores의argmax,정확한tie는가장작은class index.

## 합성 검증과 예산

Deadline2026-09-13T14:30:00Z,root단독writer. 구현파일1개/새tests1개,외부network0,
추가dependencies0,fit0,GPU0,새raw0. 새합성suite 최대3회,각60wall초/단일BLASthread,
추가 기존 관련회귀suite1회60wall초. Output새pytest cache/bytecode 제외≤1MiB.
합성 실패는수학/API 구현오류만수정가능; 실제자료를보고조건을바꾸지않는다.

- Nominal5bank의각주파수 positive canary:5/5분류,자기score≈1.
- 명시된 합성 shifted5bank[6.49,7.34,8.31,9.60,11.61]에서5/5분류; nominal과
  scores차이 존재. 두 bank의 실제 EEG 우열로해석하지않는다.
- Label-free independent least-squares projection식과score일치≤1e−10.
- 상수offset/비영scale/부호/각harmonic공통phase회전의score불변성≤1e−10.
- 직교화한negative trace는전bank의합집합에직교하므로score≈0. Pure sinusoid
  reference canary는실제 EEG성능의대리변수가아니다.
- Invalidshape/NaN/constant/비정수·비증가event/duplicate frequency/nonorthogonal
  bank가STOP;bank/scorer입력불변;querylabel을받지않는API명시.
- 새MAT/NPZ reader,actual CLI,source cohort 반복 실행 또는M gate가 없는지 독립review.

모두통과하면 `SYNTHETIC_OPERATOR_READY_ONLY`로기록한다. 실패하면합성gate실패,
최대3회소진시STOP. 실제decoder복원/M기여/보정절감/held60효능승격은모두금지다.
후속은별도실제개발계약(정확한S001a/b입력pins,지원5trial 비용,공통2bank,
단회15query/30predictions,누출검사,수치감사,중단기준)을고정할지결정하는것이다.
