# MAMEM reference 잔차 v1: 단일 source DIN 구조 관문

2026-09-13, base main5d1272f. **새 raw 읽기 전에 고정하는 별도 계약**.
원래 저보정 SSVEP에서 외부 acquisition M을 학습에 활용하여 Q·공통정보를 넘는
개선/보정절감을 검정하는 목표는 유지한다. v2 M2→PSD RETIRE 및 S001 reference
12/15→14/15의 개발관찰을 보존한다. 이번은 그 효능 실험도 재실행도 아니다.

Problem signature: class별 2초 DIN event sample 간격; 병목은 공통 설정과 전달 가능한
개인/기록 잔차의 구분; 허용은 기존 공개 source10명 a/b의 DIN-only 구조 분석;
목적은 **full classwise residual A→B 전달 경로 하나**의 eligibility; feedback은
반복변동/공통 대비 event-frequency 예측오차; 실패모드는 common 상수를 M 학습으로
포장하거나 B event를 EEG predictor에 제공하는 것이다. 실제광학Hz·clock 원인은 미확인.

## 자료·접근·예산

- 정확한20파일: S002–S011 × a/b, 기존
  `/home/whwovy/data/mamem_recorded_event_source_v2/`의 MAT.
  경로·member·bytes·SHA256·input receipt SHA는
  `configs/governance/mamem_reference_residual_v1_inputs.json`에 고정한다.
  총2,760,155,793bytes. 새다운로드/추출 없음. S001/c/d/e/held60는 접근하지 않는다.
- 모든 파일은 participant role/source 지정 그대로다. 같은10명의 재사용은 **개발 탐색**,
  신규 독립 참가자 검증이 아니다. a→b는 지정 cross-run 역할이며 실제 시간순서 및
  a의 clock이 b의 현재 acquisition metadata라는 주장은 하지 않는다.
- 각 파일 **hash 전체byte읽기1, whosmat1, loadmat1**. loadmat whitelist는
  DIN_1/samplingRate만. eeg는 header shape257×T,double,0<T≤200000만 확인한다.
  DIN header4×E,cell,0<E≤10000; samplingRate numeric scalar250; descriptor행1/3
  수치접근0. 원EEG decode/수치접근0; 기존features.npz/trial결과 읽기0.
- 기존 eventsv1/v2 및 sample_clock_frequencies 코드 SHA를 보존한다.
  v2의23group/8adaptation제외/15main/3perclass-contiguous/one-based→zero-based
  bounds 검사 뒤 include_metadata=False. M2 extractor0.
  모든main trial의 [onset0+250,onset0+750) event samples에서
  `f=250*(n−1)/(2*(last−first))`, 2events/cycle 가정. class별시간순repeat0/1/2.
  매repeat 5class bank는 기존pureAPI의0<f<62.5Hz·class순상승검사를그대로사용한다.
- A10파일을 먼저 읽어 전체150창/주파수/역할을 `a_frozen.json`으로 저장한 뒤 B10파일.
  B도150창 event-frequency를 **diagnostic target**으로 계산하므로 B추출0이라고
  기록하지 않는다. B predictor사용/EEG classification0. 읽기 완료 파일을 각각 별도
  exclusive record로 저장해 실패 시 partial provenance 보존; 재독해/교체/skip0.
- 실제1attempt, source20파일/300eventwindows, parameterfits0, EEGpredictions0.
  deadline2026-09-13T15:30:00Z, parentwall180s, workerCPU120s/AS2GiB,
  perfile1MiB/총run8MiB, reserve8GiB, BLAS/OMP1thread. GPU/네트워크/유료0.
  일반artifact는stdout/stderr각1MiB+terminal1MiB총3MiB를미리예약한다.
  manifest/start/workerclaim/terminal/pins/exclusivecreate; 실패시STOPPED_NO_RETRY.
- 실행전 합성테스트 최대3suite, suite당60wall초; 최종 관련회귀1회/60초.
  실제 저장수치 독립 검사1회/60초, raw再읽기/fit0. 검사 script도 실제 전에 동결.
  기술실패도 보존하며 실제 source 결과후 코드/threshold 변경·재실행0.

## 수학과 고정 기준

F[p,r,c,j]는 Hz, shape10×2×5×3; L=log(F). r=A/B, j=시간순repeat.
각 held-out p에 대해 **다른9명의 모든3repeat만** 사용하여 별도로
`muA[-p,c]=mean(L[others,A,c,:])`, `muB[-p,c]=mean(L[others,B,c,:])`.
Held-out의 A/B 어느 쪽도 common 값에 들어가지 않는다. B-source의 공통 설정도 허용하는
강한 source-only baseline이다. 두 run의 공통 역할 차이는 M의 개인 기여로 세지 않는다.

- A지원값 `x[p,c]=L[p,A,c,0]−muA[-p,c]` (class당1trial=총5trial).
- B평가값 `target[p,c]=exp(mean(L[p,B,c,:]))` (3repeat 기하평균).
- 공통예측 `common[p,c]=exp(muB[-p,c])`.
- full잔차예측 `direct[p,c]=exp(muB[-p,c]+x[p,c])`.
- 사람별5class 평균 squared-Hz error, 전체는10명 동등가중 평균.
  `gain_fraction=1−MSE_direct/MSE_common`; common MSE≤1e−24이면0으로정의하고
  `common_error_floor`로미통과. 사람별 strict개선은commonMSE−directMSE>1e−12.
  A/B 잔차 correlation은 단독 통과 근거로 사용하지 않는다.

반복변동과 between규모는 각run별 **Hz 산술평균** 기반으로 별도 계산한다:
`W=mean_p,c(var_j(F,ddof=1))`,
`V=mean_c(var_p(mean_j(F),ddof=1))`, `noise_of_mean=W/3`,
`excess_rms=sqrt(max(V−W/3,0))`.
두run 모두 V≥2*(W/3), excess_rms≥0.025Hz일 때만 size/repeat 관문통과.
W는 quantization/추정오차뿐 아니라 실제 trial변화도 포함하므로 순수 측정오차나
안정적 생리특성 분해라고 주장하지 않는다. iid decomposition은 작업근사다.

**주판정: size/repeat 두run통과 AND pooledMSE 10%이상감소 AND 7/10명 strict개선**.
통과=`ELIGIBLE_CLASSWISE_DIRECT_TRANSFER_ONLY`, 그외=`RETIRE_FULL_RESIDUAL_ROUTE`.
0.025Hz/2배/10%/7명은 데이터를 보기 전에 고른 탐색용 운영 기준이며 문헌에서 정한
유의수준·검정력·통계적확증이 아니다. 통과도 학습M/Q이상/보정절감 증거가 아니다.

이 gate는 full잔차 전달에 보수적이다. noisy M을 **부분 shrinkage**하면 유용할 수 있어도
full전달은 실패할 수 있다. 따라서 미통과를 모든 reference 학습/metadata무효의 증명으로
확대하지 않는다. 이번 예산에서 실패를 보고 partial계수/grid/same-run으로 구제하지 않는다.

고정nominal값(6.66/7.5/8.57/10/12)과공통bank를보고서에서나란히표시한다.
별도nominal-delta필드를감사했다고주장하지않는다.
class별 common/direct오차 및 A/B잔차는 모두 공개한다.
전5class x의 all-ones 방향 투영 에너지비
`sum_p(5*mean_c(x)^2)/sum_p,c(x^2)`와 직교 잔차 RMS를 **기술통계만** 보고한다.
분모≤1e−24면zero_residual_energy로표시하고factor비율0;수치roundoff를방향성으로세지않는다.
PCA 최선축을 clock이라고 부르지 않으며 factor값으로 별도후보 승격0. 총1trial 전파는
검정하지 않는다. B평가3repeat를 본 뒤 support 수/기준/bank정의 교체0.
공통 source를 공유하는10개fold도 통계적으로 완전히 독립적인 것은 아니다.

## 비용·출력·후속

A class당첫trial5개: selectedstimulus25s, analyzedDIN10s, 각p의
selected_elapsed_prefix=max(trial_end0)/250 및 fullA_record=T/250을 별도로 저장한다.
전체DIN 사후적격성 검사가 필요하므로 query_ready/setup UNKNOWN.
공통bank는새target의support0이며 이번diagnostic에든자료비용을 새target시간과혼동하지않는다.
이 구조통계만으로 분류정확도, 온라인실용성, 보정절감은 계산하지 않는다.

출력dir=`docs/reports/mamem_reference_residual_v1_run`:
manifest(코드/계약/inputspec SHA), started, worker_claim, record_S00xr 각각20개,
a_frozen(A10내용SHA/150창/시간), result(300F,10common/directtarget/cost,수식/판정),
terminal(완료/실패·pins·시작/끝·artifact SHA), bounded stdout/stderr.
Audit는 raw자료/producer import없이 저장event samples→300F→LOPO/moments/errors/
고정판정/cost·코드/입력핀·Aseal→B시간순서·terminal/result 연결을 별도 재계산한다.
원 DIN→event분할 재구축/독립광학측정/OS I/O trace가 아님을 명시한다.

Root solewriter(main): contract/inputspec/puremodule/runner/tests/audit/reports/SQLite.
검토3agents sharedrepo **read-only**, raw/fit/write0. 알고리즘·경계 의존성이 있어
구현lane은순차이며 새worktree0, 기존46개/8untracked보존. free293GiB,
새artifact/testworkspace<64MiB,환경공유read-only/noinstall/nosymlink/nocleanup.

통과하면 최대1개 저용량학습 후보의 새계약을 설계한다. 공통k0/nominalk0/directM 및
강한 Q/Q2/QM/SHAM, 동일지원/동일query/분할/비용을 유지하고 별도fit예산을 고정한다.
실패하면 여기서 해당full잔차 경로를종료하고 원목표하의 다음 정보원/경로를 근거로
다시제안한다. 문헌검색은 현재 필요 없으며 기존연구workspace를 참조한다.
held60/사람에게자료요청/유료는 계속별도승인. 전체연구goal을 이 구조gate로완료하지 않는다.
