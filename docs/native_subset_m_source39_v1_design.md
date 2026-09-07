# Native subset M source39 v1 — 직접 임피던스 증분 검정

2026-09-07, 사용자 '시작' 승인. 권위는 [고정 JSON](../configs/analysis/native_subset_m_source39_v1.json)이다.
Plan SHA ee9b758f755a18f7540c978e9faf18a5711e652948b94df595377bbfb83cb80a.
기존 [native 호환성 확인](author_etrca_source39_v1_results.md) 뒤 한 개의 M 연산자를 시험한다.
새 decoder 계열 탐색, 기존 후보 재개, 보존 평가60명 개봉은 하지 않는다.

## 쉽게 말하면

같은 사람의 보정 예시3개 또는5개를 받았다고 하자. 모두 쓰는 eTRCA와 한 블록씩 뺀 eTRCA를
미리 만들어 공통 후보로 둔다. **Q는 EEG와 알려진 측정조건으로, QM은 거기에 임피던스를
더해 '이번 예측에서 어떤 예시를 빼는 것이 나은가'를 판단**한다. 예상 개선이 없으면 모두 쓴다.
M이 모든 class 점수에 같은 수를 곱하는 무효과 연산이 되지 않도록 실제 후보 선택에 연결한다.

각 후보의 eTRCA 학습/예측 수식은 원본 그대로지만 최종 선택기는 새로운 adaptation wrapper다.
공통 후보는 full,drop0,...,drop(k−1); k3에서4개, k5에서6개다. 블록의12class를 함께 제외한다.
라벨을 이미 얻었으므로 빼더라도 비용은36/60labels다. '모델이 덜 사용'과 '사용자가 덜 보정'은 다르다.

## 정확히 무엇을 학습하나

개발39명을 ID목록 순서 modulo3로 나눠 다른26명에서 학습하고13명에서 평가한다.
Native250Hz/8channels/5bands/dry-wet presets,4windows0.5/.752/1/2초,앞3/5block과뒤5query를 유지한다.
어떤 유리한 cell도 새 primary로 고르지 않는다. Source39는 여전히 반복 노출된 개발자료다.

목표값은 각 omission 후보가 full보다 맞혔는지 여부의 차이(-1/0/1)다. Query 정답은 fit26명의
지도학습 목표로만 쓰며 평가13명의 fit/정규화에는 들어가지 않는다. 학습 손실은 participant×cell×k
동등 가중 MSE +0.1×slope L2다. 각 단위 안에서60queries와k개 omission을 동등 가중한다.
하이퍼파라미터 검색 없이 이 ridge 한 개만 쓴다. 예상 gain을10소수자리로 반올림하고 양수인
최상 후보 하나를 선택한다. Full의 gain은0이며 동률은full,그다음낮은block 순서다.

Q의33개 입력은 공통 protocol/order8개, full/A0 confidence6개, omission confidence/차이7개,
support label-aligned reference-quality4개, query-support logcov-distance3개,
label-aligned support consistency2개, 사전지정 interaction3개다.
QM은 channelwise impedance mismatch/상대값/query값, 유효비율, 세 요약량과 Q×M6개로36입력을 더한다.
모든 변환식·순서·결측 처리·scaling은 JSON에 있다. 낮은 impedance가 항상 좋다는 단조 가정은 없다.

입력69slots+intercept로 저장 구조와ridge/학습 질량은 같다. **Q의 실효 차원은33으로 더 낮다**.
이를 동일한 실효 용량이라고 부르지 않고69차원 support-shuffled M 재학습 대조를 별도로 둔다.
QM의Q계수도 함께 학습하므로 missing M은 단순 입력0이 아니라 기존 fittedQ의 선택으로 정확히 돌아간다.

## 무엇과 비교하나

| 방법 | 정보·역할 |
| --- | --- |
| A0_author | 같은 길이의 native 무보정 기준 |
| FULL | 모든prefix의 native eTRCA, 선택기 없이 |
| Q | EEG + 공통 interface/window/k/시간순서/headband order/period |
| QM | Q + 실제 사전 impedance |
| SHAM_REFIT | Q와query M·결측패턴은 유지, support 수치의 대응을 섞은69차원 재학습 |
| M_SHUFFLE | 올바르게 학습한QM에서 평가 support 대응만 섞음 |
| M_STALE | QM의 query M을 마지막support 측정값으로 대체; 원query결측마스크 유지 |
| M_MISSING | exact fittedQ fallback |

Shuffle은 같은participant/interface의현재prefix 안에서 동일8channel 결측패턴 그룹별로
cyclic permutation을 만들고 Cartesian product 중 전역identity를 뺀다. Train sham은 고정SHA로
그중 하나를 택한다. 평가shuffle은 모든 유효map을 각각 예측한 뒤 metric을 평균한다.
섞을 수 없거나 값이 같아 변화가 없으면 이를 보고하고39명 primary에서 사후 제외하지 않는다.
이 대조는 **query M·mask·nuisance를 조건으로 numeric support-M 대응**을 검사한다.
전체M 무작위화나 순수용량 검정이 아니며 정렬된support효과는QM>Q만으로 주장하지 않는다.
공통Q에순서정보가이미있으므로별도 Q_no_order 실험은 추가하지 않는다.

## 데이터·비용·판단 경계

이전에 검증한 source39-only projection780packets를 새로운 명시적 권한으로 재사용한다.
Raw Impedance.mat나102명 manifest를 다시 열지 않고 native환경에Pandas도 설치하지 않는다.
원 projection은float32 impedance변환을상속하므로 원정밀도 재현이라고 하지 않는다.
Zero impedance는valid, null/NaN은missing, negative/infinite는오류다. EEG raw는직접float64다.
Nativefull/A0 bandcorrelation은직전source39 cache와1e−12 이내 및exactargmax로검증한다.
이는새pairedM나새독립확인을얻은것이아니다. 공개preset의whole-cohort tuning 상속도그대로다.

Primary는8조건동등평균 QM3−Q3, 참고실용크기1pp와기술적95%CI다. Q>A0 평균성공을
입장권으로요구하지않는다. 동시에A0/full/Q5와실용성,개인harm,control,실제예측변화를보고한다.
80%최초관측0/3/5,같은window의QM3−Q5와비열등참고폭1/60,36→60labels를분리한다.
미달은관측grid상미달이지미측정1/2/4까지배제한최소k의하한이아니다.
RetainedEEG=12k*N/250초,history=12k*.14초. 측정·휴식·준비시간미확보로사용자wallclock절감은미입증이다.
4680rows/120summary를전부보고하고효능과완료상태를분리한다. 자동READY/heldunlock은없다.
이번M방식1개+원인이특정된수정최대1회뒤프로그램점검예산을유지한다.

## 소유권·검증·문헌

Base main108379392820226865e4948b09ade297fe896ec2 clean/ahead96,23기존worktrees보존.
가용약294GiB,추가checkout<=12MiB/cache1GiB/temp2GiB,CPU4/BLAS1/예상memory2GiB/runtime30분.

| Lane | 단독 소유 | 통합·검증 |
| --- | --- | --- |
| Main | 계약/설계, scripts/native_subset_m_core.py와tests, 독립auditor,문서/SQLite/실행 | cleanmain에서단회; 모든freeze를평가전publish |
| Producer | 새격리worktree scripts/run_native_subset_m_source.py, tests/test_native_subset_m_source.py | synthetic만; main환경pytest/외부nativefixture; commit반환 |
| Reviewer | main/upstream읽기전용 | native pool·Q/features·회귀·누출·control/결과검토 |

Producer worktree /home/whwovy/califreeEEG-wt-subset-m, branch codex/native-subset-m-source39-v1.
Main 공유core API: projection_arrays(payload,plan)->(z[39,2,10,8],order[39]);
fit_all(cache,projection,plan)->freezeJSON; evaluate_all(cache,projection,freeze,plan)->rows/summary/진단.
Producer는검증된원projection을projection key에담은새source-projection.json을publish한다.
순서:start→projection/raw/caches→features→fitall→fold-freezes→outerevaluate→result.
Output /home/whwovy/native-subset-m-artifacts/source39-v1의5파일은새경로/0400/exclusive/해시결속이다.
기존runner/seal/결과/environment를수정하거나실패attempt를덮어쓰지않는다.

Read-only independent native fixture: k2/4 포함80fits,4.195초CPU1/BLAS1, projectedPearson max5.55e−16,
모든drop score변화/17of768prediction변화. Human자료는열지않은합성호환성이다.
SciPy import의일시적temp만 /tmp/cfeg-selector-fixture.8NQxl6에서허용했고현재빈경로다.

`academic-research`로기존landscape/frontier를재사용하고exact-title foundation search
search:32e455897dd63526(Crossref/OpenAlex,cutoff2026-09-07,5records,noerrors)을추가했다.
[Jacobs et al.1991 공식abstract](https://direct.mit.edu/neco/article/3/1/79/5560/Adaptive-Mixtures-of-Local-Experts)는
조건부expert선택의선례이지EEG impedance효과의증거가아니다. 직접페이지재오픈403은우회하지않았다.
이번pass는abstract/구현검토이며새PDF정독은없다. Histogramboosting 문서도검토했지만고정ridge를선택해
그추가모델계열/튜닝은실행하지않는다. 핵심신규성은gating자체가아니라추가acquisition-M의반증가능한검정이다.
