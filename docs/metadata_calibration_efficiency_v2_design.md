# Metadata-assisted low-calibration SSVEP — V2 frozen design

상태: **human 방법 r6 보존 / synthetic governance V11 단회 시도 인프라 terminal·결론 불가 /
외부·held 60명 접근 금지**

Human 방법 revision r3–r6는 어떤 V2 EEG outcome도 계산하기 전에 만든 재현성 수정이다.
r3는 `effective support mass`, r4는 P1/P2 실행 수식과 fail-closed provenance, r5는
k=3/5의 중간 prefix depth 2/4 전용 prequential API를 명시했다. r6는 NumPy 1.26과
2.2 사이의 다섯 번째 filter-bank weight 1 ULP 차이를 없애기 위해 7개 weight를
명시했다. 후보·cohort·endpoint·threshold·held 경계는 바꾸지 않았다.

Synthetic DGP와 실행 governance는 별도 revision이다. v4–v6는 spatial/phase/gain/context
축, block 단위 impedance, 정확한 harmonic phase와 support-prefix 내부에서만 가능한
shuffled/stale control을 고정했다. v7은 public primitive의 reserved seed 접근을 막고
결과 경로와 독립적인 canonical seed-global claim을 도입했다. 공개 평문 seed
`20260907`은 미실행 상태로 폐기했다.

V8–V10은 모두 해당 lockbox 정보 전에 폐기됐다. V8은 잘못된 test interpreter receipt,
V9는 claim/receipt transaction 감사 문제, V10은 preparation 뒤 development replay 중
terminal precedence 문제를 발견했다. V10에는 development receipt·authorization·beacon·
claim·outcome이 없었다. V11은 별도 error-terminal이 scientific result보다 우선하도록
고친 실제 freeze다. 커밋은 `832e579c0dec8774ffe8c3ac99abf110a74aa9e3`, 태그는
`metadata-calibration-efficiency-v2-synthetic-freeze-20260906-r11`, plan SHA-256은
`9a9683141ccd3d1fe9cf094aad955c6e317d112a73e35fbcfcdd2ef7d06e1f8c`다.

V11은 2026-09-05 21:30 UTC NIST pulse를 사전 고정했고 21:30:36 UTC에 정확히 한 번
호출됐다. Global claim과 정확한 pulse receipt는 만들어졌지만, 그 뒤 authorization을
재검증하면서 development evidence의 reserved-seed 검사가 다시 full beacon validator로
들어가는 순환 때문에 `RecursionError`가 발생했다. Scientific metric/result는 하나도
생성되지 않았고 terminal은 `consumed_inconclusive_infrastructure_error`다. 계약상 이
lockbox는 소비됐으며 재시도하지 않는다. 이는 현 방법의 음성 효능 결과도, human
metadata의 무용성 증거도 아니다. 상세 포렌식과 후속 결정은
[V2 synthetic 결과](metadata_calibration_efficiency_v2_results.md)에 기록한다.

사전 계획의 hash 의미를 보존하기 위해 frozen master YAML과 synthetic contract는 사후
상태로 다시 쓰지 않는다. Terminal 분류·artifact digest·수정 후 검증은 별도
`configs/governance/metadata_calibration_v2_synthetic_v11_terminal_audit.json`이 기록한다.

기준 설정은 `configs/analysis/metadata_calibration_efficiency_v2.yaml`이다. 이
문서는 수식을 쉽게 설명하고, 왜 V1을 수선하지 않고 별도 V2 후보로 만드는지,
어떤 순서로 실험을 끝낼지를 고정한다.

## 1. 연구목표는 바뀌지 않는다

질문은 여전히 이것이다.

> 새 사용자가 SSVEP를 쓰기 위해 정답이 붙은 EEG를 아주 조금만 제공할 때,
> EEG 신호만으로 알 수 있는 것 외에 측정 전에 알 수 있는 장비·전극·접촉 상태가
> 보정 부담을 실제로 줄이는가?

`few/one-shot SSVEP` 자체가 신규성은 아니다. V2의 조건부 기여는 강한
calibration-free 기준선과 강한 signal-only few-shot 경로를 둔 뒤에도 **외부
acquisition context가 추가로 유효한지**를 반증 가능하게 검정하는 것이다.

## 2. V1에서 무엇을 배웠나

V1 source 39명 결과는 후보 선택용 데이터로 다시 쓰지 않는다. 다만 실패 요구사항은
설계에 반영한다.

- V1 `A_Q`는 k=1에서 39명 중 35명을 악화시켰다. 한 support trial이 약한 learned
  anchor를 너무 크게 움직였다.
- `A_QM-A_Q` 평균 eAUC는 음수였고 correct metadata와 shuffle이 구분되지 않았다.
- M-off 후보가 없어서 metadata branch가 해로워도 학습된 epoch를 골라야 했다.
- strict FBCCA가 V1 neural anchor보다 훨씬 강했다.

따라서 V2는 V1 가중치나 stage-2를 이어 학습하지 않는다. strict FBCCA를 절대
기준으로 두고 support와 M은 그 예측에 작은 residual만 줄 수 있다.

## 3. 세 방법을 일상어로 설명하면

### A0 — 아무 보정도 하지 않는 강한 기준선

현재 EEG를 모든 12개(외부 자료에서는 40개) 후보 주파수의 sine/cosine reference와
비교해 strict FBCCA 점수를 만든다. 정답 주파수 하나만 넣지 않고 전체 codebook을
모든 query에 동일하게 쓴다.

### AQ — 소수 calibration EEG가 알려 주는 “오류 지문”

각 calibration trial도 FBCCA로 통과시키면, 예를 들어 실제 10 Hz인데 10.2 Hz에도
높은 점수를 주는 식의 사용자별 혼동 패턴이 생긴다. V2 primary 후보는 클래스마다
이 점수 분포를 작은 prototype으로 저장하고 새 query의 점수 분포가 어느 prototype과
가까운지 본다.

구체적으로 class 수를 `C`, smoothing을 `s=0.05`, pseudocount를 `nu`, support
affinity를 `a_qi`라 하면 ideal prototype은

`u_cj = s/C + (1-s) * 1[j=c]`

이고 실제 prototype은

`r_qc = (nu*u_c + sum_{i:y_i=c} a_qi*p_i) / (nu + sum_{i:y_i=c} a_qi)`

이다. query와 각 `r_qc` 사이의 negative Jensen–Shannon divergence를 class score로
만들고, 같은 overflow-safe row-center/RMS/softmax 변환으로 `p_support`를 얻는다.

그러나 이 prototype으로 FBCCA를 대체하지 않는다. 최종 확률은

`p_AQ = (1-lambda) p_FBCCA + lambda p_support`

이고 `lambda`는 최대 0.10/0.20/0.30 중 development에서 하나만 고른다. support가
적을수록, 그리고 FBCCA 자체가 확신할수록(`normalized entropy`가 낮을수록) lambda가
더 작다. 명시적인 `lambda=0`
후보가 있으며 gate가 실패하면 산술 혼합도 하지 않고 원래 FBCCA 객체를 그대로
반환한다.

비교할 하나의 secondary 후보는 waveform target-template residual이다. V1의 standalone
template 성능이 약했으므로 이것도 작은 convex residual로만 허용한다. BETA/Dong
development에서 두 operator와 12개 고정 조합을 비교한 뒤 하나만 source gate로 보낸다.
P2는 anchor와 동일한 7개 filter band 및 정확한 subband weight를 쓴다. 각 band에서
support를 affinity-weighted class template로 평균하고, query와 template의 channel×time을
펴서 Pearson `rho`를 계산한 뒤 `sum_b w_b*sign(rho_b)*rho_b^2`를 class score로 쓴다.
사전계산 score를 사용할 때는 producer schema와 preprocessing, filterbank,
query/support partition SHA-256을 모두 receipt로 묶어야 한다.

### AQM — “이 calibration이 지금 query와 같은 조건에서 얻어졌나?”

AQM은 별도 분류기를 만들지 않는다. query block과 calibration block의 interface와
채널별 impedance가 얼마나 비슷한지 계산하고, 비슷한 support는 그대로, 멀리 떨어진
support는 덜 믿는다. impedance는 `log2(1+kOhm)` 공간의 median 차이로 비교한다.
두 배 정도 차이나면 affinity가 절반이 되는 고정 함수다.

AQ의 query별 혼합량을

`lambda_AQ = lambda_max * k/(k+1) * normalized_entropy(p_FBCCA)`

라고 하면, AQM은 클래스별 평균 query-support affinity를 다시 동일 가중 평균한
`m_context`를 사용해 `lambda_AQM = lambda_AQ * m_context`로 둔다. `m_context=1`이면
AQ와 완전히 같고, 측정 조건이 멀수록 support residual 전체를 덜 믿는다. 이 항은
P2의 k=1에서 특히 필요하다. 클래스마다 support가 하나뿐이면 그 하나의 scalar
가중치는 정규화된 waveform 평균에서 상쇄되기 때문이다.

중요한 제한은 다음과 같다.

- M은 FBCCA anchor나 query precision을 바꿀 수 없다.
- M은 support 내부 가중치와 그 가중치의 클래스 균형 평균인 residual 유효 증거량만
  바꿀 수 있다. anchor 자체나 query precision은 바꾸지 않는다.
- P1에서는 affinity가 ideal-prototype pseudocount 대비 sample 기여와 residual 유효
  증거량을 모두 줄이는 보수적 이중 attenuation으로 작동한다.
- k=0에는 support가 없으므로 pair 코드를 호출하지 않고 `AQM == AQ == A0`이다.
- 비교 가능한 M이 모두 없으면 `AQM == AQ`를 bitwise exact하게 반환한다.
- subject ID, sample ID, 파일명, row 순서는 입력이 아니다.

즉 가설은 “dry가 항상 좋다”가 아니다. **어떤 장비가 좋으냐보다 calibration과 실제
query의 측정 상태가 맞을수록 그 calibration을 더 믿어도 되는가**가 가설이다.

## 4. 왜 k=1과 k=3/5의 gate가 다른가

k=3과 k=5에서는 과거 calibration block만으로 다음 calibration block을 예측하는
prequential 검사가 가능하다. 예를 들어 k=3이면 block 1로 block 2를, block 1–2로
block 3을 예측한다. 평균 BA와 정답 log-probability가 모두 FBCCA 이상일 때만 최종
query에 support residual을 쓴다.

k=1에서는 유일한 trial을 hold-out하면 학습할 trial이 남지 않는다. HOSO와 HTCCA
원문도 각각 최소 K=2와 2 trials/class를 요구한다. 한 trial을 시간 절반으로 나누는
것은 독립 validation이 아니므로 V2 gate로 쓰지 않는다. 대신 BETA/Dong의 별도
development participant에서 k=1이 사전 비열등 기준을 통과한 고정 updater만 허용한다.

모든 `k>0` operator 호출은 gate authorization을 명시해야 한다. 빠뜨리면 support를
읽기 전에 exact FBCCA로 돌아간다. k=3/5 prequential receipt는 평가 block `b`마다
fit block이 정확히 `1..b-1`임을 기록하고 검사한다. A_Q 호출은 context 입력을 전부
거부하며, A_QM은 query/support key와 metadata packet pairing hash를 감사용으로 남긴다.

여기서 최종 query 예산과 gate를 만드는 중간 support depth를 구분한다. 최종 query
API는 여전히 k=0/1/3/5만 받는다. 별도 prequential API만 `evaluation_block=b`와
`fit_blocks=1..b-1` receipt를 받은 뒤 depth 1/2/3/4를 허용하며, 오직 그 다음
calibration block의 candidate probability를 계산한다. 이 경로는 최종 query를 받을 수
없고 출력에 depth와 receipt를 그대로 남긴다. 따라서 k=3에서 block 1로 block 2를,
blocks 1–2로 block 3을 실제로 계산할 수 있으면서도 k=2/4를 새 endpoint처럼 공개하지
않는다.

이 설계가 보장하는 것은 “미지의 사람에게 절대 손상이 없다”가 아니다. 보장되는 것은
`off`일 때의 exact fallback과 `||p_new-p_FBCCA||_1 <= 2 lambda`라는 perturbation bound다.
효능과 harm tail은 독립 participant에서 경험적으로 검정한다.

## 5. 데이터 역할과 오염 방지

### 후보/하이퍼파라미터 선택

- BETA: 이미 공개된 S1/S16을 제외한 68명 중 동결된 23명
- Dong2023: 이미 공개된 S1을 제외한 58명 중 동결된 19명
- 총 42명을 사용해 12개 고정 후보 중 하나만 선택한다.

### 독립 AQ source gate

- 위와 겹치지 않는 BETA 45명 + Dong2023 39명 = 84명
- 네 block 자료이므로 k=0/1/3, query는 block 4로 고정한다.
- 이 단계에서 실패하면 wearable held 60명은 열지 않는다.

### 새 cross-session 외부 복제

Choi et al. 2019 GigaDB 100660은 CC0, 30명×2일 SSVEP 자료다. 다운로드와 asset
감사를 마친 뒤 exact block 구조를 outcome 전에 고정하여 AQ의 k=5 및 session-shift
복제 gate로 사용한다. questionnaire/physiology는 탐색적 context이지 wet/dry·impedance
증거로 쓰지 않는다.

### metadata 직접 검증의 현재 병목

검색에서 발견한 가장 적합한 독립 자료는 Liu et al.의 16명×3일 wet/dry 연구다.
매일 headset 순서를 무작위화하고 block 전 impedance를 기록했지만 raw data는 저자에게
reasonable request가 필요하다. 라이선스, 동의 범위, 기존 Zhu 102명과 participant
중복 여부를 서면 확인해야 한다.

공개 Nakanishi 자료는 NEMAR에서 다운로드 가능하지만 license가 `Unknown`이고 M이
없다. 따라서 권한 확인 전에는 재배포하지 않으며, 확인 후에도 signal-only comparator
역할뿐이다.

## 6. 실험 순서와 중단 규칙

1. **형식/단위시험:** k=0 exact FBCCA, off exact fallback, all-missing exact AQ,
   no-query-label API, order/permutation invariance, probability simplex, convex bound.
2. **합성 development와 별도 synthetic lockbox — terminal 완료:** 도움이 되는 support, 무작위 label,
   오염 block, context-null, correct/shuffled pairing을 모두 검사한다. 생성식·독립 RNG
   key·판정 기준·개발/lockbox seed는
   `configs/analysis/metadata_calibration_v2_synthetic.yaml`에 결과 전에 별도로 동결한다.
   이것은 구현과 메커니즘의 필요조건이지 human EEG 효능 근거가 아니다. V11은
   post-claim infrastructure terminal로 끝나 scientific gate를 평가하지 못했다.
3. **BETA23+Dong19 선택 — 미실행·금지:** operator와 작은 grid에서 하나만 선택하고 즉시 동결한다.
4. **BETA45+Dong39 독립 AQ gate — 미실행·금지:** pooled eAUC lower bound>0, observed gain>=.01,
   dataset별 k=1 평균>=0 및 harm-tail 기준을 모두 요구한다.
5. **Choi 외부 복제 — 미실행·금지:** asset/protocol 동결 후 한 번만 실행한다. 실패하면 held로 가지 않는다.
6. **V2 clean tag·power receipt·새 서명 승인 — 후속 human 실행 권한 없음:** permanent outcome key가 candidate 이름과
   무관하게 같은 물리 cohort의 중복 공개를 막는지 확인한다.
7. **조건부 wearable held 60 — 미개봉·금지:** 모든 역할·budget·control을 하나의 atomic bundle로
   생성하고 한 번만 공개한다. 일부 결과만 보고 계속할 수 없다.

어느 hard gate든 실패하면 그 결과를 보존하고 현 candidate를 종료한다. threshold나
lambda를 같은 lockbox에 맞춰 바꾸려면 새 candidate와 새 미관측 cohort가 필요하다.
V11처럼 post-claim infrastructure failure가 난 경우에는 효능을 판정하지 못했더라도 같은
lockbox를 재호출하지 않는다. 후속 연구는 순환 버그 수정만으로 자동 승인되지 않으며, 새
schema·명시적 pre-outcome amendment·새 미관측 lockbox가 있어야 별도 후보로 시작할 수 있다.
현재 repository의 governed V2 effectful entrypoint에는 hash-pinned terminal deny-overlay를
적용해 synthetic 재실행과 BETA/Dong request·prediction·label join·reduction·selection·gate를
데이터 접근 전에 기계적으로 거부한다. Read-only plan/allocation/inventory/artifact 감사만 남긴다.

## 7. 최종 통계 질문

Primary는 participant별 `eAUC(AQM)-eAUC(AQ)`이다. eAUC는 k=0/1/3 calibration
curve 아래 면적이며 k=0 차이는 구조상 정확히 0이다. 평균 차이의 단측 95% lower
bound가 0보다 크고 관측 평균이 0.02 이상일 때 실질적 metadata 효과를 주장한다.

그 다음에만 calibration saving을 연다.

1. AQ k=3이 AQ k=1보다 실제로 좋아야 한다. 그렇지 않으면 줄일 calibration 자체가 없다.
2. AQM k=1이 AQ k=3보다 BA 1/60 이상 나쁘지 않아야 한다.

두 조건이 함께 맞아야 “metadata가 3 blocks를 1 block 수준으로 줄였다”는 표현이
가능하다. correct M과 pair-shuffled M 차이는 그 뒤의 mechanism family이며, primary
효과를 대신하지 않는다.

## 8. V11 실행 전 구현·worktree 경계 (historical)

V11을 만들 때 공유 계약은 main이 소유했고, 계약 커밋 뒤 새 worktree에서 다음 lane을
분리했다.

- operator lane: score prototype, template residual, convex fusion, relative affinity,
  exact fallback과 unit tests
- experiment lane: sealed allocation loader, synthetic/external runner, participant-level
  statistics, immutable receipts
- integration lane(main): candidate registry, lifecycle/permanent-key 연결, documentation,
  최종 end-to-end 검증

기존 V1 schema나 intervention 이름의 의미는 바꾸지 않았다. V2는 별도 schema/version
dispatch를 사용했다. 이 구현 경계는 V11 재실행 권한이 아니며, 조건부 external producer
worktree도 main에 병합하지 않았다.

## 9. 후속 별도 후보에서 외부로 필요한 것

현 V11 경로에서는 synthetic·signal-only·human gate를 더 진행할 수 없다. 새 candidate,
schema, seed, amendment와 future beacon을 별도로 동결한 뒤 직접적인 metadata 복제를
강화하려면 다음 한 가지가 추가로 필요하다.

- Liu et al. raw EEG+block impedance 자료의 분석 허가와 다운로드 링크, 또는
- 같은 정보를 수집할 새 연구의 IRB/동의/데이터 거버넌스 승인

이 자료 권한은 새 synthetic gate 자체의 필요조건은 아니지만, 없으면 wet/dry·impedance의
독립 human replication을 주장할 수 없다. 어느 경우에도 현 V11의 소비된 lockbox나 미실행
external 경로를 이어서 실행해서는 안 된다.
