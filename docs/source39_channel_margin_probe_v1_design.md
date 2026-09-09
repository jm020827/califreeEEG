# Source39 channel-margin probe v1 — 사전 설계

2026-09-10 KST. **설계와 순수 수학 구현 단계이며 사람자료 실행 전이다.**
원 목표는 외부 acquisition metadata가 EEG-derived Q/공통 정보 이상으로
저보정 SSVEP의 정확도와 실제 calibration label 비용을 개선하는지 검증하는 것이다.
이번은 그 목표를 대체하지 않는 **작은 기작 진단 1개**다.

## 쉽게 설명하면

기존 39명의 개발자료에서, 각 사람의 앞 3회 EEG로 12개 자극의 채널별 예시 파형을 만든다.
별도 source block5의 파형이 ‘정답 예시’와 얼마나 비슷하고 ‘가장 헷갈리는 오답 예시’와
얼마나 다른지 계산한다. 여덟 채널 중 상대적으로 유리한 채널을 예측하는 데,
EEG에서 계산한 품질 특징만 있을 때보다 실제 임피던스 기록을 더하면 도움이 되는지 묻는다.
평가할 사람의 답을 그 사람을 예측하는 모델 학습에 넣지 않는다.

별도 block와 support template의 반복 불일치를 예측한 과거 proxy 실험과 다르다. 그러나 이 표적도 **단일 채널의
예시 파형 비교**라는 한정된 표현이다. 성공해도 SSVEP 정확도/보정 절감 증거는 아니며,
실패해도 공간 공분산·채널 조합·다른 시점/측정의 metadata까지 무효가 되지 않는다.

## 과거 근거와 이번 차이

- [N1-R 실제 음성](task_trca_n1_transport_recovery_r1_results.md): M이 학습되어 점수가
  바뀌었지만 k3 정답 변화0, k5 정답1개 감소, 보정량312개 cell 모두 동일했다.
- [인공 능력 PASS](n1_metadata_generated_efficacy_v1_results.md): 유용하게 구성한 M은
  예측을 개선했다. 따라서 작은 경로가 언제나 작동 불가능하다는 설명은 좁혀졌다.
  실제 M 정보 부족이 과거 실패 원인이라고 확정한 것은 아니다.
- [기존 proxy](metadata_prior_source39_v1_results.md)는 별도 block의 반복 불일치였다.
  여기서는 정답과 경쟁자 구별을 표적으로 한다. 기존 결과를 덮어쓰지 않는다.

## 고정 입력·역할

Source ID는 [기존 명세](../configs/analysis/metadata_prior_source39_v1.json)의 39명과 동일하며
새 [prospective config](../configs/analysis/source39_channel_margin_probe_v1.json)에 복사했다.
**반복 노출된 개발 cohort**이지 새 검증 cohort가 아니다.

| 입력 | 허용할 내용 | 이번 예측 모델에서의 역할 |
|---|---|---|
| Native `x_250` | 두 interface, blocks0–2, 12classes/5bands/8channels/250samples | support template와 Q |
| Native `x_250` block5 | 같은 사람의 source EEG/정해진 class 순서 | margin 표적만 |
| Source M projection | 해당 사람/interface의 blocks0–2 packet | numeric M 또는 공통 missing mask |
| 공통 정보 | interface, order, channel identity, k, availability | 모든 모델에 동일 |
| 그 외 | blocks3–4/6–9, full/a0 arrays, raw MAT, 미래 M, held60 | 숫자 decode 금지 |

k=3, N=250(1초), dry/wet 모두, 원 5bands를 고정한다. N500을 잘라 대체하지 않는다.
주파수/채널 순서는 원 native 설정 그대로다. 표적은 band별로 계산하며 loss에서 모든 band에
동일 가중치1/5를 준다. **ETRCA band-combination weights나 분류 점수는 사용하지 않는다.**

실제 읽기 구현은 새 facade에서 `support3/source_block5/metadata3`만 제공한다.
기존 N1 `HUMAN_PROFILE`/role/global을 변조하거나 evaluation을 runtime에 fit으로 바꾸지 않는다.
원 archive 무결성 해시의 전체 bytes 읽기는 허용 block의 수치 decode와 별도로 등록해야 한다.
전체 x를 `np.load`하거나 첫5blocks를 읽는 `native_support_prefix`는 사용하지 않는다.
새 read manifest는 정확한 paths/pins, 39ID, N250/k3/block5, decode 전 durable journal을 결합한다.

## M-blind 표적 — 수식과 실패 처리

Support 평균 template를 `t[j,b,c,:]`, source block5를 `x[y,b,c,:]`라고 한다.
각 파형을 시간 평균으로 중심화한 signed Pearson 상관을 `rho`로 계산한다.

`d[b,c] = mean_y(rho(x[y,b,c], t[y,b,c]) - max_{j != y} rho(x[y,b,c], t[j,b,c]))`

Primary `z[b,c] = d[b,c] - mean_c(d[b,c])`: 사람/interface/band 내부의 상대 채널 판별력.
12classes는 동일 가중치다. 임피던스, Q, 학습된 spatial filter, query scores는 표적 함수의
인자가 아니다. Level `mean_c(d)`와 class별 margin은 기술적 진단만 보존하며 대체 primary로
선택하지 않는다. Correlation은 float64/시간 중심화 후 계산, 미세 roundoff만 [-1,1] clip한다.
중심화 norm <=1e-12/비유한 입력/순서·shape 오류는 조용히0으로 대체하거나 행을 제외하지 않고
전체 probe를 `INVALID_INPUT`으로 종료한다. 데이터 파일의 원래 class routing은 별도 감사 대상이다.

## 모델·표현·선택

이것은 N1 learner 재실행이 아닌 **joint ridge 예측 probe**다. 고정 Q15와 M2를 재사용하되
M은 첫3측정을 `log1p`한 뒤 채널중심 평균과 population SD를 계산한 두 특징이며
NaN만 missing(0 유효)이다. SD 자체를 채널중심화한 특징은 아니며 이후 design 중심화는 공통 적용한다.
`metadata_features`의 availability를 사용하여 fit-only scaling 뒤 미관측 채널을0으로 둔다.

| arm | 입력 | 명목 계수 수(절편 포함) |
|---|---|---:|
| Q | Q15 | 16 |
| Q2 | Q15 + fit-standardized Q 열1/2 각각의 제곱 | 18 |
| QM | Q15 + M2 | 18 |
| SHAM | Q15 + stratum donor M2 | 18 |

열 index는 zero-based:1=off-harmonic fraction의 log, 2=repeat disagreement.
Q2는 선형 Q에 같은 열을 다시 선형으로 넣는 중복이 아니다. 그렇다고 충분히 강한 모든
EEG-only 학습기를 대표하지도 않는다. 세 augmented 모델은 동일 joint fit/규제 후보/선택 기회다.
이전 N1의 frozen-Q +3계수 tanh 잔차와 다른 설계이며 같은 실험으로 합산하지 않는다.

각 fit에서 Q와 M scaler는 fit participants만 사용한다. Q2의 제곱은 그 Q scaler의 출력에서
계산한다. 모든 design은 채널축 평균을 빼고, 그 뒤 `fit_ridge`로 각 열을 fit-only 재표준화하여
`mean squared error + alpha * sum(coef**2)`를 최소화한다. 예측도 채널축 중심화한다.
채널 내 상수/절편 방향은 사라지므로 표의 명목 수는 유효 rank가 아니다. 각 fit의 design rank와
비활성 열을 반드시 보고한다. 대상과 loss는 모든 interface/band/channel에 같은 가중치다.
각 fit에서 **두 interface와 5bands에 하나의 계수 벡터를 공유**한다. 채널 중심화는
logk/order/interface처럼 채널 내 상수인 가산 효과도 제거한다. 따라서 이것은 shared-slope
relative-channel probe이며, interface/band별 계수·상호작용의 M 효과를 배제하는 검증이 아니다.

Alpha 후보는 큰 규제 우선 `[1, .01, .0001]` 3개만, inner participant 3-fold의 pooled OOF
26명 균등 평균 MSE(9/9/8개 fold 평균의 단순 평균 아님)로
선택하며 최소값+1e-12 이내 tie는 큰 alpha를 택한다. Alpha0/후속 탐색/모델 교체는 없다.
39ID 정렬 후 rank modulo3으로 outer3fold(26fit/13eval), 각26fit 안에서도 ID 정렬 modulo3으로
inner3fold를 고정한다. 모든 interface/band/channel은 참가자를 따라 함께 이동한다.
3outer ×4arms ×(3alpha ×3inner +1final)=**120ridge fits**다.

## SHAM과 crossfit 권한

각 inner/final의 fit와 validation partition 안에서 따로 donor를 만든다. 동일 participant의
dry/wet packet을 함께 교환하도록 두 interface 전체의 order와 `[2,3,8]` missing mask를 stratum
key로 묶는다. Stratum의 정렬 ID를 한 칸 cycle하며 singleton은 self로 남겨 공개한다.
타인/다른 partition의 label은 donor 배정에 사용하지 않는다. Raw packet 변경률은 NaN 위치가
같다는 조건에서 관측 값의 exact 비교로 별도 기록한다. Coverage gate는 raw가 아니라
**소비 design** 기준이다: 해당 fit의 SHAM scaler/중심화/재표준화를 실제 M과 donor M에 동일
적용한 두 M design 열에서 max absolute difference>1e-12가 **각 interface 모두** 성립해야
그 사람을 교체로 센다. Raw 변경만으로 gate를 통과시키지 않는다.
양 interface를 별도로 섞어 paired 구조를 깨지 않는다.

이 SHAM은 진짜 M 대응관계의 진단 대조군이다. **M|Q를 보존하는 검증된 conditional sampler나
knockoff가 아니므로 conditional-independence p-value를 계산하지 않는다.**

동일 사람의 block5는 다른 outer fold에서는 학습 표적으로 쓰인다. ‘모든 block5를 전역 모델
동결까지 미접근’이 아니라 **각 평가 모델의 모든 fitting/scaling/selection에서 그 사람 제외**가
보장 대상이다. 전체 모델/선택결과를 먼저 동결하고 pooled OOF 성적을 한 번 공개한다.
원 blocks6–9와 held60은 이 단계 내내 열지 않는다.

## 측정·선택·중단 — 결과 전에 고정

각 사람 p의 `L_arm[p] = mean_{interface,band,channel}(prediction - z)^2`.
39개 사람 loss를 같은 가중치로 평균한다. 비교 개선율은
`G(QM,C)=(mean L_C - mean L_QM)/mean L_C`, C=Q/Q2/SHAM.
분모<=1e-8은 그 비교에서 headroom 부족이며 유망 판정 불가다.

유효성 조건: 모든39명·2interfaces 완비/비유한값0; mean(z²)>1e-8;
두 interface 모두에서 M 관측 채널>=2인 사람이 적어도32명;
각 outer held partition에서 위 소비 design 기준으로 바뀐 SHAM 사람이 최소80%(13명 중11명).
Inner donor coverage/rank도 전부 기록하되 성적에 따라 partition을 바꾸지 않는다.
Coverage 미달은 `UNINFORMATIVE_CONTROL`, target 부족은 `INSUFFICIENT_TARGET_VARIATION`으로
구분하며 ‘M 정보 없음’으로 쓰지 않는다.

`PROMISING_PROBE`는 세 비교 모두 평균 MSE 개선>=2%, 각 비교에서 outer3fold 중2개 이상
양수, 그리고 두 interface 각각에서 QM−Q 개선이 양수일 때만 부여한다.
여기서 ‘양수’는 absolute mean MSE 감소>1e-12이며 1e-12 이하는 수치 tie로 둔다.
2%는 후속 예산을 배분하는 **사전 실무 기준**이며 유의확률/임상적 최소효과가 아니다.
다른 유효 결과는 `NO_PROMISING_SIGNAL_IN_THIS_PROBE`. 모든 사람의 paired loss,
fold/interface별 결과와 범위를 공개한다. 겹치는 CV 학습과 반복 개발자료 때문에 iid39 확증 CI,
trial/channel 단위 p-value 또는 일반 conditional-information 추정치를 내지 않는다.
이 규칙은 불확실한 결과도 성공으로 승격시키지 않지만 통계적 무효/동등성을 증명하지도 않는다.
복수 사유는 모두 기록하되 terminal 우선순위는 `INVALID_INPUT`/`EXECUTION_FAILURE`/
`AUDIT_FAILURE`(효능 미평가) → `INSUFFICIENT_TARGET_VARIATION` → `UNINFORMATIVE_CONTROL`
→ `INSUFFICIENT_COMPARATOR_HEADROOM` → 위 유망/비유망 판정 순이다.
성공적으로 만들어진 결과의 감사 실패는 이전 잠정 판정을 덮어쓰지 않고 terminal에서 강등한다.

## 예산·다음 분기·현재 권한

후속 제안 예산: source 특징 추출1회, primary1회(120fits), 독립 수치 감사1회, 재시도0.
Primary CPU1/30분/RSS4GiB, audit CPU1/15분/RSS4GiB, 출력1GiB, tests총15분/출력1GiB.
코드 형상 추정: EEG decode전체 약300MB, 순차처리 가능; 입출구 전체 archive 해시는 약6GB I/O.
실측 시간/peak 보증이 아니므로 supervisor가 제한을 집행하고 최초 실패를 보존한다.
GPU/설치/새 dataset/외부 요청/유료0. 테스트 seed는 구현 회귀용이며 새 효능 반복으로 세지 않는다.

- 유망: 이 표적을 channel regularization 학습에 쓰는 후보1개를 **별도** 설계한다.
  같은 Q/Q2/SHAM, 고정 k별 accuracy와 실제 calibration 비용 검증 없이는 연구 성공 선언 불가.
- 음성/불충분: 본 field/단일채널표적/ridge 방향은 종료. 강도·window·alpha·seed를 바꿔 살리지
  않는다. 다른 기작은 별도 근거/유한 예산으로만 제안하며 자동 veto하거나 자동 재시도하지 않는다.
- 입력/감사 실패: 효능 미평가로 종료. 부분 출력·실패·기존 N1-R 음성을 보존한다.

현재 사용자의 ‘계속’에 따라 설계와 순수 수학의 생성 unit tests를 진행한다. 이 문서는 이전
generated-only 계약을 사람자료 권한으로 바꾸지 않는다. **실제 source39 재읽기의 구체 범위와
예산 승인 후** 새 facade/runner/audit를 완성·검증·등록하고 위 단회 실행을 수행한다.
승인된 범위 안의 단계별 재승인은 필요 없으며 held60/외부 요청/유료는 계속 별도 승인이다.

## 문헌·작업 방식

`academic-research`로 기존 landscape/frontier와 CPI 방법을 확인했다.
Watson/Wright2021 [공식 논문](https://link.springer.com/article/10.1007/s10994-021-06030-6)의
abstract와 선택 HTML §1–2.3은 marginal/conditional 중요도 및 유효한 knockoff 조건을 구분한다.
이를 근거로 우리 cycle SHAM에 정식 조건부 독립 검정의 보장을 붙이지 않았다.
이것은 그 논문의 CPI 구현/재현이 아니다. PDF0/abstract-provisional card;
원 cutoff2026-09-04 유지, 좁은 방법 검색만 수행하고 포괄적 최신문헌 조사라고 하지 않는다.
첫 CPI 키워드 검색은 핵심 논문을 놓치고 Semantic Scholar429를 기록했다. 그 공백을 메우는
정확한 제목의 Crossref 조회1회를 추가했다(전체 검색2회/8records, 상세 검토 논문1개).

`coordinate-worktree-changes`: root 단독 작성, 두 agent는 코드/설계 읽기 전용 검토.
새 worktree0/공유 tree 동시작성0, 기존40trees와 미추적 출력 보존. 병렬 구현은 공유 통계·권한
계약을 먼저 고정해야 하므로 이번에는 하지 않는다. 현재 코드가 파일 접근 권한을 보증하지는
않으며 실제 reader/end-to-end/audit가 남았다는 것을 상태 문서에 명시한다.
