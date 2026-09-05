# Metadata-calibration-efficiency-v1 source 결과와 후속 결정

기준일: 2026-09-06

상태: **source-development 24/24 jobs 완료 / development no-go / held 60명 미개봉 / 후보 종료**

실행 tag: `metadata-calibration-efficiency-v1-freeze-20260905-r1`

## 한 문장 결론

현재 bounded calibration-prior 구현에서 올바른 wet/dry·채널 impedance metadata는 EEG-derived
Q만 쓴 같은 checkpoint보다 early-calibration 성능을 높이지 않았고, shuffled metadata와도
구분되지 않았다. 따라서 사전 규칙대로 held 60명을 열지 않았다.

이것은 `metadata는 언제나 쓸모없다`는 결론이 아니다. 다음 두 사실을 함께 보인
**candidate-specific development no-go**다.

1. 공통 A_Q calibration updater부터 k=1에서 크게 악화해 충분히 강하지 않았다.
2. M branch는 완전히 죽지는 않았지만 실제 block의 올바른 값보다 metadata의 존재에 가까운
   비특이적 precision offset을 학습했다.

## 무엇을 한 번만 실행했는가

- 데이터: `wearable_v3` 102명 중 사전 할당 source 39명
- 평가 단위: participant 39명; seed나 interface를 독립 표본으로 세지 않음
- outer split: 13명씩 3 folds
- 각 fold: 20명 inner-fit + 6명 inner-validation으로 epoch 선택, 26명 refit, 남은 13명 평가
- seeds: 42, 43, 44; query probability를 seed 사이에서 먼저 평균한 뒤 argmax
- interface: wet/dry를 별도 calibration·평가하고 participant 안에서 동일 가중 평균
- support: interface당 k=0/1/3/5 complete blocks = 0/12/36/60 labeled trials
- 고정 query: interface당 block06–10의 60 trials, participant당 120, 전체 4,680 rows
- 실행량: candidate 9 + baseline 15 = source 24 jobs
- 조건부 held: source gate가 PASS할 때만 candidate 3 + baseline 5 jobs를 실행하도록 봉인

24개 source job과 private grid는 모두 첫 실행에서 완료됐다. Source gate가 FAIL하여 `held/`
directory, held input seal, held registry claim과 held result는 생성되지 않았다.

## 사전 source gate

| Gate | 사전 기준 | 관측 | 판정 |
|---|---:|---:|---:|
| A_Q k=5 viability | BA ≥ 0.500 | 0.476282 | FAIL |
| A_QM−A_Q early eAUC 평균 | ≥ +0.010 | −0.004843 | FAIL |
| outer-fold 방향 | 3/3 strictly positive | −0.004274 / −0.004808 / −0.005449 | FAIL |
| correct−block-shuffle eAUC 평균 | ≥ +0.010 | 0.000000 | FAIL |
| shuffle 대비 음수 fold 없음 | 각 fold ≥ 0 | 0 / 0 / 0 | PASS |
| all-missing fallback | probability exact equality | exact | PASS |
| metadata-only shortcut | 구조적으로 1/12 | 0.083333 | PASS |

Primary source contrast는 39명 중 6명 개선, 7명 동일, 26명 악화였다. 평균은 `−0.004843`,
중앙값은 `−0.004167`, SD는 `0.006251`이다. 사후 기술용 95% t interval은
`[−0.006870, −0.002817]`이고 paired `d_z=−0.775`다. Gate는 이 사후 interval이나 p-value가
아니라 위의 사전 수치 조건으로 판정했다.

## Calibration curve를 쉽게 읽으면

| 방식 | k=0 | k=1 | k=3 | k=5 | early eAUC |
|---|---:|---:|---:|---:|---:|
| A_0: Q/M 없음 | 0.485684 | 0.482051 | 0.485470 | 0.486111 | 0.483796 |
| A_M: metadata만 | 0.485897 | 0.474359 | 0.480342 | 0.485043 | 0.478276 |
| A_Q: EEG-derived Q만 | 0.484402 | 0.404915 | 0.463889 | 0.476282 | 0.437821 |
| A_QM: Q+올바른 M | 0.484615 | 0.396581 | 0.461752 | 0.472863 | 0.432977 |
| A_QM−A_Q | +0.000214 | −0.008333 | −0.002137 | −0.003419 | **−0.004843** |

M은 보정 부담을 줄이지 못했다. 더 근본적으로 A_Q는 k=0에서 k=1로 갈 때 평균 BA가
`0.484402→0.404915`로 떨어졌고 35/39명이 악화했다. k=3과 k=5에서 회복하지만 k=5도
k=0보다 낮다. A_Q−A_0 early eAUC도 `−0.045976`이다. 즉 M의 증분만 약한 것이 아니라
현재 Q/support posterior update가 한 block에서 과도하거나 잘못된 방향으로 작동했다.

`A_QM(k0)−A_Q(k1)=+0.079701`이라는 표면적 절약값은 metadata가 좋은 것이 아니라
A_Q(k1)이 나빠진 결과다. Source gate 실패와 BA floor 때문에 calibration saving으로
해석하지 않는다. 반대로 Q-only `k3−k1=+0.058974`는 더 많은 label이 현재 updater의 초기
손상을 일부 회복한다는 진단이다. `A_QM(k1)−A_Q(k3)=−0.067308`이므로 사전 NI margin
`−1/60`도 크게 벗어난다.

## Metadata mechanism은 왜 실패했는가

공개 source 결과에서 correct와 block-shuffle의 participant BA/eAUC는 모든 participant와
budget에서 같았다. Correct−stale은 `−0.004380`, correct와 wrong/opposite-interface도 거의
같았다. All-missing=A_Q exact fallback과 metadata-only chance proof가 통과했으므로 단순 배선
오류나 label shortcut의 증거는 없다.

실행 종료 뒤 source의 label-free probability artifact만 사용한 명시적 사후 진단은 다음을
보였다. 이 값은 signed primary 결과가 아니라 구현 원인을 찾는 exploratory evidence다.

| Correct와 비교한 intervention | k=1 mean absolute probability difference | k=1 argmax changed rows |
|---|---:|---:|
| block shuffle | 0.0000106 | 0 / 14,040 seed-fold rows |
| all missing | 0.014222 | 431 / 14,040 seed-fold rows |
| interface only, 즉 impedance 제거 | 0.011141 | 329 / 14,040 seed-fold rows |
| impedance only, 즉 interface 제거 | 0.000190 | 6 / 14,040 seed-fold rows |

따라서 M path가 numerical no-op인 것은 아니다. 그러나 관측 metadata를 전부 없애면 반응하면서
실제 block impedance를 다른 block으로 바꾸면 거의 반응하지 않았다. Source context의 shuffle
입력도 무효가 아니었다. 780 participant-interface-block 중 donor 8채널 impedance가 완전히 같은
경우는 6개뿐이고 correct–shuffle `log1p(impedance)` 절대차 평균은 `0.156653`이었다.

가장 타당한 해석은 current encoder가 실제 impedance 값과 correct pairing보다 거의 항상 1인
availability/presence pattern에서 비특이적 precision offset을 배웠다는 것이다. 이는 모든 아홉
stage-2 job이 최소 허용 epoch 5를 선택했고, 그 선택 시 inner-validation M increment가 모두
`−0.001389` 이하의 음수이며 correct−shuffle이 사실상 0이었다는 기록과도 맞는다. 현 선택기는
zero-M인 epoch 0을 후보로 허용하지 않아 `M을 쓰지 않는 편이 낫다`고 abstain할 수 없었다.

## 기존 저보정 방법과의 비교

| 비교군 | k | source participant mean BA |
|---|---:|---:|
| strict FBCCA | 0 | **0.677564** |
| SAME3 filter-bank eTRCA component | 1 | **0.560043** |
| target filter-bank eTRCA | 3 / 5 | 0.316026 / 0.387179 |
| target template correlation | 1 / 3 / 5 | 0.141239 / 0.166453 / 0.189316 |
| Chiang-2021 LST+filter-bank eTRCA port | 1 / 3 / 5 | 0.090812 / 0.095513 / 0.098291 |

Strict FBCCA k=0은 A_Q k=0보다 평균 `0.1932`, SAME component k=1은 A_Q k=1보다 `0.1551`
높았다. 데이터가 원래 decoding 불가능해서 no-go가 난 것은 아니다. Candidate는 target-only
template/eTRCA와 현재 Chiang clean-room port보다 높았지만, 서로 다른 정보권한·source 학습
자원을 쓰므로 이 순위는 descriptive다. SAME 값도 full CSDuDoFN/OS-SSVEP 재현이 아니라
one-shot component다. Chiang k=1은 protocol-adapted이고 결과가 chance에 가까워 port 또는
protocol mismatch 가능성을 배제하지 않는다.

## 무엇을 주장할 수 있고 없는가

주장할 수 있다.

- 이 same-device 39명 cross-fitted development에서 현재 bounded precision-residual 후보는
  external acquisition context의 Q 초과 증분이나 pairing reliance를 보이지 않았다.
- 현재 A_Q common calibration updater는 충분히 강하지 않았고 한-block update가 특히 해로웠다.
- Fallback, shortcut 방지, wrong-context bounded-harm engineering invariant는 작동했다.
- Source no-go에 따라 held를 열지 않은 것은 사전 설계에 부합한다.

주장할 수 없다.

- 모든 impedance/interface metadata가 SSVEP에 무의미하다는 모집단 결론
- impedance나 interface의 독립 인과효과
- broad device/site OOD 일반화
- current candidate가 FBCCA/SAME보다 일반적으로 열등하다는 confirmatory ranking
- wet과 dry 각각의 효과; public result bundle이 condition별 summary를 내보내지 못했다
- held 60명 성능; input seal조차 생성되지 않았다

`deployment_qualified_for_robustness_claim=true`도 efficacy가 있다는 뜻이 아니다. Wrong-context
평균 harm `−0.004950`, 최악 participant `−0.022222`, severe harm 0명이 사전 bounded-harm
조건 안에 있었다는 좁은 engineering 결과다. M에 거의 의존하지 않아 안전해 보였을 가능성이
크다.

## 추가 데이터 결정

현재 로컬 BETA 70명과 Wang 35명은 electrode가 `unknown`, Dong2023 59명은
`pregelled_semidry` 한 수준이며 세 자료 모두 impedance가 없다. Ke2025 24명은 두 session,
32-channel AR SSVEP라 signal/OOD 진단에는 유용하지만 BIDS electrode impedance가 모두 `n/a`이고
interface contrast가 없다. 이들을 primary M 효과에 합치거나 source no-go를 뒤집는 용도로
실행하지 않는다.

같은 연구목표로 새 candidate를 시험하려면 **새 독립 development data가 필요하다**. 요청 또는
신규 수집의 최소 내용은 다음과 같다.

- current source 39명과 독립인 participant cohort; V1의 39명은 출발 참고일
  뿐이며 수집 전 새 가정·gate로 sample size를 다시 power함
- participant마다 wet/dry 또는 비교할 interface를 randomized/counterbalanced crossover
- interface당 12-class complete block 최소 10개, 동일한 immutable query 설계
- 매 block 전 8채널 이상 per-channel impedance와 availability/contact flag
- cap 재장착을 포함한 최소 두 day/session
- device, firmware, reference/ground, gain/filter, electrode material, room·display timing
- 실제 cue/rest/feedback를 포함한 calibration 소요시간
- IRB/동의, raw/derived 재사용·배포 권한과 version/checksum

기존 source 39명은 새 v2의 exploratory engineering에는 사용할 수 있지만 다시 unbiased promotion
cohort로 세지 않는다. 기존 held 60명은 계속 미개봉 confirmatory 자원으로 보존한다. Full-H2의
계획 민감도를 약 0.80으로 높이려면 기존 계산상 held N=83이 필요하므로, 같은 protocol의 held
participant 약 23명을 추가하는 방안도 새 power/allocation에서 검토한다.

## 같은 목표를 유지한 v2 구현 순서

1. **A_Q부터 고친다.** Strict FBCCA/SAME을 넘는 것을 무조건 요구하지는 않더라도 k=1이 k=0을
   크게 해치지 않는 common path와 fallback을 먼저 source-independent signal data에서 검증한다.
   Harmonic/FBCCA score 또는 더 강한 spectral embedding과 convex/gated support shrinkage를
   후보로 둔다.
2. **Null abstention을 허용한다.** Stage-2 epoch 0의 exact M-off 상태를 선택 후보에 넣고,
   inner-validation에서 correct M이 Q-only보다 나쁘면 residual을 0으로 유지한다.
3. **값과 pairing을 학습하게 한다.** 상수 availability shortcut을 제거하거나 별도 취급하고,
   query M과 support M의 상대적 거리·일치도를 직접 support weighting에 넣는다.
4. **Mechanism-aware training을 사전등록한다.** Correct 대 shuffled/stale context의 contrastive
   objective 또는 ranking constraint를 source inner-fit에만 사용하고, 별도 inner-validation에서
   probability-level potency와 outcome benefit을 함께 요구한다.
5. **보고 누락을 막는다.** 다음 finalizer는 wet/dry별 participant summary, seed stability,
   correct-shuffle probability displacement를 query token 없이 공개 결과에 포함한다.
6. **새 candidate와 새 data gate를 만든다.** 현재 threshold를 바꾸거나 같은 39명으로 재실행하지
   않는다. 새 candidate ID, 새 power receipt, 새 독립 source cohort를 먼저 동결한다.

연구목표는 바꾸지 않는다. 바꿀 것은 `external context가 calibration burden을 줄이는가`라는
질문이 아니라, 그 질문을 시험하기에 약했던 A_Q common path와 pairing-insensitive M operator다.

## 무결성과 재현성

- freeze commit: `56bff34dda154e12771b198cc4d316df2b3b33ee`
- source tree digest: `559ad276ad3b648bb43c7a71d245f706d2d8f8f78a51859d2123d7a6af9c0f92`
- plan: `04d77058dbec3d82d11ac15c8ad25f32fb3a3c102644e8a92a76cf759e22515a`
- source result bundle: `63bf33ae213bb5ae5ec37fcdb235e0769b23e3924e54d6d3c75e31be66203b77`
- source completion signed record: `c89d00ed01dff0a6c574fe2378fc830cb238d8a09e962e8a16c10eb9a545145e`
- lifecycle signed record: `86c4ba772539b6419053e8938c5454be2f7f93ffc47da42b59d680e8a7bfdd58`
- registry: source claim 1, held claim 0, integrity failure 0
- private keys: canonical path에 0; public keys만 보존

동결 snapshot validator로 lifecycle, source completion, private seal, 24-job/75-artifact grid,
registry와 공개 7-file result bundle을 다시 검증했다. Source query label/covariate sidecar는 암호문만
남고 plaintext는 없다. 실행 snapshot의 script mode가 read-only hardening 때문에
`100755→100644`로 보이는 mode-only 차이는 있으나 file content, commit, tag와 source-tree hash는
일치한다. Python 메모리나 외부 복사본의 포렌식 zeroization까지 보장하는 것은 아니다.

Canonical 결과:

- `/home/whwovy/eeg-results/metadata-calibration-efficiency-v1-20260905-run1/source/results/gate_or_claim_result.json`
- `/home/whwovy/eeg-results/metadata-calibration-efficiency-v1-20260905-run1/source/results/candidate_participant_contrasts.csv`
- `/home/whwovy/eeg-results/metadata-calibration-efficiency-v1-20260905-run1/source/results/baseline_summary.csv`
- `/home/whwovy/eeg-results/metadata-calibration-efficiency-v1-20260905-run1/source/results/completion_receipt.json`
- `/home/whwovy/eeg-results/metadata-calibration-efficiency-v1-20260905-run1/lifecycle_completion.json`
