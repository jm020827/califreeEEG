# V4 후속 AQ 검증 001 — 사전 계획

기존 V3 개발 실행 #5와 V4 pilot001은 종료한 채 보존한다. 이 연구는 새
`metadata-calibration-efficiency-v4-aq-study001`이며, **보정이 필요한 조건과 유용한
EEG-only 비교법을 확보하는 단계**다. Metadata 효과를 이번에 평가하지 않으며 목표를
손상 복원·OOD/discovery로 바꾸지 않는다. 과학적 기준원은
[고정 JSON](../configs/analysis/metadata_calibration_v4_aq_study.json)이다.

## 무엇을 어떻게 나누는가

```text
개발용 합성 참가자 24명
  ├─ A0 성능만으로 보정이 필요한 stable 조건 표시
  └─ decoder별 확률 온도 적합 (source query 정답 사용)
        ↓ 점수·온도·조건을 파일로 고정
독립 평가용 합성 참가자 24명: 모든 조건·방법 평가
        ↓ 참가자별 paired 차이를 검산
AQ utility 판정 / metadata·사람 EEG 효과는 아직 주장하지 않음
```

Source supervised development와 target evaluation을 구분한다. Source query 정답으로
온도를 적합하는 것은 사전 허용하지만 evaluation query 정답은 metric 계산에만 쓰인다.
Source/evaluation은 새 독립 RNG namespace를 가지며, 종료한 후보의 seed나 EEG를 재사용하지 않는다.

관측 길이는 **75/125/250 samples = 0.3/0.5/1.0초**, noise 배율은 1/2/3이다.
9조건을 stable/drift 두 family에서 모두 시험한다. 같은 참가자 내에서는 동일한 긴
signal/noise와 trial 순서를 공유해 길이·noise 효과를 paired 비교한다. 반드시 **crop 후
filter**한다. 한 초 전체를 zero-phase filtering한 뒤 자르면 짧은 window 이후의 정보가 섞인다.

이 길이 범위는 짧은 SSVEP decoding을 다룬
[TRCA 원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC5783827/) 및
[저자 FBCCA tutorial의 0.5초 예시](https://github.com/mnakanishi/TRCA-SSVEP/blob/master/tutorial/tutorial_fbcca.m)와
연결된다. 그러나 선택한 noise 배율·spatial/phase/AR 계수는 **engineering sensitivity grid**이며
사람 EEG나 impedance 물리의 실측 분포에 적합한 값이 아니다. 짧은 window의 전체 all-class Q
projection은 잔차 차원을 소진할 수 있어 이번에는 쓰지 않는다. Labeled class별 harmonic
projection을 이용한 support reliability만 block diagnostic의 가중치로 사용한다.

## 비교법과 score scale

Primary는 **class별 평균 파형(pooled)을 사용한 calibrated fusion**이다. Block별 posterior
평균 방식은 diagnostic이다. 둘 다 raw temperature .1 버전과 source-calibrated 버전을 모두
보고하며, standalone decoder도 함께 남긴다. 이전 pilot의 k5 diagnostic을 성공한 모델로
소급 선택하지 않고, 이번 새로운 자료에서 사전 지정한 primary를 검증한다.

FBCCA A0, pooled k1/3/5, block k3/5에 총 6개 temperature를 source에서 고른다.
Block k1은 pooled k1과 같은 score이므로 같은 temperature다. A0 온도는 waveform 온도와
독립적이고 k에 따라 바뀌지 않는다. 각 온도는 고정 31-point grid `logspace(-4,1,31)`에서
source 전체의 standalone NLL 최소값으로 정한다. Block은 실제 평가와 같은
`softmax → reliability weighting → posterior mixture`의 NLL을 최소화한다.
가족·noise·길이별 oracle 온도, evaluation-informed fit, lambda 탐색은 금지한다.

[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)의 temperature scaling은
confidence calibration의 근거다. CNN 연구를 EEG에 이전하는 것은 검증할 가설이며,
이 논문이 SSVEP fusion이나 metadata 효과를 입증한 것은 아니다. 양의 온도는 standalone
decoder의 class 순위를 바꾸지 않지만 두 posterior를 합쳤을 때의 결정에는 영향을 줄 수 있다.
EEG의 labeled calibration과 확률의 calibration은 서로 다른 의미다.

Fusion은 항상 `.5*A0 + .5*support`다. k0에서 raw 방법은 exact A0_raw,
calibrated 방법은 exact A0_cal이다. 두 A0의 class prediction은 동일하다.
A0_cal은 **평가 참가자의 보정은 0**이지만 source supervised development를 사용하므로
완전 무학습으로 부르지 않는다.

## 난이도와 판정 — 결과를 보기 전에 고정

Source stable A0 평균 BA가 .55~.90이고, source 참가자 절반 이상이 80%에 못 미치는
조건만 informative 집합 S로 표시한다. AQ나 metadata 효과를 보고 S를 고르지 않는다.
S가 비어도 evaluation의 18 cells를 모두 실행·보고하며 primary 결론은 내지 않는다.

Evaluation에서는 S를 다시 고르지 않는다. S의 participant×cell 중 절반 이상에서
A0<.8이어야 난이도가 재현됐다고 본다. Primary 효과는 각 참가자에서 S 조건들의
`eAUC(fusion_pooled_cal)-eAUC(A0_cal)`를 평균한 뒤 **24명**을 통계 단위로 계산한다.
24×조건 수를 독립 표본으로 세지 않는다. eAUC는 `BA0/6+BA1/2+BA3/3`.
평균 >=.01(1 percentage point), one-sided95% paired-t LCB>0, k1 평균 차이>=0을
모두 요구한다. k3−k1 향상은 보고하되, 유용한 one-shot plateau를 배제하지 않도록 필수로
요구하지 않는다. 개인별 harm 비율, drift 결과, 모든 조건의 원자료를 함께 공개한다.

가능한 상태는 `NO_SOURCE_INFORMATIVE_CELLS`, `DIFFICULTY_NOT_CONFIRMED`,
`AQ_NOT_ESTABLISHED`, `AQ_READY_FOR_CONTEXT_COMPARATOR_STUDY`다. 마지막도 metadata 기여,
사람의 효능이나 safety 증명이 아니다. Metadata를 다시 넣을 때에는 query-support EEG
context matching까지 갖춘 강한 Q 비교법을 새로 검증해야 한다.

12 classes의 k=1/3/5는 12/36/60 labeled trials다. `12*k*N/250`은 분석한 EEG 길이이지
cue·자극지연·쉬는시간·장비준비를 포함한 실제 사용자 시간은 아니다. 80% first-crossing은
관측한 k만 쓰고 미달은 `>5`, 임의 숫자나 도달자만의 평균으로 바꾸지 않는다. A0보다
AQ가 좋아지는 것만으로 calibration 비용 감소를 입증하는 것도 아니다.

## 구현·provenance·ownership

시작 base main `04218de7cbb7c54eccb40afaf94dc2227a5769c3`. 새 checkout 약 8 MB,
source/evaluation score cache와 결과 포함 1 GB budget; 여유 공간 약 300 GB.
Dependency 설치나 GPU runtime 변경은 없다. Main interpreter는 읽기 전용으로 쓰고
pytest/cache는 작업 폴더별로 분리한다. 기존 worktree를 덮어쓰거나 삭제하지 않는다.

| Lane | 단독 소유 경로 | 자원·통합 순서 |
| --- | --- | --- |
| Main | 새 plan/설계, 연구일지, 상태·결과 문서; 새 독립 auditor와 그 tests | 유일 research SQLite 작성자; 계약 먼저, 마지막 단회 실행·검산 |
| aq_implementation_map | `/home/whwovy/califreeEEG-wt-v4-aq-study`, branch `codex/v4-aq-study001`; 새 AQ module/runner/tests 3개만 | contract commit 뒤 시작; fixture seeds만 사용; 새 commit으로 반환 |
| aq_design_review | 읽기 전용 설계·구현 검토 | 공유 repo 가능; source/evaluation study RNG 실행 금지 |

Pure CCA/template helpers만 종료한 V4 module에서 읽기 전용 재사용한다. 그 module·old plan·
governance·artifact는 수정하지 않는다. 새 generator는 metadata를 생성하지 않는다.
Start → source score cache → source freeze → evaluation start → evaluation score cache →
result 순으로 exclusive create, fsync, hash 결속한다. Source freeze 전에는 evaluation RNG를
열지 않는다. Source와 evaluation의 raw score cache를 보존해 DGP 재실행 없이 온도 적합과
결과를 독립 검산할 수 있게 한다. 기존 start가 있으면 자동 retry/resume하지 않는다.
이것은 local engineering provenance이며 외부 preregistration이나 암호학적 lockbox가 아니다.
