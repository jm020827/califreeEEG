# califreeEEG 연구설계 적합성·수정안

> **역사적 감사 문서:** DEC-20260830-001이 `N=102 primary`를, DEC-20260830-002가 아래 `N=99 5-fold primary`를 대체한다. 현재는 전체 asset N=102, S1–S3 development-only, 39명 frozen training, 60명 independent lockbox다. [프로토콜 현재 상태](research_protocol_status.md)를 우선하며 아래 지적과 당시 판단은 변경 이력 보존용이다.

최초 검토 기준일: 2026-08-28  
구현·프로토콜 갱신일: 2026-08-29 (protocol 0.3 + metadata protocol 0.4-dev, confirmatory freeze 전)

대상: 저장소의 연구 질문, split/evaluation 구현, 실험 체크리스트, 공개 데이터 계획, 2025–2026 관련 연구

> **현재 상태 요약:** 연구 질문과 A0/A2 treatment는 유지한다. Query-local complex spectral backbone, paper-faithful Chen-2015 FBCCA, exact resume, confirmatory six-job execution manifest, target-free inference simulation을 구현했다. A2는 이제 공통 query-QC FiLM, external global FiLM, per-channel impedance bounded gain을 갖는 `physical_hybrid_v1`이며 A0는 같은 graph의 exact-null 경로다. 겹치는 outer-fold 모델의 의존성 때문에 5-fold primary는 폐기하고 39명 train/60명 lockbox로 바꿨다. `DEC-20260901-004`는 physical reveal #2, 10개 개발 gate와 validator-only exact restart를 승인·동결했다. S4–S102 성능은 아직 열지 않았고 confirmatory SESOI·alpha·operational threshold·multiplicity·source freeze는 별도 미결정이다.

> **Metadata 동결 상태:** 최소 정보권한과 채널별 impedance reliability 경로는 구현됐지만 완전히 동결되지는 않았다. 현 wearable primary의 treatment는 electrode type, block impedance mean/max, per-channel impedance/availability로 좁혀졌고 상수 reference/cap은 제외됐다. Six-role/18-job mechanism 결과와 충분한 외부 factor 반복이 아직 없기 때문이다. 현재 구현과 남은 gate는 [Protocol 0.4-dev 구현 기록](protocol_0_4_dev_implementation.md)에 정리했다.

> **재정독 결과:** 원자료 수치, 최신 직접 방법의 공개 코드, foundation-model 선행연구까지 다시 감사한 변경 결론은 [2026-08-29 재정독 감사](deep_read_reassessment_2026-08-29.md)가 우선한다.

## 1. 최종 판정

최초 연구설계는 **아이디어 검증용 파일럿으로는 목적에 부합했지만 그대로는 “metadata가 calibration-free SSVEP의 일반화 범위를 확장한다”는 결론을 지지하기에 충분하지 않았다.** Protocol 0.3은 split·query·평가를, 0.4-dev는 P0 결합·정보권한·physical global/channel mechanism을 고쳤다. 전체 `wearable_v3` acceptance도 완료됐다. Physical 실행의 immediate gate는 clean annotated source와 preflight이고, confirmatory 차단 사유는 physical 결과 review, protocol-matched 최신 기준선 일부, 통계·source freeze의 미동결이다.

| 평가축 | 판정 | 이유 |
|---|---|---|
| 연구 질문과 모델의 정합성 | 높음 | EEG-only와 metadata-conditioned 모델을 같은 backbone에서 비교하려는 구조가 질문에 맞음 |
| 내부 타당도 | 구현 검증 통과, 단일 development reveal 대기 | split/query/P0 mapping/full asset/physical 권한·margin은 동결; 공개 전 validator-only 정정본을 재검증함 |
| 외부 타당도 | 보통 이상 | wearable outer folds가 unseen subject를 보장하고 Dong이 실제 acquisition contrast를 제공; 완전 외부 lockbox는 남음 |
| 신규성 입증 | 아직 부족 | 2025–2026의 학습형·alignment·online-adaptive calibration-free SSVEP 기준선과의 비교가 빠짐 |
| 실행 가능성 | physical 높음 / confirmatory 조건부 | 개발 승인·판정값은 동결; confirmatory는 결과 review와 별도 통계/source freeze 필요 |
| 강한 결론 가능성 | full run 후 판정 | 39명 fixed training·60명 independent lockbox의 3-seed paired 결과와 외부 재현이 필요함 |

핵심은 피험자 수를 무작정 늘리는 것이 아니다. **같은 시험 trial, 같은 자원, 같은 backbone, 같은 target-data 접근권한**에서 metadata 하나가 무엇을 바꾸는지 분리해야 한다.

## 2. 연구 목적과 가설을 쉽게 설명하면

SSVEP에서는 사용자가 특정 주파수로 깜빡이는 표적을 보면 EEG에 그 주파수와 고조파 성분이 나타난다. 그런데 사람, 장비, 전극 위치·재질, reference, sampling rate가 바뀌면 같은 표적을 보더라도 관측되는 EEG 분포가 달라진다.

이 연구의 아이디어는 EEG만 보는 모델에 다음과 같은 **획득 조건 메모**를 함께 주는 것이다.

- 어느 채널이 실제로 존재하는가
- sampling rate와 window 길이는 얼마인가
- reference와 전극 유형은 무엇인가
- impedance처럼 물리적으로 이전 가능한 측정 특성이 무엇인가

모델이 이 메모를 사용해 “표적 때문에 생긴 신호”와 “측정 조건 때문에 생긴 변화”를 더 잘 분리한다면, 새 사용자나 새 장비에서도 별도 교정 없이 더 잘 작동할 수 있다는 가설이다.

### 권장 주가설

> H1: 현재 분류할 query 외에는 target 사용자의 labeled/unlabeled EEG나 target-set 통계를 adaptation에 사용하지 않는 inductive k=0 조건에서, transferable structured metadata를 사용하는 A2가 동일 backbone의 EEG-only A0보다 held-out target-subject balanced accuracy를 높인다.

주가설의 핵심 비교는 **A2 대 A0**이다. A4 대 A0만 비교하면 metadata, adapter, consistency, stochastic latent의 효과가 한꺼번에 섞인다.

### 보조가설

| 가설 | 검증 비교 | 반증되는 경우 |
|---|---|---|
| H2 robustness | A3/A4와 A0의 clean→perturbed 성능 하락량 | clean 평균만 오르고 채널 손실·잡음·reference 변화의 하락률은 줄지 않음 |
| H3 calibration efficiency | 고정 query에서 k=0/1/3/5의 곡선 | query를 고정하면 이득이 사라지거나 support draw마다 방향이 뒤집힘 |
| H4 replication | Wang→BETA, BETA→Wang, wearable을 분리 보고 | 효과가 한 방향 또는 한 데이터셋에만 존재함 |
| H5 nuisance mechanism | latent/condition representation의 domain·condition 및 class probe | nuisance보다 class label을 강하게 담거나 `z=0` 추론에서 효과가 사라짐 |
| H6 capability expansion | subject-condition cell의 learned−forgotten 수 | 평균 BA는 오르지만 새로 성공한 cell보다 새로 실패한 cell이 같거나 많음 |

H5가 검증되기 전에는 `latent nuisance`보다 **stochastic auxiliary latent**라는 명칭이 안전하다.

## 3. 현재 설계에서 유지할 부분

- Wang→BETA와 BETA→Wang을 별도 방향으로 평가한다.
- source validation으로 checkpoint를 선택하고 target test를 early stopping에 사용하지 않는다.
- `dataset_id::subject_id` 단위 split과 label-frequency canonical mapping을 둔다.
- A0–A4의 단계적 ablation으로 EEG-only, metadata, adapter/consistency, latent를 분리하려 한다.
- channel 수, noise, reference, sampling, metadata missing/shuffle 등 실제 실패 조건을 포함한다.
- k=0을 핵심으로 두고 k=1/3/5 few-shot calibration과 구분한다.

이 골격은 좋다. 문제는 구현된 일부 비교가 그 의도를 아직 충족하지 못한다는 점이다.

## 4. 최초 감사에서 발견한 P0 문제와 현재 조치

아래 P0-1~P0-5의 “현재 구현” 표현은 최초 감사·재정독 시점의 historical 상태다. 수정안은 2026-08-29의 0.3/0.4-dev 구현에 반영됐다.

### P0-1. unseen target category가 학습되지 않은 임의 embedding을 사용한다

전역 vocabulary에는 `wang`, `beta`, `wet`, `dry`, hardware ID가 미리 들어 있다([constants.py](../src/cfeg/constants.py)). 그러나 source-only 학습에서는 source에 등장한 embedding 행만 gradient를 받는다. condition encoder는 시험 시 target ID의 행도 그대로 조회한다([condition_encoder.py](../src/cfeg/models/condition_encoder.py)). 따라서 target dataset이나 hardware embedding은 의미를 학습한 적 없는 임의 벡터일 수 있다.

이 문제는 특히 dataset-ID-only A1을 해석 불가능하게 만든다. source dataset ID는 학습 중 상수이고 target ID는 미학습 상태이므로, A1의 성능은 “dataset metadata의 가치”가 아니라 임의 embedding 주입의 결과가 될 수 있다.

수정:

1. primary A2에서는 `dataset_id`와 opaque `hardware_id`를 제거한다.
2. training에 categorical dropout을 넣어 명시적인 `UNK` embedding을 학습한다.
3. 시험에서 학습 중 한 번도 보지 못한 category는 반드시 `UNK`로 보낸다.
4. unseen electrode 자체를 일반화하려면 이름이 아니라 impedance, 좌표, 채널 존재 mask, 접촉 방식 등 물리적 descriptor를 사용한다.
5. A1은 주효과 모델이 아니라 negative/diagnostic control로만 둔다.

현재 조치: vocabulary를 source training partition에서만 만들고 `UNK=0`을 training-only categorical dropout으로 학습한다. Primary A2는 `dataset_id`/opaque hardware ID를 제외한다.

### P0-2. k=0/1/3/5가 서로 다른 시험 문제를 푼다

현재 `_subject_calibration_indices`는 각 class에서 앞의 k개를 support로 빼고 나머지를 모두 evaluation으로 사용한다([eval_loop.py](../src/cfeg/eval_loop.py)). 따라서 k가 커질수록 query set이 작아지고 구성도 바뀐다.

수정:

1. subject×class별 **fixed query block**을 먼저 잠근다.
2. support는 query와 겹치지 않는 별도 block에서만 추출한다.
3. 모든 k와 모든 모델이 완전히 같은 query trial을 평가한다.
4. support draw는 여러 번 반복하되 query는 고정한다.
5. k=0 결과도 동일 query에서 계산해 calibration curve의 공정한 기준점으로 쓴다.

현재 조치: maximum-k support pool과 분리된 fixed query를 먼저 잠그고, 모든 k의 partition CSV·query hash·prediction을 저장한다.

### P0-3. 현재 `FBCCA`는 FBCCA가 아니다

현재 구현은 EEG를 flatten하고 harmonic reference 행을 평균한 뒤 dot-product를 계산한다([fbcca.py](../src/cfeg/baselines/fbcca.py)). canonical correlation 최적화도 없고 filter bank도 없다. 이 결과를 FBCCA라는 이름으로 논문에 넣으면 기준선의 타당성이 깨진다.

수정:

- 정식 CCA와 FBCCA를 구현하고 원 논문의 sub-band weighting 및 harmonic 설정을 기록한다.
- 입력 window, latency offset, 채널, 전처리를 neural model과 정확히 맞춘다.
- unit test에서 알려진 합성 주파수의 정답 class와 reference implementation의 score를 확인한다.
- k>0에는 eTRCA/msTRCA 또는 transfer-template 계열을 추가한다.

현재 조치: regularized CCA와 실제 sub-band filtering/가중 canonical-correlation FBCCA 및 합성 정답 테스트를 구현했다. transfer-template 계열은 확증 비교의 미완료 항목이다.

### P0-4. wearable robustness가 held-out test subject만 평가하지 않는다

`_scenarios`의 channel stress와 robustness는 manifest 전체의 `all_indices`를 사용한다([eval_loop.py](../src/cfeg/eval_loop.py)). 하나의 wearable manifest에 train/val/test subject가 함께 있으면 학습·검증 subject까지 robustness 결과에 섞인다.

수정:

- 모든 robustness 함수가 명시적인 `base_query_indices`를 필수 입력으로 받게 한다.
- source split의 `split.test`, cross-dataset target lockbox, few-shot fixed query 외에는 평가하지 못하게 assertion을 둔다.
- 결과 행마다 split hash와 query-set hash를 저장한다.

현재 조치: checkpoint 옆 `split.csv`의 test sample ID 또는 명시적 target filter가 없으면 평가를 거부하며, query identity를 artifact에 기록한다.

### P0-5. wearable impedance가 반대 electrode condition에 연결된다

Figshare version 4의 실제 `Impedance.mat` shape는 `(8,10,2,102)`이다. 전체 평균을 직접 계산하면 axis index 0은 261.6749 kΩ, index 1은 19.6317 kΩ이다. 원 논문 Figure 9의 dry 261.67 kΩ, wet 19.63 kΩ와 대조하면 실제 순서는 `[dry, wet]`이다.

그러나 현재 [wearable.yaml](../configs/data/wearable.yaml)은 `impedance_electrode_types: [wet, dry]`로 선언하고 [prepare_wearable.py](../src/cfeg/data/prepare_wearable.py)는 해당 순서의 index를 EEG condition에 붙인다. [test_wearable.py](../tests/test_wearable.py)도 이 잘못된 순서를 정답으로 고정한다.

따라서 현재 dry EEG에는 wet impedance, wet EEG에는 dry impedance가 결합된다. 기존 impedance-conditioned processed asset과 A2 pilot은 연구 증거로 사용할 수 없다.

현재 조치: config/default/test를 `[dry, wet]`로 수정하고 261.6749/19.6317 kΩ numeric signature, per-channel NaN 위치 보존, headband order 53/49, `wearable_v3` asset/checkpoint revision guard를 구현했다. 위 문단의 “현재”는 무효 `wearable_v2`에만 해당한다.

필수 수정:

1. 실제 순서를 `[dry, wet]`로 바꾼다.
2. shape만 검사하지 말고 paper-reported 261.67/19.63 kΩ signature를 검사한다.
3. per-channel impedance와 condition별 분포를 전처리 artifact에 보존한다.
4. 새 dataset revision으로 전처리하고 기존 결과를 명시적으로 무효화한다.

## 5. P1 설계 문제

### 5.1 wearable dry↔wet은 joint generalization이 아니었다

현재 `dry→wet` 설정은 전극 조건을 나누지만([wearable_dry_to_wet.yaml](../configs/train/wearable_dry_to_wet.yaml)), split 함수는 condition filter 후 source train/val만 subject별로 나누고 target condition 전체를 test로 둔다([splits.py](../src/cfeg/data/splits.py)). 같은 participant가 source dry와 target wet에 나타날 수 있으므로 “새 사람과 새 전극을 동시에 일반화”했다고 말할 수 없다.

두 실험으로 분리한다.

- **Subject DG:** participant-disjoint train/val/test를 먼저 정하고, train participant에게서는 wet/dry를 모두 보여 준다. held-out participant의 wet/dry 성능과 model×electrode interaction을 본다.
- **Joint subject-condition DG:** source participant의 wet만 학습하고 완전히 다른 target participant의 dry를 평가한다. 역방향도 별도로 수행한다.

현재 조치: participant를 먼저 train/validation/test로 분할한 뒤 source condition의 train/validation과 disjoint target participant의 target condition test를 만드는 joint split을 구현했다. 다만 dry-only source에서 wet category는 `UNK`가 되므로 이 실험은 “wet 의미를 metadata로 학습했다”는 검증이 아니라 **unseen metadata level extrapolation stress**로 해석한다.

### 5.2 한 개의 2 s window와 theoretical ITR만으로는 실용성을 말하기 어렵다

wearable 설정은 0.64 s가 window가 아니라 시작 offset이고 실제 window는 2.0 s이다([wearable.yaml](../configs/data/wearable.yaml)).

수정:

- 최소 0.3/0.5/0.7/1.0/1.5/2.0 s의 accuracy–latency curve를 권장한다. 직접 경쟁 연구와 맞추려면 0.3–0.7 s 구간이 중요하다.
- 하나의 primary window를 사전지정하고 나머지는 secondary curve로 둔다.
- 자극 시간만 넣은 값은 `theoretical ITR`, cue·gaze shift·feedback·adaptation 시간을 포함한 값은 `practical ITR`로 표기한다.
- 온라인 검증 전에는 “real-time usable” 대신 “offline latency estimate”라고 쓴다.

### 5.3 피험자가 통계 단위여야 한다

trial을 합쳐 하나의 accuracy와 p-value를 계산하면 표본 수를 과장한다.

- primary endpoint: target subject별 balanced accuracy의 A2−A0 paired difference.
- subject 안에서 seed와 support draw를 먼저 평균하거나, subject/seed/support를 보존하는 hierarchical bootstrap을 사용한다.
- 최소 3개, 가능하면 5개 training seed를 사용한다.
- 95% CI, paired permutation 또는 Wilcoxon signed-rank를 함께 보고한다.
- ablation·window·perturbation 다중 비교에는 Holm correction을 적용한다.
- MCID와 primary contrast는 target 결과를 보기 전에 정한다.
- 최종 표에는 평균뿐 아니라 median, subject distribution, worst-group, 실패 subject 수를 함께 제시한다.

정확한 power는 trial 수가 아니라 **subject별 paired difference의 분산**으로 계산해야 한다. 먼저 소규모 pilot로 이 분산을 추정한 뒤 confirmatory power analysis를 갱신한다.

## 6. “calibration-free” protocol을 네 종류로 분리할 것

최근 문헌은 서로 다른 target-data 접근을 모두 calibration-free라고 부른다. 한 표에 섞으면 공정하지 않다.

| protocol | target label | target EEG 접근 | update | 이 연구에서의 위치 |
|---|---:|---:|---:|---|
| Inductive zero-calibration | 없음 | 현재 query만; target-set 통계 없음 | 없음 | **primary k=0** |
| Transductive unsupervised | 없음 | test batch/target covariance 사용 | 정렬 또는 추론 전 적응 | 별도 보조 실험 |
| Online pseudo-label adaptation | 없음 | 과거 target stream 사용 | 매 trial 또는 누적 update | 별도 online-adaptive 비교 |
| Supervised few-shot | 있음 | class당 k개 support | adapter/head/full update | k=1/3/5 |

모든 결과표에 다음을 명시한다.

- target label 사용량
- target unlabeled trial 사용량
- batch 전체를 미리 보는지 또는 causal stream인지
- update되는 parameter
- adaptation 시간과 실패 시 rollback 규칙

## 7. 최근 직접 관련 연구가 설계에 주는 영향

### 7.1 반드시 넣을 직접 기준선

1. **Fast SSVEP Detection Using a Calibration-Free EEG Decoding Framework** (Wang et al., arXiv:2506.01284, 2025)
   - training-only Inter-Trial Remixing, per-input Context-Aware Distribution Alignment, adaptive spectral denoising을 결합한다.
   - Benchmark, BETA, Nakanishi에서 within-dataset LOSO와 0.3–0.7 s를 평가한다.
   - 직접적인 learned inductive calibration-free 기준선이다.
   - 다만 train-one-dataset→test-another-dataset은 하지 않아 califreeEEG의 더 강한 cross-dataset 목표와는 구분된다.
   - [arXiv](https://arxiv.org/abs/2506.01284)

2. **Cross-domain Correlation Analysis to Improve SSVEP Signals Recognition** (Hu et al., Biomedical Physics & Engineering Express, 2026)
   - source-subject selection, Euclidean alignment, CCA/TRCA, weighted fusion을 사용한다.
   - alignment가 target의 unlabeled covariance를 이용하는지와 사용 시점을 원문으로 확인한 뒤 inductive 또는 transductive 표에 배치해야 한다.
   - [DOI](https://doi.org/10.1088/2057-1976/ae2772)

3. **Online Adaptive Spatio-Temporal CCA** (OAST-CCA, BSN 2026)
   - pseudo-labeled online target trials로 spatio-temporal filter를 적응한다.
   - manual calibration은 없지만 frozen k=0과 같지 않으므로 online-adaptive protocol로 별도 비교한다.
   - [OpenReview](https://openreview.net/forum?id=2dcrEoX8Zo)

최소 기준선 묶음은 다음과 같다.

- Inductive k=0: CCA, 정식 FBCCA, compact scratch A0/A2, protocol-matched DG-Conformer 계열 port, Fast SSVEP. REVE는 target-exposure/license를 표시한 secondary다.
- Transductive: Euclidean-alignment/Correlation-Analysis 계열.
- Online: OACCA/OAST-CCA.
- k>0: eTRCA/msTRCA, LST/template-transfer 계열, adapter/LoRA/full fine-tuning.

모든 방법을 억지로 동일 ranking에 넣기보다 protocol별 표를 만들고, 공통 자원점에서만 비교한다.

후속 공개-code 감사에서 TFA-Net runner는 sample-level shuffled K-fold, TBMSCCN runner는 개인별 block split, SSER runner는 무작위 복원·중첩 test crop을 사용하는 것으로 확인됐다. 이 세 공개 runner는 strict participant-disjoint fixed-query headline baseline에서 제외한다. DG-Conformer는 target 접근이 가장 가깝지만 공개 구현이 Benchmark 0.8 s에 한정되고 license가 없어, 허가 또는 독립 port 후 공통 protocol로 실행한다. TST-CSFR는 원문 수식/공식 코드 확보 전 literature-only다.

## 8. 요청 논문 arXiv:2608.11829는 관련 연구인가?

**직접 EEG 관련 연구로는 아니다. 평가 설계의 비유로는 유용하다.**

[Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling](https://arxiv.org/abs/2608.11829)은 LLM on-policy distillation 논문이다. 저자들은 평균 성능과 적은 sampling budget에서는 좋아 보이지만, 큰 K에서의 `pass@K`와 문제별 retained/learned/forgotten 분석에서는 실제 capability boundary가 확장되지 않을 수 있음을 보인다.

이 연구에 가져올 수 있는 것은 다음 질문이다.

> metadata 모델이 이미 쉬운 subject-condition에서 평균만 올렸는가, 아니면 EEG-only가 실패하던 새로운 subject-condition을 실제로 성공으로 바꿨는가?

권장 분석:

1. 단위를 target `subject × dataset/electrode condition × window × perturbation severity` cell로 정한다.
2. target 결과를 보기 전에 operational success threshold `τ`를 사전지정한다.
3. A0와 A2/A3/A4를 비교해 각 cell을 분류한다.
   - retained-success: 둘 다 `BA ≥ τ`
   - learned: A0는 실패, metadata 모델은 성공
   - forgotten: A0는 성공, metadata 모델은 실패
   - retained-failure: 둘 다 실패
4. `net expansion = learned − forgotten`과 각 비율의 paired/bootstrap CI를 보고한다.
5. 평균 BA가 올라도 net expansion이 0 이하이면 “일반화 범위 확장”이 아니라 “기존 성공 영역의 효율 향상”으로 해석한다.

중요한 비유의 한계가 있다. LLM 논문의 K는 **같은 문제에서 뽑는 독립적인 stochastic inference sample 수**이고, 이 저장소의 k는 **labeled target calibration trial 수**이다. 둘을 같은 축이나 같은 metric으로 매핑하면 안 된다. EEG에 `pass@K`를 그대로 도입할 이유도 없다. 이 논문은 related-work novelty 표가 아니라 evaluation rationale 또는 discussion에 짧게 인용하는 편이 맞다.

### 8.1 현재 연구의 OOD는 무엇이고, 무엇이 아닌가

현재 저장소의 주된 연구는 **closed-set domain OOD generalization**이다. 표적 class/frequency는 이미 정해져 있고, 사람·데이터셋·장비·전극·reference가 바뀌어 입력 분포만 달라지는 상황을 다룬다.

| OOD 종류 | 예 | 현재 설계 |
|---|---|---|
| Subject OOD | 처음 보는 사용자 | 포함 |
| Acquisition OOD | 처음 보는 장비·전극·reference | 의도는 포함하지만 protocol 보강 필요 |
| Compositional OOD | 본 적 없는 `사용자×전극×장비` 조합 | 현재 불충분; 명시적 holdout 필요 |
| Corruption OOD | 채널 손실, noise, metadata missing | robustness로 포함하되 held-out query만 써야 함 |
| Class/frequency OOD | 학습하지 않은 자극 주파수 | 포함하지 않음; fixed 40/12-class head로는 발견 불가 |
| Continual OOD | 운용 중 새 domain을 감지하고 스스로 적응 | 포함하지 않음; 현재 k=0 모델은 target에서 update되지 않음 |

따라서 현재 모델이 target BETA에서 잘 맞히더라도 “새 주파수를 발견했다”거나 “새 domain을 학습했다”고 말할 수 없다. 정확한 표현은 **source에서 배운 규칙을 새 domain에 일반화했다**이다.

### 8.2 논문의 문제의식을 학습 과정에도 반영하는 방법

최종 평균만 비교하면 metadata 모델이 정말 새 OOD 영역을 얻었는지, 이미 잘하던 영역의 confidence만 높였는지 알 수 없다. 이를 학습 과정에서 확인하려면 다음 설계가 필요하다.

1. 먼저 공통 A0 EEG-only checkpoint를 source multi-domain 데이터로 학습한다.
2. 같은 A0에서 출발해 metadata module을 붙인 A2/A3를 학습한다. 별도 random initialization 결과도 보조 실험으로 남긴다.
3. source domain 중 일부 subject·condition 조합을 **development pseudo-OOD**로 처음부터 제외한다.
4. training checkpoint마다 ID validation과 development pseudo-OOD를 함께 측정한다.
5. 각 checkpoint에서 pseudo-OOD cell을 retained-success, learned, forgotten, retained-failure로 분류한다.
6. ID 평균은 오르지만 `learned−forgotten` 또는 worst-group OOD가 나빠지는 시점이 있는지 추적한다.
7. checkpoint와 hyperparameter 선택은 이 development OOD까지만 사용하고, BETA 또는 제3 데이터의 final lockbox는 마지막에 한 번만 평가한다.

이 설계는 LLM 논문의 “학습이 진행될수록 작은 예산 성능은 좋아지지만 capability boundary는 줄 수 있다”는 관찰을 EEG에 맞게 바꾼 것이다. 여기서 보는 것은 `pass@K`가 아니라 **학습 step에 따른 OOD coverage와 forgetting trajectory**다.

### 8.3 실제 capability expansion을 목표로 한 학습 변경

단순 cross-entropy 평균 최소화만 사용하면 큰 source group이 학습을 지배할 수 있다. capability expansion을 주목적으로 삼는다면 다음을 독립 ablation으로 시험할 수 있다.

- **Multi-source factorized training:** dataset ID를 외우게 하지 않고 채널 좌표, electrode type, reference, sampling, impedance 등 이전 가능한 factor를 사용한다.
- **Compositional holdout curriculum:** wet/dry, channel 수, reference, subject의 일부 조합을 의도적으로 숨기고 보지 못한 조합으로 일반화하게 한다.
- **Worst-group objective:** 평균 loss와 함께 가장 약한 source-condition group의 loss를 줄인다. 평균 상승이 쉬운 group에만 집중되는 것을 막는 목적이다.
- **Retention anchor:** A0가 맞히던 source/dev-OOD cell에서 A2가 무너지지 않도록 logit consistency나 replay anchor를 둔다. 이는 A0→A2의 순차학습일 때만 ‘forgetting 방지’로 해석한다.
- **Metadata masking/UNK training:** metadata 일부를 의도적으로 누락하고 unseen category는 학습된 `UNK`로 처리한다.
- **Mechanism probes:** condition representation이 acquisition factor는 예측하지만 stimulus class를 과도하게 예측하지 않는지 확인한다.

권장 목적함수의 개념적 형태는 다음과 같다.

`L = L_classification + λ_cons L_view-consistency + λ_worst L_worst-group + λ_ret L_retention`

모든 항을 한 번에 넣어 최종 모델 하나만 비교하면 다시 원인을 알 수 없다. A0→metadata→worst-group→retention 순서의 ablation이 필요하다.

### 8.4 모델이 OOD를 ‘발견하고 적응’하게 하려면 별도 연구 단계가 필요하다

운용 중 처음 보는 domain을 모델이 감지하고 스스로 학습하는 것을 의미한다면, 현재 연구보다 범위가 넓다. 다음 phase로 분리하는 것이 안전하다.

1. **OOD detection:** prediction entropy, energy 또는 embedding distance로 현재 trial/domain이 source support 밖인지 판단한다.
2. **Selective prediction:** OOD score가 높으면 억지로 명령을 출력하지 않고 abstain/recalibration을 요청한다.
3. **Unlabeled online adaptation:** 과거 target stream만 사용해 normalization, adapter 또는 prototype을 갱신한다. 현재 trial을 먼저 채점한 뒤 update해야 causal하다.
4. **Safety gate:** pseudo-label confidence가 낮거나 source-anchor 성능이 떨어지면 update를 중단·rollback한다.
5. **Continual metrics:** adaptation 전 성능, target history에 따른 학습 속도, old-domain forgetting, learned/forgotten cell, adaptation 시간과 abstention rate를 보고한다.

이 단계는 더 이상 엄격한 frozen inductive k=0과 같지 않다. `online calibration-free` 또는 `unsupervised continual adaptation`으로 별도 표와 별도 주장을 사용해야 한다.

### 8.5 정말 ‘새 frequency/class 발견’을 연구하려는 경우

현재 40-class/12-class head는 학습 때 정의되지 않은 새 frequency를 출력할 수 없다. 이를 연구하려면 별도 구조가 필요하다.

- class ID head 대신 candidate frequency/phase를 query로 받는 frequency-conditioned scorer를 사용한다.
- 일부 stimulus frequency와 phase를 학습에서 완전히 제외한다.
- interpolation frequency와 범위 밖 extrapolation frequency를 구분해 평가한다.
- 기존 class 정확도뿐 아니라 unknown-frequency detection과 frequency estimation error를 보고한다.
- CCA/FBCCA처럼 원래 unseen candidate frequency를 reference signal로 평가할 수 있는 방법과 비교한다.

이것은 acquisition metadata OOD보다 더 강한 **open-class SSVEP generalization** 연구이며, 현재 논문의 주질문에 넣으면 범위가 크게 늘어난다. 우선 compositional acquisition OOD를 확증한 뒤 후속 연구로 분리하는 편이 현실적이다.

## 9. 수정된 최소 실험설계

| 실험 | split | 비교 | target-data 권한 | primary 출력 |
|---|---|---|---|---|
| E0 code/data sanity | synthetic와 BETA/wearable 소수 subject | CCA, FBCCA, F0, A0, A2 | 현재 query만 | schema·label·pipeline 정답성 |
| E1 **primary wearable subject DG** | 사전 할당 39명 fixed-epoch train → 독립 60명 one-shot lockbox | parameter-matched 3-seed A0 vs A2 ensemble | 현재 query만 | N=60 subject BA paired difference; condition별 descriptive |
| E2 wearable robustness | E1의 동일 held-out query | A0 vs A3 및 missing/shuffle control | 현재 query만 | clean 대비 하락, worst-group, learned/forgotten |
| E3 joint subject-condition stress | source-participant dry → disjoint target-participant wet 및 역방향 | A0 vs A2 | 현재 query만 | unseen-level 성능; 방향별 독립 결과 |
| E4 acquisition replication | Wang/BETA/Dong2023 중 source-only train/validation → held-out acquisition | CCA, generic/paper FBCCA, F0, A0, A2 | 현재 query만 | dataset별 subject BA paired effect와 interaction |
| E5 boundary diagnostic | Wang→BETA와 BETA→Wang | A0 vs A2 | 현재 query만 | 동일 metadata에서 A2 이득은 주장하지 않음 |
| E6 transductive/online | E1–E5의 target stream | EA 계열, OACCA/OAST-CCA | unlabeled target 사용 | 별도 protocol의 성능 대 history·시간 |
| E7 few-shot | 고정 query + disjoint support | k=0/1/3/5, eTRCA/LST, adapter/full FT | labeled support | calibration curve와 support variance |
| E8 external lockbox | 모든 선택을 동결한 뒤 제3 데이터/OpenBCI | 최종 A0 vs 최종 metadata 모델 | 현재 query만 | 외부 재현 여부 |

### 공통 고정 규칙

- target test는 model selection, threshold 설정, normalization fit에 사용하지 않는다.
- 같은 실험의 모든 모델은 같은 subject, trial, window, perturbation seed를 사용한다.
- optimization seed와 `data.split_seed`를 분리하고, 모든 model seed가 동일 outer fold를 사용한다.
- class 수가 다르면 shared frequency subset 결과와 dataset-specific head 결과를 구분한다.
- A0와 A2는 같은 `physical_hybrid_v1` graph·parameter schema·초기 state·backbone·optimizer·schedule·query QC를 사용하고, `external_metadata_mode=null|observed`만 다르게 한다. 별도 학습된 두 모델의 성능 동일성을 요구하는 것이 아니라 학습 전 공정성과 정보권한을 맞춘다.
- model selection은 source-domain validation의 사전지정 metric으로만 한다.
- 모든 split/query/support manifest와 hash를 결과 artifact에 저장한다.
- failed run과 excluded subject의 사유를 숨기지 않는다.

## 10. 데이터 입수 가능성

| 자산 | 접근성 | 설계상 역할 |
|---|---|---|
| Wang2016 | 원 배포/MOABB로 공개 획득 가능 | 35명, 40 targets; BETA와의 external-domain boundary |
| BETA | Figshare CC BY 4.0, 약 4.9 GB | 70명, 40 targets; Wang과의 external-domain boundary |
| Wearable wet/dry | Figshare CC BY 4.0, 약 1 GB | **primary**; 102명, 8채널, 12 targets, wet/dry 및 impedance |
| [Dong2023](https://www.nemar.org/dataset/nm000128) | NEMAR CC BY-NC 4.0/[Zenodo mirror](https://zenodo.org/records/18847318), 약 0.7–1 GB | 59명, exact 40 targets; 8채널·Fp1·semi-dry acquisition replication |
| [Nakanishi 12JFPM](https://github.com/mnakanishi/12JFPM_SSVEP) | 공개 author repository, 약 153 MB; data license 불명확 | exact wearable 12 targets; 허가 확인 후 secondary replication |
| REVE base | Hugging Face gated access | montage-aware frozen backbone 후보; SSVEP zero-shot 성능은 별도 검증 필요 |
| 제3 SSVEP/OpenBCI | 공개 데이터 또는 신규 수집 필요 | 모든 설계 선택 후 external lockbox |

공개 원자료는 대체로 확보 가능하다. P0 mapping, 전체 wearable_v3 acceptance, physical 0.4-dev 경로와 reveal #2 승인·margin 동결은 완료됐다. 현재 첫 병목은 **clean annotated source에서 exact 18-job+single reveal transaction을 실행·판정하는 것**이다. 그 다음이 confirmatory source freeze, Nakanishi data license, protocol-matched baseline, REVE 접근·weights 조건, 대용량 반복 실행이다.

## 11. 구현·실행 우선순위

1. [x] unseen category를 trained `UNK`로 처리하고 opaque ID를 primary A2에서 제거한다.
2. [x] fixed query/support manifest를 만들어 k별 시험 집합을 동일하게 한다.
3. [x] 정식 CCA/FBCCA를 구현하고 synthetic/reference test를 추가한다.
4. [x] robustness가 held-out query 밖의 index를 받을 수 없게 한다.
5. [x] wearable split을 subject DG와 joint subject-condition DG로 분리한다.
6. [x] A0/A2 parameterization과 평가 artifact를 맞추고 OOD coverage 분석을 구현한다.
7. [x] 역사적 5-fold 구현과 Dong2023 S1 adapter pilot을 검증했다. Primary inference에는 쓰지 않으며 39/60 lockbox가 대체한다.
8. [x] **P0 code:** impedance order `[dry, wet]`, numeric-signature test, `wearable_v3` revision guard를 구현한다.
9. [x] Protocol 0.4-dev의 공통 `m_struct`, external-only treatment, A0/A2 공통 query-local QC를 구현한다.
10. [x] 전체 `wearable_v3` acceptance와 per-channel `m_quality` physical mechanism을 구현한다.
11. [재승인·기준 동결 / exact restart 대기] Clean tagged source에서 exact 18-job development와 **각 fold당 Full A2 intervention bundle 하나, 총 세 개**, aggregate와 10-gate 결과를 reveal #2로 한 번 공개한다 (`DEC-20260901-004`).
12. [ ] Fast SSVEP, 허가/독립 port된 DG-Conformer 계열을 protocol-matched로 재현한다. Paper-faithful FBCCA는 구현 완료다.
13. [ ] Confirmatory SESOI·alpha/alternative·operational threshold·multiplicity와 fairness/source lock을 잠근 뒤 wearable 39-train/60-lockbox primary를 실행한다. Physical margin과 `development_gate_only`는 이미 동결됐고, Dong/5-fold는 별도 replication/exploratory다.

## 12. 주장 가능한 범위

수정 전에는 다음 정도만 주장할 수 있다.

> metadata-conditioned calibration-free SSVEP generalization을 시험하기 위한 구현과 파일럿 protocol을 제안했다.

P0–P1 수정과 E0–E7 후에는 다음 주장을 검토할 수 있다.

> transferable acquisition metadata가 동일 backbone의 EEG-only 모델보다 held-out subject/dataset/condition에서 평균 및 worst-group 성능을 개선했고, 그 효과가 양방향·다중 seed·고정 query에서 재현되었다.

`learned > forgotten`, joint subject-condition 재현, external lockbox까지 충족해야 다음의 더 강한 표현이 가능하다.

> metadata conditioning이 calibration-free SSVEP의 적용 가능한 subject-condition 범위를 확장했다.

## 13. 검토 한계

이 문서는 코드와 설계의 audit 및 protocol amendment이지 physical A2 성능 결과 보고서가 아니다. 전체 wearable 원자료는 전수 무결성 감사했다. Physical six-role 최초 실행은 private prediction/intervention 계산 뒤 공개 전 validator에서 중단됐고 성능 outcome을 공개·검토하지 않은 채 격리했으므로 metadata 효과 크기는 아직 알 수 없다. 39명 confirmatory training과 60명 lockbox prediction도 실행하지 않았다. Strict Chen-2015 FBCCA는 구현했지만 모든 데이터셋 원 논문의 toolbox 설정까지 재현한 것은 아니다. 2026 OAST-CCA와 Cross-domain Correlation Analysis는 공식 metadata/abstract로 관련성을 확인했지만 PDF 접근 실패로 target-data timing과 세부 통계는 provisional이다. arXiv:2608.11829와 Fast SSVEP Detection은 PDF 원문을 선택 구간까지 확인했다. 따라서 검색은 최신 직접 경쟁 연구를 추가한 evidence-grounded review이지만 완전한 systematic review로 주장하지 않는다.
