# Metadata 활용 문헌검토와 Protocol 0.4 수정안

기준일: 2026-08-29

상태: **Protocol 0.4-dev reveal #2 완료 — valid assay / diagnostic no-go / confirmatory 차단**

연구주제: 변경하지 않음

> **구현·결과 갱신:** 배포 문구와 모순되는 Figshare v4 impedance numeric order를 `[dry, wet]`으로 교정하고, signature·per-channel 보존·headband parsing·`wearable_v3` deep receipt/revision guard를 구현했다. 전체 102명·24,480행 acceptance audit도 통과했다. `physical_hybrid_v1`은 공통 query-QC FiLM, external global FiLM, 채널별 impedance gain, exact-null routing으로 구현됐다. `DEC-20260901-004`의 exact restart는 완료됐지만 valid N=3 assay에서 A0 `0.1292`, Full A2 `0.1194`, Δ `−0.0097`이고 네 substantive gate가 실패했다. 현 candidate의 confirmatory는 차단한다. 상세 상태는 [Protocol 0.4-dev 구현 기록](protocol_0_4_dev_implementation.md)을 따른다.

> **후속 DEC-20260830-002:** 아래의 N=99 5-fold primary는 target-free simulation에서 model-shared dependence와 Type-I inflation이 드러나 폐기됐다. 현재 primary는 39명 fixed training·60명 independent lockbox이며 [연구 프로토콜 현재 상태](research_protocol_status.md)를 우선한다.

> 구조화된 획득조건 metadata가 target 사용자 데이터로 적응하지 않는 strict inductive `k=0`에서 처음 보는 사용자·획득조건의 closed-set SSVEP 분류를 개선하는가?

## 1. 먼저 내리는 판정

현재 실험설계는 **physical development 실행과 판정이 끝났고, 그 결과 현 candidate는 confirmatory no-go다.** Protocol 0.3에서 연구 질문과 평가 골격을 고정했고, 0.4-dev에서 physical treatment의 표현·주입·반증 경로까지 구현했다. 두 번째 development reveal의 potency는 유효했지만 correct metadata의 방향성 이득과 pairing/counterfactual reliance를 보이지 못했다. Confirmatory SESOI 등은 여전히 미결정이지만 그것만 채워 no-go를 해제할 수 없다.

| 상태 | 항목 |
|---|---|
| 유지·동결 | closed-set SSVEP, strict inductive `k=0`, target-set 통계·adaptation 금지, wearable held-out-participant 평가, A0 대 A2, subject-level balanced accuracy, fixed query, source-only model selection |
| 구현·개발 기준 동결 | metadata schema, 구조 정보와 treatment의 경계, physical A2 주입, exact-null, block-coherent shuffle, metadata-only control, wrong-metadata intervention |
| 현 candidate no-go | clean direction, counterfactual reliance, inference pairing, training pairing gate 실패; 동일 S1–S3 재튜닝·추가 reveal 금지 |
| 새 연구에서 수치 결정 필요 | MCID, capability 성공 threshold `τ`, seed 수, 외부 lockbox 사용 횟수와 실패 판정 |
| 탐색으로 유지 | A3/A4, GroupDRO·selective consistency, LUPI teacher, 고용량 domain prompt/hypernetwork, LLM 유추 |

따라서 지금은 Protocol 0.3을 그대로 확증 실행할 때가 아니다. 같은 연구 질문을 유지하려면 독립 외부 development data나 새 사전근거로 별도 candidate와 gate를 정의해야 한다. 기존 결과에 맞춰 아래 0.4 기준을 바꾸지 않는다.

## 2. 연구 가설을 가장 쉽게 설명하면

SSVEP label을 `Y`, 뇌에서 생긴 반응을 `S`, 실제로 기록된 EEG를 `X`, 획득조건을 `C`라고 하자.

```text
표적 Y → 뇌 반응 S → 기록 EEG X
                       ↑
          reference·전극·위치·접촉·필터 C
```

모델은 `X`만 보고 `Y`를 맞힐 수도 있고, `X`가 어떤 조건 `C`에서 측정됐는지를 함께 보고 맞힐 수도 있다. A2의 가설은 다음과 같다.

> 같은 신호라도 어떤 센서 구조와 접촉·reference 조건에서 관측됐는지 알면, 모델이 획득조건 때문에 생긴 변형을 덜 헷갈려 처음 보는 사람에게 더 잘 일반화할 수 있다.

이때 metadata는 정답 힌트가 아니라 **측정 과정에 대한 사전 관측 가능한 설명**이어야 한다. dataset ID, subject ID, 파일명, trial 순서, 표적 주파수처럼 정답이나 데이터셋을 암기하게 하는 값은 primary A2에 넣지 않는다.

## 3. 0.4-dev 패치 전 구현에서 발견한 식별 문제

이 절은 수정 동기를 보존하는 historical audit다. 아래의 mixed continuous vector, A2 전용 channel prompt, 모든 metadata 강제 missing 문제는 최소 0.4-dev 경로에서 수정됐고 legacy checkpoint 경로에만 남아 있다.

### 3.1 패치 당시 A0와 A2의 차이

패치 당시 A0와 A2는 같은 condition encoder와 adapter를 두고 A0에서 모든 metadata를 missing으로 만드는 parameter-matched 비교였다. A2는 `reference`, `electrode_type`, `cap_type`, `reattach_flag`, 연속값, 채널 집합을 사용했다([ablation.yaml](../configs/train/ablation.yaml)).

패치 당시 condition encoder는 범주형 embedding의 합, 다섯 연속값과 missing mask의 MLP, 채널 ID embedding의 평균을 세 token으로 만든 뒤 작은 Transformer와 prompt projection으로 융합했다([condition_encoder.py](../src/cfeg/models/condition_encoder.py)). 당시 연속값은 processed sampling rate, 채널 수, impedance mean/max, 마지막 session 이후 시간이었다([collate.py](../src/cfeg/data/collate.py)).

### 3.2 구조 정보가 중복된다

- Tiny backbone은 이미 channel ID embedding과 channel mask를 사용한다([tiny_transformer.py](../src/cfeg/models/backbones/tiny_transformer.py)).
- REVE wrapper도 channel ID를 실제 전극 이름·위치에 매핑한다([reve.py](../src/cfeg/models/backbones/reve.py)).
- 따라서 A2가 같은 채널 집합을 global prompt에 다시 넣어 얻는 이득은 “새 geometry 정보”가 아니라 **중복 conditioning 경로의 효과**일 수 있다.

채널 이름·좌표·mask는 입력의 의미를 정의하는 구조 정보다. 이것을 A0에서는 숨기고 A2에만 주면 공정한 metadata 비교가 아니라 sensor representation 비교가 된다.

### 3.3 wearable primary가 실제로 시험하는 metadata는 좁다

현재 wearable pilot에서 변하는 정보는 사실상 다음뿐이다.

- `electrode_type`: wet/dry
- impedance mean/max: block에 따라 변함

reference, cap, processed sampling rate, 채널 수와 채널 집합은 상수이고, reattach와 elapsed time은 누락돼 있다. 따라서 현 H1이 직접 검증할 수 있는 주장은 넓은 “모든 acquisition metadata”가 아니라 다음이다.

> 알려진 electrode-interface 조건과 pre-query impedance가 unseen participant SSVEP 분류에 추가 정보를 주는가?

wearable의 impedance 그룹마다 12개 label이 모두 존재해 단순한 label lookup은 아니지만, impedance를 mean/max 두 값으로 압축하면 채널별 접촉 품질과 reference mismatch를 잃는다.

### 3.4 외부 acquisition에서 범주 의미가 사라진다

source-only vocabulary에서 보지 못한 Dong의 `Fp1` reference와 `pregelled_semidry` electrode는 trained `UNK`로 간다. 이 처리는 임의 미학습 embedding보다 안전하지만, **새 reference나 electrode의 물리적 의미를 extrapolate하지는 못한다.**

따라서 외부 H3에서 의미 있는 zero-shot metadata transfer를 주장하려면 범주 이름 대신 reference 위치·가중치, 접촉 매질·재료·active/passive 같은 분해된 물리 descriptor가 필요하다.

## 4. 인접 연구가 말해 주는 것

### 4.1 뇌과학·EEG acquisition

- SSVEP 검출률이 reference 선택에 따라 달라진 직접 결과가 있다([Wu & Su, 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4123903/)). reference는 장식용 metadata가 아니라 관측 파형을 정의하는 변환이다.
- cap을 다시 씌우면 전극 위치가 수 mm 이상 달라질 수 있고, 동일한 표준 이름도 같은 피질 위치를 보장하지 않는다([Atcherson et al., 2007](https://pubmed.ncbi.nlm.nih.gov/17929157/), [Scrivener & Reader, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8865144/)).
- wearable 102명 자료는 같은 사람·장치·protocol에서 wet/dry를 비교하는 현재 가장 좋은 paired contrast다([Zhu et al., 2021](https://www.mdpi.com/1424-8220/21/4/1256/html)). 다만 wet/dry는 품질의 순서척도가 아니라 서로 다른 interface다.
- impedance 영향은 amplifier와 온도·습도·처리에 따라 달라진다([Ferree et al., 2001](https://pubmed.ncbi.nlm.nih.gov/11222977/), [Kappenman & Luck, 2010](https://pubmed.ncbi.nlm.nih.gov/20374541/)). 따라서 `1/impedance`를 곧바로 신뢰도 가중치로 쓰면 안 된다.
- EEG-BIDS는 channel과 electrode를 구분하고 reference, sampling, filter, 좌표, electrode type/material, per-electrode impedance를 별도 필드로 기록한다([EEG-BIDS specification](https://bids-specification.readthedocs.io/en/v1.7.0/04-modality-specific-files/03-electroencephalography.html)). 이 구분을 metadata schema의 출발점으로 삼는 것이 타당하다.

이 문헌들은 acquisition condition이 신호에 영향을 준다는 근거다. 그러나 **그 metadata를 neural model에 넣으면 strict `k=0` SSVEP가 좋아진다는 직접 증거는 아니다.** 그 부분이 이 연구가 새로 검증할 내용이다.

### 4.2 신호처리

- 단위·resampling·filter처럼 알려진 deterministic transform은 먼저 표준화해야 한다.
- reference는 가능한 경우 올바른 re-reference 또는 reference-robust 표현으로 처리하고, 완전히 제거하지 못한 차이만 conditioner에 남긴다.
- bad channel과 reference는 독립 문제가 아니므로 quality mask와 reference 처리를 함께 평가한다([PREP pipeline](https://pmc.ncbi.nlm.nih.gov/articles/PMC4471356/)).
- target cohort covariance를 계산하는 Euclidean/Riemannian alignment는 label-free여도 strict inductive `k=0`가 아니라 transductive protocol이다([Euclidean Alignment](https://pubmed.ncbi.nlm.nih.gov/31034407/)).
- 결측 채널 보간은 항상 무해하지 않으므로 explicit mask와 직접 비교해야 한다([EEG interpolation distortion study](https://link.springer.com/article/10.1186/s40810-015-0009-5)).

즉 권장 원칙은 **preprocessing first, residual conditioning second**다.

### 4.3 EEG foundation model

- REVE는 실제 3D 전극 좌표와 시간을 4D positional encoding으로 사용해 임의 layout을 다루도록 설계됐다([REVE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/20a917f77773ac0fa8bea2bdd6606b66-Abstract-Conference.html)).
- BIOT도 채널별 tokenization과 channel embedding으로 서로 다른 채널 구성·길이·missing value를 다룬다([BIOT, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract.html)).
- 2026년 frozen EEG foundation model의 montage adapter 연구는 coordinate-interpolated channel embedding을 이용한 unseen montage/cross-dataset 처리를 보고했다([Ma & Ruotsalo, 2026](https://pubmed.ncbi.nlm.nih.gov/41997170/)). 논문이 SSVEP로 표기한 Facecat endpoint는 실제로 frequency-coded class decoding이 아니라 RSVP/P300 성격의 event segmentation이므로 canonical SSVEP zero-shot 증거로 사용하지 않는다. 그래도 좌표→pre-query channel embedding이라는 개념적 선행기술은 강하다.

이 결과는 geometry를 A2의 global prompt만으로 넣기보다 **모든 모델의 sensor representation에 넣으라**는 근거에 가깝다.

### 4.4 ML domain generalization과 metadata conditioning

- FiLM은 conditioning 정보로 feature별 affine transform을 만드는 저용량 방법이다([FiLM, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11671)). EEG 효과의 직접 근거는 아니지만 A0로 되돌아갈 수 있게 identity initialization한 primary conditioning baseline으로 적합하다.
- metadata를 이용한 image segmentation에서도 저비용 linear conditioning의 이득이 보고됐지만 다른 modality·task의 결과이므로 설계 유추로만 사용한다([Lemay et al., 2021](https://proceedings.mlr.press/v143/lemay21a/lemay21a.pdf)).
- DomainBed는 source-only model selection과 강한 ERM baseline의 중요성을 보여 준다([DomainBed](https://arxiv.org/abs/2007.01434)). A0를 약하게 만들고 A2만 복잡하게 만들면 metadata 효과를 주장할 수 없다.
- missing modality 연구는 multimodal model이 한 modality가 사라질 때 unimodal baseline보다도 악화될 수 있음을 보였다([Ma et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html)). metadata availability mask, field dropout, graceful degradation이 필요하다.
- D³G와 continuous DG는 domain descriptor로 predictor를 조절하지만, 많은 descriptor-diverse domain과 descriptor 근방성이 필요하다([D³G, ICLR 2024](https://openreview.net/forum?id=Dc4rXq3HIA), [Continuous DG, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/35c1d69d23bb5dd6b9abcd68be005d5c-Abstract-Conference.html)). 현재 acquisition regime 수로 고용량 hypernetwork를 primary로 쓰기에는 근거가 부족하다.
- LUPI는 metadata를 training 때만 쓰고 inference에서는 제거하는 별도 방법이다([Vapnik & Vashist, 2009](https://pubmed.ncbi.nlm.nih.gov/19632812/)). 배포 시 metadata가 자주 누락될 경우 secondary teacher-student ablation으로는 가치가 있다.

### 4.5 LLM 연구의 위치

[arXiv:2608.11829 v2](https://arxiv.org/abs/2608.11829)는 LLM on-policy distillation과 test-time sampling 연구다. EEG 방법의 직접 related work는 아니다. 차용할 것은 한 가지 평가 원칙뿐이다.

> 평균 성능이 올랐다고 이전에 못 풀던 OOD 조건을 새로 풀게 된 것은 아니다.

따라서 A0 실패→A2 성공인 `learned`, A0 성공→A2 실패인 `forgotten`, `learned − forgotten`, worst-condition을 함께 본다. LLM의 `pass@K`나 sampling `K`를 EEG의 calibration `k`와 동일시하지 않는다.

NLP에서는 domain feature로 prompt를 생성해 unseen domain을 다루는 PADA가 구조적으로 더 가까운 유추다([PADA, TACL 2022](https://aclanthology.org/2022.tacl-1.24/)). 그래도 text domain과 물리 acquisition은 다르므로 직접 근거가 아니라 factorized descriptor 실험을 만드는 아이디어로만 쓴다.

## 5. Protocol 0.4 metadata contract

### 5.1 `m_struct`: A0와 A2 모두가 받는 구조 정보

- 실제 channel mask와 canonical channel identity
- nominal 또는 digitized 3D coordinate, coordinate frame
- coordinate provenance: session digitized / subject template / manufacturer template / nominal / inferred
- processed sampling rate와 model time grid

이 정보는 EEG tensor의 축과 위치를 해석하는 데 필수다. **A2 treatment로 세지 않는다.** geometry 가치를 묻고 싶다면 별도의 geometry ablation을 둔다.

### 5.2 `m_acq`: A2의 전이 가능한 획득 descriptor

- acquisition reference mode, reference electrode 위치·좌표
- ground 위치
- offline re-reference transform과 원 reference 복구 가능성
- contact medium: gel/paste, saline, hydrogel/pregelled, dry-contact, capacitive
- electrode material·geometry·active/passive
- acquisition/distributed/model sampling rate를 분리
- hardware/software high-pass, low-pass, anti-alias, notch, power-line frequency

단순 `reference="Cz"`, `electrode_type="dry"` 하나보다 가능한 물리 구성요소로 분해한다. 알 수 없는 값은 추측하지 않고 field별 `UNK`와 availability mask로 둔다.

### 5.3 `m_quality`: query 전에 측정된 채널별 품질

- per-channel log impedance와 missing mask
- median, IQR, max, reference-channel mismatch
- impedance 측정 시각·주파수·pre/post 여부
- 장비가 제공한 bad/flat/clipping flag
- 현재 query에서 causal하게 계산 가능한 flatline·line-noise·broadband-artifact 지표

현재 query의 signal에서 계산한 QC는 signal-derived feature이므로 A0에도 같은 계산 권한을 준다. 실험 전에 측정된 impedance 같은 외부 정보만 A2 treatment로 분류한다.

### 5.4 모델 입력이 아니라 분석 층으로 둘 confound

- headband order, run order, fatigue
- age, sex, vision correction
- sleep, caffeine, medication
- gaze·attention 품질

이 값은 stratification, sensitivity analysis, mixed model covariate로 쓸 수 있지만 deployment에서 항상 관측되지 않으므로 primary A2 input에 넣지 않는다.

### 5.5 primary에서 금지할 proxy

- dataset/site ID
- subject/session/file ID
- opaque hardware serial/model ID
- stimulus label, frequency, phase, trial/order-derived label proxy
- target accuracy, post-hoc rejection 결과
- target batch/covariance/statistics

manufacturer/model은 관측 가능해도 dataset token이 되기 쉬우므로 physical spec으로 분해하고 ID 자체는 diagnostic control에만 쓴다.

## 6. metadata를 어디에 넣을 것인가

권장 구조는 하나의 거대한 metadata prompt가 아니라 다음 hybrid다.

1. 알려진 deterministic 차이는 preprocessing에서 처리한다.
2. channel 좌표·mask는 sensor token/positional encoding에 넣는다.
3. per-channel quality는 bounded reliability gate 또는 attention bias에 넣는다.
4. 보정 후 남는 global acquisition descriptor만 작은 residual FiLM/conditional LayerNorm에 넣는다.
5. residual branch는 zero/identity initialization해 metadata가 없으면 A0로 자연스럽게 돌아가게 한다.

현재 primary 후보 `physical_hybrid_v1`은 이 원칙대로 구현됐다. A0/A2는 동일 module graph·parameter schema·초기 state를 쓰고 `external_metadata_mode=null|observed`만 다르다. Global 경로는 electrode type과 block impedance mean/max를 residual FiLM으로, channel 경로는 채널별 impedance/availability를 bounded gain으로 주입한다. A0 또는 all-missing 입력은 exact identity/null로 돌아간다. 기존 prompt+Transformer condition encoder는 legacy secondary architecture ablation으로만 유지한다.

### preprocessing × fusion 2×2 development 실험

| | residual fusion 없음 | residual fusion 있음 |
|---|---:|---:|
| 최소 공통 preprocessing | B0 | F |
| context-aware preprocessing/QC | P | P+F |

- `P − B0`: 물리적 preprocessing의 효과
- `P+F − P`: preprocessing 후 metadata fusion의 순증분
- `P+F − F`: 둘의 상호보완성

`P+F − P`가 0이면 실패를 숨기지 않는다. 그 결과는 metadata의 역할이 learned prompt가 아니라 preprocessing에 있었다는 기전 발견이다.

## 7. 수정된 실험 순서

### D0. confirmatory 전 source-only development

1. Figshare v4 impedance mapping을 `[dry, wet]`로 수정하고 261.67/19.63 kΩ numeric signature test를 통과한다.
2. raw metadata provenance와 배포 시 observability를 필드별 감사한다.
3. wearable에서 per-channel impedance와 headband order를 보존한다.
4. `m_struct`, `m_acq`, `m_quality`, confound, forbidden proxy를 schema에서 분리한다.
5. participant 경계를 먼저 지키며 source primitive는 모두 보되 `electrode × impedance-bin × channel-mask` 일부 조합만 숨긴 compositional pseudo-OOD를 만든다.
6. [완료] Physical six-role(A0/full/global-only/channel-only/shuffle-train/metadata-only)×3-fold 18-job과 **각 fold당 Full A2 intervention bundle 하나, 총 세 개**를 사전 margin으로 판정했다.
7. [완료·no-go] `DEC-20260901-004`의 reveal #2로 결과를 한 번에 공개했다. Potency는 유효했지만 substantive gate 네 개가 실패했다. Prompt 방식은 legacy secondary 비교로만 두고 동일 S1–S3를 다시 조정하지 않는다.

DEC-20260830-001은 S1–S3를 영구 development-only로 정했다. S1–S3에서 본 v2 1-epoch 결과는 impedance가 잘못 결합된 무효 artifact였지만, 성능 노출 자체는 되돌릴 수 없다. DEC-20260830-002는 남은 S4–S102를 39명 training·60명 lockbox로 다시 고정했다. 전체 N=102와 N=99 outer CV는 post-primary exploratory다.

### E1. primary unseen-participant 실험

상태: **현 `physical_hybrid_v1`에는 실행 권한 없음.** 아래는 superseded되지 않은 목표 설계의 기록이지만, 새 독립 development 근거와 별도 decision contract 없이는 39/60을 시작하지 않는다.

> H1-W: strict `k=0` wearable held-out-participant 평가에서 factorized electrode-interface와 pre-query contact metadata를 사용하는 최종 A2가 구조 정보를 동일하게 받는 A0보다 subject-level balanced accuracy를 높인다.

- 39명 전체에서 validation·early stopping 없는 fixed 10-epoch training
- A0/A2×seeds `[42,43,44]` 6 jobs 완료 뒤 독립 60명 lockbox를 함께 공개
- 모든 A0/A2가 같은 allocation, query, seed schedule, backbone, parameter budget 사용
- wet/dry를 각각 보고하고 model×electrode interaction을 보고
- headband order는 모델 입력이 아니라 paired/statistical adjustment에 사용

이 H1은 현재 데이터가 실제로 식별할 수 있는 범위로 좁힌 operational hypothesis이며 연구주제를 바꾸지 않는다.

### E2. metadata mechanism과 robustness

- full metadata / acquisition only / quality only
- correct metadata / `UNK` / all missing
- acquisition-block-coherent donor shuffle. Donor는 exact label/window를 맞추며 train metadata만 교환하고 validation EEG에는 clean metadata를 사용
- same subject×class×block의 counterfactual wet↔dry metadata
- field availability 0/25/50/75/100%
- impedance measurement noise와 out-of-range corruption
- channel drop, contiguous occipital drop, coordinate jitter 5–20 mm
- native/no interpolation, spline interpolation, explicit mask

### E3. acquisition replication

- Wang/BETA/Dong leave-one-acquisition-out 결과를 dataset별로 분리
- 범주형 `UNK`만으로는 physical zero-shot transfer를 주장하지 않음
- target descriptor가 source descriptor support에서 얼마나 먼지와 성능을 함께 표시
- target covariance·batch normalization fit을 사용한 방법은 별도 transductive 표로 분리

## 8. 반드시 포함할 반증 실험

| control | 통과 조건 | 실패 시 해석 |
|---|---|---|
| metadata-only classifier | balanced chance 수준 | label/dataset shortcut 가능성 |
| missingness-only classifier | 유효한 natural pattern이 2개 이상일 때 chance-equivalent | availability pattern shortcut; 단일 pattern이면 invalid assay |
| block-coherent within-class shuffle | correct metadata보다 낮음 | EEG–metadata 대응이 필요하지 않았음 |
| wrong/counterfactual metadata | 기전: correct보다 낮음; 안전: A0 대비 사전 허용손실 이내 | 같으면 metadata 미사용, 큰 붕괴면 배포 안전성 실패 |
| dataset-ID oracle | physical A2보다 우수하지 않음 | semantic transfer보다 dataset 암기 |
| A2 metadata unavailable | A0보다 materially 나쁘지 않음 | 배포 시 불안전한 fusion |
| geometry joint permutation | signal과 좌표를 함께 바꾸면 동일 logit | channel-order artifact |
| metadata support distance | 거리 밖에서도 사전지정 수준 유지 | 아니면 extrapolation 주장 제한 |
| preprocessing 2×2 | `P+F−P`로 fusion 순증분 확인 | fusion 가설 반증 가능 |

metadata shuffle 뒤에도 A2 이득이 유지되면 “metadata가 도움이 됐다”고 결론 내리지 않는다. A2가 단지 더 복잡하거나 최적화가 쉬웠던 것이다. Counterfactual이 correct보다 낮은 것은 reliance 증거지만, A0보다 과도하게 낮으면 wrong metadata에 취약한 것이므로 mechanism pass와 safety pass를 분리한다.

## 9. 평가와 동결 규칙

- primary estimand: target subject별 `BA(A2) − BA(A0)`
- seed는 subject 안에서 먼저 평균하고 독립 표본으로 세지 않는다.
- 95% paired CI와 paired permutation 또는 Wilcoxon을 보고한다.
- 평균 외에 wet/dry, worst-subject, worst-condition, NLL/ECE를 보고한다.
- capability cell은 `subject × condition × window × severity`로 고정한다.
- `learned`, `forgotten`, `learned − forgotten`을 보고한다.
- MCID와 success threshold `τ`는 final target 결과 전에 잠근다.
- source-only development에서 확정하지 못한 분석은 exploratory로 표시한다.

Protocol 0.4는 다음이 모두 기록되기 전에는 `frozen`으로 표시하지 않는다.

- [ ] metadata dictionary와 provenance
- [ ] deployment observability tier와 missing 규칙
- [x] preprocessing·structural input·A2 treatment 경계 — wearable primary에서 reference/cap 제외
- [x] primary A2 architecture와 identity initialization — `physical_hybrid_v1` exact-null 구현
- [x] per-channel impedance 보존 및 bounded channel-gain 경로
- [x] wearable impedance axis numeric-signature regression과 새 processed revision
- [ ] source-only architecture-selection split
- [x] 구현된 physical negative controls와 판정 margin의 owner 승인 (`DEC-20260901-004`)
- [x] Physical reveal #2 실행·판정 — potency-valid, substantive 네 gate 실패, diagnostic no-go
- [ ] MCID, `τ`, seed, multiplicity family
- [x] S1–S3 개발 전용 여부 — DEC-20260830-001로 영구 development-only 확정
- [ ] Dong 및 최종 external lockbox의 단일 사용 규칙

## 10. 최종 해석 범위

이 설계로 직접 주장할 수 있는 것은 다음 순서다.

1. correct metadata가 shuffle/missing/wrong control보다 낫다.
2. A2가 unseen participant의 평균과 worst-condition을 개선한다.
3. `learned > forgotten`이고 source descriptor support 밖에서도 유지된다.
4. factorized descriptor를 가진 외부 acquisition에서 재현된다.

1–2만 만족하면 “acquisition context가 효율을 개선했다”고 한다. 3–4까지 만족해야 “OOD 적용 범위를 확장했다”고 한다. 개별 reference·electrode·impedance의 인과효과는 factorial intervention이 없는 한 주장하지 않는다.

현재 physical candidate는 1과 2를 만족하지 못했다. 따라서 이 candidate에 대해 acquisition context 효율 개선이나 OOD 범위 확장을 주장하지 않으며, 39/60 confirmatory로 진행하지 않는다.

## 11. 조사 범위와 한계

2026-08-29 cutoff로 core, foundation, adjacent, analogy, contrary, implementation branch를 검색했다. 재정독 완료 시 research workspace에는 619개 후보와 33개 search run, 13개 paper card, 17개 deep-read item이 기록됐다. 다만 후보 수는 자동 검색 결과이지 정독 논문 수가 아니다.

- Semantic Scholar가 여러 broad query에서 HTTP 429를 반환해 OpenAlex, Crossref, arXiv, OpenReview와 공식 venue/publisher 페이지로 보완했다.
- direct strict-inductive `k=0` SSVEP의 preprocessing-only 대 metadata-fusion 비교 연구는 찾지 못했다.
- REVE, 2026 montage adapter, wearable 원 논문은 결정 영향이 커 targeted full-text deep-read queue에 추가했다.
- 따라서 acquisition condition의 물리적 중요성은 문헌 근거가 있지만, residual fusion의 실제 이득은 아직 **검증 대상**이지 선행연구로 확정된 사실이 아니다.
