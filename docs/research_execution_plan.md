# califreeEEG 연구 실행 계획

기준일: 2026-08-29  
프로토콜 버전: 0.3 (full confirmatory run 전 수정)

## 범위 고정

연구 질문은 바꾸지 않는다.

> 구조화된 획득조건 metadata가 target 사용자 데이터로 적응하지 않는 k=0 조건에서 처음 보는 사용자·데이터셋·전극 조건의 closed-set SSVEP 분류 성능과 강건성을 개선하는가?

다음은 이번 연구 범위가 아니다.

- 새로운 stimulus frequency/class 발견
- target stream을 이용한 online/continual adaptation
- 일반적인 open-set OOD 탐지
- LLM `pass@K`의 직접 적용

## 사전지정 가설

- Primary H1: wearable 5-fold outer participant CV의 k=0에서 A2 structured metadata가 구조와 parameter 수를 맞춘 A0 metadata-null보다 unseen-subject balanced accuracy가 높다. 모든 102명은 정확히 한 outer fold의 test가 되고, 각 test subject에는 wet/dry가 모두 있으며 condition별 결과도 분리한다.
- H2: A3 adapter/consistency가 A0보다 held-out query의 clean→perturbed 하락을 줄인다.
- H3 acquisition replication: Wang/BETA/Dong2023의 exact 40-class semantics에서 source-only validation 후 held-out acquisition의 A2−A0를 평가한다. Wang과 BETA만의 전이는 metadata 이득 식별이 아니라 boundary test다.
- H4: 사전지정 성공 threshold에서 A2/A3의 `learned − forgotten` OOD cell 수가 양수이고 worst-group 성능이 개선된다.
- H5 exploratory: stochastic auxiliary latent가 class보다 acquisition condition을 더 잘 나타낸다.

## 프로토콜 수정 기록

### 0.1 → 0.2

원래 0.1 계획은 Wang→BETA를 primary metadata 실험으로 두었다. 원 논문과 실제 MAT를 full run 전에 감사한 결과, Wang과 BETA는 SynAmps2, Cz reference, 배포 sampling rate, 64채널이 같고 electrode/cap 재질은 공개 근거가 없었다. 기존 `gel`/`wet_cap` 값도 근거 없는 동일 상수였다. 따라서 이 전이는 metadata 유용성을 식별할 수 없고 architecture 차이만 측정할 수 있었다.

연구 질문은 유지하되 primary를 source subjects에서 wet/dry와 impedance variation을 실제로 학습할 수 있는 wearable LOSO로 옮겼다. 이 수정은 102명 full data나 confirmatory A0/A2 결과를 보기 전에 이루어졌고, 당시 3명·1 epoch pilot은 두 모델 모두 chance 수준이었다. Wang↔BETA는 closed-set external-domain boundary test로 유지한다. dry→wet/wet→dry는 target category가 source vocabulary에서 `UNK`가 되는 level-extrapolation stress test이지, categorical wet/dry 의미 학습의 증거로 해석하지 않는다.

### 0.2 → 0.3

코드·문헌 peer review에서 `wearable_loso`가 실제 LOSO가 아니라 단일 60/20/20 participant holdout임을 확인했다. 약 20명의 test subject만으로 102명 전체를 대표한다고 쓰지 않도록, primary를 5-fold outer participant CV와 fold 내부 source-only validation으로 수정했다. `data.split_seed`는 optimization seed와 분리해 모든 A0/A2 seed가 같은 fold를 사용한다.

또한 Dong2023이 Wang/BETA와 같은 40개 frequency/phase semantics를 쓰면서 NeuSenW, 8채널, Fp1 reference, pre-gelled semi-dry electrode, 비차폐 환경이라는 실제 acquisition contrast를 제공함을 확인했다. 따라서 Dong2023을 acquisition-metadata external replication에 추가한다. 이 수정도 confirmatory A0/A2 결과를 보기 전에 이루어졌다. Nakanishi 12JFPM도 wearable과 exact 12-class semantics를 공유하지만 데이터 재사용 license가 명확하지 않아 허가 확인 전에는 확증 자산으로 쓰지 않는다.

## 구현 게이트

실제 데이터 결과를 만들기 전에 모두 통과해야 한다.

- [x] source training partition에서만 categorical vocabulary를 만들고 unseen target 값은 trained `UNK=0`으로 처리
- [x] training-only categorical metadata dropout으로 `UNK` 학습
- [x] k=0/1/3/5가 동일한 fixed query를 사용하고 support와 query가 겹치지 않음
- [x] channel/robustness/calibration 평가가 checkpoint의 held-out split 또는 명시적 target filter만 사용
- [x] canonical correlation과 filter bank를 실제 계산하는 CCA/FBCCA 기준선
- [x] wearable dry↔wet에서 source/validation/test participant가 완전히 분리된 joint subject-condition split
- [x] subject-level metric과 prediction artifact에 split/query identity 기록
- [x] A0와 A2가 같은 condition-encoder/adapter parameterization을 사용하고 A0만 모든 metadata를 강제 missing 처리
- [x] Wang raw channel order, BETA schema variants, wearable target frequency/phase 및 impedance wet/dry 축을 공식 자료와 대조
- [x] cross-dataset/cross-condition/standard prediction이 checkpoint `split.csv`의 test ID와 target filter를 교차 적용
- [x] optimization seed와 split seed 분리, wearable 5-fold outer participant split
- [x] corrected wearable artifact를 `wearable_v2` 경로와 revision으로 분리하고 required impedance 누락 시 중단
- [x] Dong2023 Zenodo 선택 다운로드, MD5/schema 검증 및 40-class canonical adapter

## 비교 모델

| 모델 | 역할 | Primary 여부 |
|---|---|---|
| CCA / generic FBCCA | training-free protocol-matched 기준선 | 필수; 실제 채널·filter parameter 기록 |
| paper-faithful FBCCA | Chebyshev-I와 데이터셋별 문헌 고정 설정 | 확증 전 추가 필요 |
| recent learned calibration-free baseline | 최신 직접 비교 | 필수 또는 재현 실패 사유 공개 |
| F0 backbone + linear head | backbone 자체의 효과; REVE run에서만 frozen | 필수 |
| A0 metadata-null | A2와 같은 구조·parameter 수, 모든 metadata 강제 missing | Primary control |
| A1 dataset-ID-only | opaque ID 진단용 negative control | 보조 |
| A2 structured metadata | metadata 주효과 | Primary treatment |
| A3 A2 + adapter/consistency | robustness 가설 | Secondary |
| A4 A3 + stochastic latent | mechanism 탐색 | Exploratory |

A2 primary에는 `dataset_id`와 의미가 이전되지 않는 임의 hardware ID를 넣지 않는다. 채널 좌표/mask, sampling, reference, electrode type, impedance처럼 다른 데이터셋에서도 의미가 유지되는 정보를 사용한다. `metadata_missing_all`과 `metadata_shuffle`을 필수 negative control로 실행해 A2가 실제 metadata에 의존했는지 확인한다.

## 실험 순서

### Phase 0 — 코드 타당성

1. P0 수정과 단위 테스트
2. 전체 pytest 및 ruff
3. synthetic frequency 신호로 CCA/FBCCA 정답 검증
4. split·support·query·vocabulary artifact 검증

### Phase 1 — synthetic pipeline pilot

1. synthetic asset 생성
2. Tiny backbone 1-epoch smoke
3. A0/A2 dry-run과 짧은 training
4. held-out channel/robustness 평가
5. fixed-query k=0/1/3/5 실행

Synthetic 결과는 사람 EEG에 대한 연구 증거가 아니라 pipeline 검증으로만 기록한다.

### Phase 2 — 공개 EEG pilot

1. BETA S1/S16, wearable S1–S3, Dong2023 S1로 서로 다른 raw schema·label·metadata smoke
2. 실제 CCA/FBCCA와 Tiny backbone A0/A2 실행
3. subject×electrode prediction/metric 산출 확인
4. 오류가 없을 때 wearable 102명과 Wang/BETA/Dong2023 전체로 확장

### Phase 3 — confirmatory full runs

1. wearable 5-fold outer subject DG primary: 각 fold의 F0, A0, A2; k=0; 동일 fold에서 3–5 optimization seed
2. wearable condition별 및 worst-subject 분석
3. wearable joint subject-condition dry→wet 및 wet→dry stress test
4. Wang/BETA/Dong2023 leave-one-acquisition-out replication; Wang↔BETA만의 전이는 boundary test
5. A3 robustness, A4 exploratory, k>0 secondary
6. 추가 shared-label multi-source dataset을 확보하면 leave-one-dataset-out metadata 실험
7. 선택과 threshold를 동결한 뒤 external lockbox

## 통계와 판정

- 독립 단위: target subject. 각 subject는 outer test fold에 정확히 한 번만 포함한다.
- Primary endpoint: subject-level balanced accuracy의 paired A2−A0
- Optimization seed: 최소 3개, 가능하면 5개. 동일 subject×model의 seed를 먼저 평균하며 seed를 독립 피험자로 세지 않는다.
- Split seed: 42로 고정. split 변동성은 별도의 repeated-group-CV 분석으로만 다룬다.
- 보고: effect size, 95% CI, paired permutation 또는 Wilcoxon
- 다중 비교: Holm correction
- capability 분석 단위: subject×condition×window×severity cell
- 성공 threshold와 MCID: target 결과를 보기 전에 확정

평균만 개선되면 metadata efficiency로 해석한다. `learned > forgotten`, worst-group 개선, 양방향 재현까지 충족해야 OOD 적용범위 확장으로 해석한다.

## 필수 산출물

- 실행 config와 Git commit
- source/target dataset 및 license/version
- `split.csv`, vocabulary, class map
- fixed support/query manifest와 hash
- seed별 checkpoint 및 subject-level predictions
- 환경, hardware, wall time, parameter count
- 실패·제외 run과 사유
- aggregate 표와 learned/forgotten transition 표

## 중단 규칙

- target test로 checkpoint·threshold·hyperparameter를 고르면 해당 run은 confirmatory 결과에서 제외한다.
- A2가 A0를 이기지 못하면 A4 결과로 metadata 가설을 대체하지 않는다.
- 한 전이 방향에서만 효과가 있으면 dataset-specific 결과로 제한한다.
- split/query/vocabulary provenance가 없는 run은 재실행한다.

## 2026-08-29 실행 현황

- 전체 단위 테스트: 93 passed; targeted Ruff, compileall, YAML parse, shell syntax, `git diff --check` 통과
- synthetic 640-trial pipeline: A0/A2 training, held-out robustness, fixed-query k=0/1/3 artifact와 동일 query hash 검증
- BETA S1/S16 pilot: `(320, 64, 400)`, subject당 160 trials, class당 4 blocks, 750/1000-sample 원본 schema 모두 검증
- BETA S1/S16 posterior-9, H=5 pilot: CCA BA 0.8656, generic Butterworth FBCCA BA 0.8719. 개발 sanity일 뿐이며 paper-faithful FBCCA는 아직 아니다.
- wearable S1–S3 v2 pilot: `(720, 64, 400)`, subject당 240 trials, wet/dry×class당 10 blocks, impedance 축을 원본과 수치 대조
- wearable S1–S3 official native-8, H=5 pilot: CCA BA 0.6125( dry 0.4667 / wet 0.7583), generic Butterworth FBCCA BA 0.5750(dry 0.4194 / wet 0.7306). 3명 결과이므로 성능 결론이 아니라 pipeline sanity check다.
- wearable v1 pilot에서 class frequency 순서 오류 때문에 CCA/FBCCA가 chance 수준이었고, 공식 stimulation table 순서로 고친 v2에서 회복했다. v1과 그 파생 결과는 연구 결과에서 제외한다.
- parameter-matched wearable S1–S3 Tiny 1-epoch pilot: A0/A2 모두 40,860 parameters, 동일 split hash, test BA 0.0917. 1 test subject·chance 수준이므로 효과 증거가 아니라 end-to-end control 검증이다.
- Dong2023 S1 pilot: 공식 MD5와 `[8,1250,40,4]` schema, 160 trials를 검증했다. native-8 H=5에서 CCA BA 0.7813, generic FBCCA BA 0.7563이었다.
