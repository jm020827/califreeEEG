# Calibration-Free EEG 모델·학습 타당성 분석 보고서

- 작성일: 2026-09-01
- 대상 저장소: <code>califreeEEG</code>
- 분석 범위: 현재 서버 상태, 모델 구조, 학습·평가 설계, 제안 성능 목표, 달성 가능성과 위험
- 근거: 저장소 코드·설정·테스트·실행 로그, W&B smoke run, 공개 데이터셋 및 REVE 1차 문헌

> **종합 판정: 조건부 준비(Yellow).** Tiny Transformer 기반 synthetic 학습과 W&B 로깅은 끝까지 검증됐다. REVE 가중치 로드와 FP16 autocast forward도 성공했다. 그러나 현재 학습 진입점과 평가 경로의 FP32 forward가 REVE 내부 FlashAttention 요구사항과 충돌하므로, **현 코드 그대로 REVE 실학습을 시작할 수는 없다.** Wang/BETA/Wearable 실데이터도 아직 내려받지 않았다.

## 1. Executive summary

| 영역 | 현재 상태 | 판정 | 근거 |
|---|---:|---:|---|
| GPU/PyTorch | RTX PRO 6000 Blackwell, Torch 2.9.0+cu128 | Green | CUDA forward 확인 |
| Hugging Face | jm020827 인증, REVE base/positions 완료 | Green | 두 snapshot 검증 |
| REVE 모델 생성 | 69.19M backbone 로드 | Green | d_model=512, frozen |
| REVE FP16 forward | 전체 decoder [2, 40] 출력 | Green | CUDA autocast 진단 |
| REVE 기본 학습 진입점 | 최초 FP32 forward 실패 | **Red/P0** | FlashAttention dtype 오류 |
| W&B | jmsmlove02 로그인 → jm020827 team | Green | run ejobsi42 finished |
| Tiny synthetic smoke | 1 epoch train/test/robustness 완료 | Green | 정확도 1.0, 실데이터 근거는 아님 |
| 테스트 | 39 passed | Green | 기존 suite |
| 실제 EEG 데이터 | Wang/BETA/Wearable 없음 | **Red/P0** | EEG_DATA_ROOT 사용량 0 |
| 논문 수준 재현성 | 반복·통계·고전 baseline 미완료 | Yellow | 기본 seed 42 |

핵심 판단은 다음과 같다.

1. **연구 아이디어는 타당하다.** REVE의 montage-aware 표현, acquisition-condition prompt, conditioned adapter, latent nuisance, two-view consistency가 서로 다른 domain shift를 담당한다.
2. **실험 골격도 좋다.** source subject로 train/validation을 나누고 target dataset은 test-only로 유지하며 frequency 기반 canonical label을 쓴다.
3. **현재 결과는 연구 성능 결과가 아니다.** synthetic 100%는 파이프라인 검증값이며, REVE·Wang·BETA·Wearable 성능 증거는 아직 없다.
4. **첫 조치는 dtype 경로 통일이다.** 모델·인증이 아니라 preflight·validation·evaluation·calibration forward에 autocast가 빠진 것이 직접 차단 원인이다.
5. **성공은 상대 개선으로 판정해야 한다.** A4 full model이 A0 EEG-only와 올바른 FBCCA/TRCA 계열 baseline을 양방향·복수 seed에서 넘어야 한다.

## 2. 기존 문서 abstract 요약

| 문서 | abstract 수준 요약 | 현재 분석과의 관계 |
|---|---|---|
| [README.md](../../README.md) | Frozen REVE token과 acquisition metadata prompt를 결합하고 Wang↔BETA, wet↔dry, channel/metadata stress를 평가하는 연구 코드의 사용 설명서다. Target test를 checkpoint 선택에 사용하지 않는 원칙도 정의한다. | 모델·실험의 공식 개요다. |
| [implementation checklist](../../calibration_free_eeg_codex_implementation_plan.md) | Metadata prompt, latent consistency, k=0 성능, 여러 transfer 방향의 연구 질문과 구현·서버 실험 상태를 체크한다. | 코드 존재와 연구 결과 존재를 구분한다. HF 접근은 완료됐지만 실데이터·REVE 실험·반복 통계는 미완료다. |
| [data/README.md](../../data/README.md) | Raw/processed EEG, weight, checkpoint, 로그를 Git 밖 영속 저장소에 두는 정책이다. | 현재 /mnt/ssd3/jm020827 배치가 정책을 충족한다. |

기존 문서에는 **수치 기반 성공 기준이 없다.** 뒤의 수치는 기존 약속이 아니라 제안형 go/no-go 기준이다.

## 3. 현재 실행 환경과 검증 상태

### Figure 1. 실제 REVE 실행 경로의 준비도

~~~mermaid
flowchart LR
    A["RTX PRO 6000<br/>Torch 2.9 + CUDA 12.8"] --> B["HF 인증 및<br/>REVE snapshots"]
    B --> C["REVE 생성 성공<br/>69.19M frozen"]
    C --> D{"forward dtype"}
    D -->|"FP16 autocast"| E["성공<br/>logits [2, 40]"]
    D -->|"현재 preflight/eval FP32"| F["실패<br/>FlashAttention dtype error"]
    E --> G["학습 step 가능성 높음"]
    F --> H["현 상태 실학습 차단 P0"]
    G --> I["validation/eval도<br/>autocast 수정 필요"]
~~~

### 서버 배치

| 용도 | 경로 | 상태 |
|---|---|---|
| HF 상위 캐시 | /mnt/ssd3/jm020827/cache/huggingface | 설정 완료 |
| HF Hub 모델 | /mnt/ssd3/jm020827/cache/huggingface/hub | REVE 약 265 MiB |
| Python 환경 | /mnt/ssd3/jm020827/califreeEEG/venv | repo의 .venv로 연결 |
| EEG raw/processed | /mnt/ssd3/jm020827/califreeEEG/eeg_data | 데이터 없음 |
| W&B 로그 | /mnt/ssd3/jm020827/califreeEEG/wandb | online smoke 완료 |
| SSD3 여유 공간 | 약 2.8 TiB | 충분 |

환경변수는 사용자 [~/.bashrc](/home/jm020827/.bashrc)에 영구 설정돼 있다. 새 셸에는 자동 적용되고 기존 셸에서는 <code>source ~/.bashrc</code>가 필요하다.

### W&B 연결

- 로그인 사용자: <code>jmsmlove02</code>
- 실제 run entity/team: <code>jm020827</code>
- project: <code>calibration-free-eeg</code>
- 검증 run: [smoke/ejobsi42](https://wandb.ai/jm020827/calibration-free-eeg/runs/ejobsi42)
- 원격 상태: <code>finished</code>

사용자명과 entity slug는 다르다. jmsmlove02 entity는 404, 조직 기본 slug는 403이었고, 계정이 속한 jm020827 team에서 정상 업로드됐다.

## 4. 연구 문제와 데이터 도메인

목표는 subject calibration 없이 unseen subject/dataset, 64→8→4→2 channel, wet/gel/dry electrode, sampling/reference/noise 변화, metadata 결측에 견디는 SSVEP decoder를 만드는 것이다.

| 데이터셋 | 피험자 | 채널 | class | 예상 trial | 코드 window | 역할 |
|---|---:|---:|---:|---:|---:|---|
| Wang benchmark | 35 | 64 | 40 | 8,400 | 200 Hz × 400 | controlled source/target |
| BETA | 70 | 64 | 40 | 11,200 | 200 Hz × 400 | lower-SNR real-world shift |
| Wearable | 102 | 8 | 12 | 24,480 | 200 Hz × 400 | wet↔dry, impedance shift |
| Synthetic | 8 | ≤64 | 4 | 640 | 200 Hz × 400 | engineering smoke only |

Wang benchmark는 35명, 64채널, 40-target, 피험자당 6×40 trial을 제공한다. [원 논문](https://pubmed.ncbi.nlm.nih.gov/27849543/) BETA는 70명·40-target이며 차폐실 밖에서 기록돼 Wang보다 SNR이 낮고, narrow-band SNR 3.996 dB 대 benchmark 8.157 dB를 보고한다. [BETA 원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC7324867/) Wearable 데이터는 102명·8채널·12-target, wet/dry 각각 10 block이며 개인별 dry/wet 성능 차이가 크다. [Wearable 원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC7916479/)

이 구성은 within-dataset 정확도보다 domain shift 방향성을 관찰하는 데 적합하다. Wang→BETA가 BETA→Wang보다 어려울 가능성이 높다는 것은 SNR 차이로부터의 추론이며 실제 크기는 실험으로 확인해야 한다.

## 5. 모델 구조

### Figure 2. 전체 decoder

~~~mermaid
flowchart TB
    X["EEG x<br/>B × C≤64 × 400<br/>200 Hz, 2 s"]
    POS["REVE position bank<br/>electrode name → xyz"]
    META1["Categorical<br/>dataset/reference/hardware/<br/>electrode/cap/reattach"]
    META2["Continuous + missing mask<br/>sfreq/channels/impedance/time"]
    META3["Channel IDs + mask"]

    subgraph Frozen["Frozen representation path"]
      REVE["REVE-base<br/>4D positional encoding<br/>69.19M frozen"]
      TOK["EEG tokens<br/>d=512"]
    end

    subgraph Condition["Trainable condition path"]
      CE["ConditionEncoder<br/>3 modality tokens"]
      PROMPT["4 prompt tokens<br/>4 × 512"]
      CVEC["condition vector<br/>512"]
    end

    FUSE["Prompt↔REVE cross-attention<br/>gate + residual + norm"]
    ADAPT["Condition-gated adapter<br/>512→64→512"]
    LATENT["Latent nuisance q(z|h,c)<br/>z=16, KL, z-dropout"]
    ZERO["Deployment path<br/>z=0"]
    HEAD["Classification head<br/>40-class or 12-class"]

    X --> REVE
    POS --> REVE
    REVE --> TOK --> FUSE
    META1 --> CE
    META2 --> CE
    META3 --> CE
    CE --> PROMPT --> FUSE
    CE --> CVEC --> ADAPT
    FUSE --> ADAPT
    ADAPT --> LATENT --> HEAD
    ADAPT --> ZERO --> HEAD
~~~

REVE는 4D positional encoding으로 다양한 electrode arrangement를 처리하도록 설계됐고, 92개 데이터셋·25,000명·60,000시간 이상의 EEG로 masked-autoencoding pretraining됐다고 보고한다. [REVE 논문](https://openreview.net/forum?id=ZeFMtRBy4Z), [공식 프로젝트](https://brain-bzh.github.io/reve/) 다만 대표 downstream task가 SSVEP 전이를 직접 보장하지 않으므로 이 저장소의 핵심 가설은 여전히 검증 대상이다.

### 실측 파라미터

| 구성 | 총 파라미터 | 학습 가능 | frozen | 학습 비율 |
|---|---:|---:|---:|---:|
| REVE full decoder, 40-class | 76,192,936 | 7,003,304 | 69,189,632 | 9.19% |
| REVE backbone | 69,189,632 | 0 | 69,189,632 | 0% |
| Tiny synthetic smoke | 986,084 | 986,084 | 0 | 100% |

Backbone을 고정하면 optimizer state와 gradient memory가 줄고 학습은 약 7.0M prompt/adapter/latent/head parameter에 집중된다. 그러나 frozen weight도 매 forward에서 계산되므로 학습 시간이 9.19%로 줄어드는 것은 아니다.

### Condition 정보와 leakage 통제

| 경로 | 입력 | 기대 역할 |
|---|---|---|
| categorical | dataset, reference, hardware, electrode, cap, reattach | acquisition context |
| continuous | sampling rate, channel count, mean/max impedance, elapsed time | 품질·세션 차이 |
| channel | canonical IDs와 mask | montage/layout 차이 |
| 금지 필드 | label, frequency/phase, trial, subject, session, source file | target leakage 방지 |

Dataset ID는 강력하지만 shortcut 위험도 크다. 따라서 A1 dataset-ID-only와 A2 structured-without-ID 분리가 중요하다.

## 6. 진행하려는 학습 방식

### Figure 3. 학습·선택·평가 흐름

~~~mermaid
flowchart LR
    RAW["Raw EEG"] --> PRE["6–90 Hz<br/>200 Hz resample<br/>2 s / 400 samples<br/>per-channel z-score"]
    PRE --> SPLIT{"subject-group split"}
    SPLIT --> TRAIN["Source train subjects"]
    SPLIT --> VAL["Source validation subjects"]
    SPLIT --> TEST["Untouched target<br/>dataset/condition/subjects"]
    TRAIN --> VIEWS["Two strong views<br/>channel subset/dropout<br/>noise + time shift"]
    VIEWS --> OPT["AdamW + AMP<br/>up to 100 epochs"]
    OPT --> CKPT["best checkpoint"]
    VAL -->|"accuracy early stop"| CKPT
    CKPT --> TEST
    TEST --> METRIC["Accuracy / BA / Macro-F1<br/>NLL / ECE / ITR / drop"]
~~~

두 augmentation view를 v₁, v₂라 하면 손실은 개념적으로 다음과 같다.

\[
\mathcal{L} =
\sum_{i=1}^{2}\mathrm{CE}(y,f(v_i,z_i))
+0.5\sum_{i=1}^{2}\mathrm{CE}(y,f(v_i,0))
+0.1\mathcal{L}_{repr}
+0.05\mathcal{L}_{logit\_KL}
+0.001\sum_{i=1}^{2}D_{KL}(q(z_i|h_i,c_i)\|\mathcal{N}(0,I)).
\]

| 요소 | 값 | 의도 |
|---|---:|---|
| optimizer / LR | AdamW / 3e-4 | adapter·prompt·head 학습 |
| weight decay | 0.01 | regularization |
| epoch | 최대 100 | patience 15 |
| AMP / grad clip | enabled / 1.0 | 속도·안정성·REVE dtype |
| channel dropout | 0.2 | 결측 채널 적응 |
| strong subset | 8/4/2채널, p=0.75 | 저채널 일반화 |
| noise / shift | 0.01–0.05 / ±40 ms | 측정·latency 변화 |
| latent | z=16, inference z=0 | nuisance 분리와 배포 안정성 |

### Ablation ladder

| Variant | Prompt | structured metadata | adapter | consistency | latent | 목적 |
|---|---:|---:|---:|---:|---:|---|
| A0 EEG-only | — | — | — | — | — | 순수 EEG baseline |
| A1 dataset-ID | ✓ | ID만 | ✓ | — | — | shortcut 효과 |
| A2 structured | ✓ | ID 제외 | ✓ | — | — | 일반화 metadata |
| A3 prompt+consistency | ✓ | ✓ | ✓ | ✓ | — | invariance |
| A4 full latent | ✓ | ✓ | ✓ | ✓ | ✓ | 주 모델 |
| A5 full fine-tune | ✓ | ✓ | ✓ | ✓ | ✓ | 상한·과적합 확인 |

## 7. 평가 프로토콜

| 실험 | source train/val | untouched test | class | 질문 |
|---|---|---|---:|---|
| Wang→BETA | Wang subjects | 전체 BETA | 40 | controlled→real-world |
| BETA→Wang | BETA subjects | 전체 Wang | 40 | heterogeneous→controlled |
| Wearable LOSO | 일부 subjects | unseen subjects | 12 | subject-free |
| dry→wet | dry | wet | 12 | electrode transfer |
| wet→dry | wet | dry | 12 | harder field direction |
| calibration | target k=0/1/3/5 | 남은 target trial | 12 | calibration-free retention |
| channel stress | all/8/4/2 | 같은 selection | — | montage 축소 |
| robustness | metadata/noise/rate/reference/compose | 같은 selection | — | 복합 OOD |

Target test는 checkpoint 선택에 쓰지 않는다. Source validation accuracy만 early stopping에 사용한다. 현재 선택 기준이 balanced accuracy가 아니라 raw accuracy 하나라는 점은 class imbalance가 생기면 보완해야 한다.

| 지표 | 역할 | 주의점 |
|---|---|---|
| balanced accuracy | **권장 primary endpoint** | class recall 평균 |
| macro-F1 | minority class 실패 | precision/recall 균형 |
| accuracy | 현재 checkpoint 기준 | 균형 데이터에서만 BA와 유사 |
| NLL / ECE | 확률 품질 | ECE는 binning에 민감 |
| ITR | 속도-정확도 결합 | 2초 stimulus만 포함, cue/rest 제외 |
| generalization drop | source 대비 target 저하 | 양 도메인 난이도 차이 주의 |

## 8. 현재 empirical evidence

### Tiny synthetic smoke

| 항목 | 결과 |
|---|---:|
| train/val/test | 320 / 160 / 160 |
| epoch | 1 |
| test accuracy / BA / macro-F1 | 1.000 / 1.000 / 1.000 |
| test NLL / ECE | 0.404 / 0.322 |
| 코드 기준 ITR | 60 bit/min |
| W&B | sync 성공 |

100% 정확도에도 ECE 0.322인 것은 confidence가 정답률보다 낮다는 뜻이다. 더 중요하게, synthetic 정현파 class는 실제 domain shift보다 훨씬 쉽다. 이는 배선·checkpoint·W&B·평가 코드의 동작 증거이지 실데이터 정확도의 estimator가 아니다.

### Figure 4. Synthetic channel stress

~~~mermaid
xychart-beta
    title "Synthetic smoke: channel stress accuracy (%)"
    x-axis ["All", "8 ch", "4 ch", "2 ch"]
    y-axis "Accuracy (%)" 0 --> 100
    bar [100, 100, 99.22, 93.91]
~~~

### Figure 5. Synthetic robustness

~~~mermaid
xychart-beta
    title "Synthetic smoke: selected robustness accuracy (%)"
    x-axis ["Clean", "Meta missing", "100 Hz", "Re-reference", "Combined"]
    y-axis "Accuracy (%)" 0 --> 100
    bar [100, 100, 97.81, 25, 90.63]
~~~

Metadata 결측 100%는 metadata module의 강건성보다 EEG 정현파만으로 class가 쉬웠다는 신호다. Average re-reference에서 chance 25%까지 무너진 것은 reference robustness 취약 가능성을 암시하지만 실데이터 재검증이 필요하다.

## 9. 기대효과와 반증 조건

| 설계 | 기대효과 | 관측 신호 | 반증 조건 |
|---|---|---|---|
| frozen REVE | montage-aware 표현 재사용 | Tiny/A0 대비 평균 상승·빠른 수렴 | SSVEP harmonic/phase 정보 부족 |
| metadata prompt | acquisition shift 조건화 | A2>A0, transfer drop 감소 | A1만 높으면 dataset shortcut |
| conditioned adapter | 조건별 feature correction | A2/A3 개선 | source condition 과적합 |
| latent nuisance | nuisance 흡수, z=0 배포 | A4>A3, NLL 개선 | posterior collapse/class 흡수 |
| two-view consistency | 채널·잡음 invariance | A3>A2, 4/2ch drop 감소 | clean만 하락 |
| canonical label | Wang/BETA 의미 정렬 | 양방향 confusion 정상 | raw index 혼입 시 전체 무효 |

## 10. 제안 목표치

### 절대 성능 구간

다음은 **예측값이 아니라 의사결정용 제안 기준**이다. ITR은 저장소의 2초 공식이며 온라인 overhead는 제외한다.

| 과제 | chance BA | Floor | Target | Stretch |
|---|---:|---:|---:|---:|
| Wang↔BETA 40-class | 2.5% | BA 30%, ITR 22.23 | BA 45%, ITR 42.67 | BA 60%, ITR 67.10 |
| Wearable 12-class | 8.33% | BA 40%, ITR 16.15 | BA 55%, ITR 31.06 | BA 70%, ITR 49.98 |

Within-dataset supervised 문헌 성능을 zero-shot cross-dataset 목표와 직접 비교하면 안 된다. BETA는 Wang보다 SNR이 낮고 supervised template 방법은 짧은 window에서 training-free 방법보다 강하다. 45%의 40-class zero-shot도 chance의 18배다.

### 주 연구 성공 기준

| 항목 | 최소 통과 | 권장 target | Stretch |
|---|---:|---:|---:|
| A4−A0 양방향 평균 BA | >0 pp | ≥+3 pp | ≥+5 pp |
| 개별 방향 A4 열화 | ≥−2 pp | 열화 없음 | 양방향 ≥+3 pp |
| target/source BA retention | ≥60% | ≥70% | ≥80% |
| 4ch/clean retention | ≥80% | ≥90% | ≥95% |
| 2ch/clean retention | ≥60% | ≥75% | ≥85% |
| combined/clean retention | ≥60% | ≥70% | ≥80% |
| wearable k=0/k=5 BA | ≥70% | ≥80% | ≥90% |
| ECE | <0.25 | ≤0.15 | ≤0.10 |
| 반복성 | 3 seeds | 5 seeds | 5+ + bootstrap CI |

절대 목표보다 A4−A0, target/source, k=0/k=5 retention이 핵심이다. 그래야 데이터 난이도와 모델 contribution을 분리할 수 있다.

## 11. 달성 가능성

아래 확률은 통계적 신뢰구간이 아니라 현재 구현·문헌·blocker에 근거한 **조건부 estimate**다.

| 목표 | 가능성 | 판단 |
|---|---:|---|
| REVE dtype P0 해소와 1-epoch train/eval | 90–95% | autocast forward 성공, 원인 명확 |
| Wang/BETA 전처리·검증 | 80–90% | 공개 데이터·parser 존재 |
| Wearable 전처리 | 70–85% | manual layout/파일 변형 위험 |
| A4가 A0를 평균적으로 이김 | 60–75% | 가설은 합리적, SSVEP-specific 증거 없음 |
| 양방향 A4−A0 ≥3 pp | 45–65% | 비대칭·shortcut 위험 |
| 40-class zero-shot BA ≥45% | 40–60% | 강한 신호지만 완전 cross-domain |
| wearable zero-shot BA ≥55% | 55–70% | 12-class 유리, dry shift 불리 |
| 40-class stretch BA ≥60% | 20–40% | calibration-free 조건에서 공격적 |
| 논문 수준 인과 주장 | 현재 10–20% | baseline·seed·통계가 아직 없음 |

**파이프라인 완주 가능성은 높지만 강한 양방향 우월성의 가능성은 중간**이다. 다양한 EEG task에서 REVE가 우수하다는 사실과 미세 SSVEP frequency/phase discrimination 우월성은 같은 주장이 아니다.

## 12. 주요 위험

| 우선순위 | 위험 | 증거 | 영향 | 권고 |
|---|---|---|---|---|
| **P0** | REVE FP32 forward | dry-run 실패, FP16 성공 | train/eval 차단 | preflight, evaluate_loader, collect_predictions, calibration AMP 통일 |
| **P0** | 실데이터 없음 | EEG_DATA_ROOT=0 | 연구 결과 불가 | Wang/BETA 우선 |
| **P1** | 고전 baseline 미완성 | fbcca.py가 정식 filter-bank CCA가 아니고 eval 미연결 | 우월성 주장 약화 | 검증된 FBCCA와 TRCA/TDCA |
| **P1** | REVE integration test 부재 | 39 tests가 dtype 오류 미검출 | 회귀 위험 | GPU smoke test |
| **P1** | 단일 seed | preset seed 42 | 분산 불명 | 최소 3, 권장 5 |
| **P1** | dataset ID shortcut | A1 존재 | 일반화 오해 | A2를 핵심 비교 |
| **P1** | calibration도 FP32 | 해당 경로 autocast 없음 | k curve 차단 | 동일 AMP policy |
| **P2** | 선택 지표 accuracy | best_acc 고정 | imbalance 왜곡 | configurable BA |
| **P2** | ITR overhead 제외 | trial_time_sec=2.0 | 온라인 과대해석 | practical ITR 병기 |
| **P2** | synthetic 과해석 | metadata missing 100% | false confidence | engineering test로만 표기 |

## 13. 권장 실행 순서

### Figure 6. Decision gate

~~~mermaid
flowchart TD
    P0["Gate 0<br/>AMP dtype 경로 수정"] --> T0{"REVE synthetic<br/>train+val+eval 1 epoch?"}
    T0 -->|No| STOP0["중단: runtime 수정"]
    T0 -->|Yes| DATA["Gate 1<br/>Wang+BETA fetch/preprocess"]
    DATA --> T1{"class map·split·count<br/>검증 통과?"}
    T1 -->|No| STOP1["중단: data audit"]
    T1 -->|Yes| BASE["Gate 2<br/>FBCCA + A0 baseline"]
    BASE --> FULL["Gate 3<br/>A1–A4 × 양방향 × ≥3 seeds"]
    FULL --> G3{"A4−A0 평균 >0?<br/>방향별 붕괴 없음?"}
    G3 -->|No| REVISE["가설·모델 수정"]
    G3 -->|Yes| ROB["Gate 4<br/>robustness"]
    ROB --> WEAR["Gate 5<br/>wearable + k curve"]
    WEAR --> REPORT["통계·최종 보고서"]
~~~

| 단계 | 산출물 | 통과 조건 |
|---|---|---|
| 0 Runtime | REVE 1-epoch checkpoint/eval | dtype 오류 0, W&B finished |
| 1 Data audit | Wang/BETA asset·split 표 | canonical 40-class, leakage 0 |
| 2 Baseline | 정식 FBCCA, A0 Tiny/REVE | 동일 window·split·channel |
| 3 Core | A0–A4, 양방향, seed별 run | ≥3 seeds, target selection 금지 |
| 4 Robustness | 8/4/2ch와 복합 stress | clean retention+CI |
| 5 Wearable | LOSO, wet↔dry, k=0/1/3/5 | subject-paired 결과 |
| 6 Closeout | 평균±SD, CI, effect size | 방향·seed·budget 완전 보고 |

## 14. 결론

프로젝트의 강점은 큰 backbone 자체보다 **montage-aware frozen representation + acquisition metadata + parameter-efficient adapter + nuisance-aware training + 누수 방지 transfer evaluation**을 검증 가능한 가설 사슬로 만든 점이다. 세 실데이터를 합치면 약 44,080 trial로 adapter 학습에는 충분할 가능성이 높다.

현재 상태는 다음과 같다.

- **Tiny synthetic 인프라:** 준비 완료
- **HF/W&B/GPU/저장소:** 준비 완료
- **REVE weight와 FP16 계산:** 준비 완료
- **REVE end-to-end 학습·평가:** dtype P0로 미준비
- **실데이터 연구:** 데이터 부재로 미시작
- **기대효과:** 합리적이나 미실증
- **달성 가능성:** engineering 완주는 높고 강한 성능 우월성은 중간

다음 행동은 장시간 본 학습이 아니라 AMP 경로를 바로잡고 REVE 1-epoch train→validation→evaluation gate를 통과시키는 것이다.

## 참고 문헌 및 근거

1. El Ouahidi et al., [REVE: A Foundation Model for EEG](https://openreview.net/forum?id=ZeFMtRBy4Z), NeurIPS 2025.
2. [REVE 공식 프로젝트](https://brain-bzh.github.io/reve/).
3. Wang et al., [A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces](https://pubmed.ncbi.nlm.nih.gov/27849543/), IEEE TNSRE 2017.
4. Liu et al., [BETA: A Large Benchmark Database Toward SSVEP-BCI Application](https://pmc.ncbi.nlm.nih.gov/articles/PMC7324867/), Frontiers in Neuroscience 2020.
5. Zhu et al., [An Open Dataset for Wearable SSVEP-Based Brain-Computer Interfaces](https://pmc.ncbi.nlm.nih.gov/articles/PMC7916479/), Scientific Data 2021.
6. 저장소 [README](../../README.md), [implementation checklist](../../calibration_free_eeg_codex_implementation_plan.md), 코드·config·2026-09-01 smoke 산출물.
