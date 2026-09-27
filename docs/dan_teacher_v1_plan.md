# DAN teacher v1 — 논문 기반 전이학습과 metadata 학습 표적의 유한 비교

2026-09-23 KST. 사용자 `응 그렇게 계획을 잡고 시작.` 승인에 따른 새 프로그램이다.
원 저보정 SSVEP 목표를 유지하고 B의 효과와 M의 증분을 분리한다. 기존 종료된
joint-harmonic/N1/context-template/router와 옛 DAN `NOT_RUN` 권한을 재개하지 않는다.
Root 단독 작성이며 동시 writer/worktree/agent는 사용하지 않는다.

## 질문과 사전 근거

- 표현: stimulus-locked multichannel 파형, target의 적은 labeled support, 실제 block/channel 임피던스.
- 병목: 이전 공동학습은 CCA보다 낮았고 여러 M 경로의 증분이 없었다. B 부적합과 M의
  정보 부족은 구분되지 않는다. 원 논문 숫자를 우리 결과인 것처럼 가져올 수 없다.
- 허용 연산: 공개 논문을 기반으로 별도 DAN 계열 구현, 같은 B의 Q/Q2/QM/SHAM 비교.
- 목적: 소량 보정으로 target 공간에 맞춘 source EEG를 만들 때, 측정 M이 teacher 품질을
  개선하여 같은 Q를 넘어 성능 및 획득 prefix 비용을 개선하는지 시험한다.
- 자원: 아래 후보1개/총량 한도. Scope와 한도는 첫 생성 수치 시험 전에 기록한다.
- 피드백: 구현·정보권한 검사, 원 설정과의 차이, matched 보정 곡선, 독립 저장 결과 검산.
- 실패 가능성: 부족한 source 수, 원법과의 구현/프로토콜 차이, 임피던스와 유용성의 불일치,
  teacher의 편향, 알려진 condition/Q와 M의 중복, 약한 learned baseline, 동결 전 query 노출.

기존 research-space를 먼저 조회했다(1151 papers/48 cards, 역사적 cutoff2026-09-04).
이번은 새 포괄 문헌조사가 아니라 DAN 방법·평가 절 및 공식 저장소의 한정 확인이다.
[논문 v1 III-A–F, IV-A–C](https://arxiv.org/html/2311.12666v1)에서 source→target waveform,
공유 cross-stimulus 공간변환, source 통합 pretrain/개별 fine-tune, target support 평균,
MSE와 filter-bank TRCA를 확인했다. 저자 보고 성능은 여기의 구현 성능이 아니다.

2026-09-23 `gh api` 확인: [공식 저장소](https://github.com/CECNL/SSVEP-DAN)는
revision `521cd7a46f69da1ea8c29e476fa74661cc1607ea`, GitHub license=null이다.
코드를 복사/포팅/실행하지 않는다. **공개 논문을 기반으로 독립 작성하는 protocol-adapted
DAN 계열 비교**이며 저자 코드 재현/원 논문 숫자 재현/SOTA 확보라고 부르지 않는다.
기존 NOT_RUN 기록은 이 새 구현으로 소급 PASS로 바꾸지 않는다. 라이선스 취득·연락은 하지 않는다.

## 후보1개: 신뢰도 가중 teacher를 통한 target-conditioned waveform alignment

원형 U는 target의 support 평균파형을 teacher로 사용한다. 새로운 Q/Q2/QM/SHAM은
모델 구조와 학습량을 바꾸지 않고, teacher 생성에 쓰는 block/channel 가중치만 바꾼다.
모든 class에 같은 block/channel weight를 적용한다. 현재 query 및 query M은 필요 없다.

기본 Q는 support trial의 알려진 자극 주파수+3 harmonics에 대한 투영 잔차/총파워의 log를
block/channel별 class 평균으로 만든다. Q2는 같은 support의 log 정규화4차 모멘트다.
M은 log1p(임피던스 kohm),0은 유효하고 NaN만 missing이다. 음수/inf는 실패다.
각 특징은 source fit 참가자의 첫5 block에서만 구한 channel별 mean/std로 표준화한다.
std<1e-8이면1; target이나 source validation/query 사람으로 scaler를 재적합하지 않는다.

표준화된 Q와 보조 특징은 target의 이미 지불한 prefix 안에서 block축 중심화한다.
보조 특징의 missing은 observed block 평균으로 중심화한 후0으로 둔다.
Q arm의 보조 특징은0, Q2는 추가 EEG 특징, QM은 실제 M, SHAM은 역할·headband order
내에서 사람의 전체 block/channel packet을 derange한 M이다. 동일 donor를 모든 k에 유지한다.

`logit[b,c] = clip(-Qcenter[b,c] - 0.5 * AUXcenter[b,c], -3, 3)`.
`w[b,c] = 0.5/k + 0.5 * softmax_b(logit)[b,c]`.
`teacher[class,c,t] = sum_b w[b,c] * support[b,class,c,t]`.
U는 정확히1/k다. 보조 특징이 prefix 내 상수이거나 모두 누락이면 QM은 Q와 같아야 한다.
M→weight→teacher→alignment MSE gradient→DAN parameters→추가 calibration EEG→eTRCA 경로다.
**M encoder 자체를 학습하는 것이 아니라, 실제 M을 감독 신호 구성에 사용해 B를 학습한다.**
가중치 공식/계수는 고정하며 잘 나오는 계수를 사후 탐색하지 않는다.

기존 context-template은 query와 support의 impedance 차이로 추론 시 correlation template
기여도를 바꾸었다. 이 후보는 query M 없이 target teacher를 바꾸고 source→target 변환을
학습한 뒤 transformed source와 동일한 실제 target support로 classifier를 학습한다.
N1의 native-filter covariance penalty, source-expert mixture, joint harmonic prototype
metric과도 다르다. **Reliability weighting 자체는 새 발명이 아니며 중복을 숨기지 않는다.**
Teacher 가중만으로 고정 채널 계수처럼 최적화에서 소거되지 않는지 생성 대수/gradient로 확인한다.
이 검사는 실제 M 유용성 사전 입증 조건이나 인공 효능 gate가 아니다.

## 고정 데이터·평가·baseline

- 기존 source39 ID만(동일 allowlist); dry/wet 모두. 다른 참가자/held60 접근0.
- 250Hz,원 `[8,710,2,10,12]`; sample160부터375samples=1.5초.
  각 trial의 현재 end535까지 prefix만 notch50Hz/Q35 후 filter bank. 전체710sample을 보지 않는다.
  3band: pass[8,16,24]–90Hz,stop[6,14,22]–100Hz,Cheby-I pass3dB/stop40dB;
  prefix 내 zero-phase 처리 후 crop,각channel time-center/ddof1 zscore. 모든 arm/baseline 동일.
  추가 history0.64초와 upstream downsampling 때문에 전체 온라인 인과성 재현을 주장하지 않는다.
- k2/3/5 =24/36/60 target calibration trials. k1은 실행하지 않는다.
  target support는 앞 k complete blocks,query는 고정 block6–9(0-based)의48trial.
  사용하지 않은 중간 block을 calibration으로 읽거나 query 선택에 쓰지 않는다.
- 원 논문과 달리 source39/3fold/2seed/같은 electrode 내 transfer만,양방향 cross-electrode
  전이는 추가하지 않는다. 전체102명/10repeat/원 실험의 정확한 재현이 아니다.
- sorted ID position mod3 outer fold. 각 fold의26 source 중 sorted앞5명을 source pool로 고정;
  이5명의 앞4명은 pretrain fit,마지막1명은 source validation이다. EEG 성능으로 고르지 않는다.
  Source EEG는 block0–5,source M scaler는 fit4명의 block0–4. target query는 모든 fit 종료 후 접근.
- 별도 고정 비교군: 동일전처리 FBCCA(k0),target-only eTRCA(k2/3/5),unweighted DAN(U).
  Q/Q2/QM/SHAM은 U와 동일 DAN/eTRCA 구조·source pool·seed·데이터량·계산량이다.
  모든 실제 target support를 eTRCA에 똑같이 제공하며 M으로 선택 trial 수를 줄이지 않는다.
- eTRCA는 클래스별 inter-trial covariance generalized eigenvector와 모든 class의 필터를
  결합한 correlation scorer;고정 trace ridge1e-8. 필터 가중 `(band_index+1)^(-1.25)+0.25`,
  signed-rho² 합. FBCCA는 같은 bank/3harmonics/ridge1e-8 및 rho² 합.
  독립 수식 구현/생성 검사 전에는 human decoder를 허용하지 않는다.

## 학습 명세 및 전체 예산

각 band의 alignment model은 spatial Linear(C,C,bias=False),timestamp×channel별 BatchNorm,
channel-wise Linear(C,C)→tanh→Linear(C,C). 시간 위치끼리 섞지 않는다.
BatchNorm은 train batch로만 업데이트하고 validation/transformation에서 eval 고정한다.
논문에서 완전히 명세되지 않은 초기화/BN 설정은 PyTorch 기본으로 고정한 구현 선택이다.
HTML의 learning-rate 지수 표기는 모호하므로 lr5e-4를 독립 설정으로 명시한다.
입력/teacher는 같은 전처리이고 CE가 아닌 waveform MSE로 fitting한다.

Pretrain은 fit source4명×6blocks×12class=288trials,batch96,500epochs;
매epoch source validation1명72trials의MSE를 보고 최소값 checkpoint(동률earlier)를 선택한다.
Fine-tune은 이 checkpoint에서 source5명 각각150epochs;각source의앞4blocks48trials를
train,다음2blocks24trials를validation으로 고정하고 같은 선택 규칙이다. 각미니배치의
source class에 맞는 target teacher만 제공한다. Val은 새로운 target query가 아니다.
변환은 모든source5명×6blocks의360trials를 eval로 처리하여 원target support와 결합한다.
Adam lr5e-4,weight_decay0,gradientclip5;seeds20260923/20260924,추가LR/epoch/seed 탐색0.

| 항목 | 전체 상한 |
| --- | ---: |
| 연구 후보 / M 위치 / hyperparameter 탐색 |1 /1 /0 |
| 실제 raw loads/hashes |39 /39(단일 추출 attempt) |
| shared manifest hash/첫5블록 projected read |1 /1,4680rows |
| 실제 extraction cache load |2(실행/감사),raw 재추출0 |
| Alignment adaptation cells |39×2interface×3k×5arms×2seed×3bands=7020 |
| Optimizer fits |7020×(pretrain1+fine5)=42120 |
| Optimizer updates |7020×(500×3+150×5)=15,795,000 |
| Source validation outputs |각epoch에1회:7020×(500+150×5)=8,775,000 |
| 최종 target-query 평가 |모든모델/decoder동결 후1batch |
| 전체 사람 실행 / GPU training |30wall hours /24GPU hours |
| RAM / torch GPU allocator / 새 산출물 |24GiB /10GiB /12GiB |
| 생성 시험 호출 / wall / optimizer updates |6 /1200초 /1200 |
| 구현 수정 |생성 단계 최대3round,사람실행 후0 |
| 저장 결과 감사 |1batch/3600초/재학습0 |
| held60/외부 요청·연락/유료/새dataset |0/0/0/0 |

큰 fit수는 작은 source-target/band별 독립 network 수이지 수만 개 hyperparameter 탐색이
아니다. 그래도 실험 자원 요구가 크므로 **생성 profile로 보수적 비용을 추산하여 상한 내
실행 가능하지 않으면 human 실행 전에 자원 불충분으로 중단**한다. Source 수/epoch를
조용히 줄여 더 쉬운 다른 방법으로 바꾸지 않는다. GPU가 다른 작업에 사용 중이면 침범하지 않는다.

## 판정과 중단

- Primary: k2/3 BA 평균에서 QM−Q,QM−Q2,QM−SHAM;참가자/interface동일가중.
  모든 k/전극/seed/개인별 결과를 보존. 39명 participant bootstrap2000seed20260925는 기술통계다.
- B효과: U−target-only,각DAN−FBCCA를 별도로 보고. 낮은 B성능을 이유로 M결과를 감추거나
  baseline 개선을 무한히 선행하지 않는다. 오류 없는 낮은 성능은 구현 실패로 재명명하지 않는다.
- M승격: primary QM−Q≥2pp,QM>Q2,QM−SHAM≥1pp,두seed양수;
  QM k3 BA≥80%,QM3−Q5≥−1pp,QM3−FBCCA≥−1pp;
  primary가5pp넘게악화된참가자≤20%;모든정보권한/검산PASS.
- 보정량 주장은 **사전 고정36(QM3) vs60(Q5)trial의40% 획득prefix 감소와 비열등 성능**을
  함께 만족할 때만 허용. 사후80% 최초도달 곡선은 보조 기술통계이며 온라인 중단 정책이 아니다.
  EEG창 사용시간/추가history와 전극준비·측정·휴식 포함 ready-time은 구분;후자는UNKNOWN.
- 전후processor/decoder/정보권한 생성검사 및 전체fit→freeze→score→audit 모의흐름이
  통과하기 전 human entry는 없다. 코어만 통과했다고 사람이용 실행권한을 부여하지 않는다.
- 예산·시간·유효성 실패/후보 부정 결과에서 종료. 유망하면 탐색 종료 후 독립검증 계획만 보고.
  어떤 경우든 held60은 자동 개봉하지 않고, 실패 코드/결과와 부정 결과를 보존한다.

## 이번 시작 단계

먼저 core module과 생성 검사를 구현한다: strict no-IO 학습 표적,상수/missing M→Q,
class permutation,BN eval batch independence,C/F 저장왕복,학습gradient,직접 MSE 수식,
CPU/CUDA 수치. 전처리/eTRCA/reader/전체runner/독립auditor는 후속 구현 대상이다.
현 단계는 효능 결과가 아니라 위 계약 안의 구현 착수다. 해당 구성 완료·profile 확인·
전체생성흐름 통과와 exact artifact hashes를 추가 영수증에 결속한 뒤 사람실행 여부를 정한다.
