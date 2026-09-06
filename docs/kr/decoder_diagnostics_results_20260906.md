# REVE 및 EEG 해독기 진단 결과

2026-09-06 14:11 KST, seed 42의 20개 method/protocol 조합 및 REVE calibration 평가가 모두 정상 종료했다. 계획은 [진단 실험 설계](decoder_diagnostics_plan_20260906.md), 제출용 본문은 [연구·평가 요약](research_seminar_followup_20260906.md)에 저장했다.

## 정확도

| 방법 | Wang-only → Wang 새 피험자 | BETA-only → BETA 새 피험자 | Pooled → Wang 새 피험자 | Pooled → BETA 새 피험자 | Wang → BETA 전체 |
|---|---:|---:|---:|---:|---:|
| Harmonic | 38.10% | 42.10% | 38.10% | 42.10% | 47.79% |
| Reference CCA | 72.44% | 59.11% | 72.44% | 59.11% | 66.37% |
| Frozen REVE mean + linear | 12.26% | 11.12% | 9.17% | 8.44% | 3.06% |
| Frozen REVE occ8 + linear | 31.01% | 41.96% | 23.69% | 39.73% | 2.86% |
| EEGNet-style 8,2 | 46.01% | 48.97% | 46.96% | 44.24% | 6.04% |

Wang 새 피험자는 7명/1,680 trial, BETA 새 피험자는 14명/2,240 trial이다. Pooled는 각 dataset의 같은 test subject를 유지한다. Wang→BETA 전체는 70명/11,200 trial이며 다른 열과 평가 대상이 다르다. Harmonic과 CCA는 라벨로 학습하지 않는다. 각 열의 표기는 다른 모델들과의 시험 대상 구분을 위한 것이다.

REVE 출력에서 채널별 정보를 유지하면 각 dataset 안에서는 mean pooling보다 높았다. 그러나 512차원과 4,096차원 표현의 차이도 있으므로 공간 정보 손실만이 원인이라고 단정할 수 없다. Pooled 학습도 현재 설정에서는 BETA-only를 개선하지 못했다. Dataset 간 전이는 세 learned 방법 모두 낮았으며, CCA가 이 비교의 가장 강한 비학습 대조군이었다. CCA 66.37%는 이 전처리/시간창의 내부 결과이며 공개 leaderboard 수치가 아니다.

## 동일 query 소량 보정

| Pooled 분류기 | k=0 | k=1: 피험자당 40 trial | 차이 |
|---|---:|---:|---:|
| REVE mean + linear | 8.57% | 6.90% | -1.67%p |
| REVE occ8 + linear | 40.18% | 24.40% | -15.77%p |

Test BETA 14명 각각에서 40 trial을 support로 예약하고 나머지 120 trial을 k=0과 k=1 모두에 사용했다. 각 행은 같은 1,680 query trial이다. 50 step의 head 보정은 개선되지 않았으며, 학습률·regularization·지원 데이터량의 영향은 아직 분리되지 않았다. 이 결과로 EEG의 소량 적응 자체가 불가능하다고 결론 낼 수 없다.

## 다음 단계

동일 suite를 seed 123과 456에서 반복한다. 물리 GPU 2 한 개와 tmux를 사용하며, seed마다 subject split과 classifier 초기화를 새로 생성한다. 기존 seed 42 feature cache는 라벨을 사용하지 않고 trial별로 계산했으므로 공유하되, feature 표준화 통계와 모든 learned head는 각 seed의 train subject로 다시 fit한다. 같은 seed 안의 단독·혼합 데이터 비교는 동일 test subject를 유지한다.

다음 소량 보정 설정을 고를 때에는 test 피험자가 아닌 validation 피험자의 support/query로 learning rate와 step 수를 결정해야 한다. 이번 반복 단계에서는 먼저 기존 고정 설정의 재현성을 확인한다.

원본은 `${CFEG_EXPERIMENT_ROOT}/20260906/decoder_diagnostics_v1/summary.csv`, 각 run의 `complete.json`, `split.csv`, `calibration_metrics.csv`이다. REVE pooled run은 [mean](https://wandb.ai/jm020827/calibration-free-eeg/runs/5fz03yy8), [occ8](https://wandb.ai/jm020827/calibration-free-eeg/runs/rewyx15w)에 기록됐다. Feature cache와 checkpoint는 Git에서 제외한다.
