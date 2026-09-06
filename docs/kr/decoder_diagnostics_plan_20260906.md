# REVE 및 EEG 해독기 후속 비교

## 질문

Gated 반복 실험은 harmonic 성능 손실을 줄였지만 순수 harmonic보다 높지 않았다. 다음 실험은 REVE 표현, 전체 토큰 평균 pooling, 학습 데이터 구성, 소량 calibration을 분리해 평가한다. 성능 향상은 가설이며 아직 결과가 아니다.

## 평가 구성

| 방법 | 학습하는 부분 | 입력 |
|---|---|---|
| Harmonic | 없음 | 뒤통수 8채널, 1–5차 배음 power |
| Reference CCA | 라벨 학습 없음 | 같은 8채널과 후보별 sine/cosine, 1–5차 배음 |
| REVE mean + linear | 선형 분류기만 | Frozen REVE 전체 channel/time 토큰 평균 512차원 |
| REVE occ8 + linear | 선형 분류기만 | 같은 frozen REVE의 뒤통수 8채널별 시간 평균을 연결한 4,096차원 |
| EEGNet-style 8,2 | 작은 CNN 전체 | 64채널 EEG 원시 시계열 전처리본 |

EEGNet은 시간 convolution, depthwise 공간 convolution, separable convolution, linear head를 이용한 독립 PyTorch 구현이다. 200 Hz에 맞춰 첫 temporal kernel을 100 sample로 두었다. 이 결과를 공식 EEGNet 또는 MOABB leaderboard 수치로 부르지 않는다. CCA는 orthonormal subspace SVD로 계산하며 sklearn CCA와 수치 비교 테스트를 수행한다. CCA correlation은 확률이 아니므로 NLL/ECE는 보고하지 않는다. REVE occ8도 backbone 입력은 위치를 확보한 62채널이고, 출력 pooling만 8채널에 제한한다.

## 데이터와 누수 방지

Seed 42에서 dataset별 subject를 train/validation/test 60/20/20%로 분리한다. Wang-only, BETA-only, pooled 학습은 동일한 각 dataset의 test 피험자를 유지한다. Pooled는 두 dataset의 train끼리, validation끼리 합친다. 이 구성에서 BETA를 학습에 사용해도 test 피험자가 분리되므로 새 피험자에 대한 0-calibration 평가가 가능하다. 별도의 Wang→BETA cross-dataset 비교는 기존처럼 BETA 전체를 test로 쓴다.

Wang은 현재 저장된 34명/8,160 trial, BETA는 70명/11,200 trial을 사용한다. Wang 공개 원본 35명 중 빠진 피험자가 있으므로 공개 benchmark 전체에 대한 평가와 구분한다. 모든 방법이 기존 200 Hz, 2초, 채널별 z-score 전처리본을 공유한다. REVE 공식 권장 전처리의 재현 실험이라고 주장하지 않는다. 사전학습 corpus에 Wang/BETA가 포함됐는지는 미확인이다.

Feature extraction은 trial별 독립적이며 라벨을 사용하지 않는다. Cache 생성에 test trial도 포함되지만 전역 통계, 정규화 fit, backbone 업데이트를 하지 않는다. Linear 입력의 mean/std는 train subject로만 fit한다. Metadata·dataset ID·condition prompt·adapter·latent·harmonic hybrid는 learned 비교군에서 모두 제외한다.

Linear 및 EEGNet은 AdamW lr 0.001, weight decay 0.01, 최대 100 epoch, validation accuracy patience 15로 학습한다. Batch는 linear 512, EEGNet 128이며 별도 신호 augmentation은 없다. EEGNet의 hidden dropout 0.5는 사용하며 채널 제거는 아니다. 각 방법은 같은 protocol의 validation으로 checkpoint를 선택하고 test로 조정하지 않는다. Epoch 수가 같아도 모델별 연산량과 capacity는 같지 않다. 단일 seed 탐색이며 결론은 반복 검증이 필요하다.

## 소량 보정

Pooled REVE 두 linear 분류기에 대해 test BETA 피험자마다 클래스당 1 trial, 총 40 trial을 support로 먼저 예약한다. 남은 120 trial은 k=0과 k=1에 공통된 query다. k=0도 예약 support를 시험 성적에 넣지 않는다. k=1은 pretrained head를 복제하고 그 피험자의 support만으로 AdamW lr 0.001, 50 step을 수행한다. 기존 weight와의 평균제곱 차이에 0.01 가중치를 둔다. Query label로 step, lr, checkpoint를 선택하지 않는다. 총 trial 수와 sample ID를 저장하며 이 실험을 사용자 데이터가 전혀 필요 없는 방법과 구분한다.

## 실행 및 공개 근거

GPU는 물리 2 한 개, tmux `cfeg_decoder_diagnostics_gpu2`, W&B group `decoder_diagnostics_v1`을 사용한다. 중간 결과는 `${CFEG_EXPERIMENT_ROOT}/20260906/decoder_diagnostics_v1/summary.csv`에 저장한다. 체크포인트·EEG·feature cache는 Git에 넣지 않는다. 최대 20개 method/protocol 평가이며 새 backbone을 대규모 사전학습하는 작업은 아니다.

Wang/BETA는 SSVEP 벤치마크 데이터다. [MOABB의 cross-subject SSVEP 예제](https://moabb.neurotechx.com/docs/auto_examples/paradigm_examples/plot_cross_subject_ssvep.html)는 여러 해독기를 동일한 평가 절차로 비교한다. 이번 실행은 기존 전처리본을 사용하는 내부 비교이며 MOABB의 공식 평가를 실행한 것으로 표시하지 않는다.

[REVE 모델 카드](https://huggingface.co/brain-bzh/reve-base)는 사전학습 encoder와 200 Hz 입력을 설명한다. REVE-base에는 이 프로젝트의 40개 주파수 분류기가 기본 내장되어 있지 않으므로 frozen-feature probe로 분류 가능성을 평가한다. [EEGNet 저자 구현](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py)의 구조를 참고했다.

[SmolVLA 공식 가이드](https://huggingface.co/docs/lerobot/smolvla)는 새 환경의 데모를 모아 fine-tuning하며 약 50 episode를 출발점으로 제시한다. 새 환경 데이터를 전혀 주지 않는 Wang→BETA zero-shot과는 데이터 제공 조건이 다르다. EEG의 소량 보정 가능성은 위의 동일 query k=0/1 비교로 확인한다.
