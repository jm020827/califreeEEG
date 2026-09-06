# Harmonic gated 반복 실험 결과

2026-09-06 06:26 KST 학습 및 후처리 완료. 물리 GPU 2 한 개에서 tmux로 순차 실행했다. 새 학습 5건과 평가 모두 exit code 0이다. Additive seed 42는 9월 4일 완료 실행을 재사용했다.

## 설계와 결과

Wang 6,480 trial을 학습, 다른 Wang 피험자의 1,680 trial을 validation, BETA 11,200 trial을 test로 사용했다. 각 seed에서 subject split과 초기화가 함께 바뀌므로 반복 차이는 초기화만의 효과가 아니다. Checkpoint는 Wang validation accuracy로 선택했다. BETA의 반복 평가가 후속 연구 방향에 영향을 주었으므로 완전히 미공개된 최종 시험 성적과 구분해야 한다.

| Seed | Additive Wang val | Gated Wang val | Additive BETA | Gated BETA | BETA 변화 |
|---|---:|---:|---:|---:|---:|
| 42 | 74.82% | 65.83% | 34.88% | 46.63% | +11.76%p |
| 123 | 58.63% | 49.11% | 31.90% | 47.71% | +15.80%p |
| 456 | 58.27% | 45.36% | 28.61% | 47.60% | +18.99%p |

BETA 평균±표본 표준편차는 additive 31.79±3.14%, gated 47.31±0.59%였다. Gated의 harmonic-only는 47.7768%, learned-only는 3.10–3.38%다. 세 seed 모두 gated가 harmonic-only를 넘지 못했다. 학습 경로의 부정적 영향을 줄였지만, 순수 harmonic을 초과하는 전이 성능은 확인하지 못했다.

Gated는 harmonic 점수의 scale을 1로 고정하고, 표준화한 learned logits에 입력별 scalar gate를 곱해 더한다. Gate는 0.02에서 시작하고 상한은 0.35다. Auxiliary learned CE와 spectral anchor KL을 각각 0.25 가중치로 추가했다. 따라서 gate, 정규화, scale 고정, 보조 손실의 묶음 비교이며 gate 하나의 독립 ablation은 아니다. Gate 상한은 정확도 하한이나 모든 후보 순위의 보존을 보장하지 않는다.

## 잡음 평가

| Gated seed | Clean | Gaussian 0.10 | Gaussian 0.20 | 8–16 Hz noise 0.10 | Occipital 4채널 |
|---|---:|---:|---:|---:|---:|
| 42 | 46.63% | 46.68% | 46.94% | 46.66% | 50.50% |
| 123 | 47.71% | 47.88% | 47.87% | 47.55% | 50.54% |
| 456 | 47.60% | 47.47% | 47.58% | 47.55% | 50.40% |

합성 additive noise에서 clean 대비 변화는 약 ±0.3%p 이내였다. 4채널 결과는 별도 조건이며 같은 조건의 harmonic-only 대조 결과가 없어 learned 경로의 잡음 제거 효과로 해석할 수 없다. 실제 착용 잡음이나 calibration 감소 효과는 미검증이다.

## 추적

| Run | Epoch / best epoch | W&B |
|---|---|---|
| Gated 42 | 60 / 57 | https://wandb.ai/jm020827/calibration-free-eeg/runs/vtqdoxye |
| Additive 123 | 42 / 30 | https://wandb.ai/jm020827/calibration-free-eeg/runs/mzb94er2 |
| Gated 123 | 42 / 30 | https://wandb.ai/jm020827/calibration-free-eeg/runs/804z4nlk |
| Additive 456 | 59 / 47 | https://wandb.ai/jm020827/calibration-free-eeg/runs/38ejtgjb |
| Gated 456 | 59 / 47 | https://wandb.ai/jm020827/calibration-free-eeg/runs/s7qlc9x2 |

원본 결과는 `${CFEG_EXPERIMENT_ROOT}/20260905/gated_residual_study`의 `metrics_test.json`, `metrics_val.csv`, `eval_*.csv`, `*_status.txt`다. 대용량 EEG·모델·체크포인트는 Git에서 제외한다.
