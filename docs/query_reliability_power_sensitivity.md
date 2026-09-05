# Query-reliability N=20 power/sensitivity 감사

기준일: 2026-09-04
상태: **outcome-free design audit / 마진 동결 또는 실행 권한 아님**

> 이 N=20 감사는 2026-09-04에 primary에서 철회된 query-only BETA 계획의 역사적
> sensitivity다. 새 metadata-assisted low-calibration 설계는 N=60 participant eAUC를
> 별도로 simulation해야 하며 이 문서의 margin이나 power를 재사용하지 않는다.

이 감사는 EEG signal, prediction, accuracy를 사용하지 않았다. 고정 N=20 one-sided paired-mean
t 절차의 이론적 noncentral-t power와 seed `20260904`의 Normal Monte Carlo만 계산했다.

## 무엇을 구분해야 하나

- Primary inferential null: `H0: mean(Q1_AUG−Q0_AUG) <= 0`.
- `관찰 mean >= +0.02`: 별도의 운영상 promotion screen. `mean > +0.02`를 검정하지 않는다.
- `12/20 positive`: 비추론적 이질성 screen. 독립 fair sign에서 12명 이상이 양수일 확률도
  약 0.2517이다.
- Clean null: `H0: mean difference <= −0.01`. 실패는 harm 입증이 아니라 비열등성 미확립이다.

## N=20, paired SD=0.04의 단일 t-test power

| 실제 primary 평균 | power |
|---:|---:|
| 0.00 | 0.0500 |
| +0.01 | 0.2855 |
| +0.02 | 0.6951 |
| +0.03 | 0.9437 |
| +0.04 | 0.9961 |

## 실제 clean 차이 0일 때 −0.01 비열등성 power

| paired SD | power |
|---:|---:|
| 0.010 | 0.9961 |
| 0.015 | 0.8902 |
| 0.020 | 0.6951 |
| 0.030 | 0.4179 |
| 0.040 | 0.2855 |
| 0.060 | 0.1773 |
| 0.080 | 0.1345 |

80% power의 paired standardized effect는 약 0.577이다. 따라서 true clean delta 0에서
`−0.01` margin을 80% power로 보이려면 paired SD가 약 `0.0173` 이하여야 한다. SD 0.04를
가정하면 약 N=101이 필요하다. 이는 margin을 넓히라는 뜻이 아니다. Margin은 실제 허용 가능한
utility loss로 정해야 하며, N=20을 유지하면 높은 non-promotion 위험을 명시적으로 받아들여야
한다.

## 운영상 primary screen을 모두 포함한 Monte Carlo

Normal difference, 셀당 200,000회에서 `LCB>0`, 관찰 mean `>=.02`, `>=12/20 positive`를
동시에 요구했다.

| true mean | SD | 전체 primary screen 통과확률 | MC SE |
|---:|---:|---:|---:|
| .02 | .02 | .5017 | .00112 |
| .02 | .04 | .4938 | .00112 |
| .02 | .06 | .3927 | .00109 |
| .03 | .02 | .9878 | .00025 |
| .03 | .04 | .8648 | .00076 |
| .03 | .06 | .6734 | .00105 |

True mean이 승격선과 정확히 같은 `.02`라면 관찰 mean screen 때문에 통과확률이 대략 절반이다.
따라서 `.02`를 “80% power로 검출하려는 SESOI”라고 부르면 안 된다.

## 세 inferential component를 모두 요구할 때

Primary true mean `.03`, clean 두 contrast true mean `0`, 세 SD `.04`인 Normal 모형에서 전체
IUT + 운영 screen의 통과확률은 다음과 같다. Correlation은 같은 참가자에서 생기는 세 contrast
사이의 가정이다.

| correlation | 통과확률 | MC SE |
|---:|---:|---:|
| 0.00 | .0715 | .00058 |
| 0.25 | .1048 | .00069 |
| 0.50 | .1398 | .00078 |
| 0.80 | .1974 | .00089 |

즉 clean 비열등성 두 개가 현재 설계의 주요 병목이다. 엄격한 safety gate로서 유지할 수 있지만,
결과가 좋아도 승격하지 못할 확률이 높다는 설계 비용이 있다.

## Mechanism audit

Wrong-query와 identity contrast는 `H0: participant mean <= 0`, one-sided alpha `.05`, `LCB>0`을
잠정안으로 둔다. `+.02` operational screen은 적용하지 않는다. SD .04에서 true mechanism
mean `.005/.01/.02/.03`의 power는 각각 `.1345/.2855/.6951/.9437`이다. 실패는 reliability
메커니즘 주장만 막고, 별도로 성립한 Q-bundle-conditioned system utility를 뒤집지 않는다.

## 결정

현재 `+0.02/−0.01/12-of-20`을 실행 가능한 frozen 수치로 승격하지 않는다. 다음 owner review는
다음 셋 중 하나를 outcome 전에 선택해야 한다.

1. N=20과 −1%p를 유지하고 낮은 joint promotion probability를 받아들인다.
2. 실제 utility 근거로 margin을 다시 정한다. Power를 높이려고 margin을 고르면 안 된다.
3. 독립 참가자 또는 별도 corpus를 늘려 safety precision을 높인다.

재현 코드는 `src/cfeg/analysis/query_reliability_power.py`와
`scripts/analyze_query_reliability_power.py`다. 기본 명령은 다음과 같다.

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_query_reliability_power.py \
  --draws 200000 --seed 20260904
```

최종 freeze에서는 Normal 외 standardized t(5)와 harmed-participant mixture, 고정 correlation
matrix, 셀당 1,000,000회, MC SE, NumPy/SciPy version과 script SHA-256까지 receipt에 묶어야 한다.
