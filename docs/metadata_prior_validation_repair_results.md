# M-blind 검증 설계 수리 — 구현·공학 검증 완료

2026-09-08. **이번 결과는 metadata 효능 실험이 아니다.**
[구현 전 계약](metadata_prior_validation_repair.md),
[전체 검산 JSON](reports/metadata_prior_validation_engineering_v1.json),
[종료된 합성 v1 결과](metadata_trca_prior_synthetic_v1_results.md).

## 쉽게 말하면

지난번은 시험 문제가 너무 쉬워 새 방법의 효과를 판별하기 어려웠다.
이번에는 문제를 더 어렵게 바꿔 M이 이기는 결과를 찾지 않고, **비교 방법과 검사 도구를 고쳤다.**

- EEG-only Q는 규제 강도를 고정하지 않고, source 참가자들 안의 분리된 검증으로 선택할 수 있게 했다.
  다른 채널의 평균 정보도 Q에 제공하여 M 쪽에만 채널 간 상대 정보가 들어가는 표현상의 불균형을 줄였다.
- 한 참가자의 예측을 만들 때 그 사람의 모든 채널·보정량 조건을 학습과 규제 선택에서 함께 제외한다.
  그 예측으로 후속 잔차 학습을 준비할 수 있다. 아직 M 잔차 학습기는 연결하지 않았다.
- 예측 오차의 ‘모든 채널 공통 수준’과 ‘채널 간 모양’을 나눠 보도록 했다.
  앞쪽만 좋아지면 정규화된 공간필터에는 변화가 없을 수 있다.
- 식으로 만든 작은 배열에서 규제 변경이 **필터→점수→최종 선택**까지 전달되는지 직접 확인했다.
  이 배열에는 정답을 부여하지 않았고 정확도를 계산하지 않았다.

따라서 해결한 것은 특정한 구현·검증상의 모호함이다. Q가 이제 최적이라는 주장,
실제적인 난이도 설계가 끝났다는 주장, metadata가 보정량을 줄였다는 주장은 여전히 없다.

## 구현과 수치 확인

새 module은 `src/cfeg/analysis/metadata_prior_validation.py`다. 기존 operator/v1 generator/learner/
runner/config/result는 그대로다. 실제 data adapter, source export, M learner, 학습된 Q의 기존 runner 연결은 없다.

| 검사 | 확인 결과 | 뜻하지 않는 것 |
|---|---|---|
| 같은 Q에 ridge residual 재적합 | unclipped `s → 2s−s²` 항등식 검사 통과 | Q2 이득이 새로운 정보라는 뜻 |
| Q nested selection | local/context ×6alpha, participant3-fold 내부 선택과 바깥 OOF 제외 검사 통과 | 최적 Q 또는 정보 충분성 |
| Channel-context 선형 항등식 | 9인공 group에서 OOF 최대오차1.33e−15 | 실제 EEG proxy 예측 성능 |
| TRCA S/C 직접 구성 | `S=diag(5.82,5.805)`, `C=3I`, 식 대비오차1.78e−15 | 실제 EEG 잡음 covariance 복원 |
| Bounded residual의 전달 | 고정 gamma.1, delta(−.2,+.2)에서 `[1,0]→[0,1]` 선택 변경 | 정확도 향상 또는 M 유용성 |
| 대조·불변성 | penalty trace 차이0, gamma0 filter 차이0, missing exact, uniform prior 오차1.11e−16 | 자연 결측에서의 사람 성능 보장 |
| 점수 margin 진단 | `margin > 2*max_score_change`이면 답이 그대로라는 단방향 보증 검사 | 불변인 모든 답의 원인 설명 |

선형 항등식 예제는 참가자별 offset이 다른 채널값에서 그 참가자의 채널 평균을 빼는 문제다.
그 식을 정확히 표현하는 context＋alpha0이 선택되도록 구성되어 있으므로 실제 일반화 성능을
비교하는 독립 benchmark가 아니다. OOF 학습 group6개, 최종 fit9개의 크기 차이도 기록했다.

기존 v1은 점수 벡터를 저장하지 않았기 때문에 새 margin 진단으로 당시 ceiling의 원인을
입증했다고 하지 않는다. 위상 정렬·공통 공간 gain·반복 평균에 따른 높은 유효 신호대잡음비는
코드에서 추론한 원인 후보이며, 각각의 기여를 실험으로 분해한 것은 아니다.

## 실행 기록과 검토

- 계약 먼저 `1745386`, helper/검산기/tests `45a5dfd`, 마지막 집계 loss 유한성 수정 `20339b2`.
  최종 수치 receipt 생성 source는 `20339b2de2c87e01f308c9da6fc7a6676aeecc4f`다.
- JSON SHA-256 `7355eedba42b6856e7de9d1ccdec7669b55fc3d043a33924b2a87645502808b7`.
  Python3.10.12/NumPy1.26.4/SciPy1.15.3, CPU/BLAS1. RNG 없이 고정 배열을 썼다.
  단위시험·독립검토·CLI의 같은 배열 반복 검산을 서로 독립된 연구 실험으로 세지 않는다.
- 최초 임시 receipt는 `/tmp/cfeg-prior-repair.tDbHVI/engineering-result.json`에 보존했다.
  마지막 수정은 유한한 개별 loss들의 평균이 overflow하면 거부하는 검사이며 과학 모델 변경이 아니다.
- 두 읽기 전용 reviewer가 대수와 경계를 검토했다. 초안 검토자의 S 산술오류는 root가 식으로 정정했고
  독립 cross-repeat pair-sum 검산으로 확인했다. Pair-sum과 operator 차이≤2.67e−15,
  직접 단일채널 Pearson과 operator score 차이≤1.76e−16.
- 구현 중 찾아 수정한 문제: global-min 기준과 달랐던 근접 동률 우선순위, 큰 유한 입력에서
  무한 scaler를 반환하던 경우, 유한 개별 loss의 평균 overflow. 각각 회귀검사를 추가했다.
- 제가 처음 PATH의 시스템 Python으로 실행해 CLI import/기존 cold NumPy 검사2개 실패 및
  safetensors 없는 전체 collection3개 오류가 있었다. 기존 프로젝트 `.venv`로 정정했으며
  환경 설치·기존 v1 코드 수정은 하지 않았다. 새 CLI는 source 경로를 명시하도록 고쳤다.
- 최종 관련 **118tests PASS2.84초**(새34＋기존84), Ruff3files/format/diff check PASS.
  마지막 수치 수정 후 최종 전체 **1679tests PASS202.97초**, 기존 Torch warnings68개.
  앞선 전체1678PASS220.20초는 마지막 집계 overflow 검사 추가 전 실행이며 최종 실행과 구분한다.

Main은 코드/문서/연구공간 단독 writer, 두 reviewer는 공유 저장소 읽기 전용이었다.
새 worktree 없이 기존30개를 보존했고, 새 human EEG/M/held60 접근·데이터 확보·효능 suite 실행은0이다.
`academic-research`의 기존 근거/열린 질문을 재사용하여 구현의 성립과 효능의 증거를 구분했다.
새 검색/PDF0이며 최신 문헌을 다시 포괄 검토했다는 주장은 하지 않는다.

## 남은 일

연구목표는 **적은 SSVEP 보정에서 acquisition metadata가 EEG-only Q를 넘어 추가 가치를 주는가**로 유지한다.
다음에는 M outcome을 보지 않는 별도 benchmark 설계에서, 사전 지정한 모든 난이도 조건을 보고하고
Q-only 오답·필터/점수 민감도를 확인해야 한다. 개발 자료와 최종 평가 자료를 분리해야 한다.
그다음에만 동일한 선택 기회를 가진 Q/QM/SHAM과 보정량 endpoint를 비교한다.

이번에는 현실적인 generator·난이도 범위·최종 효능 seed를 새로 정하거나 실행하지 않았다.
현재 자료가 반드시 부족해서 멈춘 것이 아니라, **새 효능 검증의 범위를 아직 고정하지 않은 상태**다.
위 공학 검증 통과를 근거로 기존 종료 결과를 승격하거나 held60을 자동 개봉하지 않는다.
