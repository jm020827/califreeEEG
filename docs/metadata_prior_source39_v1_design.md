# Metadata prior source39 — 실제 저보정 실험 종료 계획

2026-09-08. 사용자 ‘끝까지 실험 계획 세우고 실행하라’에 따라, 준비 단계가 아닌 **실제 자료의 한 후보 평가와 최종 판정까지** 실행한다.
[고정 JSON](../configs/analysis/metadata_prior_source39_v1.json)이 정확한 수식·입출력·판정 권위다.
이전 합성 v1과 engineering 결과는 보존하며 기존 종료 runner를 재개하지 않는다.

## 연구질문과 범위

새 사용자의 SSVEP 보정3블록에서, 사전에 측정한 채널별 임피던스로 공간필터 학습을 조절하면
EEG-only Q와 추가 Q 학습/Q와 같은 방식의 sham보다 정확도가 좋아지고 보정5블록을 줄일 수 있는가?
원 목표는 그대로다. 고임피던스가 무조건 나쁘다거나 최초 few-shot/최초 metadata 학습이라는 주장은 없다.

추가 자료 없이 기존 개발39명, dry/wet×0.5/.752/1/2초의8조건 전부를 사용한다.
이8조건은 이미 알려진 개발 설계의 고정 portfolio이며, 새로운 합성 난이도를 골라 성공을 찾지 않는다.
자료의 반복 노출과 공개 native preset의 whole-cohort tuning 한계 때문에 **독립 확증은 아니다**.
Held60/retired1–3, 전체102명 manifest/Impedance.mat, 옛 outcome runner, 외부 요청은 범위 밖이다.

## 보정과 평가의 분리

앞3/5 complete blocks는 각각36/60 labeled trials다. Zero-based block5는 별도의 다음 반복
proxy supervision, blocks6–9는 분류 평가다. 각 사람·조건의 query48개는 예전60개와 다르다.
평가 참가자의 proxy와 query labels는 해당 outer training/선택/정규화에 들어가지 않는다.
실제 배포 사용자는 proxy block의 label을 추가 보정으로 제출하지 않는다. Proxy는 source training supervision이며
평가 cohort에서는 진단용으로만 사용한다. Query가 나중에 있어 생기는 대기/세션 진행의 차이는 사용자 시간 절감으로 숨기지 않는다.

새 source-only exporter가 원250Hz 자료를 직접float64로 읽고, 고정 native filterbank를 각 N마다
독립 적용한다. 새 native A0/FULL reference도 같은48queries에서 계산한다. Project operator의
gamma0 correlation과 argmax가 이 native 결과에 맞아야 한다. 두 Python 환경의 수치오차 허용1e−9,
예측은 정확 일치다. 기존 저장 correlation만으로 새 filter를 복원하지 않는다.

## 학습과 공정한 비교

- 39명을 정렬 ID 순위modulo3으로26fit/13eval. 모든 window/interface/budget/channel은 사람과 함께 이동한다.
- Interface/window/band마다 Q를 따로 적합한다. Q 입력15개는 실제 EEG 품질3종,logk,M availability,
  공통 headband order/period, 알려진 channel onehot8이다. Numeric M과 oracle/query 정보는 없다.
- Q의 local/context와6alpha는 source 참가자 안에서 nested selection한다. 같은 source의 OOF Q error로
  Q2/QM/SHAM residual을 모두 고정alpha.1/intercept 포함으로 적합한다. 잔차 모델의 새 CV는 하지 않아
  validation 정답이 nuisance residual을 통해 섞이는 경로를 만들지 않는다. OOF/final fit 크기 차이는 남는다.
- Target은 기존 원리의 **native 정규화 파형 반복 불일치 log값**이다. 물리적 noise variance가 아니다.
  전체 수준/채널 모양 MSE 및 실제 prior 변화도 보고하되 raw proxy endpoint를 사후 교체하지 않는다.
- M은 앞kblock의 log1p(kOhm) 채널 중심화 평균과population SD. 실제0은 유효,NaN은 결측이다.
  Query M은 전혀 사용하지 않는다. Base clip±3/residual±.2/gamma.1/trace=8은 고정한다.
- FULL/ISO/Q/Q2/QM/SHAM_REFIT/PERMUTED/STALE/MISSING와 A0k0를 모두 보고한다.
  Q2는 차원이 다르므로 동일 용량이라 부르지 않는다. QM/SHAM은 같은2차원/fit/규제를 쓴다.

Mean/std M은 block 순서를 섞어도 같아진다. 따라서 새 sham은 같은 interface/order와 **전체prefix
결측 패턴**의 사람들끼리 packet을 정해진 cyclic derangement로 바꾼다. Fit/eval 사이는 섞지 않는다.
Singleton이나 실제 특징이 안 바뀌는 경우도 그대로 포함·보고한다. 순열은 효능을 보고 선택하지 않는다.
STALE은 첫packet 반복, MISSING은 공통 Q mask를 남긴 numeric-denial exactQ fallback이다.

## 결과를 보기 전에 정한 종료 분기

Primary는8조건을 사람 안에서 평균한 QM3−Q3이며39명 단위 기술적95%t CI를 보고한다.
실용1pp 이상/CI하한>0, QM3−Q2와QM3−SHAM CI하한>0, 평가M특징 변경coverage≥50%를
함께 통과해야 이번 개발자료에서 metadata increment가 성립한 후보로 본다. 다중 cell CI는 탐색 진단이다.

보정량 감소는 위 조건에 더해 QM3−Q5 CI하한>−1pp, QM3 평균≥80%, 두 방법 모두 도달한
case에서 평균 label차이<0 및 새도달≥도달상실을 요구한다. 1pp는 실용 참고폭이지48query 중1개라는 뜻이 아니다.
관측k0/3/5 최초80%와 미도달 전이를 전부 보고하며, 미도달을 임의의24label 비용으로 대체하지 않는다.
1/2/4shot과 실제 준비·휴식·임피던스 측정시간은 미관측이므로 최소비용·wallclock절감을 주장하지 않는다.

입력/누수/native검증 실패는 VALIDITY_FAILURE. 나머지는 METADATA_INCREMENT_NOT_ESTABLISHED,
CLASSIFICATION_INCREMENT_ONLY, DEVELOPMENT_CALIBRATION_BENEFIT_CANDIDATE 중 하나로 종료한다.
양성·음성·불확실 모두 이 후보의 최종 결과다. 실패 뒤gamma/window/M집계/seed/새모델을 바꾸지 않는다.
단순 인프라 수정은 실패 산출물을 보존한 별도 경로에서 같은 과학 설정으로만 복구한다.
양성도39명 개발상 후보이지 독립 확인/연구 전체 성공이 아니며 held60을 자동 개봉하지 않는다.

## 실행 순서·소유권

1. 계약/config commit → 인공 누수·native·cold CLI 검증.
2. Native source-only export39files(약2.97GiB waveform＋reference), 입력/코드해시 receipt.
3. Q/M feature 추출 → 모든 outer fit와freeze 게시 → 전체 분류 평가1회.
4. 독립 auditor가 저장 점수·통계·선택된 회귀식·분할을 검산 → 결과·연구일지·최종 판정.

Main d954870은 계약/공통schema/학습core/분석runner/문서/SQLite/최종실행을 단독 소유한다.
Exporter는 별도worktree의 scripts/export_metadata_prior_source.py＋tests2paths,
auditor는 또 다른worktree의 scripts/audit_metadata_prior_source.py＋tests2paths만 쓴다.
동일tree 동시 writer는 없다. Main은 contract→exporter→core/runner→auditor 순으로 통합한다.
기존30worktrees 보존, 신규2checkout 합계32MiB 예산, 전체artifact/temp8GiB 대 가용약263GiB,
RAM가용약42GiB. Existing nativePy3.9/projectPy3.10 재사용, CPU2/BLAS1,각stage최대1시간.
설치/환경교체/push/cleanup은 없다. Guard는 application 경계이지 OS 보안 격리가 아니다.

`academic-research`의 기존 landscape/qualified claims/open gap을 재사용했다. 현재 불확실성은
새 문헌 수가 아니라 실제 자료에서의 이 한 연산자 효과이므로 이번 실행 전 추가검색/PDF는0이다.
기존 데이터·toolbox의 출처/비상업 license는 [호환성 기록](author_etrca_compatibility.md)에 보존한다.
