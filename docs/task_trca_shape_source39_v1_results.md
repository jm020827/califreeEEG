# Task-shape source39 v1: 안정성 실패로 종료

2026-09-08. **현재 후보 판정은 `VALIDITY_FAILURE`다. Metadata 효과 없음이라는
판정은 아니다.** 실제 학습을 시작했지만 고정된 필터 안정성 조건을 위반해
최종 평가 전에 종료했다. 연구목표는 계속 ‘추가 acquisition metadata가 EEG-Q를
넘어 SSVEP 보정 부담을 줄이는가’이며 바꾸지 않았다.

전체 수치·해시: [terminal JSON](reports/task_trca_shape_source39_v1_terminal.json).
설계: [원 설계](task_aligned_trca_shape_v1_design.md),
실행: [별도 고정 manifest](../configs/analysis/task_trca_shape_source39_execution_v1.json),
구현·소유권: [execution build](task_trca_shape_source39_execution_build.md).

## 쉽게 말하면

이 방법은 먼저 EEG만으로 채널을 어떻게 조합할지 배우고(Q), 그것을 고정한 뒤
임피던스 같은 추가 metadata로 조금 수정한다(QM). 공정하게 비교하기 위해
metadata를 다른 사람의 것으로 바꿔 학습하는 대조군(SHAM)도 똑같이 실행한다.

이번에는 그 **대조군을 학습하는 도중**, 새 필터를 원래 기준 필터와 같은 방향·부호로
맞추는 유한한 계산값이 고정된 C-내적 정렬 하한을 위반했다. 허용한 변경량은 작아도 필터 방향까지 항상 조금만
변하는 것은 아니었다. CPU로 따로 계산해도 같은 현상이 나와 CUDA 고유 오류로
설명할 수 없다. 실패한 대조군이나 참가자를 빼고 QM만 평가하지 않았다.

## 실제로 어디까지 했나

| 구간 | 관측한 상태 |
|---|---|
| 실제 입력 | 기존 개발39명, 역할별 보정 EEG와 source block5 및 보정 구간 M만 접근 |
| 첫 외부 분할 | 내부9개+최종1개 pipeline 완료:40heads,8000optimizer updates |
| 첫 분할 검산 | 저장된 scaler·도너·inner-QCE를 별도 NumPy/SciPy로 재계산,36개 CE 비교 최대오차4.89e−15 |
| 두 번째 외부 분할 | 첫 inner fold, λ=.0001의 학습 중 안정성 검사 실패 |
| 세 모델의 전체 동결 | 미완료 |
| 최종 query6–9·FULL/A0 평가값 | **접근0, 새 정확도·보정 비용 결과 없음** |
| Held60 / 이전 후보 재개 | **0 / 없음** |

원 실행은 clean commit `50654e3f08000c7d81b50d40fb3a5ffaa79f4bb2`에서
212.780초 실행됐다. Source0/1은 각26명의 역할별 통계이며 합집합이39명이다.
완성된 첫 분할의 λ=.001은 **그 분할에서만** 선택된 값이며 다른 분할에 이식하지 않았다.
첫 분할의 학습 손실이나 λ 선택을 metadata 효능 결과로 제시하지 않는다.

사전에 선언한 원칙은 어떤 필수 arm/분할의 수치 조건 위반도 전체 attempt 실패로
처리하는 것이었다. 따라서 이는 진행 중인 미완료 작업을 성공으로 포장한 것이 아니라,
후보의 유효성 실패에 따른 종료다. Metadata 순증분·calibration 절감은 **평가되지 않았다**.

## 실패 지점의 제한적 재현

최초 오류 메시지는 `native anchor is nearly C-orthogonal to the leading direction`이다.
이 메시지는 낮은 cosine과 비유한 cosine을 함께 처리하므로 메시지만으로 원인을 단정하지 않았다.

원본을 보존한 뒤 **이미 저장된 source1 통계**로 실패한 한 pipeline만 같은 CUDA/float64,
같은 참가자 분할·λ·초기값·Adam 설정으로 재현했다. 새 후보를 평가하거나 threshold를
바꾸지 않았고, 같은 예외에서 종료해 완성 모델을 게시하지 않았다. 진단 소요12.571초;
재현에서 추가 계산된 optimizer update는685회다(완료3heads×200+SHAM85회).

- 재현 위치: outer fold1 / inner fold0 / λ=.0001 / **SHAM_REFIT의86번째 목적함수 호출,
  해당 update 전**. 그 재현에서 Q/Q2/QM은 먼저 완료됐다.
- 해당 support: 참가자41, wet,188samples(.752초),k5, band0/class4(모두 zero-based).
- GPU의 C-내적 절대cosine: **6.105745921e−7**.
- 독립 SciPy CPU의 값: **6.105745590e−7**, 고정 하한 **1e−6**보다 작다.
- 모든 cosine은 유한하다. C 최소고유값3.27673>0, leading gap 약.00700248로
  top-gap 기준도 충족한다. 이 지점의 R도 trace8·ratio≤2·max≤16/9 범위 안이다.

따라서 확인된 문제는 **허용된 R 안에서도 native 기준 필터에 대한 정렬이 거의 직교할 수
있다는 것**이다. 이는 일반 Euclidean 각도가 아니라 C-내적 기준이다. C≤B≤1.1C라는
분모 bound 자체가 깨졌거나 C가 singular라는 결과는 아니다. 특정 학습 경로 전체에서
고유값 branch가 어떻게 이동했는지는 저장하지 않았으므로 그 상세 기전까지 입증하지 않는다.

진단 원본: `/home/whwovy/task-trca-shape-anchor-failure-diagnostic-v1.json`,
SHA `86eda34fc11f9e20a3874ef0e952a2aedc8b60e7f908bdd290b84ccf84b1561b`.
코드 `cab4c048a5be441eeb9f1d24b1242bedbcb8f9ce`, 인공 직교/평행2tests PASS.
원 attempt의 실패 판정은 불변이며 이 진단은 효능 재실험이 아니다.

## 지금 고쳐야 할 부분과 바꾸면 안 되는 것

1. **목표가 아니라 필터 방향·부호를 안정적으로 정의하는 구현 설계를 고쳐야 한다.**
   현재의 ‘원 native 필터에 투영해 부호를 맞춘다’는 규칙은 허용 학습 영역 전체에서
   안전하지 않다. 후속 제안은 기준·점수 정의 또는 support-only 유효 영역을 명확히 하고,
   이미 노출된 source 유래 실패 S/C/R 사례와 별도 생성한 인공 사례를 구분해 검증해야 한다.
   노출된 실패 사례에 맞춘 수정의 통과는 신규 효능 증거가 아니다. 작은 penalty가 작은
   방향 변화를 보장한다고 다시 가정하면 안 된다.
2. **SHAM을 제외하거나 λ=.0001을 지워서 살아남는 결과만 보고하면 안 된다.**
   Threshold 완화·seed/window 변경·선택되지 않는 residual fit 생략도 현재의 종료 규칙을
   바꾸는 행위다. 이를 단순 GPU 최적화나 입력 복구라고 부르지 않는다.
3. **다음 구현에는 실패한 head/step/case/R/실제 검사값을 즉시 저장해야 한다.**
   이번 원본 로그는 head/step을 남기지 않아 제한적인 재현이 필요했다. 이는 결과 선택과
   무관한 관측 기능이지만 다음 실행 전 코드/manifest에 다시 고정해야 한다.
4. **새 완료-path 감사에는 출판 chain 검사를 더 강화해야 한다.** Code-only review에서
   cold auditor의 `globalfreeze.models`와 artifact의 직접 결합, access event의 정확한
   participant×fold×interface×window 집합, manifest 원본 bytes/고정 파일명 결합,
   과거 cold-audit failure 우선권의 추가 검사를 찾았다. Live reader의 freeze 검증은
   이미 있었지만 독립 cold 감사에서도 별도로 확인해야 한다. 이번에는 최종 query/result가
   없어 그 경로를 실행했다고 주장하지 않는다. **실제 실패 경로**에는 아래 전용 cold 감사로
   exact manifest/artifact/access 결합을 적용했다.

다음 연구 작업의 범위는 이 정렬 문제에 대한 **별도 수학·인공 검증 설계**까지가 적절하다.
현재 파일을 다시 실행하거나 held60을 열지 않는다. 더 많은 데이터가 이번 수치 실패의
직접 해결책이라는 근거도 없다. 향후 독립 paired-acquisition 자료는 일반화 검증의
별도 공백이지 이번 실패를 없애는 우회로가 아니다.

## 검증·보존 범위

- 실행 전 전체2213tests PASS251.86초,68개 기존 Torch warnings. 실패 뒤 추가한 진단2개와
  access-multiset 감사1개도 PASS; 그 뒤 전체 suite를 다시 돌렸다고 주장하지 않는다.
- CPU scalar/batch/CUDA 실제200step 인공 parity와416-case workload 측정은
  [batch receipt](reports/task_trca_shape_batch_engineering_v1.json)에 있다. 실제 GPU fit은
  정상 실행됐고 실패 뒤 사용하던 GPU 메모리는 반환됐다. 기존 다른 GPU process는 보존했다.
- 실패 원본/start/source0/source1/model0/access/events는 모두 보존했다.
  별도 process의 `failure_audit.json`은 **18.287초 PASS**: 원 manifest bytes와 내장본문,
  source/model 파일 경로·해시, 정확한1456개 접근 multiset, 첫 분할의 저장자료 검산을 확인했다.
  접근은 support832회/source block5 416회/M-prefix208회이고 query-bearing call은0이다.
- 실제 원자료 전처리의 두번째 전체 구현, 모든 Adam gradient 재현, 완성된3fold의
  정확도·보정량 검산은 이 감사의 범위가 아니다. 존재하지 않는 최종 score를 검산하지 않았다.
- artifact root `/home/whwovy/task-trca-shape-source39-v1-attempt1`, 실패SHA
  `78e540eb45e93378219127c626437a05acecac46633ae8e1fa37924a079f37ae`,
  전용감사SHA `d93575ac15604e55c9d104ff70f09aa33900d07b974456f9784a63d279c422b7`.
  최종 계산 산출물은 약406MiB로12GiB 예산 이내다.
- `coordinate-worktree-changes`로 reader/auditor 구현을 별도 tree에서 분리하고 main이
  통합·최종 검증했다. 기존34+신규2worktrees를 보존했고 설치·push·cleanup·삭제는 없다.
  `academic-research`는 검색량 확대가 아니라 측정 근거·해석 한계·다음 공백 기록에 사용했다.

**이 후보는 종료. 원 연구목표는 유지. Metadata 효능/보정량 절감은 여전히 미확립이며,
이번 실행에서는 효능 평가 자체에 도달하지 못했다.**
