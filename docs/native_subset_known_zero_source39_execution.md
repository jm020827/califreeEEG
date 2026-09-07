# Known-zero source39 — 단회 실제 평가 실행 계약

2026-09-08 사용자 ‘시작’에 따른 실행이다. [기작 명세](native_subset_known_zero_v1_design.md)의
동일답 gain=0 규칙만 실제 개발 자료에 적용한다. 목표·모델·학습·조건·대조군은 바꾸지 않는다.
이 문서는 실행 전 기록이며, 아래 인공 PASS는 실제 성능 개선을 뜻하지 않는다.

## 고정된 경계

- 실행 JSON: `configs/analysis/native_subset_known_zero_source39_v1.json`, SHA
  `7b3e478892574d3edb78f7b6659935a2215cf369c619c7bd1da5fd6b68105258`, 계약 commit `8668cab`.
- 입력은 `/home/whwovy/native-subset-m-artifacts/source39-v1-envelope-r1`의 고정된5개 산출물뿐이다.
  원본 EEG·전체 impedance·manifest·held60·retired1–3는 읽지 않는다. 기존9개 fit을 정책에서 재학습하지 않는다.
- 새 출력은 `/home/whwovy/native-subset-m-artifacts/known-zero-source39-v1`의 `start.json`, `result.json`뿐이다.
  새 시작 기록을 부모 byte 읽기보다 먼저 배타 생성한다. 파일은 생성부터0400/nlink1이며 fsync한다.
  부모5개 SHA·총117526373bytes·출처·Q 재계산을 확인하고, 기존5개 결과표를 모두 재현한 뒤 새 선택을 계산한다.
  게시 직전 부모 해시를 재검사한다. 같은 attempt의 재시도·덮어쓰기·resume은 없다.
- Main 기존 venv Python3.10.12/NumPy1.26.4/SciPy1.15.3, CPU1/BLAS1,600초,
  입력128MiB·새 출력64MiB. 이는 저장된 feature의 재평가이며 원래 native fitting 환경의 재현이 아니다.
- 원 과학 JSON, 부모 amendment, 기존 helper5개는 SHA 고정이다. 신규 IO/producer/auditor와
  설정4개를 포함한 정확한12개 source hash를 clean main commit/tree와 함께 시작 기록에 남긴다.

## 공정성과 독립 검산

Primary는 수정 QM3−수정 Q3다. Q·QM·SHAM_REFIT·stale·각 shuffle 모두 같은 규칙을 쓰며,
결측은 수정된 Q로 돌아간다. 39명×dry/wet×4window×k3/5,36/60labels와 모든 보고 조건은 동일하다.
4680rows/120summary/153contrasts/624diagnostics/1248attainment를 유지하고624개 수정 진단을 더한다.
수정 전후 후보·답 변화와 수리/손상은 전 조건을 보고하며 유리한 사례 선택에 쓰지 않는다.

독립 auditor는 producer/IO/새 vectorized rule을 호출하지 않는다. 부모 provenance·Q/M·scaler·
9개 ridge와 원 보고서를 독립 재계산한 뒤 scalar 규칙으로 모든 새 결과와 진단을 검산한다.
9개 ridge 재계산은 감사 전용이며 새 정책 학습이나 모델 선택이 아니다. Raw/native fit 재추출 감사는 아니다.

원 성능과 수정 성능을 비교하되, Q와QM이 함께 개선된 것은 metadata 성공이 아니다.
한 번의 수정 후 평가가 끝나면 결과 부호와 무관하게 이 구현 탐색을 프로그램 점검한다.
새 후보·threshold·재학습·subgroup 탐색이나 held60 자동 개봉은 없다.

## 실행 전 검증 및 작업 분리

`coordinate-worktree-changes`에 따라 기존27개 worktree를 보존하고 독립 writing lane2개를 추가했다.
약282GiB 가용 공간에서 checkout 각12MiB, 인공시험 temp 합계2GiB로 예산을 정했다.
Main은 계약/IO/연결시험/문서/SQLite/최종 실행을 소유한다. Producer와 auditor는 각각 별도 worktree의
소유2파일만 편집했고 공유 tree 동시 쓰기는 없었다. 통합 순서는 계약→IO→producer→auditor→연결시험이다.

- Producer worktree `califreeEEG-wt-known-zero-eval`: `34a1c9e` → main `0a4a594`.
- Auditor worktree `califreeEEG-wt-known-zero-eval-audit`: `b5fe037` → main `99d133c`.
- IO28tests PASS0.07초, producer13tests PASS24.36초, auditor35tests PASS2.91초.
- 실제 크기/스키마의 **인공** 부모5파일→실제 runtime guard/IO→새 정책→독립 전체 감사
  2tests PASS22.26초. 입력 권위·temp 경로/preflight만 인공 경계로 바꾸고 수학·검증 함수는 그대로 실행했다.
  잘못된 부모 해시는 새 start만 남기며, 완료 경로 재시도는 거부됨을 확인했다.
- 검토에서 library 경로의 `..`/symlink 탈출 가능성을 발견해 **사람 자료 접근 전** 차단하고 회귀시험을 추가했다.
  이 guard는 application 수준의 실수 방지이며 같은 OS 사용자를 격리하는 보안 sandbox라는 주장은 하지 않는다.
- 전체 회귀시험 및 통합 검토의 최종 기록은 연구일지에 추가한 뒤 실제 실행한다.

`academic-research`는 기존 claim/gap과 실행 전 signature를 연결하는 데 사용했다.
새 문헌 검색/PDF 정독은0이다. 인공 검증·실제 측정·metadata 효능·독립 확인을 서로 다른 증거로 기록한다.
실행 signature는 연구공간의 `native_subset_known_zero_execution_signature_20260908.md`다.
Worktree/temp/이전 산출물은 보존하며 cleanup/push/환경 설치는 하지 않는다.

## 최종 실제 접근 전 통과 기록

Clean `95db187bd40900d90b322bdfa76d9c117d5f9f4b`에서 전체 **1520tests PASS197.03초**, 기존 Torch
warnings68개. 새7개 Python 파일 Ruff lint/format, diff check, 원본 불변 검사 PASS다.
같은 source의 읽기전용 통합 검토 CODE GO와 canonical preflight PASS(12source hashes)를 확인했다.
이 시점까지 실제 부모 산출물 byte 접근과 새 실제 attempt 시작은0이다. 이 기록을 commit한 뒤
동일 코드의 단회 실제 실행을 시작한다. 실제 결과는 별도 결과 문서/연구일지에 기록한다.
