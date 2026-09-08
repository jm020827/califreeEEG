# Task-shape v1 구현 작업 기록

2026-09-08, base15f853cee236fef8e93024d5f112946e1c7d474e/main. 사용자가 승인한 다음 단계는
고정 [구현 명세](task_aligned_trca_shape_v1_implementation.md)의 learner와 인공 검증이다.
사람 EEG/M/archive, held60, 기존 outcome/geometry artifact는 이번 입력이 아니다.
설계 JSON과 이전 결과는 불변이며 새 사람 실행 manifest는 만들거나 승인하지 않는다.

## 소유권과 예산

| Lane | Writer / 경로 | 공유 계약 | 자원·검사 | 통합 순서 |
|---|---|---|---|---|
| Root | main: `task_trca_shape_learning.py`, `task_trca_shape_inputs.py`, 해당 tests, 인공 check script, docs | API·입력 역할·학습 선택·최종 검증 | 기존3.10env, GPU main만, 인공 temp | 계약 → leaf → learner → 검증 |
| Operator | `/home/whwovy/califreeEEG-wt-task-shape-operator`, branch `codex/task-shape-operator-v1`: operator 및 전용test2files만 | float64 leading projector/Gram API | CPU threads1, 별도 mktemp pytest | 첫번째 |
| Features | `/home/whwovy/califreeEEG-wt-task-shape-features`, branch `codex/task-shape-features-v1`: features 및 전용test2files만 | mask-only Q/scaler/donor API | CPU threads1, 별도 mktemp pytest | 두번째 |
| Review | shared repo read-only | native compatibility/누수/반증 | 사람 artifact·GPU·SQLite 읽기/쓰기 없음 | 통합 전후 |

기존32worktrees 보존, 신규 writing2개. 가용약235GiB 대비 두 checkout64MiB, 인공/test artifacts1GiB
예산이다. 기존 `.venv`를 명시적 executable로 재사용하되 symlink/설치/의존성 변경은 없다.
동일 tree 동시writer, SQLite 병렬writer, push/cleanup/삭제는 없다.

## 고정 API

- Operator: `leading_projector(H[...,8,8])`, `bounded_filters(S,C,anchor,R)`;
  root의 S/C는[5,12,8,8], anchor[5,12,8], R[5,1,8]이고 출력[5,12,8]을
  [5,8,12]로 transpose해서 scoring한다. `gram_statistics(templates[12,5,8,N],query[n,5,8,N])`,
  `score_gram(filters,stats,weights[5])`. `native_zero_scores(support,query,weights)`는 원 프로젝트 경로 exact delegate.
- Features: `support_q_mask(support,boolean_mask,interface,order,frequencies)`;
  `metadata_features(packet)`; `fit_scaler(values,participant_ids,fit_ids,availability=None)`와
  `.transform`; `donor_map(ids,masks,orders,interface)`는 전달된 partition 안에서만 매핑.
- Learner는 fit용 block5 task와 validation task만 받는다. Evaluation support는 supervision과 분리한다.
  이번 reader는 **인공/메모리 역할 adapter**이며 실제 archive decoder나 사람 입력 guard의 완성으로 부르지 않는다.

실제 완료/검사 수치는 실행 후 append한다. 과거1877tests PASS를 새 learner 검증으로 재사용하지 않는다.
