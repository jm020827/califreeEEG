# Headroom cold-r1 — 승인된 별도 실행, 과학설계 불변

> 완료: [실제 결과·독립 검산](native_subset_headroom_cold_r1_results.md), [전체 집계](native_subset_headroom_cold_r1_tables.md).
> 이 실행은 종료됐다. 아래는 사전에 고정한 실행 계약이며 추가 실행 권한이 아니다.

2026-09-08 사용자 ‘응’은 ‘실패 기록을 보존하고 수정 코드로 별도 진단 실행1회’에 대한 승인이다.
이 문서는 [이전 start-only 실패](native_subset_headroom_source39_results.md)의 미승인 상태를
이 별도 실행에 한해 전망적으로 대체한다. 옛 시작/과학계약/실패 판정은 그대로 보존한다.

새 plan `configs/analysis/native_subset_headroom_source39_cold_r1.json`,
SHA `e7affb66540ed68703ac8aaea53c24841f32df944feaa4e698f073863b4e69a0`.
Study는 그대로이고 attempt/output만 `native-subset-headroom-source39-v1-cold-r1` /
`/home/whwovy/native-subset-m-artifacts/headroom-source39-v1-cold-r1`로 분리한다.
고정2입력, 모든 계산/보고/참가자/조건/후보/비용/CPU1·BLAS1·120초·입력128MiB·출력4MiB 불변이다.
코드는 이전 plan byte를 SHA/commit 검증하고 실행 신원·승인 외 과학 필드 변경을 거부한다.

기존 zipfile eager import와 pytest 없는 cold-child 검증을 사용한다. 새 runner 수정은 plan binding,
불변성 검증, receipt의 attempt/이전 실패 연결뿐이다. Policy fit·metadata packet·raw·held60·
retired 접근권한은 추가하지 않는다. 기존2184개 BA를 먼저 재현한 뒤624개 cell/18summary/9cost/
1248attainment를 계산한다. Oracle action/target은 출력하지 않고, 결과와 무관하게 추가 learner를 만들지 않는다.

실패한 start SHA `f51ab2bd25ddcce6e0a3fa11e5555dfd0a078f7af3c2bcb1d8a48ef80c35e191`를
실행 전후 확인하고 보존한다. 새 root의 배타 start→동일2입력 SHA→BA재현→새집계→입력재해시→result 순서다.
새 실행도 자동 재시도/덮어쓰기하지 않는다. 실제 계산이 끝나면 독립 읽기 전용 검산과 전체표를 남긴다.

Main base41e163b clean. 약255GiB 가용, temp1GiB 이내로 별도 환경/의존성 설치나 worktree가 필요 없다.
기존29worktrees 보존, root가 계약/실행/문서/SQLite를 단독 편집하며 agents는 독립 읽기 전용 검토다.
`coordinate-worktree-changes`의 가벼운 분담을 사용하며 cleanup/push0.
`academic-research` 연구공간을 재사용하고 새 광범위 검색/PDF 없이 측정·한계·기존 M 결론을 분리한다.
