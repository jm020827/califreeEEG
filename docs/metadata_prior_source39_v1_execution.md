# Source39 metadata prior 실행 기록

2026-09-08. 이 문서는 고정 설계의 실제 실행 이력을 누적한다.
과학 권위는 [설계](metadata_prior_source39_v1_design.md)와 SHA256
`ed67cc1f361ed89c37b6f7c5df1e9170b78484e428a5105020485c8c03e3934b`의 JSON이다.

## 실제 자료 개봉 전 검증

- 계약4c825ee → 실제 자료를 보기 전 정의 명료화0d1b0cd → exporter52bcc2f 순으로 main 통합.
- 새 학습/평가 core와 runner: 인공9명에서 실제 feature 추출, nested Q/residual 적합,
  prior/TRCA 점수, 저장, guard, no-overwrite를 연결한 cold lifecycle 통과.
  이 인공 native reference는 자체 operator로 만들었으므로 독립 native 일치 증거는 아니다.
- 새 root16개+export31개+기존 Q/operator 검증을 합친144tests PASS(7.99초).
  신규 파일 Ruff PASS. 실제 native 환경의 인공 전체4window export 검증도 별도 통과.
- 읽기 전용 과학 리뷰: fold/axis/동일 OOF residual/78 coverage/312 비용 집계 확인.
  필터 각도 진단은 `0.5*||u-v||*||u+v||`로 계산하여 동일 필터의 반올림 가짜 변화 제거.
- 이 절을 작성할 때 이번 실행의 사람 raw EEG, numeric source projection, 새 outcome은 아직 열지 않았다.
  실험 실패 후 후보를 바꾸는 절차는 없다. Export→fit→outcome→audit→terminal report를 수행한다.

독립 감사기는 별도 worktree에서 작성 중이며 통합 후 결과를 검산한다.
원본/held/기존 종료 산출물과 예전 연구 결론은 변경하지 않는다.
