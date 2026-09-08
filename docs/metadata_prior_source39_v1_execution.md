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

## 실제 native 변환 중 통합 검증

- Export는 clean ed14df5에서 시작했다. 39명 외에는 읽지 않고 새 native 파일만 작성한다.
- Auditor7206df1을 main5822351로 충돌 없이 통합했다. 별도60개 인공시험 PASS.
  실제 producer가 만든 인공9명 결과를 독립 auditor가 읽는 교차 통합시험도 통과했다.
- 첫 전체 회귀검사:1709PASS/3FAIL/14ERROR/68기존warnings,232.30초.
  실패17개는 모두 기존 V2 외부 fixture의 임시 부모 경로 `cfeg-prior-source.TMPUtA`의
  `source` 단어가 기존 보호 regex에 차단된 원인이다. 이 검사는 실제 외부 데이터를 열지 않았다.
  보호장치/기존 tests/과학 설정을 바꾸지 않고 중립 경로로 검사한다.
- 신규 core/export/audit와 기존 V2 요청 검사를 합쳐133PASS14.69초.
  실제 자료에서 학습·Q+M outcome은 아직 시작 전이며 native 변환은 진행 중이다.

## Native 완료 해시 확인 오류와 동일 과학 복구

- 첫 export는39명 마지막 파일까지309.326초에 작성했지만, 완료 전 프로젝트 source 해시 확인에서
  `metadata_trca_prior.py`가 runtime 허용 경로에 없어 RuntimeError로 종료했다.
  실제 raw/native A0·FULL 계산은 이루어졌다. 학습된 Q/QM outcome과 numeric M은 아직 열지 않았다.
- 실패 경로 `/home/whwovy/metadata-prior-source39-v1/native`의 start와39NPZ를 그대로 보존한다.
  Start SHA `0af0f7d02e7a5b31494133fb1a99924bbcc55c7713df49028479552ac15df1c7`;
  결과 manifest 없음. 원 exporter SHA539566b9b7c962fc6da8fc0a1965c1f5f838f985a73d3572d754925bdfe6490c.
- 고친 부분은 최종 해시 검사용 정확한 SRC helper2경로 read 허용과 별도 native-cold-r1
  경로·복구 receipt다. 원 JSON SHA, preprocessing/weights/operator/feature/선택/분할/판정은 불변이다.
  Exporter가 같은 native 계산을 별도 경로에서 수행하고, analysis와 auditor는 실패start 해시와
  recovery 이유·새 경로·고정plan을 명시적으로 연결한다. 실패 파일의 덮어쓰기/재사용은 없다.
- 보존 파일 약2.986GiB를 전체8GiB 예산에서 차감한다. 별도39파일 약2.986GiB,
  analysis와 두 회귀검사 temp까지 예산 내다. 이는 독립 과학 반복/새 후보가 아닌 동일 입력의 실행 복구다.
- 중립 경로 전체1786PASS218.39초/기존68warnings. 이 collection 이후 추가한 recovery5검사와
  runner/auditor 수정을 포함한81tests도 PASS9.64초다. 최종 exporter 수정 통합 후 다시 검증한다.

## 실제 실행·최종 판정 완료

- 복구export97a72aa는308.83초 COMPLETE,39NPZ/3,205,578,402bytes.
  실패/복구39파일 바이트 동일. 원본 실패는 보존했다.
- 같은97a72aa의 analysis109.74초 COMPLETE. 39명×8조건×k3/5/48queries,
  5928rows/152summary/99contrast/2808attainment. 모든fit/freeze를 learnedqueryscoring 전에 게시했다.
- 원 auditor PASS:840selectedfits+분할·M projection·prior/proxy·점수·통계·보정량 검산.
  Gamma0 native correlation 최대7.96e−12/예측불일치0. 전체1797tests PASS234.16초.
- 최종 판정 METADATA_INCREMENT_NOT_ESTABLISHED. QM3−Q3 −0.006677%p,
  QM5−Q5 0%p,312개 관측 최초80%도달 단계 모두 동일, 보정label절감0.
  상세 수치·한계·조건별 결과는 [최종 분석](metadata_prior_source39_v1_results.md)에 있다.
- 추가 독립검토가 float취소오차로 일부참가자 signcount를 잘못 분류함을 발견했다.
  별도96c0ddd→main8dc17be의 정수보고검산은 모델을 재실행하지 않고4개ALLk5집계를 정정했다.
  99contrast 평균/CI 최대차5.55e−17/보정량·판정불변, REPORTING_CORRECTION_VERIFIED.
  원result/audit불변; 정정receipt SHA32a39fbe02c741280121418d21d34f9f6e523f417a7f03989fcb55efca77bbaf.
  관련104tests PASS9.85초이며 정정helper를 포함한 최종전체회귀검사는 별도 기록한다.
- 이번 후보 실행은 종료했다. 연구목표의 효능달성·독립확증과 같지 않다.
  추가실험/held60개봉/외부요청/설치/push/cleanup은 없다.

최종 정수보고 helper 포함1820tests PASS237.72초/기존warnings68, Ruff9files PASS.
원artifact·source·정정receipt12hash와 JSON원본/정정 projection 일치, 관련local links108개,
SQLitequick_check ok를 확인했다. 확인한 study/temp/2worktree 합계 약6.93GiB로8GiB예산 안이다.
기존30+신규2worktrees/실패·복구자료는 보존하며 삭제하지 않았다.

## 최종 협업·통합 인계

통합 기준은 main의 d954870이며 실제 source 학습/평가는97a72aa, 정수보고는8dc17be,
결과문서/연구일지 정리는5b33af3에 기록됐다. 모든 변경은 로컬이고 push하지 않았다.

| Lane | Worktree / branch | 단독 작성 범위 | 통합·보존 상태 |
|---|---|---|---|
| Main | /home/whwovy/califreeEEG / main | 계약·core·runner·root tests·문서·연구공간·최종 실행 | 통합·전체1820tests 검증 담당 |
| Exporter | /home/whwovy/califreeEEG-wt-prior-export / codex/metadata-prior-source-export-v1 | export_metadata_prior_source.py와 해당 test | e53bcc7→52bcc2f, 복구fde4fd3→97a72aa; clean 보존 |
| Auditor | /home/whwovy/califreeEEG-wt-prior-audit / codex/metadata-prior-source-audit-v1 | 원auditor/test, 이후 별도counts auditor/test | 7206df1→5822351,96c0ddd→8dc17be; clean 보존 |

읽기 전용 과학·결과 검토자는 shared repo를 사용했고 새tree를 만들지 않았다.
두 agent의 schema clarification cherry-pick은 main0d1b0cd와 같은 내용이므로 중복 통합하지 않았다.
텍스트 충돌0이지만 수식 연결·guard·정수 집계는 별도 의미 검토와 실제 실행으로 검증했다.
남은32worktrees는 사용자 작업/복구 이력으로 보존하며 이번 요청에 없는 cleanup은 수행하지 않았다.
