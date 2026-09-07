# 현재 자료 우선 — 연구계획 수정과 단일 선택공간 진단

2026-09-08 사용자 ‘응 그렇게하자’ 승인. 연구목표는 새 사용자의 적은 labeled SSVEP 보정이며,
acquisition metadata는 이를 위한 수단이다. **새 독립 paired-M 확보를 모든 후속 연구의 필수 입장 조건으로 두지 않는다.**
현재 Wearable의 EEG와 사전 block별 임피던스로 같은 측정환경의 좁은 가설을 연구할 수 있다.
이 판단은 모든 효과에 충분한 통계적 검정력이 있다는 보장이나 최근 실패의 원인이 표본수라는 주장이 아니다.

## 무엇을 수정하고, 무엇을 보존하나

- [Known-zero 결과](native_subset_known_zero_source39_results.md)의 ‘다음 연구에는 독립 자료가 먼저’라는
  운영상 요구를 앞으로의 일반 필수조건에서 내린다. 외부 자료는 장비/기관/획득환경 일반화 및 별도 재현용 보강이다.
- [2026-09-07 재검토](research_decision_reaudit_20260907.md)의 ‘현재 oracle 실행 미포함’을
  **아래 단일 저장 예측 진단에 한해서** 변경한다. 옛 실행계획/JSON/종료 결과는 수정하거나 재실행하지 않는다.
- 직접 M 방식1회＋기작 수정1회의 종료와 no-go 해석은 그대로다. 이번은 제3의 효능 후보가 아니다.
- 개발39명은 반복노출된 자료다. 보존60명은 최종 성능 미개봉 유지하며 새 개발용으로 전환하지 않는다.
  같은 코호트 새 참가자 확인은 외부 획득환경 재현과 다르다. 향후 공개 native 전체-cohort preset 상속 문제와
  표본 정밀도/실용효과/최종 프로토콜을 검토한 뒤 별도 권한이 필요하다.
- S1–S3, raw EEG, 전체 manifest/임피던스, 새 metadata packet, 외부 메일/다운로드는 이번 범위 밖이다.

## 한 번만 할 진단: 후보들 안에 정답이 있었나?

현재 정책은 36/60 labels를 이미 모은 뒤 FULL 또는 한 block을 제외한 모델을 고른다.
먼저 이 **고정된 선택지 자체에** 얼마나 개선 여지가 있는지 확인한다.

각 query에서 후보들이 A,A,B를 예측하고 정답이 B라면, 정답을 아는 이상적 선택은 맞힐 수 있다.
정답이 C라면 어떤 선택기도 이 후보들만으로 맞힐 수 없다. 이 이상적 선택의 정확도를 상한으로 계산한다.
실제 배포 시 정답은 없으므로 **구현 가능한 모델도, metadata로 달성한 정확도도 아니다.**

고정 계약: `configs/analysis/native_subset_headroom_source39_v1.json`.
입력은 완료 envelope-r1 `features.npz`와 known-zero `result.json` 두 SHA고정 파일만이다.
**Known-zero 수정 후** Q/QM 최종 actions의 BA를 해당 결과와 먼저 맞춘다. 원 envelope-r1의
수정 전 actions와 혼동하지 않으며, 새 선택 gain/학습/metadata변환은 계산하지 않는다.
기존 signed native 가중치·float64 dot·argmax 동률순서·class/block 순서를 그대로 쓴다.
k3는 FULL＋3개 omission, k5는 FULL＋5개 omission이며 padding/A0를 후보에 더하지 않는다.

보고할 것은 전39명×8조건×k3/5의624행이다.

1. 후보의 예측 불일치, 모든 후보 오답, 불일치하지만 모두 오답인 빈도.
2. 어떤 후보든 정답인 hindsight 상한과, 고정 Q/QM이 놓친 회복 가능 비교 수.
3. Oracle3−고정Q5, 36/60labels 및 기존0/3/5 grid의 최초80% 도달 **가능성 상한**.
4. 개인 안에서8조건을 평균한 뒤39명 기술적 구간과 모든8조건 표. Window/query를 독립 사람으로 세지 않는다.

Oracle3/5 후보군은 서로 중첩집합이 아니므로 비용 증가에 따른 단조성을 강제하지 않는다.
K0는 원 A0를 사용하며, 최초도달은 이후 성능 유지나 미측정k1/2/4의 최소비용을 보장하지 않는다.
원래 획득한 label 비용은 빼지 않으며 사용자 총 소요시간 절감으로 바꾸지 않는다.

## 결과 해석과 중단

| 관측 | 허용되는 해석 | 금지되는 결론 |
| --- | --- | --- |
| Oracle조차Q보다 거의 못 올라감 | 이 고정 후보공간의 남은 이득이 작다 | metadata 일반에 정보가 없다 |
| Oracle는 높지만Q/QM은 낮음 | 후보 안의 정답을 선택하는 데 미해결 여지가 있다 | M이 그 정답 후보를 식별한다 / 새 모델이면 성공한다 |
| Oracle3도Q5에 못 미침 | 해당 자료/후보공간에서 선택만으로5→3 비용 대체에 한계 | 모든 low-calibration 방법이 불가능하다 |
| Oracle3이Q5에 근접하거나 능가 | label-aware한 관측자료 상한은 허용한다 | 실제 M 기반 보정 절감 달성 |

어느 결과든 **이 진단 보고로 멈춘다.** 새로운 learner/threshold/후보pool/조건선택을 자동 추가하지 않는다.
다음 기작 연구가 필요하면 ‘M이 Q 외에 무엇을 알려주며 어떤 학습 결정을 바꾸는가’라는
가설을 새로 제시하고, 정보 접근/대조군/실용성/중단 기준을 별도로 고정한다.
이 진단은 M의 조건부 정보 여부를 직접 검정하지 않는다. 개선 상한은 그 질문의 답이 아니다.

## 실행·검증·기록

새 시작은 입력 byte 읽기보다 먼저, 새 출력 root `headroom-source39-v1`의 start/result만0400 배타 게시한다.
CPU1/BLAS1,120초,입력128MiB/출력4MiB. 소스/설정/입력 SHA와 commit을 기록하고 원본 해시를 재확인한다.
입력 검증 실패는 과학적 실패가 아니다. 같은 attempt 재시도·추가 입력으로 복구는 하지 않는다.
독립 scalar vs vectorized oracle, 실제 dot 순서와 독립 score 일치, 원 BA 재현, 모든 선택 인덱스/grid,
동답/모두오답/회복가능/중복·순열/범위오류 사례를 인공 테스트한다. Oracle action/훈련 target은 게시하지 않는다.

`academic-research`: 기존 측정/문헌/가설을 분리하고 external-gap을 보강 역할로 수정한다.
새 광범위 문헌 탐색이나 추가 자료 확보가 필요하다고 가정하지 않는다. 이번 새 검색/PDF는0이다.
`coordinate-worktree-changes`: base main cfc1f25, clean. 약263GiB 가용, 기존29worktrees 보존.
명세/코드/테스트/문서/SQLite는 main 단일 writer이고 두 reviewer는 공유 repository 읽기 전용이다.
작업이 같은 명세에 의존하므로 별도 writing lane/worktree는 추가하지 않는다. Temp예산1GiB, 환경 설치/push/cleanup0.
