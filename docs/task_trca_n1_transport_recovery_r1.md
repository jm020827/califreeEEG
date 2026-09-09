# N1 source39 transport recovery r1

최종 상태(2026-09-10 KST): **24,000updates·전체 평가·단일 독립 cold 완료**.
Metadata 추가 이득 미확립, 관측 보정량 절감0으로 후보를 종료했다.
[최종 결과와 후속 제안](task_trca_n1_transport_recovery_r1_results.md).
아래는 실행 전 계약과 진행 이력이며, 현재 상태는 최종 결과가 우선한다.

2026-09-09 사용자 “고치고 계속.”에 따라 출력/작업수명 복구와 동일과학 새 실자료1회,
그 terminal 감사를 승인 범위로 적용한다. [고정 envelope](../configs/analysis/task_trca_n1_transport_recovery_r1.json).

직접 근거는 이전 failure의 stdout `print` BrokenPipe다. 기대되는 차이는 **출력 수신측이 사라져도
파일 로그와 계산이 유지되는 것**이며, 정확도 향상을 예측한 방법 수정이 아니다.
원36개 학습·평가·준비·감사·과학 pin은 전부 불변이다. 기존 자원800/생성24k+독립cold PASS를
그대로 재사용하고, 새24k 인공 재학습으로 보고하지 않는다. 생성 단절 검사와 관련tests 합300초/
128MiB, 새human1/24k/CPU1/6h/RSS16GiB/출력12GiB, terminal최대1/30분, 신호유예10초,
총새출력16GiB다. 새로운 학습실패나 효능 음성을 복구 이유로 바꾸지 않는다.

## 변경과 계보

기존 `EventJournal`은 수정하지 않는다. 새 launcher/supervisor가 stdin을 DEVNULL,
stdout/stderr를 일반 파일에 연결하고, supervisor와 그 child 각각 새 session을 소유한다.
따라서 원 코드의 print 목적지는 대화 pipe가 아니며 출력 소비자의 생존에 의존하지 않는다.
관련 파일의 단순 같은-tree 순차 변경이므로 root가 단독 작성한다. 두 agent는 읽기 전용으로
프로세스 수명·권한/계보를 검토하며 새 worktree를 만들지 않는다. 기존40trees/4untracked보존.
처음 가용167,931,512KiB에서 새16GiB 한도를 감당할 수 있다.

원 prepare는 빈 parent를 요구하므로 새 envelope를 먼저 저장소에 고정하고, 코드/tests를
검증·커밋한 다음 원 prepare로 새 manifest를 작성한다. Parent는
`/home/whwovy/task-trca-n1-source39-recovery-r1-1i9cau`로 유일하게 고정한다.
이후 parent/supervisor의 `recovery.json`에 원실패/감사/36pins, 새manifest, 새코드/설계/test
receipt hashes를 묶는다. 배타적인 `launch_claim.json`과 `supervisor_start.json`은 실행1회를
제한한다. Existing manifest의 정확한 schema/primary1 basename은 변경하지 않는다.
`old_attempts_reopened=false`는 옛 output/model을 재사용하지 않는다는 뜻이며, 이번 인프라
재실행 이력을 숨기는 뜻이 아니다. 새 권한은 외부 recovery envelope가 맡고 기존 cold는
원 과학 실행만 검증한다. 마지막에 이 두 계보를 함께 확인한다.

PID·stdout/stderr·종료/실행수명 receipts는 모두 primary output **밖의** supervisor 폴더에 둔다.
각 child의 유일 reaper가 wait4로 exit/rusage를 기록한다. Timeout은 소유 process group에만
SIGINT→10초 유예→SIGKILL이며 재시도하지 않는다. Primary failure가 있으면 failure-only,
정상exit0+result이면 complete cold를 최대1회 실행한다. 강제종료/저장장치 장애로 둘 다 없으면
supervisor-only 실패를 남기고 감사 미실행·효능 미평가로 닫는다. Producer evidence를 만들어내지 않는다.
새 session은 임의의host/cgroup/메모리강제종료 생존을 보증하지 않는다.

## 검증·중단·해석

새 코드의 테스트는 실제 원EventJournal을 일반 stdout 파일에 연결한 채 launcher의 출력 pipe와
부모를 종료시켜 자식이 계속 기록하는지 확인한다. 같은 EventJournal의 직접 closed-pipe 대조는
기존 실패에 민감해야 한다. 배타 실행, 올바른 감사선택, receipt 누락, disk-error 전파,
소유 group watchdog도 검사한다. 사람 EEG/metadata/성능은 이 검사에 사용하지 않는다.
고정36pins·원실패불변·새권한binding을 실제 실행 전과 종료 후 재검증한다.
원목표는 유지하며 outcome은 전체cold 통과 후에만 해석한다. Held60/외부요청/유료/GPU0,
추가복구0, 모든 이전 음성·실패 보존이다. Academic 근거맵에는 이전 결과와 분리해 기록한다.

## 쉽게 보는 이번 실험

질문은 **“같은 양의 보정 EEG를 쓸 때, 전극 접촉 상태의 수치까지 알면 더 잘 학습할 수
있는가? 그 차이가 보정 EEG를 덜 모아도 될 만큼 큰가?”**이다. 손상 EEG 복원이나
새로운 연구주제로 바꾼 실험이 아니다. 이번 복구는 그 질문을 그대로 두고 실행 중 로그
연결이 끊겨도 계산이 유지되도록 한 것이다.

- **Q**는 보정 EEG 자체에서 계산한 15개 특징을 사용한다. 결측 여부·측정 순서 같은
  공통 정보는 비교군 사이에 맞춰 둔다.
- **QM**은 먼저 학습한 Q를 고정하고, 보정 시점까지의 채널별 임피던스 수치에서 얻은
  2개 특징을 추가한다. 이 특징은 채널별 정규화 강도를 조금 조절하고, 그 결과 SSVEP
  분류에 쓰는 공간 필터가 달라질 수 있다. 임피던스만으로 자극 정답을 예측하는 모델은 아니다.
- **Q2**는 같은 크기의 추가 학습 모듈에 EEG 특징을 넣는다. **SHAM**은 조건을 맞춘
  다른 사람의 임피던스를 넣고 별도로 학습한다. 단, 조건이 맞는 사람이 없는 단독 그룹은
  본인 값을 유지하며, 실제 값이 바뀐 비율도 함께 검사한다. QM이 이들보다도 나아야 단순히 추가
  파라미터나 공통 정보 때문이라는 설명을 좁힐 수 있다. 모든 EEG 정보를 통제했다거나
  임피던스의 인과효과를 증명했다는 뜻은 아니다.

39명을 세 묶음으로 나누어, 매번 26명의 자료로 학습하고 다른 13명을 평가한다. 세 묶음의
모델을 모두 확정한 뒤에만 최종 평가 블록에 접근한다. 그러나 이 39명은 연구 과정에서
이미 여러 번 사용한 **개발자료**이므로, 이 분할만으로 독립 확증 자료가 되지는 않는다.

먼저 36개 보정 trial을 사용하는 QM과 Q/Q2/SHAM의 정확도를 비교한다. 그다음 QM의
36개 결과가 Q의 60개 결과 및 기존 분류기와 비교해 사전 허용폭 1%p 이내인지 확인한다
(참가자별 차이의 기술통계용 구간 하한이 −1%p보다 커야 한다). 별도로
0·36·60개라는 관측 지점 중 48개 평가 trial에서 39개 이상 맞힌 첫 지점을 비교한다.
양쪽 모두 목표에 도달한 조건의 보정량 차이와 새로 도달/도달을 잃은 조건을 함께 보고한다.
이는 **관측한 보정 trial 수의 절감** 검사이며, 실제 사용 중 언제 측정을 중단할지 정하는
정책이나 전극 준비·임피던스 측정까지 포함한 전체 시간 절감을 검증하는 것은 아니다.

## 실행 전 검사 기록

구현·실행 전 검증 완료. 새human numeric reads0, registered recovery primary0.
원EventJournal 직접 closed-pipe 대조는 예상대로 실패했고, 새 일반파일 경로에서는 launcher의
console/부모 종료 뒤에도 마지막 이벤트가 보존됐다. 관련301tests(새26+기존275) PASS29.16초,
종료분기 테스트 민감도를 보강한 최종 새26tests PASS5.39초다. 네 pytest 호출 합45.07초에
기타검사20초를 예약해도300초 이내, 임시출력 약53.0MiB<128MiB다. 원36pins전부불변/Ruff PASS.
[원 검사 기록](/home/whwovy/task-trca-n1-transport-tests-tc0N2Q/preflight.json).
수학 관련3575전체검사는 이전 원프로그램의 증거로 유지하며 이번에 다시 실행했다고 하지 않는다.
독립 읽기전용 검토가 발견한 종료 직전 한도/감사watchdog/잔여자손3항목은 실자료 시작 전 수정했다.
결과는 새 manifest/복구 receipt를 고정하고 단1회 실행한 뒤 추가한다.

### 실제 시작

Clean91b7282에서 원prepare를1회 실행해 새manifest SHA
`e940f436781292b32a8e6fffb32f8573069767472ea1aad9f8337b90ebff996c`를 고정했다.
원manifest와 출력 경로 외 모든 필드가 같음을 실행기가 검사했다. Recovery receipt SHA
`ed8dfe176d418a3280a96c2a4a3d34d66fbe67a0cdfb3ebf64fc906e7c81d179`에 원36pins와
새코드/tests/검사근거/원실패/새권한을 결합했다. Prepare는numericreads0/process.23초다.
2026-09-09T12:31:10Z 단1회 launch. Supervisor684632(PPID1), primary684633 각각 독립SID/PGID,
실제 /proc/fd0은/dev/null, fd1/2는supervisor/primary.stdout.log·stderr.log 일반파일임을 확인했다.
첫416sourcecases/728fit읽기 준비 후 학습 중이며 이 시작 snapshot에서는query0/완료updates0이다.
이후 완료 카운터는 primary의events, 종료/자원/감사는supervisor receipts가 기준이다.
**자동 감사는 supervisor가 소유하므로 수동으로 두 번째 감사를 실행하지 않는다.**
현재efficacy/calibration미평가. [기계 상태](reports/task_trca_n1_transport_recovery_r1_state.json).

13:33:17Z에8,800updates 완료와다음inner_start(seq74)를기록했다. 이전실패의seq73출력지점을
통과했으며 model0저장/1456fit-only읽기/query0이다. 전체학습완료·효능성공을뜻하지않는다.

14:50:17Z 확인에서는20,000/24,000updates 완료, 마지막 outer의inner학습 중이었다.
2개 outer모델 저장/2184fit-only읽기/query0이며 supervisor와primary는생존해있었다.
이 진행 수치는완료결과나효능판정이아니며최종receipt가우선한다.

15:22:08Z supervisor가complete cold PASS를결합해정상종료했다. Primary24,000updates,
query-bearing1248,access4213/events182,terminal `METADATA_INCREMENT_NOT_ESTABLISHED`다.
전체312조건에서Q와QM의관측보정량이같았다. 실행실패가아니라유효한개발음성결과다.
원36pins불변/추가실행0이며숫자·한계·예산·다음제안은최종보고서에기록한다.
