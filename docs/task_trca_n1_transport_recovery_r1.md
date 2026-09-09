# N1 source39 transport recovery r1

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

## 상태

구현·실행 전 검증 완료. 새human numeric reads0, registered recovery primary0.
원EventJournal 직접 closed-pipe 대조는 예상대로 실패했고, 새 일반파일 경로에서는 launcher의
console/부모 종료 뒤에도 마지막 이벤트가 보존됐다. 관련301tests(새26+기존275) PASS29.16초,
종료분기 테스트 민감도를 보강한 최종 새26tests PASS5.39초다. 네 pytest 호출 합45.07초에
기타검사20초를 예약해도300초 이내, 임시출력 약53.0MiB<128MiB다. 원36pins전부불변/Ruff PASS.
[원 검사 기록](/home/whwovy/task-trca-n1-transport-tests-tc0N2Q/preflight.json).
수학 관련3575전체검사는 이전 원프로그램의 증거로 유지하며 이번에 다시 실행했다고 하지 않는다.
독립 읽기전용 검토가 발견한 종료 직전 한도/감사watchdog/잔여자손3항목은 실자료 시작 전 수정했다.
결과는 새 manifest/복구 receipt를 고정하고 단1회 실행한 뒤 추가한다.
