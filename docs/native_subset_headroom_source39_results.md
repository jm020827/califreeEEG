# 현재 자료 선택공간 진단 — 실행 오류 종료, 새 과학 결과 없음

2026-09-08. 상태: **infrastructure-inconclusive / start-only**.
연구계획 정정은 완료했지만, 실제 headroom 진단은 계산 전에 중단됐다.
상한이 낮거나 metadata 가설이 실패했다는 결과가 아니다.

## 연구 방향에서 바뀐 것

[현재 자료 우선 계획](current_data_first_research_plan.md)에 따라 독립 paired-M 확보는
모든 개발의 필수조건이 아니라 외부 획득환경 재현을 위한 선택적 보강이다.
기존 Wearable로 같은 측정환경의 좁은 연구를 할 수 있다. 개발39명은 반복노출됐으므로
독립 확인으로 부르지 않으며, 보존60명도 공개 whole-cohort preset 상속의 한계가 있다.
기존 실패의 원인을 표본 부족이라고 확정하지 않는다.

이번 질문은 ‘저장된 후보들 중 정답을 맞힌 후보가 얼마나 있었나?’였다.
실제 배포 시 모르는 정답을 이용한 사후 상한으로, metadata 자체의 유용성이나
보정 label 절감을 입증하는 실험이 아니다. 기존 known-zero의 거의0인 M 증분은 그대로다.

## 어디서 중단됐나

1. Clean main `9d32ee2edee6ee77d4d7b21cfdc48a7c27e0d2d9`에서 새 start를 배타 게시했다.
2. 허용된 두 입력의 byte를 읽고 SHA를 검사했으며, known-zero JSON의 기존 provenance를 확인했다.
3. NPZ를 열 때 NumPy가 뒤늦게 표준 라이브러리 `zipfile`을 import하려 했다.
   정확한 입력 경로만 허용한 접근 제한이 이 미리 준비하지 않은 import를 차단했다.
4. 따라서 cache 배열 로딩·native 점수·기존 BA 재현·oracle·새 집계·result 게시에는 도달하지 않았다.

오류는 `ValueError: Undeclared input: /usr/lib/python3.10/__pycache__/zipfile.cpython-310.pyc`다.
Wall0.37초/user0.26/system0.10/maxRSS215236KiB/exit1. 실제 시도1회, 새 과학 집계0회,
추가 metadata/raw/held60/S1–S3 접근0, policy fit0, 재실행0이다.
두 승인 입력은 읽었으므로 ‘사람 자료를 전혀 읽지 않았다’고 표현하지 않는다.

인공 child가 pytest와 test fixture를 먼저 import하면서 ZIP 의존성을 미리 로드한 것이
사전1555개 테스트가 이 cold-start 차이를 놓친 이유다. 이는 실행·테스트 구현의 오류다.
경계 검토자가 start receipt와 실행 코드 순서를 독립 확인했다. Receipt만으로 운영체제 전체의
runtime read trace를 독립 증명한 것은 아니며, 위 순서는 실행 오류와 소스 점검에 근거한다.

## 보존한 증거

| 항목 | 값 |
| --- | --- |
| 고정 plan SHA256 | `d60578316bf55819f3ae19ae17f1dc43c506fed3fe8aaafd2dff8306d87644f6` |
| 실행 source tree | `eb71dc895ae7e717e0e9b362500b9bc64fd669b8` |
| 출력 폴더 | `/home/whwovy/native-subset-m-artifacts/headroom-source39-v1` |
| 유일한 파일 | `start.json`, 2256bytes, mode0400, nlink1 |
| start SHA256 | `f51ab2bd25ddcce6e0a3fa11e5555dfd0a078f7af3c2bcb1d8a48ef80c35e191` |
| source features SHA256 | `3e70b32a77bbaf10cd77fac603b7e808183134f27ceb5f839c475c69161523ae` |
| known-zero result SHA256 | `d49d8e7c0b367c0eca15a5504e25dbc9b5ee20ac13f20ef0d2792cd5f2999f4f` |

실패 뒤 두 원본 해시는 불변임을 확인했다. `result.json`은 없으며 성공 receipt나 숫자를 만들지 않는다.
옛 과학계획·도구·결과를 덮어쓰지 않고 실패 기록도 삭제하지 않는다.

## 수정 및 다음 경계

ZIP 모듈을 접근 제한 전에 명시적으로 로드하게 고쳤다. 인공 subprocess도 pytest/test fixture를
전혀 import하지 않는 새 interpreter로 바꿨으며, 성공·초기해시 실패·후기재해시 실패와 재시도 차단을
포함한 관련35개 테스트가9.63초에 통과했다. 입력·가설·후보·계산·보고 방법은 바꾸지 않았다.
수정 runner SHA256 `91d5a8dd7b73f4a53a764fb2f4eeae65519b1629d0735f8d6ef591b46426e57e`,
수정 tests SHA256 `d7215bfb3614a3a512e58ba564021fe2e205f956971dfe69692311d1f5b16d09`다.
수정 후 전체 회귀검사도 **1555 passed, 기존 warnings68,198.97초**로 통과했고 Ruff lint/format과
diff check를 통과했다. 이것은 코드 검증이며 실제 headroom 결과를 대신하지 않는다.

현재 attempt는 시작 기록을 남긴 채 종료했다. 수정 코드가 있어도 이 폴더를 재사용하지 않는다.
새 실제 집계에는 실패를 보존한 별도 execution attempt에 대한 명시적 승인이 필요하며,
현재 재실행은 승인되지 않았다. 새 데이터·새 모델을
먼저 구하거나 추가하는 일이 아니라, 같은 고정 과학 진단의 실행 문제를 해결하는 단계다.
현재 결과로 선택공간이 충분한지, metadata가 좋은 후보를 구별할지에 관한 결론은 아직 못 낸다.

`academic-research`에 따라 운영상 계획 정정, 기존 과학 측정, 새 실행 오류를 분리해 연구공간에 남긴다.
`coordinate-worktree-changes`에 따라 main만 편집하고 두 agent는 읽기 전용 검토를 수행했다.
기존29개 worktree는 보존했으며 새 worktree/환경 설치/외부 요청/cleanup/push는 없다.
