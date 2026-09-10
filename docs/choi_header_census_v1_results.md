# Choi 구조 점검 종료 — 명시적 처리 이력은 확인하지 못함

2026-09-10. **한 파일의 이름·형식만 확인했다. 새 학습/효능 실험은 아니다.**

## 확인한 것

이미 헤더를 본 `S1/Day1/cnt_LOW(1).mat`을 읽기전용으로 한 번 열어 root와 `cnt`의
직접 자식 이름을 빠짐없이 열거했다. 해당 객체의 attribute 이름과 dataset shape/dtype도 확인했다.

| 범위 | 실제 키/attribute 이름 |
| --- | --- |
| root 직접 자식 | `#refs#`, `cnt` |
| root, `#refs#` attribute | 없음 |
| `cnt` 직접 자식 | `T`, `clab`, `fs`, `x`, `yUnit` |
| `cnt` attribute | `MATLAB_class`, `MATLAB_fields` |
| 다섯 dataset의 attribute | `H5PATH`, `MATLAB_class`; `yUnit`에는 추가로 `MATLAB_int_decode` |

**이 이름 공간에는 sensor mapping/export history/filter/marker 설정을 명시한 이름이 없었다.**
`#refs#` 내부, attribute 값, dataset 값, MAT user block은 읽지 않았다. 따라서 이 결과는
파일 전체·다른 참가자·모든 공개 코드에서 이력이 없다는 증명이 아니다.
`Gyro`는 별도 `cnt/Gyro` dataset이 아니라 기존에 확인한 `cnt.clab`의 채널 이름이다.
최소 문의 초안의 잘못된 경로 표기를 수정했으며 label 값을 이번에 다시 읽지는 않았다.

## 연구에 미치는 의미

원 목표는 바꾸지 않는다: **외부 acquisition metadata를 학습에 넣어 Q·공통 정보 이상의
이득과 실제 보정량 감소가 있는지 검증**하는 것이다. 이번에는 그 효과를 측정하지 않았다.

움직임 후보는 계속 **DEFER**다. 이유는 GPU나 학습기 부재가 아니라, 배포된 채널이 무엇을
측정했고 EEG와 어떻게 정렬·변환됐는지 아직 확정하지 못했기 때문이다. 이 상태에서
`Gyro`라는 이름만 믿고 특징을 넣으면 예상한 움직임 기작을 시험했다고 해석하기 어렵다.

다만 모든 metadata 방식에 절대 g 변환이나 support 이전 측정이 필요한 것은 아니다.
허용된 보정 EEG와 동시에 기록된 M을 보정 prefix 뒤 학습에 사용하고 첫 query 전에
모델을 고정하는 설계도 원 목표에 부합할 수 있다. upstream 미래 정보 혼입을 모르면서
이를 안전하다고 가정하지는 않는다. 기존 부정 결과·V2 종료·held60 보호는 그대로다.

## 다음 행동과 재개 조건

이 유한 점검은 종료한다. 새 단서 없이 anonymous reference/파형을 계속 뒤지거나
같은 종료 후보를 다시 맞추지 않는다. 다음으로 [미발송 문의](choi_export_minimal_clarification_20260910.md)의
두 주제를 저자에게 확인하는 것을 제안한다.

1. `Gyro` 채널과 실제 IMU 출력의 대응, EEG sample 정렬.
2. 실제 1,000→200 Hz export와 marker 변환 코드/설정, smoothing·지연·run 경계 처리.

요청 대상은 기존 설명/코드이며 새 참가자 자료나 개인정보가 아니다. **발송 승인은 아직 없고,
메일을 보내지 않았다.** 확인된 메일 발송 도구도 없어 자동 발송을 약속하지 않는다.
사용자가 직접 전달하거나 별도로 발송 경로를 정할 수 있다.

답변/코드가 오면 먼저 측정 정체성과 허용 시점의 의존성을 검증한다. 적격하면 기존 목표 아래
별도의 concurrent-support 후보·Q/Q2/SHAM 대조·실제 수집비용·후보 및 실행 예산을 사전 고정한다.
답변이 없거나 정보가 부족하면 unknown/DEFER를 유지한다. 답변 자체를 효과 입증으로 취하지 않는다.

## 실행·검증 기록

- Config `5a92853`, reader `0fcb9d06505c8d85040c17edc2ce102d08d0fc99` 동결 후 실제 1회 실행.
  0.001854초, stdout 2,858 bytes. 입력 크기 30,552,599 bytes 및 device/inode/mtime_ns 전후 일치.
  MAT 전체 checksum은 읽지 않았으며 stat 일치가 전체 내용의 암호학적 동일성 증명은 아니다.
- 생성 self-test 2/2호출, 각각 동일 4검사 PASS, fixture 6원소/디스크 fixture 0.
  Getter 차단은 Dataset/AttributeManager `__getitem__` 경로를 검사한다. 일반 보안 sandbox가 아니다.
  Ruff format/check와 diff 검사 PASS; 전체 repository pytest는 미실행이다.
- 읽기전용 검토는 값 읽기/역참조/soft·external link 추적/하위 그룹 재귀 경로가 없음을 확인했다.
  출력 최종 직렬화 크기와 null shape를 실행 전에 보완했다. 초과 이름을 하나 받은 후 중단하는
  일반 경계 및 named datatype의 타입 보고 한계는 남아 있다. 이번 실제 group 최대 5키,
  object 최대 3attribute이며 객체는 group/dataset뿐이어서 해당 한계는 실제 결과에 발생하지 않았다.
- 파형·marker·attribute 값/역참조/fit/outcome/held60/네트워크/PDF/GPU/발송/유료 접근 0.
  값 접근 0은 검토한 코드 경로를 뜻하며 HDF5 내부 물리 IO를 계측한 값은 아니다.
- `coordinate-worktree-changes`: main 단독 작성, agent 읽기전용 검토, 새 worktree 0.
  기존 41 worktrees/8 untracked 출력, 실패·부정 결과·과거 fixture 상한 이탈을 보존했다.
- `academic-research`에는 이 범위의 관측과 한계를 함께 기록했다. 전체 연구 goal은 active이며
  이 점검의 완료가 전체 목표 달성은 아니다. 근거: [관측 JSON](reports/choi_header_census_v1_observation.json),
  [기계 상태](reports/choi_header_census_v1_state.json).
