# Acquisition context의 시간경계·실제 보정비용 — 인공자료 구현 계약

2026-09-10. Active goal은 **계속 연구**다. 직전 Choi 검토는 근거와 다음 행동을 바꾼 progress이며,
새 goal 전체를 완료했다고 처리하지 않는다. 이전 유한 루프의 완료 기록도 소급 변경하지 않는다.

이번은 [후속 제안](choi_aux_context_feasibility_v1_results.md)의 구현 단계다. **사람 자료 효능 실험이 아니다.**
측정 연결/단위/다운샘플/marker 미확정은 그대로이며 인공 검사 PASS로 Choi 후보를 활성화하지 않는다.
새 논문 검색보다 이미 확인한 시간·수집비용 결함을 검증 가능한 코드로 바꾸는 것이 이번의 다음 행동이다.

## 범위와 예산

[사전 계약](../configs/analysis/acquisition_boundary_generated_v1.json)을 구현 전에 고정한다.
신규 작업트리1개(64MiB 예약), 테스트/출력256MiB, 전체384MiB 예약;
검사 호출 총10회/누적1800초/각180초 상한, 인공 scenario≤32/각배열≤200000원소.
Preflight의 코드 오류는 같은 불변조건 안에서 수리하고 실패 출력도 보존한다.
한도에 닿으면 멈춰 미완료를 보고한다. 사람 파형/학습/outcome, 다운로드/검색/외부발송/유료/GPU0.

## 공통 계약과 소유권

| Lane | 소유 파일 | 작업 위치 | 통합 순서 |
|---|---|---|---|
| root | `data/acquisition_context_contract.py`, `data/acquisition_context_boundary.py`, 해당 tests, 통합 검사·기록 | main | 공통 계약 → temporal → 최종 통합 |
| cost writer | `analysis/calibration_collection_cost.py`, `tests/test_calibration_collection_cost.py` | 별도 worktree | 공통 계약 뒤, temporal과 독립 구현 → root 통합 |
| reviewer | 코드·기작·반례 검토만, 파일/DB 쓰기 없음 | 공유 repository | 사전·최종 검토 |

Paths는 `src/cfeg/` 기준이다. Root만 공통 API/문서/DB와 최종 검증을 쓴다.
별도 환경 설치/기존 worktree 재사용·정리 없이 기존 main Python 환경을 실행 파일로만 사용한다.
각 lane의 테스트는 자기 src를 import하고 자기 출력 directory를 사용한다. 환경 symlink는 만들지 않는다.

공통 dataclass:

- `RunKey(dataset, participant, day, band, session)`: 문자열 식별자 모두 비어 있지 않아야 한다.
  파일 이름만으로 chronology를 추측하지 않고, 참가자/날짜/band/session이 하나라도 다르면 다른 기록이다.
- `SampleSpan(start, stop)`: 0-based half-open, Python int, bool 거부, `0 <= start < stop`.
  `length`와 `contains(other)`를 제공한다. 부동소수점을 암묵적으로 반올림하지 않는다.

## 시간경계 API / root

`TimingEvidence`는 run, 정수 sampling rate, export의 좌/우 dependency envelope(동일 grid의 sample 수),
확인한 marker anchor와 evidence ID를 가진다. unknown을0으로 메우지 않는다.
Evidence ID는 caller의 provenance 선언이지 사실 검증/실행권한 증명이 아니다.

`plan_context_window(...)`는 sample cutoff 직전의 output span, causal FIR taps의 history,
export upstream 좌/우 의존 구간을 계산한다. Raw1000Hz와export200Hz 인덱스를 섞지 않는다.
이 단계의 envelope는 **동일 export-grid에 투영되어 검증된 상한**이라는 전제다.
가드 폭은 임의로 추정하지 않는다. 원 변환의 시간 지지가 모르면 reject한다.

전체 dependency가 run의 허용 구간 합집합 안이고 cutoff 미만이어야 한다.
허용하지 않은 틈/다른 run/처음 trial의 부족한 history/미확정 marker를 거부한다.
허용된 exported slice만 요청하는 callable reader로 finite causal FIR을 계산하고,
local state는 그 허용 history에서만 만든다. Session 전체 filter state를 숨겨 이어받지 않는다.
배열 slice 접근 계약은 OS sandbox가 아니며 reader 구현 자체의 무단 파일 접근을 증명하지 않는다.
현재 파일 reader는 구현하지 않는다.

기대 검증: 요청 밖 future/gap을 매우 큰 값·NaN으로 바꿔도 결과 동일;
허용 값 변화에는 결과 민감; full-run 양방향 참조라는 의도적 잘못된 대조는 미래 교란에 민감.
정확한 경계/off-by-one/잘못된 shape/nonfinite/불일치 run/unknown timing은 fetch 전에 거부
(shape/nonfinite는 fetch된 허용 배열에서 검사). 단위 물리 의미나 sensor 효능의 검증이 아니다.

## 보정비용 API / cost writer

모듈 자체의 frozen dataclass `CollectedTrial(run, trial_id, label, onset_sample, end_sample)`와
`CollectionCost`를 제공한다. 동일 RunKey의 chronological trials만 허용한다.
같은 onset·중복 ID·겹친 trial·잘못된 label/시간·class 수 불일치는 reject한다.
입력 순서를 조용히 정렬해서 chronology 오류를 숨기지 않는다.

`balanced_prefix_cost(trials, *, run, classes, k, sfreq, collection_start_sample,
setup_seconds=None, extra_baseline_seconds=None)`는 각 class의 첫 k개를 선택하고,
모든 class가 k개 모인 시점까지 **실제로 실시한 prefix 전체**를 비용으로 센다.
`selected_trial_ids`, `selected_labels`, `collected_trial_ids`, `collected_trials`,
`collected_labels`, `elapsed_recording_seconds`, `setup_seconds`, `extra_baseline_seconds`,
`total_seconds`, `attained`를 반환한다. 모든 trial은 이 최소 계약에서 label이 있는 task trial이다.
`None`인 setup/extra baseline은 total에서도 `None`; 0이라고 채우지 않는다.
Elapsed는 collection_start부터 마지막 수집 trial의 end까지며 그 안의 rest도 포함한다.
Extra baseline은 그 elapsed 밖의 추가 시간만; 중복 계산하지 않을 책임을 문서화한다.
k=0은 selected/collected0, elapsed0; setup/추가 baseline은 여전히 별도 비용이다.
미도달 k는 attained=false, 목표 선택/도달비용은 None, 관측된 수집 prefix와 경과시간은 보존한다.
이 값은 observed schedule의 비용이지 prospective 적응형 중단 성공의 증거가 아니다.

## 인공 검증 후 실제 연구로 이어지는 조건

통합 fixture는 chronological schedule에서 선택된 support와 같은 run의 pre-support metadata 창을
연결한다. 시간경계 검사와 비용 보고를 함께 실행하고 cross-run 교환은 실패시킨다.
이 단계는 Q/QM learner나 새 효능 표적을 추가하지 않는다. 특히 k1 weight-only 동일성은
직전 설계 조건으로 유지하고 인공 경계 PASS를 metadata 성능 이득으로 부르지 않는다.

다음 사람 연구 전에 실제 export/marker 근거, 값의 유효변동, 별도 입력 권한·참가자 분할·기작·대조군·
효과/비용 기준 및 유한 fit/reveal 예산이 필요하다. 기존 문서 질문은 계속 미발송이다.

## 사전 공학 검토에서 보완한 binding (동일 invariants, 새 효능 설정 아님)

2026-09-10, 첫 root unit9PASS 뒤 독립 코드 검토에서 두 누락을 찾았다.
Frozen dataclass도 중첩 필드가 변형될 수 있어, `validate_run`/`validate_span`으로 fetch 전에
필드 자체를 다시 검사한다. 또한 특징 receipt에 `clock_id/sfreq/cutoff_sample`을 보존하고
비용 receipt에도 `clock_id/sfreq/collection_start_sample`을 남긴다.
Cost 함수의 `clock_id=None`은 시간차 계산 자체에는 허용하지만, metadata와의 통합 join에서는
알 수 있는 동일 clock/rate를 필수로 요구한다. RunKey만 같은 wrong-grid join을 허용하지 않는다.
이 보완은 frozen 범위의 clock/pairing 검사를 구현하는 것이며, 사람자료 사용권한/과학 threshold는 바뀌지 않는다.
