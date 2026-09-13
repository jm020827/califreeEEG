# S001a DIN-only 라벨 진단 v1 — 실행 계약

2026-09-13, basec4d6084/main. 직전 source-learning v1은 label guard 실패로 종료했고
realfit0을 보존한다. 이 진단은 실패 위치·크기를 확인하는 별도 개발 단계이며 그
성능 시도를 재시작하거나 임계값을 수정하지 않는다. 원 저보정 SSVEP 목표 유지.

Problem signature: representation=recorded-event timestamp/sample groups;
bottleneck=nominal guardband label semantics unknown; allowed operation=development-only
DIN summary; objective=identify exact failed group and distinguish adaptation/main;
feedback=23uniform summaries and old guardband membership; failure mode=threshold rescue
or inferred labels falsely called ground truth. Observed cause remains unknown before read.

단회 S001a reader1, file hash57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a,
137357437bytes. 기존 development_role hashdc26f85099ed667208ec7b992430d257d01f2a4aa727b7b98f8a5195f6d9af18.
S001의 모든 기록은 efficacy에서 제외한다. 추가 file/extraction/network/EEG/M/fit0.

전체 마감2026-09-13T13:00:00Z; process2GiBAS/60CPU/90wall, 각 stdout/stderr64KiB,
진단JSON64KiB, 모든code/test/report<=512KiB, free8GiBreserve. RawDIN은 child만 읽고
어떤 값도 대조군/학습기로 보내지 않는다. Parent exclusive STARTED/childclaim/terminal,
producer와plan/parserhelper SHA를 먼저 manifest에 고정한다. 실패/시간초과 후 재시도0.

Reader: pinned MAT stat/hash/role검사, EEG shape/class는 whosmat header만 확인,
`loadmat(variable_names=['DIN_1'])`만 수행. Descriptor rows1/3은 decode만 되고
해석하지 않는다. Numeric rows2/4 전체를 검사하고 timestamp gap>2000ms로23groups.
각 group에 동일하게 index, adaptation/main역할, event수, 첫/끝 sample과 상대시각,
구간 duration, Δtimestamp mean/median/min/max, 고정f_est=1000/(2meanΔt), 기존
nominal±quarter-spacing guardband에 해당하는 Hz목록, 기존window가능여부를 기록한다.
비정상·짧은 group도 삭제/복구하지 않는다. Freq추정 불가능하면 null과 사유를 기록한다.
정답class를 새로 배정하거나 FFT peak/EEG/분류오류로라벨을 추론하지 않는다.

Main만 정상이라는 결론을 미리 가정하지 않는다. 실패가 adaptation인지main인지,
global period-shift인지국소group문제인지도 관측 뒤 제한적으로 해석한다. 이후 새
parser가 정당화되더라도 별도 protocol/테스트/예산을 먼저 고정하고v1실패는 그대로 둔다.
독립 label의 근거가 없으면 효과실험을 보류하며 임의nearest/order label을 만들지 않는다.
후속 규칙 고정 전에는 S002–S011의 DIN timestamp/sample·frequency·label 통계도
열지 않는다. 문서화된 inferred-label protocol 자체를 금지하는 것은 아니며,
독립 ground truth와 구분하고 개발 데이터 밖의 값으로 규칙을 고치지 않는다.

Root만 새script/tests/report/SQLite쓰기; agent들은 shared repo/기존source읽기전용.
새worktree0, 기존43worktrees/8untracked보존,cleanup/install/push0. 기존v1producer,
계약, manifest,결과,held60및보호4문서내용은 수정하지 않는다.
