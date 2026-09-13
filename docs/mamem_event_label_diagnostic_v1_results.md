# S001a 라벨 진단 결과와 source-backed v2 변경 이유

2026-09-13. 전체 연구는 계속 진행 중이다. Source-learning v1의 실패를 보존한 뒤
DIN-only 진단을 별도 계약4dbe325로 한 번 실행했다. 12:37:41.547599–
12:37:41.761676UTC,0.214077초,학습0회다. 새 EEG 변수는 로드하지 않았고 EEG는
header만 확인했다. `DIN_1` 전체 decode 및 timestamp/sample행만 수치 처리했다.

## 무엇이 실패했나

23개 이벤트 구간의 수와 수치적 순서가 확인됐고, 모든구간에 고정2초 분석창이
들어 있었다. 이전 guardband에 들어오지 않은 구간은 다음5개다.

| group index | 역할 | mean interval(ms) | f_est(Hz) | 고정 MOABB 변환 key |
|---|---|---:|---:|---:|
| 2 | adaptation | 51.9569892473 | 9.6233443709 | 9 |
| 15 | main | 60.2716049383 | 8.2957804179 | 8 |
| 17 | main | 52.1290322581 | 9.5915841584 | 9 |
| 18 | main | 51.9263157895 | 9.6290289884 | 9 |
| 19 | main | 52.1397849462 | 9.5896061044 | 9 |

처음 실패한 것은 adaptation group2지만 main에서도4개가 실패하므로, adaptation만
건너뛰는 변경으로 해결됐다고 할 수 없다. 그렇다고 EEG손상·metadata무효라는 증거도
아니다. 이 진단은 temporal summaries만 관측하며 실제 display/뇌 주파수의 원인은
식별하지 않는다. 표의 key열은 진단 후 별도source-backed v2계약ce05361을 고정한
뒤 저장 interval 요약에 그 고정식을 적용한 계산이며, 진단 parser의 정답 출력이 아니다.

## 기존 소스가 말하는 것

1. 저자 `Session.m` DatasetI branch는 별도 label 없이 DIN으로 구한
   `1000/(2*meanΔt)`를 Trial 생성자에 전달한다. DatasetII의 별도 label을 I에
   가져오면 안 된다. Source revision5a03abe2a6a874e9adaceea29a52c2fce35d8a03.
2. 이번에 같은revision의 `Trial.m`을 official GitHub API에서1회 읽었다.
   Line24의 `T.label = label`은 전달값을 그대로 저장하며 nominal class 변환은
   없다. 전체37lines/1044bytes를 확인했고 Git blobde32cd15f00c8a5b5c515bdd361a966ad38800ff를
   로컬 decode후 git hash-object로 재검증했다. Network1/code1044bytes, PDF0.
3. 보존된 MOABB `mamem_event`는 interval 평균을 먼저 정수floor하고,2배한 뒤
   `1000 // (...)`를 적용한다. key6/7/8/9/11→ID1/2/3/4/5→nominal
   6.66/7.5/8.57/10/12Hz로 연결한다. 이 규칙은 floor(f_est)와도 다르며,
   정확히10Hz나12Hz인 이상적 interval에는 지원하지 않는 key가 생길 수 있다.

따라서 기존 quarter-spacing guardband가 ‘저자 검증 허용오차’라는 해석은 잘못이다.
이는 우리가 보수적으로 선택한 추가 정책이었다. v2는 관측값에 맞춰 그 폭을 늘리는
대신, 공개 구현의 한 가지 고정 변환을 **compatibility policy**로 채택한다. 이 라벨은
여전히 inferred nominal label이며 독립 ground truth나 실제 자극주파수 확인이 아니다.

MOABB/저자 loader의 짧은 group 처리에서 frequency와range가 어긋날 가능성은
복제하지 않는다. 새 parser는 group별 label/window를 함께 검증하고, 알 수 없는 key,
부족한 group, class반복수 오류를 그대로 중단한다. 순서로 정답을 강제하지 않는다.

계약ce05361을 고정한 뒤 저장된 개발 요약으로 계산한 mainkeys는
`6,6,6 / 7,7,7 / 8,8,8 / 9,9,9 / 11,11,11`이다. 이것은 개발 규칙의 일관성 확인이지
나머지10명에 대한 검증은 아니다. S002–S011의 DIN/EEG/M/label통계는 아직 열지 않았다.

## 다음 실험과 보존 경계

[v2 계약](mamem_recorded_event_source_v2_contract.md)은 metadata 후보·features·Q/Q2/
QM/SHAM·source-only 학습·효능 기준을 바꾸지 않는다. 라벨 연결만 위source에 따라
별도버전으로 고정했다. 가상 통합0추가fits, 실제최대80fits이며 한명이라도 임의로
대체하거나 소스결과를 본 뒤 label규칙을 바꾸지 않는다. v1실패/0realfit 및 이번진단
원artifact는 모두 보존했다.

진단 코드에 대한 읽기전용 검토는 manifest에서누락pin차단, parent의 outputscope
검증, directoryfsync, singleton의null사유를 보강할 것을 지적했다. Root 자체검토와
86생성tests 후 실행이 이미 끝난 시점에 도착한 지적으로, 동결 producer를 사후수정해
재실행하지 않았다. 실제 저장결과의 정확한pin목록·scope·23indices는 별도감사하며,
이를 일반적 fail-closed 운영 검증과 혼동하지 않는다. 후속v2운영기에는 해당 검증을
구현하도록 계약에 명시했다.

독립 사후 감사는 실제3PINS/current+4dbe325해시,23indices/S001/MAT/scope를 확인해
PASS했다. 저장mean으로 oldmembership도 재계산했고 failure5개가 일치했다. Output
17,124bytes, stdout/stderr0bytes, terminal hash와result hash 일치다. 원MAT/EEG/
targetcohort/fit/network 재접근0이며, 빠진 예방장치가 이미 있었다고 주장하지 않는다.

Artifacts: [진단](reports/mamem_event_label_diagnostic_v1_run/diagnostic.json),
[terminal](reports/mamem_event_label_diagnostic_v1_run/terminal.json),
[manifest](reports/mamem_event_label_diagnostic_v1_run/manifest.json).

```text
diagnostic SHA256 9512d87e19e6b14b3e5d6af3b70bfca95b39f1a4fcca226067ead958f0073cd0
manifest SHA256 fef4e7c83824bb8daa69d53a542e324e075d9bde39d4462699a17cd6593106d6
Trial decoded SHA256 87a0ae2aea77513cc229ede972a02c0b4359f3bf9e6f82503c9f333a0d23d078
Trial wrapper SHA256 b84bd3936201bcfde18d5295238a7a495e0f7c2eacef2913fb363a2980ced612
```

새 출처: [같은 revision의 Trial.m](https://github.com/MAMEM/eeg-processing-toolbox/blob/5a03abe2a6a874e9adaceea29a52c2fce35d8a03/%2Beegtoolkit/%2Butil/%40Trial/Trial.m).
