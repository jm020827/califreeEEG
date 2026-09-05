# Metadata-calibration-efficiency-v2 synthetic V11 terminal 결과

기준일: 2026-09-06  
상태: **단회 lockbox 소비 / infrastructure inconclusive / scientific result 없음 /
external·human EEG outcome 미실행**

## 한 문장 결론

V11 독립 synthetic lockbox는 미래 비콘과 전역 claim까지 정상적으로 봉인했지만,
post-beacon evidence validator의 순환 호출로 efficacy metric을 만들기 전에 종료됐다.
따라서 이 시도는 양성도 음성도 아닌 **결론 불가 인프라 terminal**이며, 같은 lockbox는
재실행하지 않는다.

## 무엇을 검증하려 했나

상위 연구 질문은 바뀌지 않았다.

> 강한 calibration-free FBCCA와 signal-only support updater가 있을 때도, query 전에
> 관측한 acquisition context가 새 사용자의 labeled calibration 부담을 추가로 줄이는가?

Synthetic gate는 사람에서 이 질문에 답하는 효능 시험이 아니다. 다음 외부 EEG 단계로
넘어가기 전에 구현이 최소한 아래 성질을 보이는지 확인하는 필요조건이다.

- 도움이 있는 세 family 중 적어도 둘에서 eAUC 하한이 0보다 크고 관측 이득이 0.01 이상
- correct query-support context pairing이 pair-shuffle보다 eAUC 0.01 이상 우수
- adversarial family에서 prequential gate 기권률 0.95 이상
- context-null family에서 margin 1/60 비열등
- 사전 지정 8개 contrast의 심각 손상률 상한이 0.10 미만
- k=0/off/missing exact fallback, 순서 불변성, class equivariance, label/identity 비접근 등
  hard invariant 전부 통과

## 실행 전에 완료한 것

| 항목 | 결과 |
|---|---|
| Frozen source | commit `832e579c0dec8774ffe8c3ac99abf110a74aa9e3`, tree `d26656acd69c88e62cc5dd336103ddd76f33022f` |
| Frozen tag | `metadata-calibration-efficiency-v2-synthetic-freeze-20260906-r11` |
| Synthetic plan | SHA-256 `9a9683141ccd3d1fe9cf094aad955c6e317d112a73e35fbcfcdd2ef7d06e1f8c` |
| 독립 코드 감사 | `CODE-GO` |
| 전체 repository test evidence | `682 passed`, exit 0 |
| Development replay | 5,888 metric rows, 19 contrasts, `engineering_only` |
| 단회 승인 | synthetic lockbox만 승인; human-held-data 권한 없음 |

Development는 구현 재현성 진단일 뿐 lockbox 표본에 합치지 않는다. 그 시드에서는 B1
P1 eAUC 이득 `0.005686`, B2 P2 `0.001866`, B3 correct AQM−AQ `0.000130`,
correct−pair-shuffle `0.000260`, adversarial 기권률 `0.800781`이었다. Null 비열등성과
severe-harm screen은 통과했지만 실질 이득·pairing·기권률 기준은 못 넘었다. 이것은
독립 scientific 판정이 아니라 사전 공개된 `engineering_only` 진단이다.

## 단회 실행에서 실제로 일어난 일

| 시각(UTC) | 사건 |
|---|---|
| 2026-09-05 21:30:00.000 | 사전 고정 NIST pulse target |
| 2026-09-05 21:30:36 | frozen CLI 단 한 번 호출 |
| 2026-09-05 21:30:37.002 | global O_EXCL claim 생성; `network_access_before_claim=false` |
| 2026-09-05 21:30:38.532 | 정확한 target pulse fetch·protocol output 검증·receipt 생성 |
| 2026-09-05 21:30:51.446 무렵 | `RecursionError` terminal 원자 공개 |

Canonical scientific result 경로에는 파일이 없다. 별도 terminal receipt의 exact 상태는
`consumed_inconclusive_infrastructure_error`, error type은 `RecursionError`, message는
`maximum recursion depth exceeded in comparison`, automatic retry는 `forbidden`이다.

## 왜 실패했나

비콘 자체나 네트워크가 실패한 것이 아니다. 정확한 펄스와 derived-seed digest는 정상
검증됐다. 문제는 비콘 receipt를 읽은 뒤 sealed authorization을 다시 검증하는 경로가
development result 재계산에 들어가고, development seed가 reserved인지 확인하는 코드가
다시 full beacon validator를 호출한 순환이다.

```text
beacon evidence
  -> claim
  -> authorization
  -> development evidence
  -> metric summary
  -> reserved-seed check
  -> full beacon evidence  (순환)
```

이 순환은 efficacy participant 생성과 metric 계산 전에 `RecursionError`를 냈다. 그러므로
관측된 synthetic promotion 값, PASS/FAIL contrast 또는 calibration-saving 결과는 없다.

## 포렌식 검증

Frozen validator 자체는 같은 순환 버그 때문에 post-beacon terminal을 끝까지 읽지 못했다.
따라서 별도 읽기 전용 감사에서 stdlib canonical JSON hash와 비콘의 순수 계산을 독립적으로
재검산했다.

- preparation/development/test/authorization/claim/beacon/terminal의 모든 내부 self-hash 일치
- 모든 canonical artifact가 owner-only mode `0400`
- authorization의 세 evidence file hash와 receipt hash 일치
- terminal→authorization→claim→beacon digest 결속 일치
- source commit/tree/source-bundle, plan, environment 결속 일치
- claim 시각은 target 이후이고 beacon fetch보다 앞섬
- exact target pulse, full-response hash, pulse field echo 일치
- deployed NIST outputValue와 derived-seed digest 재계산 일치
- scientific result 경로 부재

수정 뒤에는 canonical beacon receipt에 이미 결속된 frozen git identity와 environment를
메모리에만 주입해 실제 terminal→authorization→claim→beacon→development evidence 전체를
읽기 전용 재검증했다. 재귀 없이 동일 terminal 상태를 반환했고 RNG·participant·outcome은
실행하지 않았다.

이 검증은 로컬 custody와 내부 일관성 증거다. Self-hash는 외부 전자서명이 아니므로 제3자
부인방지 증거로 과장하지 않는다. 기계 판독 가능한 전체 기록은
`configs/governance/metadata_calibration_v2_synthetic_v11_terminal_audit.json`에 있다.

## 핵심 artifact digest

| Artifact | File SHA-256 | 내부 receipt SHA-256 |
|---|---|---|
| preparation | `fd565d38eee2b709439cde1d3520d5d27f00b201a43a19bd00981941c2c87a1b` | `d30eb21b8a6c82cde64bebad5b011d025c4dda6a1bfa2bf122c7d3e1b9948a02` |
| development | `4bcaf57357fd4093fe41985fc71e73a70ac44d531b08ae23fc938390350c4963` | `5843769380cc302735fe5ae6857476440b8e611005bb046f8ab07b9bf0361f2a` |
| full-suite test | `16e4917a300d459d779886cb8e4c627673f68ac8f1a41f1b1f04f26eac7b0a4c` | `b5e0f257c29fa852af11840172d561481cc596a69740175b86205750a5e6e9b3` |
| authorization | `04272478fa05b28b0190aca37dde9a0d0eeb6f5d9200496cb9e2388f977d1095` | `ec6809d96524e37a3e7dc10f48a319a8c5ee205de98c895dd7afb29cd4f52273` |
| global claim | `4a15828be8e6bda37e0fbb0bd03247f0a455f925a664b14f43c8e6474236032d` | `e53e78f87b6a0537dcc1c6f90df5038570be6c6fd74b92ddcd6e3f30844b8876` |
| beacon | `b1fa04e71058cbd236b22fb2b343324bcd3cbffc73796fc01a7682f3a8f39d81` | `7b9fbf9aee6be0321cae10a70e503e3224dda6d993f85b637768bddd2e1f7ca5` |
| terminal | `6a2c2d21de7c796d5a96c4fef4a23f08ffdd81684ed12e9defce7fcffc796b14` | `e2382761a3171fd08451221bdcf59a64ed977dadc5d6f987c425b22fdbd8c857` |

## 수정과 검증

Terminal 뒤 소스에는 두 층의 수정을 했다.

1. 고정 development seed와 이미 파생된 dynamic seed는 비콘을 다시 읽지 않고 reserved로
   판정한다.
2. Public seed guard는 비콘의 exact schema/self-hash/full response/pulse/output/seed만 보는
   순수 intrinsic validator를 사용한다. Claim·authorization·prelockbox evidence 전체 검증은
   terminal/scientific artifact evidence 경로에만 남긴다.

비콘 파일이 존재하는 상태의 완전한 development-result validation과, prep→development→
test→authorization→claim→beacon→error-terminal 전체 validation fixture를 회귀 테스트로
추가했다. 집중 synthetic suite는 `40 passed`, V2 terminal·analysis·request·synthetic 묶음은
`115 passed`, 사후 전체 repository suite는 `706 passed`(기존 Transformer warning 68개)였다.
Frozen master plan은 원본 그대로 두고 사후 상태는 별도 terminal audit에 기록했다. 이
수정은 V11을 재실행하지 않으며 이미 공개된 artifact를 변경하지 않는다.

추가로 SHA-256이 소스에 고정된 terminal deny-overlay를 두었다. Repository가 제공하는 V2
operational entrypoint는 synthetic 재실행, BETA/Dong request·prediction·label join·outcome
reduction·selection·independent gate를 입력 파일 접근 전에 거부한다. Choi/held V2 runner는
현재 없으며 future capability도 deny 목록에 예약했다. Plan/allocation validation, inventory
schema와 artifact integrity 같은 outcome-free read-only 감사만 허용한다. 이는 repository의
governed path에 대한 보호이며 같은 Unix UID가 임의의 새 코드를 쓰는 것까지 막는 enclave는
아니다.

## 연구 판단과 중단 규칙

이 결과로 주장할 수 있는 것은 다음뿐이다.

- V11 one-time attempt가 실제로 claim과 미래 비콘을 소비했다.
- 과학 결과 생성 전에 재귀적 evidence validation bug로 terminal이 됐다.
- 따라서 V2 P1/P2/AQM의 독립 synthetic 효능은 **미검증**이다.

다음은 주장할 수 없다.

- 현재 후보가 synthetic gate에서 과학적으로 실패했다거나 통과했다.
- acquisition metadata가 유용하거나 무용하다.
- human EEG에서 calibration을 줄인다.
- OOD/new-user 일반화가 개선된다.

Frozen 계약은 post-claim infrastructure failure도 lockbox를 소비하고 후속 external·held
단계를 금지한다. 이에 따라 BETA/Dong 선택·독립 gate, Choi cross-session replication,
wearable held 60은 실행하지 않는다. 별도 worktree의 external hardening 커밋
`2d6b44db104bfe3cfe40c02d99008785f1b65225`는 감사·테스트만 완료했고 main에 병합하거나
outcome 경로를 실행하지 않았다.

상위 연구목표는 유지할 수 있지만 이 시도를 기술 수정 후 자동 재시도해서는 안 된다.
후속 실행이 필요하다면 새 candidate/schema, 명시적인 pre-outcome amendment, 새 clean
freeze·전체 테스트·승인, 그리고 새 미관측 future beacon을 갖춘 별도 연구 루프여야 한다.

## 정보 접근 감사

| 데이터/결과 | V2 outcome 접근 |
|---|---:|
| Synthetic development seed | 예 — `engineering_only` |
| Synthetic V11 scientific lockbox result | 없음 — 생성되지 않음 |
| BETA V2 selection/independent outcome | 없음 |
| Dong2023 V2 selection/independent outcome | 없음 |
| Choi2019 classification/replication outcome | 없음 |
| Wearable held 60 outcome | 없음·미개봉 |
