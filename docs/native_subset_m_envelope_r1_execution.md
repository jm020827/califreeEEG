# Direct-M envelope-r1 — 입력 연결만 수정한 별도 실행

2026-09-07 사용자 '응' 승인. [실패 기록](native_subset_m_source39_v1_results.md)은 그대로 보존한다.
권위는 [새 execution amendment](../configs/analysis/native_subset_m_source39_envelope_r1.json)이며
[원 과학 계획](../configs/analysis/native_subset_m_source39_v1.json)의 수치·모델·split·SHAM namespace는 변경하지 않는다.
새 attempt_id/output 경로는 envelope-r1, 과학적 study_id는 원 native-subset-m-source39-v1 그대로다.

## 수정 범위

원본 byte hash와11-key envelope/5 provenance 값을 모두 검증하고, 원본 전체를 새 projection artifact에 보존한다.
Core에만 검증된6-key content를 명시적으로 전달한다. 독립 감사도 별도 envelope 검증 후 동일 수학을 재계산한다.
기존 producer/core/auditor 파일은 해시로 고정·수정하지 않는다. 이번 승인에서 고정 함수 정의의 import를 명시적으로
허용하되 기존 run/main/audit entrypoint 호출과 global monkeypatch는 하지 않는다. 이는 옛 attempt 재실행이나 seal 해제가 아니다.
새 producer가 새 실행 경계와 publication을 소유하고, 원 native 추출·pure core·독립 audit 수학을 그대로 호출한다.

Start/freezes/result에 execution_plan_sha256과attempt_id를 추가하며 plan_sha256은 기존 과학 계획을 가리킨다.
Start→원projection검증/보존→raw39/기존baseline검사→features→allfoldfreezes→outer평가→result.
출력은 새 경로의 기존5파일만0400/exclusive/fsync. 새 실패도 보존하며 같은attempt overwrite/resume은 하지 않는다.
기존 실패 directory는 이번 실행에서 내용을 열지 않으며 cleanup하지 않는다. Held60/retired/fullmanifest/Impedance.mat 접근 금지.

## 소유권과 비용

Base main8ad32c4a4d770005f64032f0709c37d080460254 clean/ahead103,25기존worktrees보존.
가용약290GiB. 신규auditor checkout12MiB/temp1GiB, output1GiB/main verificationtemp1GiB;
기존main/native venv는읽기전용재사용, CPU4/BLAS1/메모리2GiB예상/30분applicationceiling.

| Lane | 단독 소유 | 검증/통합 |
| --- | --- | --- |
| Main | amendment/문서/SQLite, scripts/run_native_subset_m_envelope_r1.py, tests/test_native_subset_m_envelope_r1.py | main단독실행; 실제loader→adapter→core 연결 및 전체검증 |
| Auditor | 별도worktree scripts/audit_native_subset_m_envelope_r1.py, tests/test_native_subset_m_envelope_r1_audit.py | producer/core import 금지; 고정독립audit helper만사용; synthetic만; commit반환 |
| Reviewer | 공유repo읽기전용 | envelope/과학적불변성/namespace/실행경계/최종결과검토 |

Auditor worktree /home/whwovy/califreeEEG-wt-subset-envelope-audit,
branch codex/native-subset-envelope-audit-r1. 통합순서계약→mainproducer→auditor→전체검증→실제단회실행.
같은tree의동시writer없음. 기존모든코드/결과/worktree보존하며최종main이end-to-end 검증한다.

## 필수 연결 시험

인공11-key 원본 파일→실제 byte loader→새 엄격 adapter→실제 core→독립 auditor.
원본출처/unknown key/누락key/packet key변조 거부; metadata 검증 실패시 raw0;
정상 원본에서 projection 보존, allfoldfreeze 전에 outer평가0, namespace 및fitted결과변경0.
실행기 경계만mock한 합성 lifecycle과 frozen native 실제 인공 입력 검사를 사용한다.
기존historical failure test는종료된기존코드를겨냥한채유지하고새성공경로와구별한다.

## 연구 해석

Academic-research 기존landscape/frontier와 실패 claim을 재사용한다. 새문헌검색/PDF읽기나
가설수정은이번입력연결문제를해결하지않으므로추가하지않는다. Source39는반복노출개발자료이고
성공/실패모두기술적결과다. M방식수정횟수0이며이번은동일방식의첫완료효능평가를목표로한다.
