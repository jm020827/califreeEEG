# 라벨 source 연결 완결 — 코드 읽기 전 기록

2026-09-13. DIN-only 개발 진단은 종료(1회/0fit)했고 guardband 실패는
group2(adaptation),15/17/18/19(main)이다. 모든고정window가포함된다.
Guardband가 저자 규칙이 아닌 우리 선택임을 기존 Session.m/MOABB코드에서 확인했다.
이 관측을 이용해 허용대를 맞추는 대신, 아직 빠진 저자 Trial 생성자의 변환 의미를
기존 고정revision5a03abe2a6a874e9adaceea29a52c2fce35d8a03에서 확인한다.

Signature: continuous recorded frequency→author stored target/class;
bottleneck=Session passes f_est but final class conversion unobserved;
operation=one official code file read plus optional parent-directory listing;
objective=source-grounded decoder, not data-fitted label rescue;
failure=unverified conversion or undocumented nearest/order assignment.

예산: public official GitHub source 최대2requests/각128KiB/30wall,
전체13:05UTC마감, 새paper/PDF/MAT/EEG/M/fit/participant0, 외부연락/유료0.
Trial.m exact path first; 없는 경우 directorylisting1만 허용하고 임의source확대0.
main만sourcefile/provenance/SQLite쓰기, agent는기존자료읽기전용.
원 v1중단 및 진단 producer/manifest결과는 수정하지 않는다.

이후 source-backed label policy가 복원되면 개발에만 근거하여 별도v2를 고정할 수
있지만 sourceS002–S011의 DIN/EEG/M/label통계는 그 전에는 열지 않는다.
독립groundtruth가 없는 inferred-label 한계는 유지한다.
