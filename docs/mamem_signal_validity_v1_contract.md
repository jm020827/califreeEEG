# S001a 신호 연결 진단 v1 — 실행 전 고정 계약

2026-09-13, base824039a/main. 이전 goal turn은 실제80-fit/독립검산/부정결과라는
progress다. 원래 저보정 SSVEP acquisition-M 목표는 유지한다. 이 단계는 v2 후보의
재학습·구제가 아니라 후속 구현의 기본 입력 타당성을 확인하는 진단이다.

## 질문과 범위

[직전 초안](mamem_signal_validity_next.md)의 problem signature를 따른다. 이 문서는
추가 S001 신호 진단 수치를 보기 전에 고정한다. 질문은 nominal reference와 기록된
event 주파수의 관계, EEG 투영 에너지, 고차원 CCA 선택성이다. 실제 M 이득의 새 검정,
accuracy 비교·bootstrap·학습·parameter 선택은 하지 않는다.

입력은 `/home/whwovy/data/mamem_i_v1_20260913/development_first.mat` 단 한 개,
137357437bytes, SHA57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a.
같은 폴더의 development_role.json SHA는
dc26f85099ed667208ec7b992430d257d01f2a4aa727b7b98f8a5195f6d9af18이다.
S001의 모든 기록은 개발용이며 독립 test가 아니다. 원본 role/member/path를 확인한다.
header eeg257×117917double, DIN_1 4×1966cell, samplingRate scalar250을 확인한 뒤
세 변수만 한 loadmat 호출로 decode한다. DIN descriptor row1/3 해석은 하지 않는다.
EEG 수치 처리는 v2 parser(include_metadata=False)가 고정한15개 main창 [1,3)초,
첫256row뿐이다. Adaptation/다른 EEG구간/row257 값 QC·처리·출력 금지.

## 정확한 계산 (모든 15개 group에 동일)

1. 수정하지 않은 events_v2 parser의23group/8adaptation/3perclass/inferred label/
   window 검사를 통과해야 한다. M2 extractor를 호출하지 않는다. Group마다 timestamp
   t(ms)·sample index s를 정렬된 동일 DIN event로 묶는다.
2. m_t=mean(diff(t)), m_s=mean(diff(s)); f_time=1000/(2m_t), f_sample=250/(2m_s).
   nominal f는 parser가 추론한 class의 [6.66,7.5,8.57,10,12]Hz다. 세 값을 보고하며
   서로 대체하지 않는다. 추가로 max(abs(diff(t)−4diff(s)))ms, m_t/m_s,
   event count와 window를 보고한다. 두 주파수는 같은 DIN의 재표현이며 독립 센서
   검증이 아니다. 평균의 일치는 국소 오차·offset·edge 의미를 입증하지 않는다.
3. X는 선택256×500만 복사 후 channel mean을 빼고 공통 RMS 정규화한다.
   기존 signal_v1과 동일하며 no filter/reref/montage/unit conversion이다.
4. 각 f∈{nominal,f_time,f_sample},h∈{1,2}에 대해 U는500sample의 centered
   sin/cos thin-QR basis. E(f,h)=||XU||_F²/||X||_F². 각 쌍 nominal↔time,
   nominal↔sample,time↔sample의 overlap은 ||U_aᵀU_b||_F²/2, 그리고 두 squared
   singular value를 보고한다. Freq peak/grid search·best-frequency 선택은 하지 않는다.
5. 기존 signal_v1 analyze_window의 nominal5×2 projection energy와 neighbor-log-ratio,
   Q2와 covariance를 사용한다. Nominal energy score_j=(E_j1+0.5E_j2)/1.5.
   Nominal CCA score5개는 기존 ridge1e−6·trace(C)/256의 그대로인 계산이다.
   각 score vector에 top1−top2 gap과 inferred-label score−max(other4) margin을
   기록한다. Argmax label/accuracy·통계적 유의성·최선 설정을 출력하지 않는다.
6. 공분산 eigenvalue 비율을 정렬하여 rank indices[1,2,4,8,16,32,64,128,256]의
   값과 cumulative mass, min/trace, entropy effective rank 및 regularized condition
   number를 보고한다. 이것은 spectrum 요약이지 독립 cortical source 수가 아니다.
7. seed20260914의default_rng로500sample permutation 한 개를 생성하고 모든 channel과
   모든15창에 공통 적용한다. 순열 배열/hash를 보존한다. Original/permuted 양쪽의
   nominal score·energy/Q 요약을 같은 방식으로 기록한다. Covariance relative
   Frobenius difference<=1e−10이어야 한다. Permutation은 spatial covariance를 보존하지만
   temporal spectrum/의존성도 바꾸므로 순수 phase-null·유의성 검정은 아니다.
8. Fixed seed synthetic B에 pi/3의2×2orthogonal R을 곱한 BBᵀ 불변 canary를 단위검사한다.
   공간 상대 phase나 query energy까지 사라진다고 해석하지 않는다. 실제 EEG phase
   reconstruction/ODE/추가 decoder/gate fit은 없다.

## 운영 예산·중단

이번 별도 진단 deadline2026-09-13T14:00:00Z. Raw worker 1회,2GiBAS/90CPU/120wall,
BLAS1, outputJSON<=128KiB, stdout/stderr 각각<=128KiB, free reserve8GiB.
추출/newnetwork/PDF/설치/삭제/held60/sourcecohort/학습0. S001b 및 S002–S011
원자료·featurecache·새 결과 접근0. 실제v2의80-fit terminal/result/정책은 바꾸지 않는다.
실패 시 원인과 partial terminal을 보존하고 재시도/다른file/기준확대하지 않는다.

Root 단일 writer가 pure module `src/cfeg/mamem_signal_diagnostic_v1.py`, runner
`scripts/analysis/diagnose_mamem_signal_v1.py`, 각 tests와 이 계약을 소유한다.
기존events_v1/v2/signal_v1는 수정하지 않는다. 새worktree0, 이전45worktree/8untracked
보존. Subagents는 원문·수식·코드 read-only review만 하며 SQLite는 root만 쓴다.
관련 생성 단위검사와 사전 review 후 코드 pin keyset/계약/입력hash를 manifest에 고정한다.
`docs/reports/mamem_signal_validity_v1_run`에 freeze→execute→worker, exclusive
STARTED/claim/terminal, parent PID·manifest hash binding, directory fsync를 적용한다.
Parent는 exit code뿐 아니라15group·scope·score dimensions·finite·covariance
invariant·입력/pins를 확인하고 마지막deadline검사 후 COMPLETE를 기록한다.

생성검사는 실제 파일 접근0, M미호출·excluded poison·변환/overlap·고정perm·full-rank
signal/noise canary·생성 CCA를 독립 direct linear algebra와 대조·단회/shape/hash 오류를
확인한다. 결과 사후 감사는 저장값 산술과 역할/ledger만, raw再讀/새fit0,
30CPU/60wall/2GiBAS 한도로 1회 허용한다. 원 EEG→수치 전체 재현 감사로 승격하지 않는다.

## 관측 뒤의 행동

전체 결과와 competing explanation을 함께 보고한다. 확인된 시간 단위 모순은 source
확인으로, baseline 선택성 부족은 기준 구현/검증 montage 계획으로, 식별불가는 추가
효능fit보류로 이어진다. 이 진단 하나로 원인·유효한decoder·새M효능을 확정하지 않는다.
S002–S011 outcome을 이미 보았고 같은사람의미개봉run도 독립참가자확인이 아니다.
다음 새학습은 기작·공통front-end 대조군·독립검증·비용과 새예산을 먼저 고정해야 한다.
