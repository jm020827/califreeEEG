# Source-only Welch/SVM 기준선 — 구현 전 계약

2026-09-14 KST / 2026-09-13 UTC, base b6d8c4a. 직전 중단 turn은 읽기 점검만
진행해 NO_PROGRESS였다. 이번은 원 저보정 metadata 연구의 공정한 비교기준을 구현하는
단계다. 기존 사람 실험·중단 기준·결과를 변경하거나 재개하지 않는다.

## 질문과 유한 예산

Representation=단일채널 spectral features; bottleneck=저자 PWelch/SVM 숨은 기본값과
재현 가능한 source-only fitting; operation=고정 revision source 판독과 reader-free 구현;
objective=후속 Q/common/M 비교를 위한 검증 가능한 공통 기준;
feedback=독립 spectral 수식 검산과 source/target 분리;
failure=논문/코드 불일치 은폐, target normalization/label 누출, 합성 정확도를 실제효능으로 과장.

이미 받은 author tree revision `5a03abe2a6a874e9adaceea29a52c2fce35d8a03`에
존재하는 다음6개 code 파일만 공식 GitHub contents API로 읽는다:
PWelch.m, PSDExtractionBase.m, LIBSVMFast.m, LIBSVM.m, DigitalFilter.m, Rereferencing.m.
Exact tree path/blob를 먼저 확인. 최대6GET/각30초/128KiB/총768KiB/redirect0/retry0,
전체10분. 401/403/429/challenge 중단, 실패경로 재시도·임의대체 소스 없음.
원 artifact 보존 후 base64 decode와 Git blob 일치를 검사한다. MATLAB 원코드 복제는
하지 않고 source-derived 설정과 독립 Python 구현의 차이를 기록한다.

Raw/PDF/숫자EEG/개인M/held60/사람요청/유료/패키지설치/삭제/push0.
Python environment는 이미 설치된 scipy/sklearn을 확인해서 사용하며 원 .venv 변경0.

구현은 코드판독 후 exact convention을 부록에 고정한 다음 시작한다. 완전한 MATLAB
filter/SVM 재현에 부족한 부분을 임의로 채워 저자재현이라고 하지 않는다. 구현 불가능한
부분은 명시적으로 미구현으로 남긴다. 파일 reader·batch run CLI·사람데이터 registry 연결0.
테스트는 합성 입력만 사용, 최대3회 suite/합계120초/각60초, BLAS1. 총 새출력64MiB 이내.
고정된 하나의 구현만 검증하며 성능을 보고 FFT/filter/C/normalization sweep을 하지 않는다.

Root 단독 writer: 계약·source receipts·src/cfeg/baselines/welch_source.py·
tests/test_welch_source_baseline.py·결과문서·SQLite. Agents는 read-only source/protocol/
code검토만 한다. 학습/특징/정보권한이 서로 의존해 한 writing lane으로 유지하며
새 worktree0, 기존46worktrees/8untracked 보존, 약289GiB 여유.

실제사람 비교는 이 계약에서0이며, 통과해도 M효능·보정절감·새방법성공 판정은 하지 않는다.

## 코드 확인 후 구현 정의 — 테스트 실행 전

6개 파일200/Gitblob일치 확인. Code-default는 elliptic filter, PWelch는 MATLAB의
빈 window/overlap 인자, LIBSVMFast는 probability OVR이고 일반 LIBSVM은 margin OVR다.
완전한 저자 filter coefficient/MATLAB pwelch 기본값/확률보정 재현은 이번 범위에 없다.

따라서 **독립 Welch/SVM core**만 만든다. 이를 완전한 baseline이나 저자 재현으로
등록하지 않는다. Filtering은 호출자 책임이며 검증된 전처리가 마련되기 전 실제 실행0.
새 논문이나 실제효능이 아니라 후속 비교에 필요한 구현 기반이다.

- 순수 `WelchSpec(fs,samples,nperseg,noverlap,nfft)`는 모든 값을 필수로 받아
  frozen으로 보존한다. Segment<=samples,nfft>=segment,0<=overlap<segment 검증.
- 입력은 실수 finite (trial,time) 배열. Trial별 전체시간 평균 제거, symmetric Hamming,
  segment별 추가 detrend 없음, one-sided density 평균, 전체0..Nyquist PSD 출력.
  자동 log/주파수선택/차원축소/필터/진폭정규화 없음. Tail 남으면 Welch의 명시적
  full-segment 규칙으로 FFT에서 제외하되 전체 trial 평균에는 포함한다.
  다중 trial의 통계로 특징을 바꾸지 않는다.
- 예측모듈은 source labelled windows와 source participant IDs, 사전선언한 target IDs를
  받는다. 문자열ID·source/target 비중복·source>=2사람·각class>=2source사람을 검증.
  Target windows/labels로fit하는 API는 제공하지 않는다. Query ID는 source에 없어야
  하고 선언한target에 있어야 한다. IDs는 caller 주장일 뿐 원파일 provenance 증명은 아니다.
- Feature StandardScaler는 source만 fit. C=1 linear SVC를 class별 one-vs-rest,
  probability=False,tol=1e-6,max_iter=100000, class_weight=None,shrinking=False로 고정.
  정수classes를 정렬하고 +1 decision margin 최대값으로 선택한다. 이 source-scaling과
  명시적solver 설정은 우리의 선택이며 LIBSVMFast probability 재현이 아니다.
- 모델은 source fit 뒤 frozen coefficient/intercept/scaler로 예측한다. Scaler/SVC
  재학습·query 일괄 정규화 없음. Fit 미수렴/비유한 특징·계수는 에러이며 우회하지 않는다.
- 합성 입력의 spec은250Hz/500samples/segment128/overlap64/nfft512로 고정.
  이는 MATLAB5초 기본 pwelch의 동일성 주장도 실제자료에 최적화한 선택도 아니다.
  Source3사람×5class×2trial,query2사람×5class×2trial,seed20260914를 사용한다.
  주파수는5개 분리된 정수합성tones. 최대4회fit(test내순서변경 포함)/suite,
  최대12fit총. 정확도>=.90은 단순toyfixture 구현sanity이며 뇌자료효능이 아니다.
- 독립 directDFT/Parseval검사,shape/NaN/constant/role/classcoverage거부,source순서변경,
  query변경시모델불변,querybatch/단일일치,positive-label방향,caller배열불변을 검사한다.

현재 사용할 환경은 기존 system Python3/scipy1.15.3/sklearn1.7.2/numpy2.2.6이다.
Repo .venv에는 sklearn이 없었으며 이를 설치하지 않는다. 원 .venv/CUDA변경0.
소스6GET 예산은완료. 구현 API 문서의 정확한 공식페이지2개(welch/SVC)만 web으로
추가 확인할 수 있다. 도구 내부GET/byte는측정불가이며 위6GET한도에숨겨포함하지 않는다.

추가 구현 경계(실행 전): expected integer classes도 caller가 사전선언하고 전class가
source에존재해야한다. 위4회fit은 pipeline 기준이며 five-class OVR는pipeline당
5 binary SVC fits,최대20 binary fits/suite·60총이다. 미수렴거부용mock은별도표시한다.

독립 코드검토 뒤 마지막 suite 전 보완: 평균 roundoff 때문에 0.1 상수도 잔차가
생길 수 있어 중심화 전 원값의 동일성을 검사한다. 변하는 실제 신호의 특징 정의는
바꾸지 않는다. Export 생성자의 shape/finite/양수scale와 deep tuple 복사는 두 번째
suite에서 보강했다. 남은 suite1회는 이 상수입력 회귀검사까지 포함하며 추가fit탐색0.
