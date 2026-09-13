# Welch/source-SVM 구현 분기 종료 — 합성 core 검증, 실제 효능 아님

2026-09-14 KST / 2026-09-13 UTC. Base `b6d8c4a`.

## 결론과 연구목표에서의 위치

**독립 Welch/source-SVM core를 구현하고 마지막 합성 suite 48개를 통과했다.**
상태는 `SYNTHETIC_CORE_QUALIFIED_NOT_HUMAN_BASELINE`이다. 완전한 저자 재현,
현대의 강한 기준선 확보, 실제 EEG 정확도 향상, metadata 효능을 뜻하지 않는다.
이번 유한 구현 분기는 종료한다. 실제 데이터 runner나 추가 decoder를 자동으로
붙여서 같은 개발 참가자에게 다시 실행하지 않는다.

[사전 계약](welch_source_baseline_v1_contract.md),
[구현](../src/cfeg/baselines/welch_source.py),
[시험](../tests/test_welch_source_baseline.py),
[출처·실행 상태](reports/welch_source_baseline_v1_state.json)를 함께 보존한다.

목적은 앞서 확인한 MAMEM의 source-trained spectral baseline 계열을 혼동 없이
다룰 작은 공통 기반이다. 원래 목표는 여전히 외부 acquisition M의 **Q·공통 정보
이상 이득과 보정량 감소**다. 이 core는 M 입력을 받지 않으며 그 목표를 달성한
것으로 세지 않는다. Q baseline 개선을 모든 M 실험의 무한 선행조건으로 삼지 않는다.

## 저자 코드에서 확인한 경계

MAMEM/eeg-processing-toolbox revision
`5a03abe2a6a874e9adaceea29a52c2fce35d8a03`의 정확한 6개 파일을 읽었다.
GitHub contents 응답과 decoded source의 길이·SHA-256·Git blob을 보존했다.
6 GET 모두 200, redirect/retry 0, 응답 body 합계 31,178 bytes,
decoded source 16,625 bytes다. 이번 마감에서는 저장된 원본의 해시·길이·HTTP
상태를 다시 검산했으며 네트워크 요청은 하지 않았다.

- `PWelch.m`은 빈 window/overlap 인자를 MATLAB `pwelch`에 넘긴다.
  따라서 SciPy 기본값을 그대로 쓴 것을 MATLAB 재현이라고 부를 수 없다.
- `LIBSVMFast.m`은 기본 linear C=1의 class별 이진 probability 경로를 사용한다.
  이번 코어의 명시적 양성 class margin OVR와 같지 않다. 별도 `LIBSVM.m`은
  margin 경로다. 저자 probability 열의 방향을 실제 실행에서 검증한 것은 아니다.
- 별도 `DigitalFilter`와 PWelch 내부 optional filter의 호출 방식이 다르므로
  계열 이름만 맞춰 전체 전처리를 재현했다고 볼 수 없다.
- `Rereferencing.m`의 meanSignal은 시간 평균 제거 경로다. 이번 6개 파일만으로
  저자 전체 pipeline의 모든 scaling 유무를 확정하지 않는다.

앞서 확인한 exampleDefault는 elliptic filter 파일을 지목하지만 원 보고서의
Table III 기본 설정은 Chebyshev I다. 이번에는 실제 filter 계수를 읽거나 원 MATLAB
pipeline을 실행하지 않았다. [앞선 비교기준 검토](mamem_baseline_contribution_review_v1.md)의
5초·LOSO와 우리 2초 reference 진단의 차이도 그대로 유지한다.

공식 API 문서는 SciPy 1.15.3 `welch`와 scikit-learn 1.7 `SVC`의 정확한 두 페이지만
확인했다. Web 도구의 내부 요청 수·전송량은 미측정이며 위 6 GET에 합산하지 않는다.
새 논문 발견·PDF 정독·실제 EEG 자료 수집으로 세지 않는다.

## 구현과 정보권한

- 호출자가 채널·시간창·필터·단위·원자료 participant ID를 책임진다. Reader/CLI/
  acquisition registry 연결은 없다. 따라서 ID 검사는 원 EEG provenance 증명이 아니다.
- 모든 Welch 길이·sampling rate를 필수 지정한다. Trial 전체 평균 제거,
  symmetric Hamming, complete segment의 one-sided PSD density 평균을 계산한다.
  남은 tail은 전체 trial 평균에는 들어가지만 불완전 FFT segment로 사용하지 않는다.
- Source에서만 StandardScaler와 class별 linear SVC를 fit한다. 이 scaling과
  probability=False margin OVR는 독립 구현 선택이지 저자 LIBSVMFast 재현이 아니다.
- Source/target 선언 ID 비중복, source 최소 2명 및 class별 최소 2명 coverage,
  expected integer classes를 검증한다. Query는 선언 target만 허용한다.
- Fit 뒤 scaler와 head를 불변 tuple로 내보내고 query에서 다시 fit하지 않는다.
  Export 직접 생성도 shape·finite·양수 scale·역할을 검사하고 mutable 입력을 복사한다.
  Margin은 확률이 아니다. 비수렴·상수 입력·수치 overflow는 거부한다.

## 합성 검증과 예산

사전 최대 3회 suite는 모두 사용했다. 관측 순서는 37/37(0.45초),
45/45(0.47초), 48/48(0.48초) PASS다. 첫 suite 뒤 ruff import-order 오류를
수정했고, export 검증과 상수 0.1의 중심화 roundoff 회귀 검사를 보강했다.
최종 ruff/diff check도 통과했다. **이번 마감에서 pytest를 재실행하지 않았다.**
시간은 pytest 보고값이며 전체 작업 경과 시간이 아니다. 별도 pytest stdout
파일을 새로 만들거나 독립 실행 증거처럼 재구성하지 않았다.

실제 합성 fitting은 suite당 2개 pipeline, 전체 6개 pipeline/30개 binary SVC다.
비수렴 거부 mock 3회는 별도이며 실제 fit으로 세지 않는다. 합성 5명·5class·각
2trial 중 source 3명, query 2명이고 seed는 20260914다. 분리된 정수 tone의
정확도 >=90%는 구현 sanity에 불과하다. 사람 SSVEP의 난도나 M 효과를 대변하지 않는다.

검사 범위는 직접 DFT/Parseval, odd/even nfft, scaling·입력 불변, 역할/class 거부,
source 순서 변경, query 일괄/개별 일치, 양성 class 방향, 실제 SVC와 export margin
일치, query 재fit 금지, export deep copy, 상수 입력의 Welch 호출 전 거부다.
독립 reviewer의 최종 검토는 코드/시험/상태의 정적 일치 확인이며 추가 실행은 0이다.

환경은 기존 `/usr/bin/python3` 3.10.12, NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.7.2, BLAS 1 thread다. Repository .venv에 sklearn이 없어 기존
system 환경을 사용했으며 설치·CUDA 변경은 없다. 전체 repository suite는 미실행이다.

## 최신 실제 결과와 다음 행동

현재 병목을 옛 MobileBCI의 self-only 포화 상태로 설명하지 않는다.
[실제 block-scaled 후속](block_scaled_router_human_v1_results.md)은 144 fits에서
학습 손실이 모두 감소했고, 36개 QM fit 모두에서 M이 gate와 class margin을 바꿨다.
그럼에도 저보정 QM−Q는 −0.162pp, 기술적 paired bootstrap 95% 구간
[−0.585,+0.304]pp이며, 획득 보정량 절감은 0%였다. 같은 16명의 개발 결과이지
독립 확증이나 모든 M의 무용성 증명은 아니다. 원 후보는 종료 상태 그대로다.
[N1 source39 임피던스](task_trca_n1_transport_recovery_r1_results.md)와
[MAMEM 결과의 기여 경계](mamem_baseline_contribution_review_v1.md)도 보존한다.

따라서 core 통과가 새 사람 실험의 충분조건은 아니다. 현재 검토 기록에서
**새로 적격화된 실제 M 후보는 없다.** EEG 총량·GPU 부족만으로 현재 음성을
설명하지 않으며, 측정된 M의 추가 정보, 활용 기작, pairing, 독립 검증을 구분한다.

다음 실제 실험에는 이전 음성과 구별되는 기작, 실제 pre-query M의 출처·시간 정렬,
동일 Q/common/label 비용의 Q/Q2/QM/조건부 SHAM 및 필요한 direct-M 비교,
효과·악화·획득비용·중단 기준과 미사용 검증자료 역할이 필요하다. 기존 자료로
이 조건을 충족할 수도 있으므로 새 데이터 확보를 모든 후보의 필수조건으로 만들지
않는다. 다만 원본에 없는 metadata를 추정값·공통 상수로 대체해 외부 M이라 부르지 않는다.

종료한 optimizer rescue·접근 실패 경로·같은 참가자에서의 무한 baseline 탐색을
자동 재개하지 않는다. Held60·사람 요청·유료 자원은 별도 승인이다. 이번에 새 raw/
M/사람 fit/예측/held60/요청/유료 사용은 0이며 전체 연구목표는 아직 미완료다.

Root가 main에서 코드·문서·근거맵의 단독 writer였고 reviewer는 읽기만 했다.
추가 worktree는 필요하지 않았다. 기존 46개 worktree와 사용자 untracked 시험 디렉터리
8개를 보존하며 cleanup/push는 하지 않는다.
