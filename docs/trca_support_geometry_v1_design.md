# TRCA support-only geometry 진단 — 결과를 본 뒤의 별도 분석

2026-09-08. 원 연구목표는 acquisition metadata가 EEG 정보 이상의 저보정 이득을 주는지 검증하는 것이다.
[이전 후보](metadata_prior_source39_v1_results.md)는 효과 미확립으로 종료했다. 이번 작업은 후보 재개나
새 효능실험이 아니다. **Metadata를 넣기 전 공통 규제부터 성능이 떨어진 이유를 좁히는 진단**이다.

## 질문과 정보 경계

관측한 ISO−FULL 손실을 이미 알고 세운 post-outcome 분석이다. 사전등록 확증으로 부르지 않는다.
이번 새 support 접근 전에 [고정 JSON](../configs/analysis/trca_support_geometry_v1.json)을 commit한다.
기존 source39 전원, dry/wet, 125/188/250/500 samples, k3/5를 모두 포함한다.
39×2×4×2×12 classes×5 bands = 37,440개의 FULL/ISO **쌍** 기록을 빠짐없이 낸다.
같은 사람의 반복 기록이므로 독립 표본37,440개로 신뢰구간이나 유의성을 계산하지 않는다.

허용 입력은 정확히 pinned native manifest와 그39개 archive다. ZIP 전체를 전후 SHA-256 검증하는 것은
허용하지만, 수치로 해독하는 것은 `x_N[interface, blocks0..4]`뿐이다.
NPZ의 모든 배열을 `np.load`로 펼치지 않고 ZIP_STORED의 실제 header/offset에서 두 support 구간만 읽는다.
ZIP64 local header와 short read를 검사한다. Query bytes까지 hash한다는 사실과 query 값을 해독하지 않는다는
사실을 구별한다. Full/A0 점수, proxy, query, 숫자M, raw MAT, held60, retired1–3은 해독하지 않는다.

## 고정 비교와 해석

Native와 같은 support의 S,C를 사용한다. FULL은 `(S,C)`, ISO는 `(S,C+τI)`의 leading generalized eigenvector다.
`τ = 0.1 tr(C)/8`만 사용하며 gamma sweep·Q/QM 적합·query 분류는 없다.

- C 고유값 λ의 평균 정규화 값, 조건수, participation-ratio effective rank=`tr(C)^2/tr(C²)`를 기록한다.
- 각 방향의 상대 규제는 `τ/λ`다. γ=.1은 모든 방향10% 규제를 뜻하지 않는다.
- FULL/ISO 단위 방향 u의 `ρ=τ||u||²/(uᵀCu)`와 부호 불변 각도 sin을 기록한다.
- 동일 ISO 방향을 C-norm에서 B-norm으로 바꾸는 배율 `a=(1+ρ_ISO)^(-1/2)`를 기록한다.
  이는 FULL에서 ISO로의 전체 필터 크기비가 아니라 **ISO 방향을 고정했을 때**의 정규화 효과다.
- 동일 사람/조건/k/band의12 class 사이 a의 CV·min/max를 계산한다. Native flattened eTRCA에서는
  class별 상대 크기가 점수에 영향을 줄 수 있다. 공통 배율 불변성과 혼동하지 않는다.

모든 조건/k의 min/p10/median/p90/max와 participant별 평균 분포를 보고한다.
`τ/λ≥1, ≥10`, `sin≥.1, ≥.5`는 기술적 빈도이지 효능 임계값이 아니다.
S,C, 두 단위 방향은 별도 geometry.npz에 보존해 원 EEG 없이 독립 재계산할 수 있게 한다.
고유방향 값은299,520개,12-class CV 그룹은3,120개다. CV는 모집단 표준편차(ddof=0)/평균,
분위수는 NumPy linear interpolation이다. 전체 scope의 참가자 평균은8조건×12classes×5bands를,
조건별 scope는12classes×5bands를 평균한다. 이후 참가자39명의 분포를 별도로 보고한다.
NPZ의 S,C는 `[37440,8,8]`, full_unit/iso_unit는 `[37440,8]`이고 result.rows와 행 순서가 같다.
행 순서는 subject→samples→interface→k→class→band이며 각 행에 이6개 식별자를 저장한다.
회전/상대 규제가 작으면 큰-기하변화 설명이 약해진다. 크면 연산자 변화는 확인되지만,
**어느 방향이 분류에 유용했는지, 이것이 정확도 손실의 원인인지, 다른 γ가 성공하는지는 모른다.**
어떤 진단 결과도 metadata 추가 효용이나 calibration 절감의 증거로 승격하지 않는다.

## 구현·검증·중단

원 operator/runner/config/result는 수정하지 않는다. 새 module/runner/tests만 추가한다.
먼저 인공 isotropic C의 공통 배율, ill-conditioned C의 상대 규제, 공통 파형 scale 불변성,
native operator와의 동일 방향, class별 정규화 효과 및 부호 불변 각도를 검증한다.
NPY prefix loader는 비허용 블록을 NaN으로 채운 인공 ZIP64 fixture, header/dtype/shape/압축/
중복/경로/해시 오류, 불완전 읽기, 배타 출판 및 start-before-input lifecycle로 검증한다.
실제 실행은 clean commit·고정 source hash·기존 Py3.10/NumPy1.26.4/SciPy1.15.3·CPU/BLAS1,
600초/증분1GiB 상한 아래 한 번만 수행한다. 오류는 보존하고 자동 수식/강도 rescue를 하지 않는다.
완료 표시는 GEOMETRY_DIAGNOSED이며 학습 성공 표시가 아니다.
실패 시 별도 failure.json을 보존한다. 실제 입력 전에 이 파일 허용과 집계/배열 축을 JSON에 명료화했다.
두 고유문제에서 finite/real eigensystem, 양의 C/B 고유값, leading gap>
`1e-10*max(1,max(abs(eigenvalues)))`를 원 operator와 동일하게 요구한다. Jitter/pseudoinverse/gap rescue는 없다.

## 작업 소유권과 연구 근거

`coordinate-worktree-changes`: base f5454bd, integration main. 기존32worktrees와 사용자 환경을 보존한다.
약245GiB 가용공간에서 증분1GiB만 책정한다. Loader/geometry/runner/test는 같은 작은 입출력 계약에 의존하므로
main이 순차 작성·commit·실행·통합검증을 전담한다. 별도 writer/worktree는 만들지 않는다.
prior_operator는 수식/native 코드, next_step_data_check는 archive layout, prior_sensitivity_review는
후속 독립 검산을 shared repository에서 읽기 전용으로 맡을 수 있다. SQLite 연구공간도 main 단독 writer다.
설치·push·외부 요청·worktree cleanup은 없다.

`academic-research`: 기존 landscape의 실제 음성결과 claim:6c575153b9f5a422,
technique:feb2f44f47a68444, 과학 gap:6a8254f849b36b2f를 재검토한다.
이번 질문은 자체 수식과 이미 있는 support에서 직접 검사할 수 있어 새 검색/PDF는0으로 둔다.
CSP→TRCA 문헌 비유는 진단 가설의 배경이지 우리 metadata의 효능 증거가 아니다.
결과 후 연구공간에는 기하 관측과 미확인 인과/효능을 분리해 기록한다.
