# 필터 부호 독립성: 별도 수학·인공 검증

2026-09-08. Base `ea33fcc4a87acdb77ec91cbd266f874b89b03ef3`, main 단독 writer.
원 source39 후보는 `VALIDITY_FAILURE` 종료 상태 그대로다. 이번 후보는 **새 점수 정의의
ENGINEERING_ONLY**이며 사람 학습/최종 query/held60 실행을 허용하거나 자동 시작하지 않는다.

## 질문과 고정 범위

원 저보정 SSVEP 목표는 유지한다. 현재 질문은 metadata가 유용한지가 아니라,
필터의 임의 부호 때문에 계산이 깨지지 않는 방법을 수학적으로 정의할 수 있는가다.
고유벡터는 v와 −v가 같은 방향인데, 이전 방법은 native 필터와 같은 쪽으로 향하도록
맞추려다 거의 직각인 지점에서 멈췄다. 부호를 선택하지 않고 vvᵀ를 이용하는 방법을 검산한다.

인공 seed `20260908`, float64, CPU1thread, 기존 CUDA 환경만 사용한다.
독립 SciPy 고유문제, 직접 투영 후 시간평균 제거 점수, 부호/offset 반례, 1차 gradient,
고정200step 인공 optimizer를 검증한다. 별도로 이미 노출된 실패 지점 JSON
`/home/whwovy/task-trca-shape-anchor-failure-diagnostic-v1.json`
(SHA `86eda34fc11f9e20a3874ef0e952a2aedc8b60e7f908bdd290b84ccf84b1561b`)
의 S/C/R와 저장된 native anchor·cosine·위치 정보를 시험한다.
이것은 독립 인공 자료가 아닌 **알려진 실제 실패점 재검산**이다.
Source NPZ·raw·M 수치·query·held60은 읽지 않으며 실패 pipeline을 재학습하지 않는다.
조건/seed/학습률/규제량 sweep, 원 anchor threshold 변경, SHAM 제외는 없다.

## 수식과 바뀌는 의미

`B=C+tau diag(R)`, `tau=.1 lambda_min(C)/(16/9)`, `L Lᵀ=B`는 그대로다.
`P=top_projector(L⁻¹ S L⁻ᵀ)`, `K=L⁻ᵀ P L⁻¹`, `F=K/tr(CK)`로 둔다.
P=uuᵀ, v=L⁻ᵀu이므로 F=vvᵀ/(vᵀCv): 부호에 무관하고 C-trace가1이다.
`C<=B<=1.1C`에서 `1/1.1<=tr(CK)<=1`이라 anchor와 직각일 때 생기는
정규화 분모0은 없다. 이는 top-gap·SPD·분산·부동소수점 문제가 모두 사라진다는 보장은 아니다.

기존 global Pearson에는 `H=W(I−11ᵀ/classes)Wᵀ`를 통한
`N mean_xᵀ H mean_t`와 분산 보정항이 있다. W의 열별 부호가 바뀌면 H도 바뀔 수 있다.
따라서 F만으로 기존 점수를 모든 입력에서 그대로 복원할 수 없다.

새 점수는 **각 투영 성분의 시간평균을 뺀 뒤** 전체를 펼쳐 Pearson을 계산한다.
`J_band=sum_class F`와 시간중심 Gram으로
`rho=<J,Gxt>/sqrt(<J,Gxx><J,Gtt>)`를 계산하고 기존 signed linear band weights를 적용한다.
성분별 분산 정규화나 성분별 Pearson 평균은 하지 않는다. S/C 및 template 생성도 바꾸지 않는다.
모든 입력 채널의 해당 창 시간평균이0이면 같은 필터에 대한 원 점수와 같지만,
native 전처리의 긴 창 중심화 후 crop은 이 조건을 보장하지 않는다.
**단순 속도 개선/동일 scorer의 버그수정이 아니라, offset 정보 제거를 포함한 별도 scorer**다.
Offset 불변성은 필터를 고정한 점수에만 해당한다. Offset을 바꿔 support부터 재적합하면
원 S/Q 특징도 바뀔 수 있으므로 전체 pipeline 불변성은 주장하지 않는다.

공식 [PyTorch v2.2.2 eigh 계약](https://github.com/pytorch/pytorch/blob/v2.2.2/torch/linalg/__init__.py#L627-L647)은
고유벡터의 부호 비유일성과 근접 고유값 미분 문제를 경고한다(공식 pinned source와 설치본 확인).
위 C-normalized projector 및 새 Pearson 동등성/비동등성은 이 프로젝트의 대수적 유도이며
논문이 metadata 효과를 입증했다는 뜻은 아니다. 새 PDF/광범위 검색 대신 기존 landscape를 재사용한다.

## 변경·검증 소유권

Root가 새 `task_trca_shape_signfree.py`, 인공 tests/CLI, 문서와 연구 SQLite를 순차 수정한다.
수식/score 계약이 결합되어 별도 구현 worktree가 필요 없다. 두 reviewer는 공유 repo 읽기 전용,
사람 자료 접근·파일 작성·GPU 사용 없음. 기존36worktrees·다른 GPU process는 보존한다.
새 출력예산1GiB, GPU 추가할당1GiB 이하, 여유 디스크 약232GiB. 설치/push/cleanup 없음.
원 실행 manifest가 pin한 파일/과학 config는 변경하지 않는다.

Optimizer probe는2bands×8의 자유 logits16개와 난수 feature2개의 residual3개를
각200step 적합하는 연결성 검사다. 실제 shared Q15 learner/metadata/nested selection이
아니며 loss는 `CE(scores/.1)+.001 mean(coef²)`로, 기존 사람 loss의 band-weight합 정규화도
포함하지 않는다. 고정 인공 입력은 초기 CE가 거의0인 쉬운 사례여서 작은 감소를
학습 유용성·분류 개선·metadata 효과로 부르지 않는다. 다른 seed/난이도를 찾지 않는다.

## 검증 뒤의 결정 경계

인공 통과는 metadata 효능·실제 source39 전체 안정성·보정량 절감을 뜻하지 않는다.
사람 후보를 만들려면 새 scorer를 Q/QM/Q2/SHAM 등 모든 matched arm에 공통 적용하고,
원 native FULL/A0를 보존하며 별도 temporal-centered FULL로 점수 변경 효과를 분리해야 한다.
원 FULL의 대체 이름을 그대로 붙이지 않는다. 먼저 새 계약과 독립 감사·실행 경계를 검토한다.
그 과정 없이 실패 후보를 재개하거나 held60을 열지 않는다.

검증 결과와 수치 receipt는 실행 후 아래에 추가한다.
