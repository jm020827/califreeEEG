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

## 실제 검증 결과

Runtime commit `f00ec3f464c18b8cc3727686e262f6707efa0207`.
저장소 전체 **2248tests PASS, 313.18초**, 기존 Torch warnings68개.
실행 중 수정은 문서뿐이며 runtime/tests는 f00ec3f와 동일했다. Ruff/diff check PASS.
요약 [JSON](reports/task_trca_shape_signfree_v1_engineering.json),
[전체 JUnit](/home/whwovy/task-trca-shape-signfree-engineering-UQgWDi/full_tests.xml)
SHA `27d29bdfe1c124b186a61436690d3e02dff9a28561a99bf741e79d580b289537`.
원본 [검증 receipt](/home/whwovy/task-trca-shape-signfree-engineering-UQgWDi/verified.json)
SHA `0467941977362b51cd906a88eeda262526ade507fedf79ef65dc8c3c62887cbf`, 권한0400.
상태 `SIGNFREE_ENGINEERING_PASS`; 기존 source39 상태는 여전히 `VALIDITY_FAILURE`다.

| 검증 | 결과 | 해석 한계 |
|---|---|---|
| 신규 인공 tests | 32 PASS, 1.36초 | 사람 자료를 읽지 않는 단위/gradient 검사 |
| 독립 SciPy projector / literal 점수 | 최대 오차3.47e−18 / 6.66e−16 | 같은 생성 배열의 독립 계산 |
| CPU/CUDA400steps씩 | 계수 최대차2.62e−14, 점수4.45e−16, argmax동일 | 자유 logits+난수 residual, 실제 Q/QM learner 아님 |
| 알려진 실패 지점의 새 F | C-trace .9999999999999982, SciPy차4.33e−15 | 단 한 지점의 재검산; 새 독립 검증 sample 아님 |
| 같은 지점의 1차 미분 | gradcheck PASS, CPU/CUDA gradient차9.19e−17 | 경로 전체나 top-tie에서의 안정성 보장 아님 |

인공 CPU400updates는 .83584초, GPU400updates는1.71443초이며 작은 입력에서 GPU가
더 빠르다는 결과가 아니다. GPU peak allocation17,320,960bytes / reserved23,068,672bytes다.
인공 base loss `4.34566024e−6 → 4.34564000e−6`, residual loss
`4.34561977e−6 → 4.34561955e−6`은 매우 작은 변화다. 다른 난이도/seed를 찾아
성공 사례로 바꾸지 않았다. 실패 행렬에서는 학습 업데이트/분류 점수 계산0이다.

첫 검사에서 상수 시계열에 기존 scaled-mean 감소의 반올림 잔차가 남아 가짜 양의
분산으로 통과하는 문제를 발견했다(27PASS/1FAIL). 새 통계 builder만
`positive scaling → 첫 샘플 빼기 → 시간평균 빼기`로 수리해 정확한 상수 채널의
분산0 거절을 확인했다. 모든 projection-null/cancellation 사례에 대한 수치 보장은 아니다.
기존 pinned operator는 변경하지 않았다. 이어진32tests에는 dense 반복 하위 고유값의
비영 R미분, top-tie 거절, channel permutation, fixed-filter offset/scale 불변성,
회전된 완전 frame의 개별 F변화와 J/점수 불변성, 파일 pin/symlink 거절도 포함한다.

정확한 반례: 투영된 두 성분이 `X=[[1,3],[3,5]]`, `T=[[1,3],[-3,-1]]`일 때
원 global Pearson은 `−1/√10`, 두 번째 필터의 부호를 양쪽 모두 뒤집으면 `+1/√10`이다.
새 점수는 두 경우 모두1이다. 각 성분을6번 복제한12-filter 검사에서도 동일하다.
따라서 새 계산의 부호 독립성을 기존 native exactness와 동시에 주장하지 않는다.

예비 [순수 인공 receipt](/home/whwovy/task-trca-shape-signfree-engineering-UQgWDi/synthetic.json)
SHA `df60bce9fd9ddf42679861b98b0f5f4a8aa948b7194beb1c2684d559af86c9b2`도0400 보존했다.
이후 CLI 검증을 `python -O`에서도 생략되지 않는 명시적 예외로 바꾸고,
`Q`라는 과도한 이름을 `BASE_LOGITS`로 고쳤다. 위 최종 receipt는 같은 생성 입력/학습법을
재검산하고 알려진 실패점을 추가한 것이다. 총 인공 optimizer 업데이트는 예비800+최종800이며
사람 학습은0이다. 결과 기반 hyperparameter/seed 선택이나 old candidate 재실행은 없다.

공식 pinned Torch 원격 source와 설치 source는 SHA
`d0a40f59375463726f9dbe5b758505966b3a6c0fe71eda600bfded3ecbd3fc7a`로 동일했다.
근거 범위는 `eigh` sign/gradient 경고이며 새 문헌의 효능 결과가 아니다.
`academic-research`의 현재 implementation gap을 좁히는 검증이며,
native-compatible 방향 정의의 기존 공백은 이 scorer 변경만으로 닫지 않는다.

다음 연구 단계는 **새 scorer를 공통 적용한 Q 대 QM 설계와 native/centered FULL 비교를
명세하는 것**이다. 기존 Q feature 생성·head·band-sum-normalized loss·nested 선택의
실제 결합 검증, 새 독립 감사와 별도 실행 manifest가 남아 있다. 보정비용/metadata 개선
기준을 낮추거나 현재의32tests를 그 전체 pipeline 검증으로 대체하지 않는다.
