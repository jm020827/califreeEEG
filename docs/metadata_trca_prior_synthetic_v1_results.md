# M-prior 합성 v1 — 구현 완료, 분류 개선 검증 미확립

2026-09-08. 공식 고정 screen: **SYNTHETIC_CANDIDATE_NOT_ESTABLISHED**.
사람 EEG 효과나 metadata 무용성의 판정이 아니다.
[고정 계약](metadata_trca_prior_synthetic_contract.md),
[전체72summary·24contrast](reports/metadata_trca_prior_synthetic_v1_summary.json).

## 쉽게 말하면

Metadata로 공간필터의 채널별 규제를 조절하는 코드는 만들었고, 기존 eTRCA와의 호환성도 확인했다.
그런데 **이번 검증용 합성 문제가 너무 쉬웠다.** 주요3시나리오는 M 없이도 전부 정답을 맞혔다.
M이 반복 불일치를 예측하는 데 조금 도움이 돼도, 분류 정확도나 보정 예시 절감으로 이어지는지를
판별할 여지가 없었다. 이 난이도 문제는 이번 검증 설계의 한계다.

또한 같은 EEG Q를 한 번 더 학습한 Q2가 M을 추가한 QM보다 proxy를 더 잘 예측했다.
따라서 **‘M이 Q를 도왔다’는 제한된 proxy 결과만으로 metadata 고유의 가치를 입증했다고
말할 수 없다.** 구현과 단순 수학 확인은 통과했지만, 연구목표에 필요한 효과의 연결은 미확립이다.

## 실제 실행과 보존 기록

- 계약·JSON을58522c6에서 먼저 고정했고, 효능 seed26090817 공개 전 구현과 대조군을 검토했다.
  참가자24×4독립 인공 시나리오×k3/5×9arms, 12class/8channel/1synthetic band/0.5초다.
  이24개는 실제 참가자가 아니다. 시나리오 간 동일 index도 동일 사람의 반복 측정이 아니다.
- 첫 start는1ad25a9에서 게시됐으나, 첫 informative 평가 참가자의 k3 MISSING 검사에서
  NumPy testing lazy import가 SVE 확인 subprocess를 호출하여 guard에 걸렸다.
  일부 인공 fit/예측은 계산됐지만 최종 결과는 없었다. 실패를 숨기거나 그 경로를 덮어쓰지 않았다.
- Exact 비교를 `np.array_equal`로 바꾼 인프라 복구 외에 seed/생성기/학습/예측/기준 변경은 없다.
  새 clean98826734ceb032eeea073f688d1907fb5c6b7220에서 cold-r1을 완료했다.
  실행 wall5.707초, Python3.10.12/NumPy1.26.4/SciPy1.15.3, BLAS/OMP/MKL 각1thread.
- 총 시작2회(실패1＋복구 완료1), **같은 고정 suite의 완성 결과1개**다.
  별도의 독립 실험 두 개로 세지 않는다. CPU 순차 실행이며 GPU 효능/속도 비교는 하지 않았다.

| 산출물 | SHA-256 |
|---|---|
| 최초 실패 start | `8b2a818d487b813772a0fcf0695daa9e9f218ad0828873a8e9d8021b84bd2e8c` |
| cold-r1 start | `deaa57afc8299e53e63d23215e0d96b6461542b1010487dbfc63a106f912363a` |
| cold-r1 result | `29706bb23559257783894f973c69df4896b21310397399de42f7cb1e0701beba` |
| frozen JSON | `3dd7c5fa38d2b5416a2e1bd166b5b50fcaec7c00af527691d8bd4aa3b81a9da2` |

Local root `/home/whwovy/metadata-trca-prior-synthetic-v1-cold-r1`의 start779bytes와
result2,863,773bytes는0400으로 보존했다. 이 권한은 일반 수정 방지이지 절대적인 불변 저장소가 아니다.
기존 실패 root에는 start만 남아 있다. 모델·숫자 입력을 사후 선택하거나 바꾸지 않았다.

## 모든 분류 결과

아래 묶인7arms는 정확도뿐 아니라 **개별 query의 class 예측도 전부 같았다**.
각 시나리오·k에서24인공 참가자×24query이며, query를 독립 참가자로 세지 않는다.

| 시나리오 | k | FULL 정확도% | ISO 정확도% | Q=Q2=QM=SHAM=PERM=STALE=MISSING 정확도% |
|---|---:|---:|---:|---:|
| M에 추가 정보가 있도록 생성 | 3 | 100 | 100 | 100 |
| M에 추가 정보가 있도록 생성 | 5 | 100 | 100 | 100 |
| 독립 M | 3 | 100 | 100 | 100 |
| Oracle-Q 제공 | 3 | 100 | 100 | 100 |
| 독립 M | 5 | 100 | 100 | 100 |
| Oracle-Q 제공 | 5 | 100 | 100 | 100 |
| 위상 변화, 일정한 잡음 크기 | 3 | 98.78472 | 98.61111 | 98.78472 |
| 위상 변화, 일정한 잡음 크기 | 5 | 98.61111 | 98.61111 | 98.61111 |

따라서 QM−Q, QM−Q2, QM−SHAM은8개 시나리오×budget cell 모두0pp,
각 cell의 참가자 help/tie/harm은0/24/0이다. 모든 residual arm의 Q 대비 prediction change도0이다.
FULL은 phase-only k3에서 Q와 예측2개가 다르지만 평균 정확도는 같고,
ISO는 예측1개 차이로 전체576개 중 정답1개가 적었다.

Q/QM의 처음 관측된80% 도달은4시나리오×24개 **96cell 전부 k3**로 같다.
이 suite는 k0/k1/k2를 관측하지 않았으므로 최소36labels가 필요하다는 뜻은 아니다.
보정량 감소, 실제 EEG 성능 또는 실제 사용자 시간을 절감했다는 결과는 없다.

## 반복 불일치 proxy 결과

MSE는 낮을수록 좋다. 이는 log repeat-disagreement 예측 오차이며 정확도%나 물리적 잡음 분산이 아니다.
FULL/ISO에는 proxy 모델이 없어 MSE는null이다. 나머지 모든 값은 linked JSON에 있다.

| 시나리오 | k | Q | Q2: Q만 추가 적합 | QM | SHAM_REFIT |
|---|---:|---:|---:|---:|---:|
| informative | 3 | 0.01289561 | 0.00514736 | 0.00976243 | 0.01292369 |
| informative | 5 | 0.01309186 | 0.00523083 | 0.01000706 | 0.01322356 |
| independent | 3 | 0.01264126 | 0.00604990 | 0.01300116 | 0.01282424 |
| independent | 5 | 0.01266086 | 0.00608397 | 0.01282647 | 0.01281636 |
| q_sufficient | 3 | 0.00991244 | 0.00560511 | 0.00795968 | 0.00999173 |
| q_sufficient | 5 | 0.01008756 | 0.00574554 | 0.00813094 | 0.01009872 |
| phase_only | 3 | 0.01038571 | 0.00960128 | 0.00968758 | 0.01046239 |
| phase_only | 5 | 0.00939108 | 0.00845703 | 0.00872257 | 0.00939216 |

Informative k3에서 QM은 Q보다 MSE가24.3% 낮지만, Q2는60.1% 낮다.
이 차이는 고정된 선형 Q가 충분히 적합되지 않았음을 시사하며 Q2 대조군이 필요했던 이유다.
새 baseline을 Q2로 소급 교체하거나, 그 뒤 M을 다시 적합해 유리한 결과를 찾지 않았다.

Oracle-Q에서도 M proxy 이득이 보인다. 전체 oracle-Q 벡터가 정보를 포함하더라도,
제한된 채널별 선형 Q 모델은 M의 채널간 중심화 변환을 같은 방식으로 표현하지 못할 수 있다.
이를 ‘충분한 Q 밖에 새 정보가 생겼다’고 해석하면 안 된다.
Phase-only도 M proxy가 좋아지지만 생성상 변화 원인은 측정 잡음이 아니라 위상이다.
Proxy 예측이 좋아졌다는 사실만으로 전극 접촉 잡음을 알아냈다고 말할 수 없는 이유다.

## 검증과 독립 검산

- 순수 operator63tests와 최종 consumer/runner21tests **84PASS1.75초**, Ruff6files/format/diff check PASS.
- 전체1644tests PASS243.81초/기존 Torch warnings68. 이 전체 실행은 cold assertion 복구와
  새 child test 추가 전 collection이며, 복구 후 최종 상태는 위84tests로 다시 확인했다.
  최종 전체1645tests를 실행했다고 잘못 합산하지 않는다.
- 실제 pinned ETRCA와 인공4cases(k3/5×centered/nonzero offset) 비교에서96query 예측 일치,
  correlation maxerror8.88e−16. Native raw preprocessing과 실제 사람 결과 재현은 아니다.
- 읽기 전용 auditor가 saved predictions로1728행 정확도를 exact 재계산했고,
  전72summary/24contrast/864attainment/12fit/eval·derangement를 독립 검산했다.
  Summary maxerror1.11e−16, contrast8.67e−19. 모든192 MISSING↔Q 쌍은 정확히 같다.
- Root는 같은 고정 artificial generator만 재사용하고 Q 특징·least-squares reference projection·
  correlation·proxy·저장계수 예측·prior를 별도 계산했다. 전1344개 proxy arm에서
  MSE maxerror2.50e−16, prior2.22e−15. 새 후보/새 독립 simulation이 아닌 검산이다.
- Prior trace error≤2.67e−15, 같은 C에서 regularized arms의 penalty trace 상대 차이≤5.90e−16.
  최소 분모 고유값4.08471, 최소 top gap0.241061, norm error≤6.67e−16으로 수치 붕괴는 없다.
  Auditor는 저장 고유값 진단을 점검했지만 전체 S/C·고유벡터를 독립 재계산하지는 않았다.

연구공간 `provenance/metadata_trca_prior_synthetic_v1_audit_20260908.json`에 검산 범위를 보존했다.
원문 재검색/PDF 추가0, 실제 사람 데이터 접근0, 추가 데이터 확보0이다.

## 판정과 다음 경계

고정 screen의 informative proxy 조건은 통과, 분류 개선 조건은 실패했다.
Null accuracy bound는 통과했지만 ceiling 때문에 허위 이득을 잘 검출하는 대조였다고도 말할 수 없다.
**이번 후보는 효과 미확립으로 종료하며 사람 EEG 실험으로 자동 승격하지 않는다.**

보존할 성과는 native-compatible operator, 누수 없는 입력 경계, Q2/SHAM 등 비교 구현과 검산이다.
고칠 연구상의 문제는 두 가지다: downstream 검증의 난이도 민감도와 충분히 적합된 Q 대조군.
앞선 V4에서도 ceiling이 있었으므로, 이번에 같은 한계가 반복된 것을 명시적으로 남긴다.
Seed/잡음/규제 강도를 바꾸며 양성이 나오는 설정을 찾지 않는다.

향후 별도 설계를 한다면 M 효과를 보기 전에 독립적인 engineering fixture에서 baseline 오답과
필터 변화의 점수·결정 민감도를 확인하는 기준을 먼저 정해야 한다. 그 조건을
이번 결과에 맞춰 사후 ‘성공’ 기준으로 바꾸거나 현재 source/held를 개봉하지 않는다.
새 외부 dataset 확보는 이 합성 ceiling을 해결하지 않는다. 연구목표는 그대로지만,
지금은 데이터 양보다 **효과를 판별할 검증 설계**가 다음 검토 대상이다.
