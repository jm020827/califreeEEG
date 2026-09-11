# 실제 형상 통합 검증: metadata 경로 작동, 학습법 우월성은 미확립

2026-09-11. **24/24 fits·2400 proposals·720 ridge solves 단회 완료.**
수치 관문은 통과했으나 원 결과의 역할 기록 덮어쓰기 결함을 별도로 보존·검산한다.
새 사람 EEG/metadata 접근·학습은0이다. 기존144/336-fit 부정 결과는 변하지 않는다.

## 무엇을 배웠고 무엇을 구현했나

지난 단순 수치 진단을 실제 adapter 형태로 옮겼다.13개 expert, Q123/aux125,3개class,
source ridge→harmonic projection→router→probability mixture를 연결했다. Q54/Q2/M2는
인공 값이므로 실제 EEG/IMU 특징 추출 전체를 검증한 것은 아니다.

세 학습기를 비교했다. ORIGINAL은 기존 합산/AdamW, SCALE은 표준화한 블록별 평균,
SAFE는 SCALE에 source학습loss 비증가와 한 번의centered logit변화≤1 조건을 더했다.
SAFE만 사전에 후보로 지정했고 결과가 가장 좋은 ablation으로 바꾸지 않았다.
거부 시 파라미터만 복구하고 Adam moments/timestep은 진행하는 별도 제안 알고리즘이다.

## 실험 결과: 유익한 정보와 무관한 정보를 구분

인공20명/60runs 중bank4명,학습8명,평가8명을 분리했다. 평가와학습은 각(h,m,r)의8cell을
완전균형으로 갖는다. POSITIVE는m가유효공간그룹을알려주며NULL은독립h가그룹을결정한다.
모든조건에3class를두번씩포함했다. 아래수치는우호적으로구성한인공신호의결과이지사람예상성능이아니다.

| 인공 조건·학습기 | Q 정확도 | Q2 정확도 | QM 정확도 | SHAM 정확도 |
|---|---:|---:|---:|---:|
| POSITIVE ORIGINAL | 0% | 0% | 100% | 51.39% |
| POSITIVE SCALE | 4.17% | 4.17% | 94.44% | 20.83% |
| POSITIVE SAFE | 4.17% | 4.17% | 94.44% | 20.83% |
| NULL ORIGINAL | 0% | 0% | 0% | 0% |
| NULL SCALE | 4.17% | 4.17% | 3.47% | 3.47% |
| NULL SAFE | 4.17% | 4.17% | 3.47% | 3.47% |

POSITIVE SAFE에서 Q NLL1.069309,QM .379880으로차이.689429이고정확도+90.28pp다.
Q2대비도사전.02NLL/.10정확도기준을넘었다. M대응만바꾸면gate변화.575511,
classmargin변화1.095383으로최종확률까지영향이전달됐다. Query정답을M에넣지않았다.

NULL의 SAFE QM은 Q보다NLL이.000650낮지만정확도는.6944pp낮아사전허용범위내다.
NULL에도M개입의작은gate .000966/margin .002052변화가남는다. **입력작동=유용성은아니다.**
완전균형인공null이어도유한학습과추가좌표가같은학습경로를보장하지않으므로
통계적조건부독립검정이나정확한효과0증명으로쓰지않는다.

## 중요한 반대 근거도 남긴다

1. **ORIGINAL QM의양성성능이더좋았다.** SAFE의보편적우월성이나실제EEG이득을주장할수없다.
2. SAFE800개제안은전부factor1로수락됐다. SCALE과최종결과가같으므로이번통합에서
   backtracking의독립효과는측정되지않았다. 축소·거부경로는비학습unit검사범위다.
3. ORIGINAL의한step중심logit변화는최대약19.48,SAFE는≤.558885였다.
   ORIGINAL 일부는중간NLL악화가있었고SAFE는없었다. 그러나이는더좋은최종일반화의보증이아니다.
4. Q가매우낮은것은양쪽공간그룹중잘못된고주파성분이강하게나타나는설계된phantom때문이다.
   그수치로실제SSVEP의EEG-onlybaseline을평가하지않는다. 이phantom을새연구목표로삼지않는다.
5. SHAM은고정순환대응이며source고유특징과부분관계를학습할수있다. 독립null은별도NULL세계다.

## 보고 결함과 복구 경계

실행동결785ad62의runner에서전체 `roles` 사전을마지막한run의 `roles`로덮어썼다.
따라서원 `result.json`은96개역할행대신v00/속도0의한행만저장했다. 학습·추론이끝난뒤
발생한보고변수결함이며모델에들어가는bank/recipient구성과혼동하지않는다. 원수치관문을
통과했어도원산출물의완전한역할기록요구는충족하지못했다.

원산출물을e905472에보존하고변수명을고쳤다. **재학습·실험재실행0**이다.
Role복구는고정코드와저장generated배열에서별도로추론가능한범위만기록하고,
당시완전한실행ledger가있었다고소급표시하지않는다. 수치감사와역할감사의범위를분리한다.
추후humanrunner는출력schema/rolecount를성공발행전에검사해야한다.

[역할감사](reports/block_scaled_router_v1_protocol_audit.json)는.116279초에고정코드pin과
저장배열의역할/좌표/균형을확인했다. [재구성96행](reports/block_scaled_router_v1_roles_reconstructed.json)은
`reconstructed_from_frozen_code_and_saved_arrays`로표시했다. [독립수치감사](reports/block_scaled_router_v1_numeric_audit.json)는
.152423초에24모델/3456예측/800SAFE경계/다섯gate를재구성했다. 학습NLL최대오차3.33e−16,
평가NLL4.44e−16이다. SAFE/SCALE의공유bias7쌍은동일하지않았으나softmax에서상쇄되는
방향이고최종확률차이≤4.44e−16이었다. 파라미터까지동일하다고표현하지않는다.

## 자원·검사·다음 결정

계약da11bf3→코드/검사동결785ad62→단회학습3.041282초(최종report쓰기전),
실행/시도2400proposals일치,SAFE후보검사800/실제trialloss800,720/720ridge.
일반학습loss평가4800회와추가trial평가를구분한다.4producer파일총1985512B,
NPZ는로컬ignored감사자료이고3text산출물은git보존했다.

비학습unit6cases를2/2suitecalls에서통과(.003/.004초),optimizer0. Ruff/compile검사통과,
전체repo suite0. 실행전review로h/m·evaluation균형검사/role전체검사/querylabelprob불변/
attempted-completed회계/출력용량사전검사를보강했지만변수덮어쓰기는사후발견했다.
보고변수수정후Ruff/compile만확인했고unit예산을늘리거나학습을다시돌리지않았다.

[결과](reports/block_scaled_router_integration_v1/result.json),
[전체training journal](reports/block_scaled_router_integration_v1/trajectory.jsonl),
[사전계약](block_scaled_router_integration_v1_contract.md),
[단위검사](reports/block_scaled_router_v1_unit_tests.json).
ResultSHA `fa3ed60ce3973917709b3c3dae3df007b48783d99dd7ae4077f4904bb17c2b39`.

이번24-fit예산은종료한다. [사람후속초안](block_scaled_router_human_v1_draft.md)은
**DRAFT_NOT_EXECUTABLE**다. 독립감사·역할보고수리·새runner의고정계약과생성mock검사뒤,
실제gyro가Q/Q2/SHAM보다정확도와실제보정prefix를개선하는지를한번만시험하는다음단계다.
이번합성성공을보정량감소로보고하지않는다. NeuralODE·새timingM·raw재추출0,
held60·외부요청·유료는별도승인,Choi보류/source39종료유지다.
