# 실제 source-borrowing v1 종료: metadata 이득·보정 절감 없음, self 선택 포화

2026-09-11. **CLOSED_NEGATIVE / NO_DEMONSTRATED_M_ACTUATION**.
이는 이번 고정 학습기의 결과다. 외부 acquisition metadata에 유용한 정보가 전혀 없다는
증거나, source 차용이 본질적으로 불필요하다는 결론은 아니다.

## 쉬운 설명

“내 EEG만으로 애매하면 metadata를 보고 다른 사람의 판독기를 골라 빌리자”를 실제
16명 자료에서 시험했다. 하지만 학습이 끝난 선택기는 모든 방법에서 사실상 **내 판독기만
선택**했다. Metadata는 내부 선택 점수를 바꾸었으나 다른 사람에게 주는 가중치가 이미
거의0이라 최종 답은 바뀌지 않았다. 합성 실험의100%는 실제 EEG로 이어지지 않았다.

더 중요한 한계는144fits 중32개에서 초기 균일 차용보다 최종 class-balanced 학습NLL이
나빠졌다는 점이다. 따라서 “최적으로 학습해 보니 source/M가 쓸모없었다”라고 말할 수 없다.
현재 evidence는 **이 parameterization·학습 설정이 self 선택에 포화된 채 끝났다**는 것이다.
학습률·최적화·특징 중 무엇이 원인인지는 저장된 최종 결과만으로 확정하지 않는다.

## 실행과 공정 비교

- 원 저보정 closed-set SSVEP 목표와 지난336-fit 부정 결과를 보존했다.
- 실행계약2fbd549 → 생성검사·최종코드/config 동결adb7ee3 → 실제1회 실행.
  고정cache1checksum/1numeric load,raw0,576/576classridge solves,
  144/144router fits·14400updates,기록된 계산13.809137초,exit0.
  이 시간 필드는 마지막 result 쓰기 직전이며600초timer는쓰기가끝날때까지적용됐다.
- 같은16명/48run, k1/2/3/5,기존support/query40..59/EEG9채널/500Hz/1초를 유지했다.
  Source4사람×3속도+self13experts. Recipient 전 조건과 해당validation/test는bank에서제외.
- Q는효과적으로작동하는EEG/common123입력, Q2/QM/SHAM125입력이다. 복사한공통열이
  softmax에서소거되지않도록pairwise차이와selfinteraction을사용했다. Q2는EEG-derivedaux2.
- SHAM은targetM유지/sourceM만같은속도내4사람순환. M2벡터쌍multiset을보존하면서
  source EEG–M 대응을깬다. 실제aux변화검사통과. 모든M를없애는chancebaseline이아니다.
- Inner96fits/9600updates후전체12개arm/fold보정정책을저장·fsync/hash로잠갔다.
  그후outer48fits;각context의4heads모두동결후outerqueryexpert점수/예측을계산했다.
  QueryM/labels/index를router입력에넣지않았다. 이16명은이미본개발자료다.

## 수치 결과

| 방법 | k1/2 평균 정확도 | source에서 선택한 보정량 | 그 정책의 정확도 |
|---|---:|---:|---:|
| Q: EEG/common 선택기 | 67.8548% | k5/class | 70.5795% |
| Q2: EEG 보조입력 추가 | 67.8548% | k5/class | 70.5795% |
| QM: 실제 gyro metadata 추가 | 67.8548% | k5/class | 70.5795% |
| SHAM: source metadata 대응 교란 | 67.8548% | k5/class | 70.5795% |
| 학습 선택기 없이 target-only | 67.8548% | QM과같은k5 | 70.5795% |

정확도는사람·속도에동일가중한balanced accuracy다. 모든학습arm의모든k 예측은
target-only와같았다. Low-kQM−Q/Q2/SHAM은0pp,고정예측의pairedbootstrap95%CI도[0,0]pp.
이는metadata효과의모든가능성을좁은구간으로배제했다는의미가아니다.

모든source fold/arm은80%에못미쳐k5 fallback을선택했다. 실제획득prefix는평균
**21.8542trials**,recording-readyproxy203.2615초,80%run도달률41.6667%로동일했다.
**Q 대비 보정trial 절감0%**. 고정query까지의gap과sensor설치시간을빼고전체online시간이
절감됐다고해석하지않는다. Sourceoffline자료비용도targetcalibration비용과구분한다.

균일13expert차용은k1/2/3/5각각66.0425/66.7022/67.0784/67.3523%다.
이표의평균정확도만으로개별episode에서차용이전혀도움없다고결론내리지않는다.
기존hash고정동일query의zeroCCA63.0274%는기술적anchor이며M효능대조의대체물이아니다.

## 왜 metadata가 최종 답을 못 바꿨는가

저장된outer평가768episode×arm 가중치에서self는float64표현상모두1.0이다.
다른source12개의합은최대1.02279e−28,개별최대8.26393e−29였다.
36개QMfit의source에서M만바꾸면centeredlogit은약0.951~1.652만큼바뀌지만,
gate변화는최대2.40603e−27이고class margin변화는전부0이다.
즉M입력선이끊긴것은아니지만 **최종전달경로가실질적으로작동하지않는다**.
사전기준gate·margin>1e−12를36fits전부에서만족하지못했다.

최종NLL이초기보다높은32/144fits,최대악화0.02104도함께보존한다.
이32개는모두innerfits다. 개별sourceweight가0으로underflow된경우는없으며아주작은
양수가최종확률혼합에흡수된것이다. 독립재구성에서도outer혼합확률과self단독확률의차이는0.
QM의표준화M계수는비영값이었다. 최종selflogit우위의블록별중앙기여는base5=1.916,
pair-Q59=12.028,self×target59=60.273,M2=.218이었다. 자기정보상호작용블록의큰기여는
최종파라미터에서확인되지만이분해만으로초기학습과정이나인과원인을입증하지않는다.
다른112fits의NLL감소가최적해도달을증명하지는않는다. 모든균일차용이최선이거나
나쁘다고단정하지않으며,최종학습기와초기모델의loss비교는분류정확도와구분한다.

## 검사·근거·한계

Generated adapter7testsPASS,full-path integration2testsPASS(144mockfits×2updates),
추가atomic/CCAidentity/pin경계3testsPASS. Adapter1/3·runner2/3suitecalls,
fixture165312elements/1322496B. 실제합성capacity재실행0. 전체repositorysuite는실행하지않았다.
Mean support scaling/500sample literalprojection/selfcompatibility/불균형loss/은행분리/
policy≥.80/정책전역동결/head동결/JSON원자발행을검사했다. 테스트범위를독립효능반복으로세지않는다.

[실행관측](reports/mobilebci_source_borrowing_human_v1_observation.json),
[fit/evaluation journal](reports/mobilebci_source_borrowing_human_v1_fits.jsonl),
[source정책](reports/mobilebci_source_borrowing_human_v1_choices.json),
[검사기록](reports/mobilebci_source_borrowing_human_v1_tests.json),
[실행계약](mobilebci_source_borrowing_human_v1_contract.md)을보존한다.
읽기전용 [프로토콜 독립 감사](reports/mobilebci_source_borrowing_human_v1_protocol_audit.json)는
outer1200/inner1536개run예측과5376개role기록을검산해불일치0이었다. 원metadata의비용만
추가참조해절대prefix/ready도확인했다. Numeric본문0.306초,원cache/raw/추가fit0.
정책잠금은journal289행,첫outer시작290행,첫outer평가298행이다.
Feature/W/router 수학검산은별도model audit 범위로구분한다.

[모델 독립 감사](reports/mobilebci_source_borrowing_human_v1_model_audit.json)도PASS다.
원helper를import하지않은NumPy로144모델/46080trial예측/5376role기록과source·evaluation
각36개QM개입을재구성했다. Scaler최대오차1.42e−13,NLL오차2.22e−16,
저장W의normal-equation상대잔차9.63e−17이었다. 새ridge풀이/optimizer/원cache/raw읽기0.
감사식의초기einsum오류와후속구문오류를수정했으며이는사람실험재시도가아니다.
완료계산.8024초,이전실패계산약.03초포함총수치계산1초미만이었다.
Six producer artifacts는합7502313B로128MiB한도내이며감사보고서는별도소량이다.

## 이번 결정과 다음 검증의 조건

등록한실용기준중low-k2pp추가이득/Q2우위/SHAM우위/정책80%/비용10%절감을실패했다.
M-specific실효작동도실패해이144-fit후보를종료한다. 재시도·학습률탐색·M특징교체·유리한
속도/사람만고른재분석은실행하지않았다. **실제부정결과보존과M정보자체에대한판단유보를구분**한다.

다음실제효능후보를늘리기전에해결할것은router의self선택포화·최적화다. 필요하다면
사람outcome을추가읽지않는유한수치검증을별도설계해,알려진최적해가있는경우목적함수를
내릴수있는지와M의유효한최종전달이유지되는지먼저시험해야한다. 이는아직실행하지않은
후속조건이며현재실험을되살리는허가가아니다. 좋은결과가나올때까지사람자료를반복하지않는다.

ChoiPARKED,held60미개봉,source39종료,이전336-fit/합성결과보존. 새외부연락·유료·GPU0.
`academic-research`는이결과를조건부측정근거로기록하고,문헌의source차용기작과실제M효능을
분리하는데사용했다. 추가문헌검색/PDF0,전체landscapecutoff2026-09-04는불변이다.
