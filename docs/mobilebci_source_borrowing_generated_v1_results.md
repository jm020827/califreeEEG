# Source borrowing 새 기작: 합성 관문 통과, 사람 효능은 미검증

2026-09-11. 원래의 저보정 closed-set SSVEP 목표는 유지한다. 이전336-fit gyro→diagonal
prior 후보는 **CLOSED_NEGATIVE**이며 보정량 절감0 결과를 변경하지 않는다.

## 이번에 실제로 한 일

새 후보는 내 EEG 규제를 바꾸는 대신, 다른 사람의 class별 판독 결과를 얼마나 빌릴지
metadata로 학습한다. Source의 공간 metric 정보를 옮기며 내 판독기를 남겨 둔다.
문헌·과거 구현을 먼저 검토해 최대2기작 중 이 하나를 선택했고, 새 trial-resolved feature를
기존 실패 prior에 넣는 후보는 보류했다. 비공개 자료를 기다리거나 새 사람 자료를 받지 않았다.

선택·검증 순서: 범위계획 `2fa9b14` → generated 계약 `0c0dd18` → 최종 코드/receipt/config
동결 `098df32` → 정확1회 실행. 합성 결과를 보고 seed/DGP/threshold를 고치거나 재학습0.

| 고정 방법/개입 | 생성 평가 정확도 |
|---|---:|
| Q: EEG 기반 source 선택 | 33.33% |
| Q2: EEG 보조입력 추가, QM과 같은 용량 | 33.33% |
| QM: 올바른 metadata로 source 선택 | 100.00% |
| SHAM: 균형 있게 엇갈린 metadata로 별도 학습 | 33.33% |
| 균일 차용 / 내 판독기만 | 각각33.33% |
| 학습된 QM 고정, metadata 대응만 엇갈림 | 50.00% |
| QM 고정, source W의 class 대응만 순환 | 66.67% |

각 평가에는8generated episodes×6queries=48개가 쓰였다. 학습8episodes와 query noise를
분리했지만 공통 source bank·고정 생성 구조를 공유한다. 독립 사람48trial 실험이 아니다.
목표 support는1/class이며 두 채널 그룹을 구분하지 못하도록 만들고, M은 맞는 그룹을
알려주도록 설계했다. **M가 원래부터 정답에 필요한 정보를 가지는 인공 능력 검사**다.
100%라는 숫자를 실제 gyro 성능이나 예상 효과크기로 쓰면 안 된다.

## 작동과 검증

- 4router fits/400optimizer updates, 별도9class-ridge solves,0.479128초CPU1.
  Attempted/completed 모두4/4와9/9. 실패·재시도0.
- M만 바꿨을 때 weight 최대변화0.9730702, 상대 class margin 최대변화1.8824455.
  k1에서도 입력이 실제 선택과 예측에 도달한다.
- Source 순서 재배열 오차3.33e−16, reference 우측 직교회전 오차1.33e−15, energy floor0.
  Rotation/scale은 표현 불변성이지 실제 모든 onset 변동에 대한 보장이 아니다.
- Unit suite2/3호출: 첫8PASS, 회계 보강 뒤9PASS. 반복검사를17개 독립 실험으로 세지 않는다.
  Partial-update 계수 보존·nonfinite/negative mixture·self/heldout bank 거부·exact no-transfer
  등은 SHA고정 unit receipt를 최종 gate에 연결했다. 전체 repository test suite는 실행하지 않았다.
- 읽기전용 reviewer가 저장된 waveform/W/router를 NumPy 시간영역 QR로 독립 검산했다.
  57항목 불일치0, 최대오차1.02435e−12. 원 scorer·optimizer·ridge fit·DGP 재실행0.
  Optimizer 경로·ridge fitting 자체를 독립 재현했다고 주장하지 않는다.
- 생성 fixture101504numeric elements/812032numeric bytes, 압축766808B.
  Fixture+result+journal848959B, config/receipt 포함851297B.8MiB상한 내다.

[관측](reports/mobilebci_source_borrowing_generated_v1_observation.json),
[독립 감사](reports/mobilebci_source_borrowing_generated_v1_audit.json),
[단위검사](reports/mobilebci_source_borrowing_v1_unit_tests.json),
[실행 계약](mobilebci_source_borrowing_generated_v1.md)을 보존한다.

## 관련 연구를 다시 읽어 바꾼 판단

[SS-iTRCA](https://arxiv.org/abs/2506.10933)는 source TRC와 target EEG의 상관으로 source를
선택하고, 여러 사람의 정보와 target TRCA를 결합한다. 기존 v1 PDF의 pp3–5/eqs6–21을
이번에 표적 재독하고 pp4–5의 식·그림을 직접 확인했다. Source selection은 이미 있는 방법이다.
모든 source 유사도가 trigger γ=.5 이하일 때 모두 빌리는 규칙을
“no-transfer 안전장치”로 오해하면 안 된다.
Target TRCA 경로도 이 실험의k1 reference-ridge와 다르다. 그 논문은 외부 gyro M가 Q를
넘는다고 입증한 연구가 아니다. 새 external-M router는 **우리의 transfer 가설**이다.

[CSDuDoFN](https://arxiv.org/abs/2311.07932), [SSVEP-DAN](https://arxiv.org/abs/2311.12666),
[cross-domain LST](https://doi.org/10.1088/1741-2552/abcb6e)의 기존 card와 공개 초록도
이웃 기작으로 재검토했다. Source 모델 조합·파형 정렬·target-label 기반 전달을
외부 acquisition-M 효과로 혼동하지 않는다. 추가 후보 T-ASS/DC-TRCA-net
([PubMed](https://pubmed.ncbi.nlm.nih.gov/41360014/))는 공개 검색 초록에서 source-selection
기작을 확인한 수준이다. 원문 열람은 PubMed empty/IOP robots 제한으로 실패했으며,
그 연구의 정확한 label 권한·fulltext 성능은 이번에 검증하지 않았다. 비공개 원문을 기다리지 않았다.

`academic-research`가 작업에 미친 영향은 “더 많은 논문 수”가 아니라, **차용 자체를 신규성으로
주장하지 않고 EEG-only source router를 필수 대조로 잡은 것**이다. Root structured search0/3,
표적 PDF재독1/2; scout2queries/최대4primaryHTMLopens, 새PDFdownload0. 기존 전체 landscape
cutoff2026-09-04를 최신 전수조사라고 바꾸지 않았다. 좁은 검색에서 직접 M 효과 논문을
찾지 못했다는 것은 그런 논문이 전혀 없다는 증명이 아니다.

## 다음 실제 실험은 무엇이 다른가

[사람 검증 초안](mobilebci_source_borrowing_human_v1_draft.md)을 작성했다. 아직 실행용
reader/trainer/manifest가 완성된 계약이 아니며 이번에 실제 cache/raw 접근·human fits0이다.

초안은 기존16개발 참가자·동일 support/query/cost로,123차원 EEG/common 대조군과
동일 용량 Q2/QM/SHAM을 비교한다. Target Q/common을 복사만 하면 shared-linear softmax에서
소거되므로 pairwise 차이와 self interaction으로 바꿨다.4source사람×3조건+self=13experts를
모든 단계에 맞추고 recipient 전 조건을 bank에서 제외한다. SHAM은 target M를 유지하고
source M만 속도 내 순환하여 metadata 거리 벡터 집합을 보존한다.

제안 예산은144router fits/14400updates,576unique class ridge solves,전체600초다.
2pp추가 이득·80%정확도·10%실제 prefix절감 등 기존 실용 기준을 완화하지 않는다.
통과해도 개발자료 후보일 뿐이며 독립 사람 확인 계획이 별도로 필요하다.

## 경계와 최종 판정

이번 단계 판정은 **GENERATED_MECHANISM_PASS / RETAIN_FOR_HUMAN_CONTRACT_ONLY**다.
실제 M 효과·보정 절감·원 연구목표 달성은 미확립이다. 합성 성공만으로 새로운 유망 인간
효능 후보를 확보했다고 보고하지 않는다. 이전336-fit·source39 부정 결과와 실패는 보존한다.
ChoiPARKED/held60미개봉/외부연락0/유료0/GPU0. Root만 main/SQLite를 작성하고
3agents는 읽기전용으로 검토하여 새 worktree·데이터 복제 없이 수행했다.

산출물 SHA256:

```text
result  2c93f071fd36c5591b1fc90de211ac46b365db03e3ad225f33d8ba33bcde5f34
fixture 2fb1e4e8f49690fcb9bd1d926838a5f426aa5c34a18b469825441d43a55ccd89
journal 5001b80354e641dbd223ff7233d62ecb084048bb6650dc2c4d6bfbfd3d21840b
config  d7519b2a19c27f4bf628cdc7a2cca296890768559c686abebbe8e086b0f05414
```
