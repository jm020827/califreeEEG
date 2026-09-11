# 다음 실제 검증 초안: metadata로 다른 사람의 EEG 정보를 빌릴지 학습

2026-09-11. **DRAFT_NOT_EXECUTABLE**. 합성 관문 다음의 설계이며 이 문서 작성 중
사람 cache/raw를 새로 읽거나 실제 학습하지 않았다. 아래 과학설정을 최종 검토하고
role-aware runner·검사·고정 config/hash를 갖춰 실행 계약으로 전환하기 전에는 실행하지 않는다.
기존336-fit 후보 종료를 취소하는 문서가 아니다.

## 쉬운 설명과 반증할 가설

이전 방식은 “머리가 움직인 정도를 보고 내 EEG 학습 규제를 조금 바꾸자”였다.
이번은 “내 보정 EEG만으로 애매할 때, 이미 학습한 다른 사람들의 판독기 중
지금 취득 상태가 맞는 쪽의 판단을 더 빌리자”다. 항상 내 판독기도 남겨 둔다.

차용·source selection 자체는 기존 연구다. 검증할 부분은 **같은 EEG 품질·속도·순서
정보까지 쓰는 선택기보다, 실제 head-gyro metadata를 더 받은 선택기가 새 target의
정확도와 보정 trial 비용에서 더 좋은가**이다. 이 코호트의16명은 이미 본 개발 자료라
성공해도 독립 검증이라고 하지 않는다. 실제 gyro2가 source 간 유용성을 설명한다는
근거는 아직 없고, 합성 성공은 이 가설의 생리학적 증명이 아니다.

## 자료와 비용을 바꾸지 않는다

- 사람: s01,s03–s17, 세 속도, 총48run. 기존 배제 s02를 결과에 따라 복귀시키지 않는다.
- 기존 cache만 사용: `/home/whwovy/eeg-data/mobilebci-features-v1-xm26hN/features.npz`,
  SHA256 `6e93b4b1a16d69c64938cee283e79cfbbc3f8683ec35ec732d71fb2e14b99dfb`.
  Schema/행 대응은 기존 extraction manifest와 대조한다. 새 raw decode/추가 자료0.
- k=1/2/3/5의 같은 support, query40..59, 같은500Hz/1초/9채널/3harmonics.
  기존 marker 매핑·필터·M/Q/common 정의를 고정한다. 새 window/feature/shift 탐색0.
- M은 유료인 해당 support 구간의 head gyro2다. Support 이전에 공짜로 아는 M이 아니다.
  Query IMU/labels/sequence index를 선택기에 넣지 않는다. Source supervised labels는
  학습에 쓸 수 있지만 target query labels는 평가만 한다.
- Target 비용은3k가 아닌 full chronological prefix trial 수. Ready-time은 보조 proxy이며
  preroll과 support–query gap 한계를 유지한다. 기존 online 시간 절감으로 부르지 않는다.
  Source의5/class 자료 비용은 offline 비용으로 별도 명시한다.

## 판독기와 source bank

각 source run의5/class로 class별 reference-ridge W를 만든다. Target은 해당 k만 사용한다.
Identity/covariance-scale ridge λ=.1, Gram-corrected projection score와 softmax(10×score)를
모든 arm에 공통 사용한다. λ·temperature 탐색0. W나 파형을 사람 사이에서 평균하지 않고
class probability만 convex mixture한다. Transfer 대상은 W Wᵀ 공간 metric이며
실제 모든 latency 변화에 불변이라는 뜻은 아니다.

Source eligibility는 **사람 단위**다. Source training episode의 recipient 자신은 세 속도
전부 bank와 M-donor pool에서 제외한다. Inner-validation·outer-test 사람 역시 해당
training pool에 들어가지 않는다. Target 자신의 판독기는 명시적 self expert 하나뿐이다.

매 episode에서 eligible training pool 중 정확4사람×3속도=12source experts와 self1개를
쓴다. 참가자 sorted-position outer3/inner2 분할에서 최소4eligible sources가 가능한지
생성 ID 검사로 확인한다. Source4명은 `SHA256("mobilebci-source-borrowing-v1|" +
recipient_id + "|" + candidate_id)` 오름차순, 충돌은 ID 사전순으로 고른다.
IDs는 분할/고정 추출 용도뿐이며 feature가 아니다. 같은 recipient는 k/arm 간 같은 bank.
EEG/M/성능으로 bank 사전선택0. 모든 단계의13experts를 맞춰 source-training/신규-target
간 bank 크기 차이를 제거한다. 데이터 일부만 빌리는 고정 bank-size 실험이라는 한계를 보고한다.

## 실제로 작동하는 Q 대조군

Shared linear router에 target Q/common을 모든 expert 행에 복사하면 공통 logit으로
소거된다. 따라서 “rich Q를 넣었다”는 선언만으로 공정한 대조가 되지 않는다.
다음처럼 expert별로 달라지는 입력을 고정한다.

| 입력 | 차원 | 용도 |
|---|---:|---|
| self indicator, own-class support compatibility 평균/표준편차/최솟값, normalized covariance distance | 5 | 실제 EEG transfer 관련 정보 |
| absolute target–expert 차이: 기존 Q54+common5 | 59 | EEG 품질·속도·순서·k·prefix·ready 차이, self는0 |
| self indicator × target(Q54+common5) | 59 | target 상태에 따라 차용 총량 대 self 판단을 바꿀 수 있게 함 |

Q는 위123차원이다. Common의 source 값은 expert를 만든 k5 상태, target은 해당 k다.
Compatibility는 cache의 own-class cross로 계산한 coherent pooled projection이며
평균/표준편차/최솟값은 세 class의 pooled score 사이 요약이다. 관측하지 않은
per-trial compatibility 분산으로 부르지 않는다. 이는
full multiclass support 정확도가 아니다. Cache에 없는 wrong-class cross나 target query
정답으로 competence를 만들어내지 않는다.

Q2/QM/SHAM은 같은 기본123차원+aux2의125차원, 동일 shared-linear 용량이다.
Q2 aux는 기존 EEG-derived q2 두 값의 absolute target–expert 차이,
QM aux는 실제 M2의 차이, SHAM은 아래 재배정한 M2의 차이다. Self aux는 모두0.
모든 arm은 같은13expert score·학습 labels·k·λ·100steps를 받는다.
임의의 모든 Q 함수에 대한 조건부 독립 검정이 아니라 이 강한 고정 대조군과의 비교다.

SHAM은 target의 실제 Mk를 유지한 채, 선택된4source IDs를 사전순으로 한 칸 cyclic
이동하여 같은 speed의 source M5만 재배정한다. Source EEG/Q/common은 제자리에 둔다.
이렇게 하면 각 target/speed의 source별 pairwise M2 벡터 **집합이 정확히 보존**되며
어떤 source EEG에 그 metadata가 연결되는지만 깨진다. Self aux는 그대로0이다.
Target와 source M를 작은 pool에서 각각 대체할 때 생길 수 있는 인공적인 donor 충돌·
zero-distance 증가를 피하기 위한 실행 전 설계 수리다. 추출된 사람 수치에 따른 수정은 아니다.

Donor는 해당 bank의 training 사람만 사용한다. Target M가 남아 있으므로 모든 metadata
정보를 없앤 대조가 아니라 **source EEG–metadata 대응의 추가 가치**에 대한 보수적 대조다.
완전한 조건부 독립이나 무작위 permutation 검정이라고 하지 않는다. Donor ID·fixed-point0·
실제 aux 변화·source별 대응·within-speed multiset 동일성을 보고한다. 재배정 전후 값이
같아 contrast가 소실되면 효능 판정을 보류하고 mapping을 사후 수정하지 않는다.
Multiset은 각 열이 아니라 두 값의 벡터쌍 전체로 확인한다. Fixed-point0은 실제 값 변화와
구분한다. Q/source 특징으로 고정 순환을 일부 복원할 수 있어 chance baseline을 가정하지
않는다. 여기서 보수적이라는 표현은 M 정보를 모두 제거하지 않는다는 뜻이지 통계적 보수성
보장이 아니다.

## 학습·예산·누수 차단

이전과 같은 outer3/inner2 participant folds, k4개. λ를 고정하므로 grid search를 없앤다.
각 split/k에서 Q,Q2,QM,SHAM 각각100 full-batch AdamW steps, lr=.05,weight_decay=.01,
zero init,float64 CPU1. Source training query의 class-balanced NLL을 run별 계산 후
사람·세 속도에 동일 가중. Standardizer도 source episode에 동일 가중, source-only이며
한번 정한 mean/std를 validation/test에 그대로 적용한다. 상수 std floor1e−12.
Query 특징/label은 router 입력이 아니며 expert mixture weight는 run/k 내 query 공통이다.

- Inner:3outer×2inner×4k×4arms=96router fits.
- Outer-source refit:3×4×4=48fits. 총144fits/14,400updates, 예산 증가·seed 반복·restart0.
- 전체48run×4k×3class=576unique ridge class solves 상한. 같은 own-support W 재사용을
  허용하되 어떤 source bank에도 excluded person의 W가 들어가지 않음을 별도 검사한다.
- 실제 실행1회, CPU1/전체 wall600초, 신규 artifact128MiB. 이 시간에는 expert/query
  score 계산·학습·bootstrap·직렬화까지 포함한다. Cache checksum1회와
  archive numeric load1회; raw decode·외부다운로드·GPU·설치·외부연락·유료0.
- 역할 adapter+trainer generated suite 각각≤3호출, fixture≤200000elements/2MiB,
  실제 capacity experiment 재실행0. 단위검사부터 별도 실행 manifest로 수량을 고정한다.
- Fit/class solve attempted/completed와 매 update를 기록한다. 모든 source-only policy를
  저장·hash고정한 뒤 최초 outer outcome을 계산한다. 첫 예상 밖 오류는 중단·보존하며
  자동 재시도0. 결과 재조정/특정 source·속도·참가자 사후 제외0.

## 무엇을 보면 유지하거나 끝낼 것인가

기존336-fit에서 달성 못한 실용 기준을 약화하지 않는다. Primary는 사람별 세 속도와
k1/2 평균 balanced accuracy, QM−Q/Q2/SHAM. Participant bootstrap2000,seed20260911의
고정 cross-fit prediction 조건부95%구간을 보고한다. 재학습/후보 탐색 전체의 불확실성을
포함한 독립 superiority 검정으로 해석하지 않는다.

모든 outer 결과를 보기 전에 각 arm은 inner OOF 평균80%를 처음 넘는 가장 작은 k를
선택하고, 없으면 k5 fallback+source_target_unmet를 기록한다. Outer에서 그 선택을
고정해 정확도,80%reach,실제 prefix trial과 ready proxy를 평가한다. Query 보고 선택한
first-hit 비용은 실용 stopping policy로 인정하지 않는다.

유지하려면 모두 필요: low-k QM−Q≥2pp, QM−Q2와QM−SHAM>0; QM−Q95%하한>−2pp;
정책 QM 정확도≥80%와Q대비 harm guard를 만족하며 prefix 비용≥10%감소·reach비열화;
k3/5 QM−Q≥−1pp;16명 중low-k>5pp 손실≤3명; 유효한SHAM과M-only 작동 진단.
M-only 진단은 frozen QM에 정해진SHAM을 넣어 source/outer의 실제 pairwise aux,
expert-centered logits·gate·score 변화를 측정하며
재학습이나 다음feature선택에 이용하지 않는다.

Target-only/균일13expert mixture는 추가 fit 없는 기술적 대조다. 기존 zero-CCA는
이미 저장된 일치 query 결과를 참조하는 descriptive anchor로 구분한다.
학습된 Q가 고정 λ의target-only보다 못한 경우를 숨기지 않는다.

모든 기준 통과 시에도 **개발자료상 후속 독립 확인 후보**일 뿐이다. 미통과면 해당144-fit
후보를 종료하고 장점만 골라 새 이름으로 재실행하지 않는다. 유망 후보의 확인에는 새로운
사람/취득 맥락과 사전 계획이 필요하며 held60·외부 요청·유료는 별도 승인이다.
