# Source39 channel-margin probe v1 — 결과와 후속 판단

2026-09-10 KST. **이번 고정 probe에서 metadata의 추가 예측력은 확인되지 않았다.**
`NO_PROMISING_SIGNAL_IN_THIS_PROBE`, 독립 감사 `PASS`로 단회 종료했다.
원 저보정 SSVEP 목표와 모든 이전 결과를 유지한다. 이 결과는 새 분류 정확도나 보정량
감소를 측정한 것이 아니며, 모든 metadata에 정보가 없음을 증명한 것도 아니다.

[사전 설계](source39_channel_margin_probe_v1_design.md),
[새 승인·실행 계약](source39_channel_margin_probe_v1_execution.md),
[상태/수치 요약](reports/source39_channel_margin_probe_v1_execution_state.json).

## 쉽게 설명하면

앞 3회 EEG에서 만든 정답/오답 예시 파형으로, 별도 source block에서 여덟 센서 중 어떤
센서가 정답 구별에 유리한지 계산했다. 이를 새로운 사람에게 예측할 때, EEG 품질 특징에
실제 임피던스 두 특징을 더해도 더 잘 예측하지 못했다. **Metadata를 넣지 않아서 실패한
것이 아니라, 넣어 학습했지만 이 표적에서는 기준 Q를 넘지 못한 것**이다.

‘섞은 metadata보다 조금 좋다’ 또는 ‘추가 Q2 모델보다 좋다’만 선택해 성공이라고 하지 않는다.
가장 중요한 기준인 원 Q보다 좋아져야 한다는 사전 조건을 충족하지 못했다.

## 사전 비교의 전체 결과

39명의 out-of-participant 예측 오차를 균등 평균했다. MSE는 낮을수록 좋다.
각 사람의 interface/band/channel은 반복 측정이지 별도의 독립 사람으로 세지 않는다.

| 모델 | 평균 channel-shape MSE | 의미 |
|---|---:|---|
| Q | 0.0041857876173 | EEG-derived Q 기준 |
| Q2 | 0.0048279569224 | 같은 두 개 추가 특징 수의 비선형 Q 대조 |
| QM | 0.0041940346302 | 실제 임피던스 M2 추가 |
| SHAM | 0.0042192670122 | 다른 사람의 조건·결측 패턴을 맞춘 M 추가 |

| QM의 비교 | 상대 MSE 감소율 | 개선한 outer fold |
|---|---:|---:|
| Q 대비 | **−0.197024%**: 오히려 오차 증가 | 1/3 |
| Q2 대비 | +13.130239% | 2/3 |
| SHAM 대비 | +0.598028% | 2/3 |

위 %는 **회귀 오차의 상대 변화율**이며 SSVEP 정확도의 %p 변화가 아니다.
Q2 자체가 Q보다 나빴으므로 QM의 Q2 대비13.13% 이득을 M의 고유한 추가 정보 증거로 쓰지 않는다.
SHAM 대비0.60%도 사전2% 기준보다 작다. 세 비교 모두2% 개선·각2개 이상 fold 양수·두
interface 각각 QM>Q라는 조건 중 세 항목이 실패했다.

| 조건 | Q MSE | QM MSE | 판단 |
|---|---:|---:|---|
| dry | 0.0064665160619 | 0.0064910360768 | 증가 |
| wet | 0.0019050591726 | 0.0018970331837 | 소폭 감소 |

이 조건별 차이는 기술적 요약이다. Wet만 선택한 새 성공 가설이나 다음 실험 조건으로
자동 승격하지 않는다. Fold별/사람별 전체 loss와 모든 비교는 원 result.json에 보존했다.

## 실행 오류나 무효 대조 때문인가

아니다. 사전 유효성 관문은 모두 통과했다.

- 39명 모두 양 interface의 M 관측 채널>=2 조건을 충족했다.
- SHAM의 raw packet뿐 아니라 **실제 소비하는 M design도 모든 outer 평가 사람의 양
  interface에서 변경**됐다. 각 fold 교체율100%였다.
- Target energy0.0128135152313으로 variation 기준을 넘었고, 세 comparator loss도 headroom 기준을 넘었다.
- 3outer ×4arms ×(3alpha ×3inner +1final)=120ridge fits를 완료했다. 최종12모델은 모두
  inner selection에서 alpha=.01을 택했다. 사전 후보[1,.01,.0001]를 바꾸지 않았다.
- Q 최종 design rank는 모든 fold10, Q2/QM/SHAM은12였다. 비활성 열은[3,4,5,6]
  (logk/availability/order/period)이다. Channel centering으로 상수 방향을 제거하는 설계이며,
  명목16/18계수와 유효rank를 혼동하지 않는다.
- QM의 최종 M2 계수는 fold0 `[0.0004453102,0.0024369108]`, fold1
  `[-0.0002923473,0.0029225591]`, fold2 `[0.0035338728,0.0016022304]`로0이 아니다.
  이는 fit-standardized/채널중심화/restandardized design 계수이며 물리적 인과효과가 아니다.

## 범위·시간·독립 검산

코드62a04716261a1b9fb9bf0782cd277a1fee03795c의439개 code/config/design/test pins를
입력 manifest·preflight와 등록했다. Source39의 nativeN250 지원 blocks0–2와 source block5,
M0–2만 수치 해독했다. 별도234 before-decode 기록,398 journal events,120fit events다.
전체 archive 무결성 byte hashing은 했지만 금지된 NPY 값·rawMAT·미래 numericM는 해독하지 않았다.

UTC17:45:01.385752 추출 시작 →17:45:07.083716 첫 fit 완료 →
17:45:07.472242 모든 모델 동결 이벤트 →17:45:07.495892 평가 완료 →17:45:09.256056 감사 종료.
동일 사람의 block5는 다른 fold의 학습 표적이 될 수 있으며, 각 평가 모델의 학습·scaling·선택에서
그 사람을 제외한 것이다. Block5 전체가 전역 동결까지 미접근이었다고 주장하지 않는다.

Primary6.622945초/peak494616KiB, audit1.607690초/476484KiB; 각 exit0/watchdog없음/잔류group없음.
Supervisor2224295/primary2224343/audit2225382 모두 종료를 확인했다. 실행1/감사1/재시도0이며
primary120fits와 **감사용 독립120ridge 재구성**을 구분한다. Primary30분·audit15분·각RSS4GiB
상한 이내다. Study295112KiB, test snapshot808720KiB로 각각1GiB 상한 이내다.

감사는 저장된 허용 source 배열에서 독립 Pearson 표적/M 통계/scalers/donors/normal-equation
ridge/선택/OOF/coverage/terminal을 재계산했다. 전4검사 PASS, 최대 전체 차이7.51239e−14,
target4.44089e−16, prediction3.49720e−15다. Q15 함수는 공유하며 독립 Q 구현은 아니다.
원 rawMAT 전처리 재현도 아니다. 파일/역할/저널 동결 연결은 별도 wrapper 검증이며 OS 보안
sandbox나 외부 행위의 암호학적 non-access 증명으로 과장하지 않는다.

실험 부모: `/home/whwovy/channel-margin-probe-v1-FNBdZf`.
원 data9파일은0400이고 completion에 해시를 결합했다. Child 로그·종료 영수증도 보존했다.
Supervisor 자신의 빈 stdout/stderr는0664이며 completion inventory는 data파일을 대상으로 한다.

| 기록 | SHA-256 |
|---|---|
| registration.json | `f740b8df2c7c73f99c281fd4d7eaf2026264d95da1f395f01a0c6e1820919ef3` |
| result.json | `93e4db18d87011db112727eff2e3bf6e2d37aa76b4d695f4ab3af382f44f0db7` |
| audit.json | `6a6c2005484228e1c6637fb1bb36bfb8e00ace4ae0a91539f624f755cbc4a5a5` |
| completion.json | `e9f0169168ab89b5858c2c9e1e98bc069128f92ad94a66d5fa9c619fa43fc60c` |
| preflight.json | `aec17d280016d3af10aa12ace27ec07f96373a926b16629b8adfa8ccdfbbc80b` |

## 구현 검증과 보존한 실패

최종 통합·영향받은 회귀·lifecycle486PASS14.39초, 전체 repository suite는 이번에 실행하지
않았다. 이전 root47/110/165와 agent63/46은 겹치는 검사로 독립 반복 효능 검증이 아니다.
등록전 legacy 회귀313PASS/2FAIL6.13초는 caller의 no-bytecode 설정 누락으로 사전 환경 관문이
일찍 거부한 결과였다. 보호코드 수정 없이 올바른 CPU1/no-bytecode 환경의 최종486이 통과했다.
Learner2파일 format 실패도 등록전에 형식만 고친 뒤 재검증했다. 실패 XML/이전 실행은 보존했다.
사람 등록 후 source/test/config 수정·추가 fit·추가 수치 감사0이다.

`coordinate-worktree-changes`로 기존 두 tree를 재사용해 disjoint learner/auditor를 작성하고
root가 입력·실행·통합을 소유했다. 문자충돌0; 추가 read-only runtime검토가 pin 누락/저널 결합/
감사 실패기록/입력오류 분류를 개선했다. Worktree추가/삭제/설치/push/GPU/외부요청/유료0이다.

## 결론의 한계와 다음 분기

이 **현재 임피던스 M2→shared-slope 단일채널 template-margin** 경로는 종료한다.
같은 데이터에서 window·alpha·seed·표적을 바꿔 양성을 찾는 반복은 하지 않는다.
남은 시간 예산이 있어도 사전 단회 attempt 예산을 다 썼으므로 자동 추가 실험은 없다.

원 목표는 바꾸지 않는다. 다만 다음 연구를 정당화하려면 이번에 검사하지 않은 정보를
구체적으로 지목해야 한다. 다음 우선 검토는 독립 paired acquisition에서 접촉 상태의 시간 변화,
움직임/접촉 압력 같은 **EEG-derived Q와 다른 측정 정보가 실제로 기록되어 있는지**다.
이는 필요한 측정에 대한 제안이지 현재 보유·접근 가능성이 확인된 새 dataset 목록이 아니다.
외부 요청/신규 수집/유료는 별도 승인 대상이다.

독립 acquisition 측정이 확보되면 시점·단위·공통 정보·Q baseline·삽입 기작을 먼저 정한 후보1개를
검토하고, 그 후보는 실제 k별 정확도와 0/36/60labels 같은 **보정 비용**까지 평가해야 한다.
새 paired자료 없이 현재 M의 모델 복잡도만 올릴 근거는 이번에서 얻지 못했다.
반면 공간 공분산·interface별 상호작용 등 다른 기작은 이 음성 결과만으로 금지하지 않는다.
별도의 구체 근거와 유한 예산이 필요하다는 뜻이다.

반복 노출 source39/겹치는 CV fit/고정 representation이라는 한계 때문에 일반 조건부 독립,
통계적 동등성, 인과적 무효를 선언하지 않는다. CPI 관련 읽기에서 이미 구분한 대로 SHAM도
정식 conditional sampler가 아니다. **실제 metadata의 저보정 효용은 여전히 미확립, 전체 goal은 미완료다.**

`academic-research`는 기존 근거·가설과 결과의 범위를 연결하는 데 사용했다. 이번 실행에는
신규 검색/PDF0, 기존 cutoff2026-09-04를 유지했다. 새 관측 근거는
`claim:2f1785de12d070cf` QUALIFIED/4evidence로 기록했고, 원
`gap:6a8254f849b36b2f`를 OPEN으로 갱신했다. Root만 연구 DB에 기록했으며 render와
SQLite quick_check를 통과했다. 현재1073papers/83searches/43cards/17techniques/
56claims/136evidence/38gaps/36deep-read items/5analogies다. 목록의 논문 수는 정독 수가
아니며 기존 source429·미확인 문헌 공백을 그대로 유지한다. 이전 N1-R 실제 음성
`claim:23d98ce865902ae8`, 인공 능력 PASS `claim:1c18ea450cdfb82d`, 당시 구현만 완료한
역사적 claim `claim:ffadce94fecf7502`도 보존했다.
