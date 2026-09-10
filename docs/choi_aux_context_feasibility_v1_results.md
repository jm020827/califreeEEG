# Choi 보조센서: 저보정 metadata 후보의 사용 가능성 검토

2026-09-10. **검토 완료 / 후보 활성화 보류 (`DEFER_FOR_MISSING_MEASUREMENT_EVIDENCE`).**

연구목표는 그대로다. **새 사용자가 적은 SSVEP 보정 예시로 쓸 만한 정확도에 도달하도록,
EEG에서 계산한 품질 Q 외에 측정환경 정보 M이 실제로 도움이 되는지** 묻는다.
이번에는 새 정확도 실험을 하지 않고, 이미 보유한 Choi2019 자료의 움직임·온도 채널이
그 질문에 쓸 수 있는 측정인지 확인했다. 기존 유한 연구 루프의 완료와 부정 결과는 바뀌지 않는다.

## 쉽게 말하면

기존 후보는 주로 전극 접촉 상태를 활용했다. 다음 가능성은
**“보정 예시를 얻기 직전 머리가 움직였다는 정보로, 어떤 예시를 얼마나 믿고 학습할지 정할 수 있을까?”**다.
움직임 센서는 EEG 밖에서 얻은 측정이므로 acquisition context 후보가 될 수 있다.
하지만 움직였다는 사실이 EEG 품질 Q보다 더 유용한지는 아직 모른다.
머리 움직임은 전극 접촉 불량 자체의 측정도 아니다.

이번에 확인한 것은 **논문에 실제 머리 움직임 센서의 부착이 보고되어 있다**는 점이다.
아직 확인하지 못한 것은 **배포 파일의 어느 값이 어떤 물리량이고, 보정/예측 전에 안전하게 사용 가능한지**다.
따라서 “좋은 후보를 찾았다”가 아니라 “확인할 가치가 있는 측정은 있지만 지금 바로 효능 실험에 넣지는 않는다”가 결론이다.

## 근거와 남은 공백

| 항목 | 확인한 근거 | 판단과 한계 |
|---|---|---|
| 실제 센서 | 논문 Methods는 Cz–CPz 사이 IMU 부착과 EEG/보조신호 동시 기록을 보고 | 단순 채널 이름보다 강한 근거. 원시 `Gyro`와 논문 trace의 정확한 대응은 미확정 |
| 물리량·단위 | Figure 8의 head movement는 g. 로컬 두 cnt의 공통 `yUnit`은 `uV` | 각속도라고 부르지 않는다. 공통 export 필드일 수 있어 논문/파일 어느 쪽이 오류인지 단정하지 않는다. 모델·축·합성 방식·변환계수 미확인 |
| 온도 | README와 DataCite는 body temperature를 언급하고 두 헤더에 `Temperature` 존재 | 논문 visible text 검색/Methods/Fig.8에서 확인 못함. 연결·부위·단위·변동 미확인이지, 미측정/placeholder 확정이 아님 |
| 시간 정렬 | 논문은 같은 증폭기, 1,000 Hz 동기 취득과 200 Hz 배포 변환을 보고. 두 헤더 `fs=200` | 원 다운샘플 필터의 미래 참조 범위·위상, marker의 정확한 사건 시점은 미확인 |
| 현재 EEG 전처리 | 전체 run을 `filtfilt`/`sosfiltfilt` 처리한 뒤 trial crop | 엄격한 사전 정보만 쓰는 새 계약에 그대로 재사용할 수 없음. 옛 offline 실험 전체가 무효라는 뜻은 아님 |
| 실제 값·pairing | S1/Day1/LOW의 두 cnt 파일에 33 EEG+6 auxiliary 채널 존재 | 파형·marker 값은 읽지 않았으므로 변동, 포화/상수, trial 정렬, 다른 참가자 동일성은 미검증 |
| 이용 조건 | DataCite 등록과 기존 README는 CC0 | 공개 재사용 조건은 확인했지만 기존 outcome 금지와 별도 실행 범위를 대체하지 않음 |

주 근거: [Choi 데이터 논문, Methods·data format·Figure 8](https://academic.oup.com/gigascience/article/8/11/giz133/5641733),
[공식 DataCite 등록](https://api.datacite.org/dois/10.5524/100660).
Figure 8을 직접 표시해 다섯 패널과 head movement의 g 표기를 확인했다. 그림의 곡선에서 새 수치를 추출하지 않았다.
PDF 정독 완료라고 부르지 않으며, 공식 HTML의 선택 구간 검토다.

**절대 단위를 모른다는 이유만으로 모든 움직임 학습을 배제하지 않는다.**
연결·채널 의미·시간·유효 변동을 확인하면, 절대 g가 필요 없는 변동성 특징도 별도 설계할 수 있다.
그러나 정규화는 잘못된 채널 대응이나 미래 정보 유입을 해결하지 않는다.
현재 고정한 의미/단위 관문을 결과 확인 후 완화해 PASS로 바꾸지는 않는다.

## 인접 연구가 말해주는 것과 말해주지 않는 것

[Mihajlovic 등, EMBC 2014](https://pubmed.ncbi.nlm.nih.gov/25571131/)의 저자 초록은
머리 움직임에 의한 EEG 변화가 접촉 임피던스만으로 전부 설명되지 않는다고 보고하고,
adaptive filtering을 통한 artifact 감소를 다룬다.
이것은 **움직임과 임피던스가 같지 않을 수 있다**는 인접 근거다.
초록만 확인했으므로 정확한 입력 센서·필터 구조·효과크기는 확정하지 않는다.
검색어에 accelerometer가 있었다고 그 논문의 학습 입력이 accelerometer였다고 쓰지 않는다.

또한 **같은 시점 EEG의 잡음을 제거하는 것**과
**사전 측정으로 적은 보정 예시의 학습을 개선하는 것**은 다른 실험이다.
이 논문이나 Choi 자료 논문이 Q 대비 M의 추가 효용·실제 보정량 감소를 입증했다고 해석하지 않는다.
ECG/호흡/체온을 자동으로 더해 일반 생리상태 예측 연구로 옮기지도 않는다.

## 다음 설계에서 바로잡아야 할 두 가지

### 1. 사용 시점과 전처리 경계

현재 `prepare_choi2019.py`는 run 전체를 필터링하고 marker 이후 창을 자른다.
새로운 strict-prequery 실험에는 별도 입력 경로가 필요하다.
전체 run 미래 부분을 바꾸었을 때 이미 허용된 시점의 특징/학습 결과가 바뀌지 않는지,
인공 신호로 먼저 검증해야 한다. 단방향 필터라도 상태에 금지된 trial을 읽어 넣으면 안 된다.
배포 전 1,000→200 Hz 변환이 이미 미래를 섞었다면, 로컬 필터만 바꿔 해결했다고 주장할 수 없다.
원 변환 코드를 확인하거나 증거에 근거한 안전 여유 구간이 필요하다.

처음 trial의 사전 구간 부족, run 시작, timestamp 단위, 반올림,
marker가 cue 시작인지 자극 시작인지도 fail-closed 규칙으로 명세해야 한다.
현재 post-stimulus crop 범위 검사는 pre-stimulus 창을 보증하지 않는다.

### 2. 선택한 label 수와 실제 수집 부담

기존 Choi pseudo-block은 각 class의 j번째 trial을 하나씩 묶는다.
실제 시간상 연속한 네 trial이라는 뜻이 아니다.
예를 들어 순서가 A,A,B,C,D이면 네 class에서 한 개씩 골라도 이미 다섯 trial을 실시했다.

따라서 다음에는 `선택된 4k labels`, `그 시점까지 실시한 전체 trials/labels`,
`경과시간`, `센서 설치·추가 baseline 비용`을 따로 기록한다.
파일에 없는 설치시간은 unknown이지 0이 아니다. 세 band는 별도로 수행한 **4-class task 세 개**이고
동시 12-class task가 아니므로 band별 예시 수/시간을 섞어 저보정이라고 부르지 않는다.
Query를 보고 보정량을 고르는 것은 사후 cost curve이지 실제 온라인 중단 정책이 아니다.

## 조건을 충족했을 때의 후보 하나 — 아직 실행 계약 아님

제안 이름: **pre-support motion-conditioned learning**.

- 기작 가설: 예시 직전 움직임 상태가 바로 뒤 보정 EEG의 신뢰성에 지속적으로 영향을 주고,
  그 영향 일부가 support EEG-derived Q에 남지 않아 학습 가중치에 추가 정보가 된다.
  이 지속성·추가 정보가 없으면 후보를 종료한다.
- 삽입 위치: source 참가자에서 학습한 작은 motion 잔차로 support 예시의 학습 가중치를 조정한다.
  예측 중 query 움직임으로 EEG를 복원하는 방법은 이 후보에 포함하지 않는다.
- Q와 QM은 같은 EEG, label, 공통 획득/자극 정보, 전처리, 시간 제한, 선택 절차를 사용한다.
  비슷한 추가 학습 용량을 Q에만 준 Q2, 사전 고정한 pairing-sham, M missing fallback을 함께 둔다.
- 각 class의 가중치 합을 Q/QM 공통 상수(예: 1)로 고정하고, 공통 source EEG/Q 모델·초기상태·규제도
  같게 유지하는 버전을 우선 제안한다. M은 이 정규화된 예시 가중치에만 작용한다.
  이 조건의 버전은 **class당 예시 1개면
  가중치를 바꿀 여지가 없어 k=1의 M 효과를 주장할 수 없다.** k=0/1은 동일성/anchor 검사이고,
  실질 가설은 k=3/5다. One-shot 효과를 원하면 별도 기작이 필요하다.
- 최소 분할 제안: Day1의 band별 session01에서 support를 얻고 session02의 고정 40 query에 평가한다.
  참가자 단위 outer 분리와 source-only 선택을 명세한다. 두 session 번호의 실제 chronology는
  marker/취득 문서로 다시 확인해야 한다. **Day2의 기존 80 query/band를 재사용하지 않는다.**
- 이 분할은 기존 Day1→Day2와 다른 새 개발 설계다. Day2도 같은 사람들의 자료이므로
  새 참가자 독립 확증이라고 부를 수 없다. 후속 held60 개봉과도 별개다.
- 성공은 움직임과 Q의 상관, 가중치 변화, 평균 정확도 한 번의 상승이 아니다.
  같은 k의 Q/Q2/SHAM 대비 실용적 이득과 더 적은 **실제 수집 비용**으로 목표 성능 도달을
  참가자 단위 불확실성·악화 사례와 함께 평가해야 한다.

정확한 창 길이·특징·학습식·후보 수·참가자 수·효과 기준·학습/개봉 예산은 아직 동결하지 않았다.
측정 문제가 풀리기 전에 임의로 정하고 실행하지 않는다. 기존 V2 terminal deny는 그대로이며
새 분할을 제안했다는 이유로 우회하지 않는다.

## 지금의 다음 일과 중단 기준

1. [미발송 확인 질문](choi_aux_context_clarification_draft.md)을 준비했다.
   센서 전체 자료를 추가 수집하자는 요청이 아니라, 이미 공개된 채널 의미와 변환/marker 문서를
   확인하는 최소 질문이다. 외부 발송은 별도 승인 전 하지 않는다.
2. 사람 파형을 읽기 전, 위 시간경계·paired join·수집비용 계산의 **인공자료 전용 구현/검증**을
   별도 유한 범위로 설계할 수 있다. 이것이 센서 근거를 대체하지는 않는다.
3. 연결/시간/사전 창 근거가 끝내 없으면 이 strict-prequery 경로는 계속 보류한다.
   효과가 나올 때까지 같은 결과를 열거나 설정을 바꾸지 않는다. 반대 증거가 나오면 해당 기작을 종료한다.
4. 단위만 미해결이고 나머지가 확인되면 unit-agnostic 후보를 별도로 사전 등록할 수 있다.
   이번 보류를 소급 PASS로 바꾸거나 과거 후보를 재개하지 않는다.

## 실행 범위·검색·보존 기록

사전 계약: [review config](../configs/analysis/choi_aux_context_feasibility_v1.json),
동결 commit `eed75afb1ad3837b16da329a2f6b2bc91cae6fed`.
SHA-256 `77db5c4954e0a16d7935262cdc26cd9c4a77f1fd2310f23c1ae5b5a8f347f71f`.
별도 문제 signature를 검색 전에 저장했다. 전체 이전 goal은 완료 상태를 유지한다.

- Structured search 2회: Choi 정확한 제목(implementation),
  `EEG accelerometer motion artifact adaptive filtering`(adjacent).
  Workspace cutoff 2026-09-04 유지, retrieval 2026-09-10.
  새 frontier 전체 조사나 신규성 검토의 완결을 주장하지 않는다.
- 기존 논문 Choi를 주 근거, EMBC 2014를 인접 초록 근거로 선택했다.
  검색의 나머지 후보는 이 센서/시간 결정에 추가 기여가 불명확해 정독하지 않았다.
  검색 중 Semantic Scholar 429, GigaDB SPA 본문 부재, web 도구의 DataCite/CDN 접근 오류,
  PubMed 직접 open의 reCAPTCHA를 기록했다. 일반 공개 API/CDN 요청으로 얻은 자료만 사용했고 우회하지 않았다.
- 두 **서로 다른 cnt 파일**의 헤더를 검사했다. 단위를 추가 확인하려고 같은 파일을 한 번씩 다시 열어
  파일 open은 총 4회다. `cnt/x`와 marker 값 읽기, 새 dataset 다운로드, fit, outcome 모두 0이다.
  공개 논문 그림 1개 다운로드는 dataset 다운로드가 아니다.
- 공식 문서 target 최대6와 표적 follow-up 최대4의 범위에서 종료했다.
  이 값은 개별 HTTP 요청/adapter 내부 재시도 수를 뜻하지 않는다.
  새 output 한도32MiB; 수집한 evidence 파일/그림과 문서만 추가했다.
- `academic-research`: abstract-provisional card 2개, qualified claim `4baf6f4df7e40484`
  (5 evidence), open gap `b654467d1d604de0`를 저장했다. 선택 HTML 근거는 카드의 PDF 정독으로 위장하지 않았다.
  관련 연구를 근거로 센서 이름의 단정과 artifact 제거→저보정 효용의 비약을 피하고 시간·비용 설계를 수정했다.
- `coordinate-worktree-changes`: root만 main/연구DB 작성, 두 agent는 자료·코드 독립 읽기전용 검토.
  새 worktree/동시 shared writer/설치/삭제/push/GPU/외부 요청/유료 0.
  기존40 worktrees와8 untracked 출력, 원 실패/부정 결과/held60/기존 Choi V2 deny를 보존했다.
  두 읽기전용 검토에서 차단 오류는 없었고, k=1 동일성의 공통 학습상태 조건과
  같은 사람의 band 반복 측정을 통계적 독립으로 오해하지 않도록 하는 표현을 보완했다.

상세 source 경로·SHA·관문은 [기계 판독 상태](reports/choi_aux_context_feasibility_v1_state.json)에 있다.
