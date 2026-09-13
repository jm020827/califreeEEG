# 공개 acquisition 자료 탐색 round4: 새 schema 후보는 확보하지 못함

2026-09-14 KST / 2026-09-13 UTC. 사전 고정한 **검색 2회·공식 원문 target 4개**를
진행했다. 검색에서 24개 서로 다른 DOI 기록을 받았으나, 이번 범위에서
**새 학습용 파일/schema 확인 단계로 추천할 후보는 0개**다.
이는 공개 acquisition 자료가 존재하지 않거나 metadata가 무용하다는 결과가 아니다.
원문 접근 실패가 있어 **검증 범위가 부족한 탐색 결과**다.

## 검색을 어떻게 했나

[사전 계약](public_acquisition_round4_contract.md)을 **a14929a**로 커밋한 뒤 실행했다.
두 query 모두 `academic-research space discover`, OpenAlex/Crossref 각 최대8개,
merged 최대12개, `published-before=2026-09-14`, unknown dates 제외 옵션을 사용했다.
SQLite 충돌을 막기 위해 root가 순차 실행했고, 기존 경로 제외 검토는 읽기 전용 agent가
병렬로 수행했다. 기존 Semantic Scholar429를 재시도하지 않았다.

| 검색 | 이유 | 검색 ID / 저장 시각 UTC |
| --- | --- | --- |
| `SSVEP dry electrode dataset impedance` | 실제 접촉/interface 측정과 공개 EEG pairing | `868060228aabfb30` / 16:11:45.398996 |
| `SSVEP motion accelerometer dataset` | 지시된 움직임이 아닌 실제 외부 센서와 EEG 대응 | `36448168ed46a976` / 16:12:27.582389 |

각 provider의 반환은 8개씩, 두 query의 merged 결과는 각12개였다. Provider 오류는
기록상 0이다. 하지만 검색 품질/관련성을 보장하지 않는다. 저장소의 총 metadata record가
1134→1151로 늘어난 것은 **새로운 적격 연구 17편을 정독했다는 뜻이 아니다**.
이번 검색은 접촉과 움직임에 한정됐으며 광학 timing 전체를 조사한 것도 아니다.

## 왜 24개를 후보 24개라고 하지 않나

[전체 triage](reports/public_acquisition_round4_triage.json)에 모든 DOI·제목·판정을 보존했다.

- **기존 경로 2개:** Wearable wet/dry 자료와 MobileBCI. 이미 사용/검토한 자료이며
  새 독립 paired source로 세지 않는다. 접근 가능성과 종료한 학습 기작은 구분한다.
- **Metadata 수준 예비 단서 10개:** 실제 SSVEP 또는 접촉·자극 기작과 관련될 수 있으나,
  논문 전문과 공개 실측 필드의 연결을 이번에 확인하지 못했다. 그중4개의 primary 접근을 시도했다.
- **반환 metadata상 과제/기록 유형 부적합 12개:** 심사·decision 기록, SSVEP가 아닌
  동작인식·sEMG·auditory/rest 연구, SSVEP 과제가 입증되지 않은 artifact-removal 자료 등이다.
  이 판정은 현재 shortlist의 제외이지 읽지 않은 논문 전체나 모든 후속 용도의 기각이 아니다.

특히 Crossref 결과에는 논문 대신 peer-review/decision DOI가 들어 있었다. 움직임 검색의
6개 기록은 날짜가 `1970-01-01`로 표시됐다. 이를 실제 출판일이라고 믿지 않으며, upstream
자료 문제인지 local normalizer의 기본값인지 이번에는 진단하지 않았다. 날짜 필터 옵션을
썼다는 사실과 **모든 반환 기록의 출판일을 검증했다는 주장**은 구분한다.

## 직접 원문 확인을 시도한 네 대상

아래 설명은 보존된 검색 metadata/색인된 초록에 근거한 **선정 이유**다. 원문을 읽고
확인한 실험 결과나 data-availability 판정으로 쓰지 않는다.

| 대상 | 확인하려던 질문 | 이번 접근 결과 |
| --- | --- | --- |
| [A Pre-Gelled EEG Electrode and Its Application in SSVEP-Based BCI](https://doi.org/10.1109/tnsre.2022.3161989) | 임피던스 비교와 SSVEP가 함께 언급돼, 실제 값/EEG의 공개 pairing 여부 확인 | 도구 safe-open 오류 |
| [Spatial quality-aware SSVEP classification for motion artifact suppression in scenario-care BCI](https://doi.org/10.2139/ssrn.5622956) | 새 외부 motion 측정인지 EEG-derived Q 방법인지, 공개 release 존재 여부 | 도구 safe-open 오류; 제목 metadata 수준 |
| [EEG dataset and OpenBMI toolbox for three BCI paradigms](https://doi.org/10.1093/gigascience/giz002) | 공개 다세션 EEG에 실측 acquisition 값이 포함되는지 | OUP CDN redirect의 safe-open 오류 |
| [Improving user experience of SSVEP BCI through low amplitude depth and high frequency stimuli design](https://doi.org/10.1038/s41598-022-12733-0) | 공통으로 알려진 자극 설정 외에 독립 실측 context가 배포되는지 | 도구 safe-open 오류 |

첫 DOI 요청에서는 TNSRE 부분의 대문자 표기를 사용했다. 네 결과의 도구 반환을
[접근 실패 기록](reports/public_acquisition_round4_primary_failures.json)에 그대로 보존했다.
**원문 성공 0/4**, 재시도·다른 endpoint·추가 일반 검색·PDF 다운로드는 0이다.
도구의 비재시도 오류와 OUP redirect 오류를 저널 서버의403/계정 요구/비공개 확인으로
바꾸어 해석하지 않는다. 실제 underlying HTTP 수·wire bytes·정확한 각 HTTP 시각은
측정하지 않았으며 네 개는 도구의 distinct target 수다.

## 연구 설계에서 유지할 경계

이전 경로 검토는 아래 사실을 분리했다.

- Liu2022의 요청 의존 데이터는 자동 문의하지 않는다. 논문의 임피던스 상한이나
  wet/dry 명칭만으로 공개 channelwise/pre-block 측정값을 확보했다고 보지 않는다.
- MobileBCI13604078/v1에는 실제 EEG–IMU GET 성공 이력이 있다. 이를 접근 실패로
  바꾸지 않지만, 종료한 gyro source-routing을 새 이름으로 재개하지도 않는다.
  다른16669072 배포본의 시간 설정을 그대로 이식할 수 없다.
- Choi의 private 문의 branch, Eye-BCI의 익명 파일403, MMV의 inventory/schema 공백은
  각각 보류 유지다. 서로 다른 사유를 “모든 자료 비공개”로 합치지 않는다.
- 세션 내 상수라는 이유만으로 실측 개인 M을 배제하지 않는다. 반대로 device label만으로
  정해지는 값은 공통 정보를 넘는다는 근거가 없다. 미래 구간이 있는 데이터셋 자체가 아니라
  **support-only 절단/정렬이 가능한지와 실제 사용 방식**을 검사해야 한다.
- 강한 Q, 용량 대조 Q2, 적절히 조건화한 SHAM, common/direct-M 기준과 전체 setup/
  재측정/버린 support 비용을 유지한다. Hardware 성능 비교는 metadata 학습 이득이 아니다.

## 종료 판단과 다음 한 가지

판정은 **`NO_QUALIFIED_NEW_SCHEMA_CANDIDATE_WITHIN_ROUND`**다. 검색 2회와
primary target 4개 예산은 모두 사용했다. 새 EEG/metadata 숫자·학습·예측·보정량 평가는 0이다.
같은 keyword의 검색을 즉시 더 하거나, 원문 확인에 실패한 자료를 학습 후보로 올리지 않는다.

포화 점검: 최근 검색은 기존 Wearable/MobileBCI를 다시 찾았고 관련 없는 records도 많았다.
남은 불확실성은 새로운 모델을 만드는 것보다 **구체적인 한 논문의 Methods/Data Availability**를
읽는 것으로 더 직접적으로 줄일 수 있다. 따라서 Pre-Gelled 논문을 **독서 후보 한 개만**
등록했다. 색인 초록만 읽은 provisional card와 targeted deep-read queue이며, 새 데이터셋
추천이나 raw/schema/learning 권한이 아니다. [다음 단일 확인 초안](pregel_public_fulltext_next.md).

이번 라운드의 진전은 탐색을 실제 실행하고, 무엇을 확보하지 못했는지·다음에 읽을 질문을
정확히 좁힌 것이다. 원 저보정 목표는 아직 미완료/active다. 이전 실패·부정 결과 및
held60·외부 사람 요청·유료 사용 별도 승인 조건은 그대로 유지한다.

## 기록과 검증

[독립 읽기 전용 검토](reports/public_acquisition_round4_review.json)는 두 보고 문서와
triage/실패 JSON, 두 검색 manifest의 DOI·제목·날짜·초록 출처/유무를 전수 대조해 PASS였다.
계산0.000614초이며 새 원문/네트워크/원자료/학습 검산은 아니다. Root가 agent의 반환을
기록한 것으로 별도 실행형 auditor를 구현한 것은 아니다. [상태](reports/public_acquisition_round4_state.json).

Academic-research의 포화 점검에 따라 broad search를 멈추고 질문이 정해진 한 편의
독서 queue를 등록했다. Claim `61d1820c940cfb50`(QUALIFIED,근거3개)와 기존 gap
`3f973164c88eabce`를 갱신했다. Coordinate-worktree-changes에 따라 root solewriter와
읽기 전용 병렬 검토를 사용했고, 새 worktree/코드 변경/pytest/설치/삭제/push는0이다.
기존46worktrees와8untracked는 유지했다. 저장 수치 검토 PASS를 저보정 연구 성공으로
해석하지 않는다.

마감 workspace: catalog1151/search92/card47/technique18/claim99/evidence263/gap45/
analogy6/deep-read39(completed27/queued11/skipped1). 이는 읽은 논문1151편이나
PDF39편이라는 뜻이 아니다. Render11views, SQLite quick_check=ok, 디스크 약290GiB,
보호 대상6개 hash 불변을 확인했다.
