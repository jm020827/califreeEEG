# MMV·MAMEM 병렬 검토 결과

2026-09-12. **두 자료 모두 이번에는 사람 학습으로 넘기지 않는다.** 공개 파일을
다시 찾았다는 것보다, 잘못된 metadata 이득을 만들 수 있는 경로 두 가지를 실제
loader 코드로 좁혔다는 점이 이번 진전이다. 자료가 없거나 metadata가 무용하다는
결론은 아니다. [사전 계획](paired_metadata_parallel_preflight_v1_plan.md)은
aca7cd7로 고정했고, 원래 저보정 SSVEP 연구목표는 유지했다.

## 무엇을 병렬로 했나

| 작업 | 확인한 것 | 판정 |
|---|---|---|
| MMV 공개 배포·loader | 정확한 배포처 접근 실패; 공개 EDF 변환기의 stream-gap 의미 확인 | PARK_ACCESS_UNRESOLVED; 실제 offline validity·license·파일쌍 미확인 |
| MAMEM 기록 출처 | 저자 Session.m의 DIN timestamp→주파수→label 경로 확인 | PARK_ACCESS_UNRESOLVED; DIN 생성기와 `.flash` 변환 출처 미확인 |
| 독립 기작 검토 | one-shot 가중평균 취소, 고정 순서 누출, 전체 보정비용 | 학습 전 관문에 반영; 효능 검증 아님 |

세 agent는 읽기 전용으로 작업했고 root가 단독으로 문서와 연구 DB를 수정했다.
별도 worktree를 만들지 않았다. 새 raw EEG·annotation·PDF·학습·held60·외부요청·유료0.

## 1. MMV: ‘데이터가 끊겼다’의 의미부터 다르다

[MMV 논문](https://www.nature.com/articles/s41597-024-03729-8)은 offline40 SSVEP
trial, CNT/EDF의 대응 event와 장시간 과제의 tracking-loss/PERCLOS 보완을 설명한다.
이번에 exact DOI와 DataCite record는 도구 안전성 오류, schema table4/5는 Nature의
cookie-authorize redirect 오류로 읽지 못했다. 이는 실제 서버가 비공개·로그인을
요구했다는 확인이 아니며, 원시 파일 목록·데이터 라이선스는 여전히 미확인이다.

대신 저자가 인용한 [EDF 변환기 소스](https://raw.githubusercontent.com/uzh/edf-converter/master/@Edf2Mat/Edf2Mat.m)를
읽고 저장했다. Root 보존본 기준:

- L192의 `LOST_DATA_EVENT`는 **data-stream gap**을 뜻한다. 광학 추적 실패,
  blink, EEG 측정 품질을 뜻한다고 이 이름에서 추론할 수 없다.
- L374–376은 importer의 `FSAMPLE`을 `Samples`로 전달한다. L285–287 및
  L555–570의 Linux/oldProcedure와 기본 경로는 같은 필드를 보존한다고 보장되지 않는다.
- L447–449의 normalized timeline은 자체 시작점을 빼는 연산이다. 이것만으로
  EEG와 eye-tracker 사이의 clock offset·drift가 보정됐다고 할 수 없다.

이것은 현재 공개 loader의 코드이지 MMV가 쓴 정확한 revision이나 실제 EDF
필드의 증거는 아니다. Scout의 browser line 번호와 root 일반 GET의 line 번호가
달라 root가 실제 저장한 소스 위치를 사용했다. `master`의 commit은 이번에 별도
조회하지 않았고 내용 hash로 고정했다. Loader MIT가 MMV 데이터 라이선스도 아니다.

따라서 기존 PERCLOS, gaze좌표, pupil, blink를 M로 대신 넣지 않는다. 실제 offline
독립 validity와 EEG 학습 신뢰도를 잇는 기작은 아직 확인되지 않았다.

## 2. MAMEM: 시간 이벤트가 이미 정답을 만드는 데 쓰인다

고정 revision의 [Session.m](https://raw.githubusercontent.com/MAMEM/eeg-processing-toolbox/5a03abe2a6a874e9adaceea29a52c2fce35d8a03/+eegtoolkit/+util/@Session/Session.m)을
새로 읽었다. Scout는 전체523줄, root는 데이터 경로·분할·label 부여 부분을 직접 확인했다.
코드는 실행하지 않았다.

1. L308–324: Dataset I은 `eeg, DIN_1`, II는 `eeg, DIN_1, labels`를 전달한다.
2. L460–503: DIN의 timestamp와 sample index로 구간을 나누고, 이벤트 평균 간격에서
   `1000 / (2 × 평균 간격)`을 계산한다.
3. L508–515: I은 그 계산한 주파수를 label로 사용하고, II는 별도 `labels`를 사용한다.

즉 **Dataset I의 query DIN 간격을 입력하면 EEG 없이도 코드가 쓰는 정답 주파수를
알 수 있다.** 이를 metadata 학습 효과로 세면 안 된다. 이 지적은 코드 경로에 대한
검토이며 실제 사람 데이터에서 shortcut 정확도를 측정한 실험은 아니다.

그러나 이 코드는 이벤트를 *읽는* loader다. DIN을 실제 빛 측정으로 생성했는지,
소프트웨어 예정 event인지, PhysioNet `.flash`와 어떻게 연결되는지는 설명하지 않는다.
Dataset II도 별도 label이 있다는 이유만으로 DIN이 target-independent라고 할 수 없다.
III의 별도 event/label 경로를 I/II와 혼합하지 않는다.

Dataset I DOI 조회는403, 문서 연결 폴더는 restricted URL 오류였다. 재시도·우회하지
않았다. 저자 보고서 HTML은 초기 일부만 읽었고 후속 section 접근이 실패했으므로
acquisition 방법을 정독했다고 표시하지 않는다. 공식 repository tree와 고정 source는
일반 GET으로 성공했다. Tree 전체 body는 root가 보존하지 않았지만 Session source의
SHA-256과 Git blob을 검산해 읽은 코드를 고정했다.

## 3. 설계에 실제로 반영한 것

[학습 전 관문](paired_metadata_learning_gate_v1.md)에 다음을 반영했다.

- **one-shot에서 작동하는지 먼저 확인:** 같은 class support의 정규화 가중평균은
  예제가1개이고 가중치 a>0이면 `a*x/a=x`다. 이런 연산만으로는 M를 넣어도 변화가 없다.
  Source prior 대비 support 불확실성 조절은 가능한 후속 가설일 뿐 자동 재실행은 아니다.
- **고정 순서도 누출:** MAMEM exact trial index/absolute elapsed를 모든 arm에 공통
  제공해도 정답 shortcut이 해결되지 않는다. 비용 ledger와 예측 입력을 분리한다.
- **공통 보정과 학습 이득 분리:** 실제 timing M가 확인돼도 deterministic correction을
  공정한 공통 기준선에 넣고, 그 이상의 Q/Q2/QM/SHAM 차이를 비교해야 한다.
- **실제 획득 비용:** 버린 support도 비용이다. Tracker 준비·9점 보정·재보정 시간이
  없으면 labeled-trial 절감과 순 보정시간 절감 주장을 구분한다.

## 종료·다음 조건

두 경로는 이번 예산에서 보류한다. 같은 실패 endpoint를 반복하거나 비공개 요청을
보내지 않는다. MMV는 실제 공개 배포와 offline validity의 독립 정의, MAMEM은
DIN 생성 및 `.flash` 변환의 공개 provenance가 확인될 때에만 다시 판단할 근거가 생긴다.
둘 중 하나가 공개 pairing을 보이더라도 별도 bounded schema와 학습 기작 관문이 남는다.
새 모델·새 dataset 탐색으로 이번 예산을 자동 연장하지 않는다.

[상세 실행 상태](reports/paired_metadata_parallel_preflight_v1_state.json)에 접근 URL과
실패·시각·수량을 남겼다. 각 scout6targets+1locator, root 성공 source 보존GET2회다.
마지막 외부 응답09:25:45.463UTC로09:45 종료 경계 안이었다. Browser underlying
bytes는 측정되지 않아 전체 네트워크2MiB 제한 준수라고 쓰지 않는다. 일반 GET4건은
각2MiB 미만이며 root 보존 artifact2개는53,650bytes다. 신규 scholarly search/PDF card0.
직전 라운드의 시간·수치행 검색 이탈과 기존 부정 결과는 그대로 보존한다.

## 검증·연구 기록

독립 최종 검토는 실제 저장 코드와 line locator, 미확인/실패 구분, 원목표 유지,
wrapper/body hash와 Git blob을 확인했다. 가중평균 상쇄의 적용 조건(a>0, 분모는
가중치 합)을 보완했다. 이는 문헌·코드·설계 범위의 검토이지 전체 효능 검증이 아니다.
Academic claim `71c921f7cd5e953b`는 근거3개로 qualified, 기존 gap
`3f973164c88eabce`는 새로 확인한 코드 경로와 남은 불확실성으로 갱신했다.
누적79claims/209evidence이며 신규 PDF card나 효능 결과는 없다.
Root도12target/2locator 계수, 두 source wrapper/body hash·크기, MAMEM Git blob,
세 claim artifact hash, 새 문서 링크와 기존 보호 문서4개의 hash 불변을 확인했다.
양의 가중치3개×스칼라3개 대수 sanity check도 통과했다. 코드 변경이 없어 전체
pytest는 재실행하지 않았으며 이 검사를 EEG 학습 결과로 세지 않는다.
