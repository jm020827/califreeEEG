# MobileBCI 시간 metadata 복구 결과

2026-09-11. **파서 수리와 한정 재검사는 성공했다.** 이전 v1 실패는 그대로 보존한다.
이는 paired event metadata 확보이며, 실제 metadata 효능이나 물리적 동기화의 완전한 입증은 아니다.

## 실제 관측

| 항목 | 확인 결과 |
| --- | --- |
| EEG–IMU 대응 |60개 event의 className/decs/onehot 순서가 동일,3class 각20개 |
| MATLAB 내부 형식 |event.className은3항목 cell; v2가 처리. v1의 당시 실패 경로는 미기록 |
| 상대 epoch축 t |두 파일 모두10,20,…,4000;100Hz와 조합하면 millisecond 해석에 일치 |
| raw event축 후보 |`(event.time−1)×fs/1000`이 두 장치 모두 정수 sample grid와 일치 |
| 시간 차이 |ms 해석 아래 IMU−EEG −9.3125~+5.3125ms,절댓값 최대9.3125ms |
| raw 길이 |IMU575.40625s,EEG575.410s |

[원시 metadata 관측](reports/mobilebci_event_time_v2_observation.json)과
[저장 JSON에서만 계산한 요약](reports/mobilebci_event_time_v2_timing_summary.json)을 남겼다.
독립 읽기전용 검산에서 class/onehot/격자/차이/보정prefix 계산이 일치했다.

## 저보정 실험에 바로 반영되는 내용

| class당 최소 k | 그만큼 모일 때까지 실제 trial수 | prefix의 각 class수 |
| --- | --- | --- |
|1|6|3,2,1|
|2|8|4,2,2|
|3|11|5,3,3|
|5|22|8,5,9|

따라서3k를 실제 수집량이라고 쓰면 안 된다. 향후 모델에 첫k개씩만 넣을지 prefix의 모든
예시를 넣을지도 고정하고, 사용하지 않는 중간 예시를 포함한 실제 수집시간·trial수를 비용으로 센다.
이 표 자체는 보정량 감소 결과가 아니다. 필요한 목표 정확도 도달 여부를 아직 평가하지 않았다.

## 확정한 것과 가정인 것

관측은 **자극별 metadata 대응과 ms/sample-grid의 강한 내부 일관성**을 지지한다.
초기 MAT의 export코드 없이 정밀한 physical clock 동기화, marker가 실제 자극 onset인지,
zero/one-based raw index 원점을 모두 검증했다고 하지 않는다. ±1sample 검사는 index원점
차이만 다루며, marker 의미 자체의 불확실성을 덮지 못한다.

이 때문에 모든 구현을 멈출 필요는 없다. 다음은 outcome과 무관하게 가정을 명시한
**자기 장치 marker 기준 support 구간 추출**을 구현한다. 예컨대 marker-relative0.5–3.5s는
onset 해석 아래 중앙 구간 후보이지 검증된 stimulus interval은 아니다. 전체 run의 미래
marker로 affine clock을 맞추지 않고, query IMU로 모델을 갱신하지 않는다. 정확한 학습·평가
창 길이는 효능 계약에서 고정한다. 초기 MAT4초 축에 최종 BrainVision5초 설정을 이식하지 않는다.

## 구현·실행 기록

- Planaaa65df → reader9173bac → exact run manifest2880bdf 순서로 실제 접근 전에 고정했다.
- 인공 test-suite2/3회,15/18testsPASS. 중첩 MAT roundtrip, shape/UTF8/깊이/원소/출력,
  임의 객체 거부, checksum 선검사, 첫 실패 중단, 실행 중 파일 변경을 검사했다.
- 실제 recovery1회,loadmat2calls/checksum2passes,각0.007892/0.018716s,visitedunits1418,
  문자열20UTF8bytes. 모양은 scipy의 squeezed representation이며 원MAT shape라고 하지 않는다.
  Visitedunits는 원시 decoded allocation 계측이 아니다.
- 메모리768MiB/30s-per-file/출력64KiB 제한. 파형/새자료다운로드/fit/outcome/held60/유료/외부발송0.
  전체repo suite는 미실행. Root만 수정·DB작성,agent는 읽기전용; 기존 tree/출력 보존.

## 다음 연구 단계

첫 한 쌍의 대응 검사는 이제 충분히 진행됐다. 다음은 source-matched 전체 공개 목록으로
**0/0.8/1.6m/s 모두 EEG–IMU가 있는 실제 참가자 수와 용량**을 확인하고,
support-only 추출·one-shot reference-ridge/Q/Q2/QM/SHAM 구현을 준비한다.
기존2191byte API projection에는전체file목록이없어서18명모두사용가능하다고가정하지않는다.
Standing1명자료만으로metadata학습효과를판단하지않고,다른speed/session을독립사람으로세지않는다.
