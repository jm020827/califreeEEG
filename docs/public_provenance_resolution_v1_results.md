# 공개 provenance 후속 결과: MMV 등록 정보와 광학 DIN 근거 확보

2026-09-12. **두 가지 불확실성을 줄였다.** MMV의 공식 registry에서 배포 주소와
라이선스 선언을 확인했고, MAMEM 계열 단일 자극 실험의 DIN이 광센서에서 왔다는
저자 보고를 찾았다. 새 EEG를 다운로드하거나 metadata 효능을 확인한 결과는 아니다.
[사전 계획](public_provenance_resolution_v1_plan.md)은 f390e3c다.

## 1. MMV: 비공개로 단정할 문제가 아니었다

이전 browser safe-open 오류는 실제 서버의 접근 정책을 확인한 것이 아니었다.
이번에 명시적으로 고정한 일반 GET 1회는200으로 성공했다.
[DataCite 공식 record](https://api.datacite.org/dois/10.57760/sciencedb.ai.00010)의 선언은 다음과 같다.

- DOI `10.57760/sciencedb.ai.00010`, 제목 *MultiModal Vigilance dataset*.
- 버전 V3, CC BY 4.0, 등록된 크기265,453,813,779bytes와3,905files.
- 논문 DOI `10.1038/s41597-024-03729-8`의 supplement 관계.
- 정확한 [ScienceDB 배포 페이지](https://www.scidb.cn/detail?dataSetId=270bcdeaab0c48adb8eee700479daafd).

이는 **registry의 데이터 라이선스·파일 수 선언**이지 모든 파일이 익명 다운로드
가능하다거나3,905개 파일 목록을 직접 검산했다는 뜻은 아니다. 약265GB 전체를
현재 약19GB의 여유 공간에 받는 계획도 세우지 않는다. 필요하다면 최소 공개
문서·파일쌍만 다루는 별도 byte 예산이 필요하다.

배포 페이지에도 일반 GET을 시도했으나 큰 응답을 도구 출력으로 전달하는 과정에서
출력이 잘려 JSON 해석에 실패했다. 이 응답의 status·원문·정확한 크기는 보존하지
못했으므로 페이지 내용을 읽었다고 하지 않는다. HTTP 접근 거부로도 해석하지 않는다.
다시 받거나 추측 API를 호출하지 않았고, 실제 inventory·offline validity·cue/clock은
여전히 미확인이다. DataCite의 성공으로 기존 오류를 지우지 않으며 진단 예산을
같은 방식으로 반복 개설하지 않는다.

## 2. DIN이 실제 점멸을 감지했다는 원문 근거

저자 기술보고서/preprint [*Comparative evaluation of state-of-the-art algorithms
for SSVEP-based BCIs*, arXiv1602.00904v2](https://arxiv.org/html/1602.00904v2#S5.SS2)의
V-B Acquisition Setup은 다음을 보고한다.

- Microsoft Visual Studio/OpenGL로 자극을 만들고 LCD에 표시한다.
- **Cedrus StimTracker ST-100과 모니터에 부착한 light sensor**로 EEG에 DIN
  marker를 추가한다. 센서가 실제 자극의 점멸을 감지한다는 설명이다.
- DIN으로 자극 구간과 주파수를 식별하고 offline EEG를 분할한다.

따라서 이 실험의 DIN을 “예정 주파수 annotation일 뿐”이라고 단정하면 안 된다.
**독립 광학 이벤트의 저자 보고가 확보됐다.** 다만 marker의 정확도·실제 파일·연속
luminance 신호의 공개를 직접 검증한 것은 아니다. 센서 접촉 impedance의 상한도
서술돼 있으나 실제 채널별 측정값이 공개됐다는 근거로 쓰지 않는다.

중요한 범위는 V-C Stimuli Layout, `S5.SS3.p3.1`이다. 이 보고서는 **한 개 상자**가
깜빡이는 실험이며 여러 표적의 동시 자극은 향후 실험이라고 명시한다. Dataset I
맥락의 provenance를 보강하지만 이를 Dataset II의 센서 구성으로 옮기지 않는다.
정확한 파일 release의 연결과 PhysioNet `.flash` 변환 규칙도 별도 확인 사항이다.

Root는 V-B의 텍스트와 V-C Stimuli Layout 문단만 보존해 직접 읽었다. 표·그림은
읽기에서 제외했으며 PDF 완독이나 최종 저널판 검증으로 기록하지 않았다. Scout의
초기 넓은 paragraph 추출은 일부 결과 설명까지 노출했지만 판정·튜닝에 쓰지 않았고
원시 데이터나 사람별 수치 배열은 읽지 않았다.

## 3. 우리 학습 연구에는 무엇이 달라지나

새 근거는 “물리적으로 측정한 시간 정보가 실제 존재할 수 있다”는 근거다.
그러나 [이미 읽은 loader](paired_metadata_parallel_preflight_v1_results.md)는 Dataset I에서
DIN 간격으로 label을 만든다. 따라서 query DIN을 넣는 방식은 여전히 정답 누출이다.

다음에 확인할 **조건부 가설**은 이것이다.

> 비용을 지불한 support의 광학 이벤트에서 공통 자극 주파수·고정 지연을 제외한
> 측정 변동을 분리할 수 있다면, 그 정보가 EEG-only Q보다 support 반응 추정의
> 불확실성을 잘 알려주어 적은 보정으로도 학습을 개선하는가?

이는 저자의 입증 결과가 아니라 우리의 후속 가설이다. 우선순위를 MAMEM I의
**exact-release/schema 확인**으로 좁힐 근거가 생겼지만, 아직 구현·fit 승인 관문은
통과하지 않았다. 실제 시간 변동이 전혀 없거나, 정답 schedule의 다른 표현일 뿐이거나,
공통 보정만으로 모두 설명되면 이 학습 후보를 종료한다. One-shot에서 취소되는
가중평균이나 상수 위상에 불변인 scorer에 M를 형식적으로 붙이지 않는다.

Q/Q2/QM/SHAM, 공통 deterministic correction, support-only 정보권한, 참가자 분리,
전체 획득 prefix·장치 준비 비용을 유지한다. 원 연구목표와 기존 부정 결과는 불변이다.

## 실행 한계·다음

Root metadata targets2/3(그중 이전 tool-only 실패의 단회 진단1), 성공 scout source
보존GET1/1; scout2/3targets와1/1locator를 사용했다. 끝까지 소진하지 않고 새 근거로
현재 질문을 정리했다. Raw/PDF/annotation/fit/held60/발송/계정·DUA/유료0이다.
공식 registry wrapper와 표적 HTML 발췌 wrapper는15,991bytes다. ScienceDB 응답
capture 실패 때문에 전체 HTTP body 총량·모든 응답 hash가 보존됐다고 주장하지 않는다.

다음은 MAMEM I의 **광센서 DIN이 들어 있는 정확한 공개 파일과 schema의 연결**을
확인하는 별도 제한 단계다. 막혔던 DOI403을 재시도하거나 저자 요청을 자동 발송하지
않는다. 실제 자료 확인 전에 모델을 구현하거나 예전 실패 실험을 재개하지 않는다.

독립 최종 검토는 registry 선언과 실제 파일 접근의 구분, optical DIN의 저자보고/
단일자극 범위, 원목표·누출 조건·기존 실패 보존에 대해 범위 내 PASS였다. 보존된
두 wrapper와 registry body hash를 검산했다. MAMEM 전체 HTML은 발췌만 남겼으므로
그 전체 response hash를 로컬에서 다시 계산했다고 주장하지 않는다.
Academic claim `bb1eb1cb03c8788b`(qualified,근거3개)와 기존 gap
`3f973164c88eabce`를 갱신했다. 누적80claims/212evidence이며 신규 PDF card는 없다.
