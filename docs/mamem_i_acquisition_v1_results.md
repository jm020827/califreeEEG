# MAMEM I 원자료 확보 및 DIN-only 점검 결과

2026-09-13 KST. **필요한 원자료 압축본2개, 합계6,585,019,498bytes(6.59GB)를 모두
확보했다.** 각 파일은 배포처 MD5와 일치했고 독립 agent의 MD5/SHA256 재검산도
통과했다. 내부에는48개 MAT, 파일명 기준11개 참가자 ID가 있다. 전체 자료의
생물학적 타당성·분류 성능을 검증했다는 뜻은 아니다.

- [동결 계획·카탈로그 기반 한도 조정](mamem_i_acquisition_v1_plan.md)
- [기계 판독 상태](reports/mamem_i_acquisition_v1_state.json)
- [원자료·획득 영수증](/home/whwovy/data/mamem_i_v1_20260913/acquisition.json)
- [내부 파일 목록·첫 기록 선택](/home/whwovy/data/mamem_i_v1_20260913/inventory.json)

## 무엇을 받았나

논문 arXiv1602.00904v2의 V-B 공개자료 문단과 reference44가 정확히
`10.6084/m9.figshare.2068677.v1`을 가리킨다. [논문](https://arxiv.org/html/1602.00904v2#bib.bib44)과
[공식 v1 카탈로그](https://api.figshare.com/v2/articles/2068677/versions/1)를 연결했다.
카탈로그는 CC BY4.0을 명시한다. 별도 PDF는 이번 질문에 필요하지 않아 받지 않았다.
광센서+StimTracker의 생성 출처는 [직전 선택 원문 검토](public_provenance_resolution_v1_results.md)의
저자 보고이며, 이번에 정확한 배포본 연결과 실제 DIN 필드 존재를 추가 확인했다.

| 파일 | 크기(bytes) | 배포처 MD5와 로컬 값 |
|---|---:|---|
| EEG-SSVEP-Part1.rar | 3,279,547,652 | c4c7a749b5e7834e2f90c19aff507a2c |
| EEG-SSVEP-Part2.rar | 3,305,471,846 | 62d6f841bc736c0b861b99bf9486b3e0 |

두 파일은 각각24개 MAT를 담은 서로 다른 묶음이다. 압축 해제 전체 크기는
목록상6,645,250,642bytes지만 **전체 해제는 하지 않았다**. S001/S003/S008은 각3기록,
S004는4기록, 나머지 S002/S005/S006/S007/S009/S010/S011은 각5기록이다.
파일명에서 센 참가자 ID이지, 서로 다른 실제 사람이라는 신원 검증은 아니다.

익명 파일 GET은 각1회, 둘 다 HTTP200/redirect1회였다. 전송·검산 시간은
292.541초+317.521초이며 부분 파일은 남지 않았다. 다운로드 종료 free10,092,568,576bytes로
8GiB 여유 규칙을 지켰다. 공유 disk의 다른 작업 변화는 이 연구에 귀속하지 않는다.
보관 위치는 `/home/whwovy/data/mamem_i_v1_20260913`이며 Git에는 raw를 넣지 않았다.

## 실제로 어디까지 열었나

두 archive를 다시 SHA256 검산한 뒤 합친 내부 경로 목록에서 첫 MAT인
`EEG-SSVEP-Part1/S001a.mat`을17:51:54.612UTC에 고정했다. **S001의 모든 기록을
개발용**으로 지정했고 미래 독립 query 참가자로 취급하지 않는다. 이 MAT 하나만
137,357,437bytes 추출해 archive CRC32와 SHA256을 확인했다.

Header-only 점검 결과:

- `DIN_1`:4×1966 MATLAB cell.
- `eeg`:257×117917 double. 값은 materialize하지 않았다.
- `samplingRate`, age/capsize/gender/hairtype/handedness는 이름·shape·class만 확인했다.
  수치/개인 특성 값은 읽지 않았다.250Hz는 여전히 저자 loader의 선언을 사용한다.

SciPy는 **DIN_1 변수 전체를 decode**했다(보수적 메모리 추정1,085,542bytes).
수치 계산에 쓴 범위는 첫 continuous group72개 이벤트뿐이다.73번째 timestamp는
gap>2000ms 경계를 검출하는 데만 썼고 해당 sample과 이후 값은 검사하지 않았다.
EEG나 다른 변수는 loadmat 요청에서 제외했다. Header 스캔 중 compressed bytes
처리는 있을 수 있으므로 ‘EEG 바이트를 전혀 풀지 않았다’고 주장하지 않는다.

[DIN 결과](/home/whwovy/data/mamem_i_v1_20260913/din_probe.json):

| 지표 | 첫 묶음 관측 |
|---|---:|
| 이벤트/인접 간격 수 | 72 / 71 |
| 간격 최소–최대 | 61–88ms |
| 간격 평균 / 모집단 표준편차 | 67.8169 / 5.9299ms |
| `Δtimestamp − 4×Δsample` 절댓값 최대 | 3ms |

선택된 sample은 양의 정수·증가 순서이며 EEG header의 시간축 길이 안이다.
사전 clock tolerance4.001ms를 통과했다. Probe 종료17:53:04.994UTC로 전체
18:26:35UTC 경계 안이다.2GiB AS/90CPU초/120wall초 제한과 부모 영수증을 적용했다.

## 연구적으로 의미하는 것과 아닌 것

이제 ‘실제로 얻을 수 있는 EEG–광센서 이벤트 파일인가?’에는 **그렇다**고 답할
근거가 있다. 다만 이 좁은 검사는 장비의 물리적 정확도나 여러 기록의 일관성을
검증하지 않는다. `Δt−4Δsample` 최대3ms는250Hz sample-clock 대응과 양립한다는
뜻이지, neural latency·display latency·정답과 독립인 jitter를 분리했다는 뜻이 아니다.

61–88ms 간격 변동은 정수 video-frame 구성, 이벤트 검출, timestamp/sample 양자화와
실제 변동이 섞인 결과일 수 있다. 단일 묶음 요약만으로 원인을 식별하지 않는다.
표준편차5.93ms를 그대로 새 M의 효과나 뇌 반응 지연으로 해석하지 않는다.
Query DIN 간격으로 target을 추정하는 기존 label shortcut은 여전히 금지한다.

원 연구목표는 그대로다: **적은 labeled calibration으로 SSVEP를 잘 분류하도록
외부 acquisition M이 Q와 공통 정보 이상으로 돕는가?** 새 자료 확보와 clock/schema
통과는 이 질문의 실행 준비일 뿐이다. 이번 학습/accuracy/보정량 절감 평가는0이다.
기존 gyro/source39 부정 결과를 뒤집거나 새 유망 후보를 확보했다고 하지 않는다.

## 새로 확인한 다음 병목

1. **파일 ID 대응:** 실제 archive는S011a–e인데 저장한 저자 Session.m의 DatasetI
   마지막 참가자는S013a–e다. 같은 사람/단순 오타라고 가정하거나 자동 rename하지 않는다.
2. **채널 대응:** 카탈로그는256electrodes라고 하고 첫 EEG는257행이다. 추가 행의
   역할과 channel/reference mapping이 미확정이다. 임의 삭제나256채널 좌표 부착 금지.
3. **잔여 정보:** 공통 stimulus schedule/clock 보정 뒤 support-only M에 무엇이
   남는지 아직 모른다. Deterministic correction은 Q/Q2/QM/SHAM 모두에 동일하게 주고,
   그 위의 학습 이득만 별도로 검증해야 한다.
4. **실제 획득비용:** 초기 휴식·적응·버린 trial·고정순서에서 균형 support를 모으기
   위한 대기와 추가 센서 설정비를 포함해야 한다. k개만 회계에 남기지 않는다.

Part1은 폴더 아래 MAT지만 Part2는 폴더 없는 member 경로다. 후속 join에서
basename과 archive 출처를 함께 보존해야 한다. 현재 합친 경로의 첫 S001a 선택에는
영향이 없다. Nominal60Hz 외에 timestamp 분해능/clock/event polarity의 실제 처리도
아직 검증하지 않았다. 단일 자극 자료의 진단을 다중표적 온라인 BCI 증거로 확대하지 않는다.

[다음 유한 관문 초안](mamem_i_timing_identifiability_v1_draft.md)은 이 네 문제를
먼저 풀도록 구성했다. 이번에는 추가 raw 기록 값/학습을 자동 실행하지 않았다.

## 검증·이력·한계

- 사전 계획97ca67f, catalogue Amendment A+fetcher b421c22, probe fbb89b4,
  출력 capture/시간 제한 수리 d60bf18. 실제 MAT 값은 그 뒤에 처음 읽었다.
- 생성자료 기반37tests PASS. 실제 경로는 두 archive 재검산→48 MAT 목록→1개
  추출→header→DIN parent/worker 영수증까지 완료했다. 전체 저장소 test suite는
  실행하지 않았다. 기존 학습기 변경은 없다.
- 독립 read-only review로 frozen deadline/바깥 오류 영수증, sample의 EEG 길이
  경계, pipe capture 사전 한도를 고쳤다. root만 repo/data/SQLite를 썼다.
  새 worktree/설치/삭제/push0, 기존41worktrees/8untracked pytest directories 보존.
- Latest-article API 첫 Node GET의 `TypeError: fetch failed`는 실패 기록으로 남겼다.
  HTTPstatus/원인은 미확정이다. 정확한v1의 별도 공식 URL은 일반GET200이었다.
  재시도나 접근 우회는 하지 않았다. Root catalogue2/3, scoutsource1/3,
  locator0/1, root 성공 source 보존1/1; 새 scholarly discovery/PDF0.
- 카탈로그·HTML은 선택 필드/문단 wrapper를 보존했다. 전체 응답 SHA는 전송 당시
  계산했으나 전체 body 미보존이라 사후 fullbody 재검산은 불가능하다. Wrapper hash와
  raw archive hash는 재검산했다. 초기 Part1 수동 header preview는 제한된 도구출력으로
  확인했고, 영속 inventory는 두 stream을 각각 bounded capture하는 수리본으로 만들었다.
- 기존 보호문서/negative report4개 hash 불변. Held60/외부발송/계정·DUA/유료0.
  파일 확보 성공을 근거로 이 승인 경계를 바꾸지 않는다.
