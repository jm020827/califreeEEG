# 공개 paired acquisition-M 2차 조사 결과

2026-09-12 KST. **학습에 바로 투입할 새 후보는 아직 없다.** 다만 무작위로
데이터셋을 더 늘리는 대신, 다음 확인 대상을 **MMV의 추적 실패 기록**과
**MAMEM I/II의 flash annotation 출처**로 좁혔다. 원 저보정 SSVEP 목표는
바꾸지 않았다. 이번에 내려받은 것은 문서·공개 파일 목록이며 EEG 파일이 아니다.

## 후보 판정

| 후보 | 직접 확인한 근거 | 판정·남은 확인 |
|---|---|---|
| MMV 2024 | 원문에 SSVEP·EyeLink 동시 수집과 장시간 과제의 추적 실패 기록 설명 | **조건부 metadata 확인 대상**. 실제 파일 접근·보정구간의 독립 validity·정답 label 미확인 |
| MAMEM | 공식 PhysioNet 문서의 I/II `.flash`, III 공개 MAT 목록 | I/II의 시간 annotation 출처 확인 대상. photodiode나 독립 acquisitionM라고 확정하지 않음 |
| SpiralE | 원문의 실제 실험 중 접촉 impedance 측정 | 원시 EEG가 reasonable request 방식이라고 명시. 요청 없이 보류 |

세 후보는 feasibility 검토 목록이지 효능 후보 shortlist가 아니다. 적합성 관문
통과0·새 사람 fit0·새 rawEEG 다운로드0이다. 다른 visual 후보의 제외 사유도
아래 보존한다. “공개 데이터가 전혀 없다”는 결론은 내리지 않는다.

## 1. MMV: 많이 기록돼 있지만 무엇을 M로 쓸지가 핵심

Wei et al., [A MultiModal Vigilance (MMV) dataset during RSVP and SSVEP
brain-computer interface tasks](https://doi.org/10.1038/s41597-024-03729-8),
Scientific Data11,867(2024). 원문 Methods/DataRecords/record declaration/
CodeAvailability/reference33을 HTML 선택 독해했다. PDF·표/그림 육안 검증은 아니다.

저자 보고상18명·4세션 중 SSVEP와 RSVP가 포함되고, SSVEP는8/9/10/11Hz의
4방향 과제다. 각 SSVEP 세션은 offline40trials 뒤90분online으로 이어진다.
EEG/EOG는CNT,눈 추적은EDF에 저장하며 대응 event가 있다고 설명한다.
**모든4세션을 SSVEP로 세거나 online cursor 명령을 자동으로 정답으로 쓰면 안 된다.**

가장 중요한 발견은 *Data record declaration*이다. 저자는 눈 추적이 끊긴
시간을 실험자가 기록하고 feature의 PERCLOS를 보완했다고 설명한다. 이 설명은
장시간 과제에 관한 것이며 offline40 보정구간에도 기록이 있다는 증거는 아니다.
따라서 배포된 PERCLOS는 그대로 acquisition 품질 M가 아니다. 60초 smoothing과
사후 보완이 섞여 있어 현행 prequery 설계로 곧바로 가져올 수 없다.

다음 판단은 우리 추론이며 논문 결과가 아니다.

- 우선 확인할 M는 **보정구간의 원래 tracking-validity/lost-track 상태**다.
  논문에 그런 독립 필드가 배포됐다는 보장은 없다. Blink와 tracking loss도
  같은 사건이라고 간주하지 않는다. Gaze좌표/방향·pupil·PERCLOS로 슬쩍 대체하지 않는다.
- EOG/ECG/EMG 등을 합쳐 vigilance를 예측하는 것은 이 논문의 과제이며,
  우리 acquisition-M 저보정 효과를 직접 증명하는 실험이 아니다.
- Offline40trials에도 EEG와 눈 추적이 대응하고 원래 cue가 남아 있는지 먼저
  확인한다. 그 구간에서 support/query를 나눌 수 있는지도 아직 모른다.
  9점 eye-tracker calibration을 포함한 추가 준비 부담도 비용이다.
- 원 데이터 전처리/feature를 그대로 쓰면 시간 경계를 침범할 수 있다.
  독립 quality가 없거나 offline pairing이 없으면 후보를 종료한다.
- Validity 필드가 있어도 눈 추적 장치 자체의 실패를 예측하는 과제로 바뀌면
  안 된다. 그 정보가 왜 EEG 보정학습의 신뢰도·전달성에 영향을 줄지 별도의
  기작 가설이 필요하다. 문서상 pairing 확인만으로 학습 후보를 승인하지 않는다.

본문의 일반 ScienceDB 링크는 홈페이지로 연결돼 파일목록을 얻지 못했다.
Reference33의 정확한 데이터 DOI는 **10.57760/sciencedb.ai.00010**이다.
이번에는 해당 DOI endpoint까지 조회하지 않았고, private/request-only라고
단정하지도 않는다. [다음 metadata-only 확인 초안](mmv_public_metadata_preflight_v1_draft.md)을
작성했으며 실행 상태가 아니다.

## 2. MAMEM: 독립 시간 정보인지, 정답 annotation인지 구분

[PhysioNet MAMEM SSVEP Database v1.0.0](https://physionet.org/content/mssvepdb/1.0.0/)
문서는 I/II에 개별 flash 위치인 `.flash`와 구간/주파수인 `.win`이 있다고
설명한다. 실제 주파수의 작은 변동도 언급한다. III의 파일 설명에는 `.win`만
나오므로 I/II의 `.flash`를 III에 있다고 옮겨 적지 않는다.

이것은 접근 가능한 공식 **schema 설명**이지 actual annotation 값이나 생성
출처 검증이 아니다. 자극 프로그램 event인지, 광출력 측정인지, EEG로 역산한
정보인지 미확정이다. 정답 주파수가 들어 있는 query `.win`을 M로 넣으면
분류 정답이 새는 문제가 생긴다. 공통 reference 정보로 이미 제공해야 하는
것인지도 따져야 한다. 따라서 현재는 **provenance-only 후속 대상**으로 남긴다.
공식 문서상 주파수 제시 순서가 session마다 같고 Experiment1은 단일 주파수를
개별 제시한다. 따라서 **query `.flash` 간격과 trial 순서·경과시간도 정답을
노출할 수 있다.** 출처가 확인돼도 자동으로 적격 M가 되지 않는다.

별도 [Figshare III API](https://api.figshare.com/v2/articles/3413851)는 익명 조회에
성공했다. Version3,123files(개별MAT121+RAR1+PDF1)이고 첫 MAT는
`U001ai.mat`,file27802707,1,247,120bytes다. 목록의 checksum은 repository가
제공한 값이며 다운로드한 파일의 검산이 아니다. [보존 목록](reports/public_paired_round2_mamem3_listing.json).
파일 실제열/접촉quality/주파수 검증과 raw다운로드는 하지 않았다.

MAMEM PhaseI의 EEG·눈 추적 공개 설명과 MAMEM SSVEP I–III는 같은 자료가
아니다. 검색 결과의 두 자료를 합쳐 “SSVEP eye-validity가 있다”고 하지 않는다.
저자 toolbox README는 읽었으나 Session.m blob API 접근은 실패했다.

## 3. 접촉·광출력 후보를 남기거나 제외한 이유

[SpiralE](https://doi.org/10.1038/s41467-023-39814-6)는 피부 접촉 impedance를
실제 EEG 실험 중 측정했다고 보고한다(Validation/Methods). 그러나
DataAvailability는 raw EEG를 저자에게 요청해야 한다고 명시한다. 공개 코드나
그림 source data가 paired raw EEG–impedance라는 뜻은 아니다. Zenodo7748035
페이지도 scout 접근 오류였으며 재시도·저자 요청은 하지 않았다.

Visual scout의 독립 조사에서는 다음을 보류/제외했다. 아래는 scout의
선택 HTML/검색 발췌 보고이며 main의 fulltext 독해 완료로 세지 않는다.

- Gu2024 [1–60Hz dataset](https://doi.org/10.1038/s41597-024-03023-7):
  photodiode 장치 검사는 실험 전이라는 단서. 매 EEG trial의 센서 로그 공개 미확인.
- Mu2024 [multi-frequency dataset](https://doi.org/10.1038/s41597-023-02841-5):
  exact page 접근 실패, 독립 M 판단 불가.
- [Renton2019](https://doi.org/10.1038/s41598-019-55166-y): 공개 BCI 코드와
  개인 template 설명만으로 공개 EEG–timing pairing을 확정할 수 없음.
- [OLED characterization2026](https://doi.org/10.3758/s13428-026-03034-9):
  실제 photodiode/photometer 측정은 있으나 scout가 읽은 flicker 검사는 장치
  시험이고 사람 실험은 intra-saccadic 과제. 현재 multiclass SSVEP 자료로 채택 안 함.

## 이번 실행 한계와 보존 기록

[사전 계획](public_paired_acquisition_round2_v1_plan.md)은2039e85로 고정했다.
Structured queries2/3(각20records,중복제거 신규34),web locator5/6을 사용했다.
Exact target 할당16회분을 사용(root10/contact2/visual4;contact잔여2를root가인수),
SpiralE가두담당자에겹쳐고유URL은15개다. 같은성공문서 내구간 조회는별도로
URL을늘리지않았다. 추가보존GET2/4(PhysioNet성공,NatureClientChallenge실패),
raw/PDF/fit/외부요청/유료0. 남은검색을억지로소진하지않고세후보평가로마감했다.

**절차 한계도 있다.** 첫/둘째 scholarly 수집시각07:07:11/08:28:48UTC 사이에
81분 이상 간격이 있어30분 목표를 지키지 못했다. 에이전트capacity오류도 있었지만
그것이 전부의 원인이라고 단정하지 않는다. Root의 한 번의 과도한 `rg`가 숫자
부분문자열까지 검색해 기존 report의 예측/label 수치 행을 출력했다. 새 raw 파일이나
held60을 읽은 것은 아니고 새 계산/튜닝에 사용하지 않았지만 **수치행 비열람
계획은 미준수**다. 이후 `.md`로 범위를 제한했다. 따라서 전체 protocol PASS라고
표시하지 않는다. 또 확인되지 않은 PMCID를 직접 입력한 페이지는 CAPTCHA여서
MAMEM 논문으로 사용하지 않았고, 원시 GitHub tree body는 tool 출력 일부가
잘려 payload 전체의 독립 보존 검산을 하지 못했다.

학술 claim에는 실제 저장 artifact hash와 읽은 범위/실패를 연결한다. MMV는
공식 HTML의 browser 추출본을 보존했으며 PDF card를 만들지 않았다. SpiralE
보존GET의3038bytes는Client Challenge이지논문이아니다. 이를근거로쓰지않는다.
논문/파일수를효능증거로세지않고,기존gyro/source39부정·Eye/XR/Choi보류와held60
경계를보존한다. Root단독main/DB쓰기,새worktree/설치/삭제/push없음.

읽기 전용 최종 검토는 보존된 MMV/PhysioNet 추출본과 결과·후속 초안을
대조했다. 장시간 추적실패와 offline구간의 구분, MAMEM의 추가 정답누출 경로,
MMV 후보 승격 전 기작 관문을 지적했고 모두 반영했다. 이는 문헌·설계 검토이며
앞서 실패한 별도 protocol agent의 감사 완료나 전체 절차 PASS를 뜻하지 않는다.

Academic workspace에는 qualified claim `86ddfa5ef6bce2bd`와 근거4개,
후속 gap `3f973164c88eabce`를 저장했다. 누적1133서지/90검색/45reading cards/
78claims/206evidence/45gaps이며 이번 신규 PDF card는0이다. 많은 서지 수를
충분한 정독이나 후보 효능의 근거로 세지 않는다. SQLite quick_check는ok였고
원 연구설계·closeout·deny overlay·직전144fit 보고서의 hash는 불변이었다.
