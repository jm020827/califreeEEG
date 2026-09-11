# XR 화면 타이밍 조사: 기작은 참고, 현재 학습 후보로는 보류

2026-09-12 KST. 원 저보정 SSVEP 목표는 유지한다. 이번은 **공개자료 가능성·
설계 검토**이며 새로운 사람 EEG 확보·학습·정확도 실험은 아니다.
직전 [Eye-BCI 접근 보류](post_gyro_candidate_feasibility_v1_results.md)와
[gyro source-routing 부정 결과](block_scaled_router_human_v1_results.md)를 보존한다.

## 쉬운 설명과 결정

12 Hz로 깜빡이도록 설정한 화면이 실제로는 다른 속도로 동작한다면,
EEG를 비교할 기준 신호부터 실제 속도에 맞춰야 한다. 이번 논문은 이러한
**자극 생성 오차를 software frame timestamp로 추정해 고치는 방법**이다.
모든 사람에게 같은 숫자를 붙이는 metadata보다, 신호가 달라지는 원인에 직접
연결된 측정이 유용할 수 있다는 설계 단서다. 그러나 이것만으로 우리 가설인
**“M을 학습에 넣으면 Q·공통 정보를 넘어 필요한 보정 trial이 감소한다”**가
증명되지는 않는다.

판정은 `NOT_EXECUTABLE_WITH_CURRENT_PUBLIC_EVIDENCE`다. 실제 paired EEG와
frame timestamp의 공개 파일 경로를 확보하지 못했고, 논문의 current-window
정보 조건도 우리 support-only/prequery 조건과 다르다. 비공개임을 확인한 것은
아니지만 이번 유한 접근 예산을 모두 사용했으므로 자동 mirror 탐색·저자 요청·
새 사람 실험은 하지 않는다. 연구 목표 전체의 무의미함이나 불가능함은 아니다.

## 읽은 근거와 논문이 보인 것

Angrisani et al., [Online Compensation of Systematic Effects in Stimuli
Generation for XR-Based SSVEP BCIs](https://doi.org/10.3390/s26030766),
Sensors 26(3),766,2026-01-23. Europe PMC의 [공식 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12899059/fullTextXML)을 보존하고
§§1–3,4.1–4.3,5,Data Availability,reference31을 선택적으로 읽었다.
PDF 정독·그림 육안 확인·전체 원문 독해 완료로 기록하지 않는다.
아래 수치는 저자 보고이며 재현 결과가 아니다.

| 항목 | 원문 근거 | 우리 연구에서의 해석 |
|---|---|---|
| M | §3: 각 자극 window의 rendering frame timestamps | 외부 software telemetry이나 photodiode로 측정한 실제 광출력은 아님 |
| 연산 | §2 Eq.(2),§3: `f_actual = f_nominal × RR_actual / RR_nominal` | 모든 후보 주파수의 reference를 수정. 실제 정답 label을 넣는 연산은 아님 |
| 비교 | §4.2: FBCCA, 두 방식 각각 source 참가자 LOSO grid 선택 | source hyperparameter 선택은 있지만, M-conditioned 학습기나 target few-shot learning을 평가하지 않음 |
| 오류가 큰 장치 | §4.3/Table3: Moverio9명,23.3%→45.0% | 이 설정의 정확도 개선 보고. +21.7%p이며 여전히 실용 정확도가 충분하다는 결과는 아님 |
| 오류가 작은 장치 | §4.3/Table2: HoloLens30명,71.2%→71.1% | 모든 장치에서 이득이 나는 것이 아님 |
| 평가 범위 | §4.1: 각 trial 첫1.25초; §4.3 jitter는 후속 과제 | 보정량 curve·실제 획득비용·target adaptation 결과는 없음 |

Table2/3의 ±는 저자 표기상 standard uncertainty이며 우리가 새로 계산한
95% CI나 SD로 바꾸지 않는다. 표는 XML 값으로 읽었고 그림은 검증하지 않았다.
HoloLens와 Moverio는 사람·자극·장치·채널 구성이 달라 두 장치 차이만으로
timing 효과의 크기를 인과적으로 정량화할 수도 없다.

## 우리 설계에 반영할 부분 — 아직 실행 계약이 아님

1. **측정 오차의 종류를 구분한다.** Complete sin/cos reference에서 상수 위상
   회전은 같은 부분공간에 머물 수 있지만 frequency/timebase 변화는 다른 문제다.
   이 구별은 우리 설계 추론이며, 임의의 phase metadata를 추가하면 좋아진다는
   증거가 아니다. Software fps와 physical refresh/light output도 구분한다.
2. **단순 보정을 먼저 공정한 기준선에 넣는다.** 나중에 timing M 학습기를 만들면
   nominal reference와 deterministic timing-corrected reference를 별도 anchor로
   둔다. Q/Q2/QM/SHAM은 같은 허용 시간·공통 보정·EEG·label 비용을 공유해야 한다.
   실측값을 기준선에도 주어 얻은 개선을 학습된 M 고유 이득이라고 세지 않는다.
   추가 학습 잔차에만 M을 넣는다면 그 잔차 연결을 바꾸는 SHAM과 전체 timing
   pairing을 바꾸는 기작 검사를 구분해 사전 고정해야 한다.
3. **지원 데이터와 평가 데이터의 시간 경계를 유지한다.** 논문은 분류할 window의
   timestamps를 함께 쓴다. 우리 현행 조건에서는 paid support 또는 query 전에
   실제로 끝난 측정만 학습에 사용한다. Support timing으로 미래 query timing을
   예측할 수 있는지는 미확인이다. Whole-run 평균·query timestamp를 몰래 넣지
   않으며 동시 telemetry 후보로 바꾸려면 별도 정보권한 설계가 필요하다.
4. **구현 전 물리량·식·효과 경로를 검증한다.** Eq.(3) XML은 M개 frame에 대해
   M−1개 interval을 합하고 1/M으로 나눈다. 표기상 정규화 불일치가 있어 그대로
   복사하면 안 된다. Mean instantaneous fps와 elapsed-time frame rate도 구별한다.
   실제 저자 코드가 어떻게 계산했는지 확인되지 않았으므로 이것이 발표 성능을
   바꿨다고 단정하거나 보정된 수치를 재현값처럼 제시하지 않는다.
   비례 보정도 frame-index 기반 자극 생성이라는 전제가 필요하다. 독립 검토는
   §2.2의 정수 cycle 설명과 일부 주파수 조합의 불일치, EEG 첫1.25초와 timestamp
   집계 범위의 일치가 불명확한 점도 남겼다. 또한 기존 nominal-frequency
   cross-product cache만으로 바뀐 주파수의 cross-product를 정확히 복원할 수
   있다고 가정하지 않는다. 이런 구현 조건은 실제 배포 자료/코드로 확인해야 한다.
5. **학습 후보는 작은 경로와 반증 조건부터 정한다.** 실제 pairing이 확보될 때에만
   support timing 특성으로 phase/harmonic template 잔차의 신뢰도를 조절하는
   작은 학습 경로를 고려한다. 예측 정보가 Q·공통 보정을 넘지 못하면 종료한다.
   그 다음에도 unseen-person source-only 선택,동일 label prefix 비용,미도달·harm를
   검정해야 한다. 지금 Neural ODE나 큰 Fourier network를 추가할 근거는 생기지 않았다.

## 데이터 접근 확인 범위

Data Availability(`notes4`,associated-data `_adda93_`)는
“Data are contained within the article.”이다. XML에는 결과 표가 있지만
trial별 EEG–timestamps raw 파일 URL은 확인되지 않았다. Europe PMC의
`hasData:Y`를 raw 공개의 증거로 쓰지 않는다.

§4.1/reference31은 *An Open Steady-State Visually Evoked Potentials Dataset
for Augmented Reality-Based Brain–Computer Interfaces*,
[DOI10.1109/JSEN.2025.3605813](https://doi.org/10.1109/JSEN.2025.3605813)을
인용한다. 해당 DOI 접근은 도구의 non-retryable safe-open 실패여서 내용·raw
링크를 검증하지 못했다. 제목의 Open만으로 사용 가능한 자료라고 선언하지 않는다.

[사전 예산](xr_timing_candidate_feasibility_v1.md)의 exact targets6/6사용:
root MDPI429,EuropePMC metadata성공,EuropePMC XML성공,PMC CAPTCHA;
scout OpenAlex exact record/위 reference31 DOI 각각 safe-open실패.
실패 경로를 재시도하거나 우회하지 않았다. Broad search0,PDF0,raw EEG0,fit0.
Root는 단독 문서·workspace 작성,scout와 기작 검토자는 읽기전용이었다.
독립 검토로 timing/학습/정보권한 구별을 확인했고 실제 연산 재현은 하지 않았다.

## 다음 단계와 중단선

이 후보는 지금 구현하지 않는다. 후속은 **이미 공개된 EEG와 독립 acquisition
측정이 같은 trial/session에 대응하고 prequery 사용이 가능한 다른 자료**를 찾는
새 유한 feasibility 라운드다. 단순히 공개 EEG 숫자만 늘리는 것은 병목을 풀지 못한다.
파일 접근·실제 열·clock·support 경계·추가 준비 비용부터 확인한 후 학습 후보를
동결한다. 이번 라운드에는 그 다음 검색·새 모델 예산을 자동으로 추가하지 않는다.
held60·외부 요청·유료 사용은 여전히 별도 승인 사항이다.

## 재현 기록

- [기계 판독 상태/접근 목록](reports/xr_timing_candidate_v1_state.json).
- 보존 XML: `/home/whwovy/research-spaces/califree-eeg-experiment-design/sources/xr_timing_2026_fulltext.xml`.
- HTTP 원문138829bytes SHA256 `258d63bd51d603c5cb59d3b588a6c0cfadc00e760c9b5ad10940f495f8b8163a`.
  저장시 말미 newline1byte 추가된 실제138830bytes SHA256
  `a474e818e8fde5eef93505629d0cfd596c19d1ba40c3c7e741fddd69d4d2e046`.
  Workspace claim에는 저장 artifact hash를 사용한다. 수집시각2026-09-11T18:22:50.454597Z.
- [공식 metadata](reports/xr_timing_europepmc_v1.json)의 SHA256
  `146e4c856e9e9d37ec0a533bf6a762abc5c86c8ef9b8680a08dafc90736002c6`.
