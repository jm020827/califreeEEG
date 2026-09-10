# 새 후보: 머리 움직임으로 one-shot SSVEP 보정 학습 보조

> 최신: [336회 실제 학습·평가 종료 결과](mobilebci_prior_efficacy_v1_results.md).
> 전체 16명 적격 확인·raw 특징 추출·Q/Q2/QM/SHAM 비교를 완료했다. M은 계산을 바꿨지만
> 실용 이득·보정량 절감은 없어 현재 후보를 종료했다. 아래는 후보 선정 당시의 제안/상태다.

> 2026-09-11 후속 상태: [공개 pair 확보·시간 검사 결과](mobilebci_public_pair_results.md).
> 두 파일43.4MB와 채널/rate 검증은 성공했다. Event/t reader는 첫 파일 내부 dtype에서 중단해
> 시간 대응은 미확정이며, 생성 파서 수리가 다음이다. 최종 논문과 초기 MAT 배포본을 구분한다.
> 아래는9월10일 후보 선택 시점의 설계/제안이며 미확보 표시는 당시 상태다.

2026-09-10. 사용자의 “공개 자료를 구하기 힘들면 놔두고 다른 후보” 지시에 따른 전환.
**Choi는 보류 기록으로 남기고, 저자 답변을 기다리는 경로에서 제외한다.**
이번 상태는 후보 선택·설계이며 데이터 적격 PASS나 실제 성능 결과가 아니다.

## 쉽게 말하면

기존 질문은 “전극 임피던스가 적은 EEG 예시로 학습하는 데 도움이 되는가?”였다.
새 질문은 **“보정 예시를 기록할 때 측정한 머리 움직임이, 그 예시를 이용한 공간필터 학습을
EEG만 볼 때보다 개선하는가?”**다. 여기서 공간필터는 여러 전극의 신호를 섞는 가중치다.

걸으면서 측정해도 과제는 여전히 세 가지 SSVEP 자극 중 하나를 맞히는 것이다.
보행 속도 분류·손상 EEG 복구·OOD 탐지로 연구목표를 바꾸지 않는다.
M은 보정 구간에서만 얻고 첫 query 전에 모델을 고정한다. Query 때 IMU를 계속 넣는
멀티모달 분류기가 아니라 **metadata-assisted calibration learning**을 시험한다.

## 후보 세 개의 선택

| 후보 | 결정 | 이유 |
| --- | --- | --- |
| Kwak2017 보행 SSVEP | 제외 | 공식 Data Availability가 IRB 제한·저자 요청을 요구한다. 사용자 지시대로 추적 종료 |
| Lee2021 명령 보행 속도만 M으로 사용 | primary에서 제외 | 속도와 session 순서가 대응한다. Q에도 알려진 조건을 QM만 독점하면 추가 측정 효과가 아니다 |
| Lee2021 실제 head-IMU + one-shot 공간회귀 | **접근·schema 사전검증 대상으로 선택** | 실제 움직임은 같은 속도 안에서도 달라질 수 있는 외부 측정. 공개 paired 파일 목록이 있으며 k1에서 정의되는 연산자로 시험 가능 |

[Kwak2017](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0172578)의 공개 제한을
다른 연구의 제한으로 확대하지 않는다. Lee2021에서 IMU가 Q보다 유용하다는 결론은 아직 없다.

## 사용할 자료와 확인 수준

[Lee2021 논문](https://www.nature.com/articles/s41597-021-01094-4)은 EEG와 머리/양 발목 IMU,
동시 trigger, 같은 자극을 여러 고정 순서의 보행 조건에서 기록한 설계를 보고한다.
읽기전용 검토에서 IMU128Hz/EEG500Hz와 raw/100Hz 전처리 export 구분을 확인했다.
원문 전체 정독·축/단위 표·실제 clock drift 검증 완료로 취급하지 않는다.

- [Figshare v1 API](https://api.figshare.com/v2/articles/13604078/versions/1): 현재 조회 성공,
  CC BY4.0 선언,445파일/8,953,349,317bytes/SSVEP191파일. `scalp`와 `IMU` 파일 쌍이 있다.
- 이 버전 설명은 **18명**이다. 최종 논문의24명/SSVEP23명과 동일한 release라고 가정하지 않는다.
  별도 버전으로 고정하고, 중복 참가자로 보이는 OSF/NEMAR를 독립 표본처럼 합치지 않는다.
- 샘플 목록: `s01_IMU_SSVEP_0.0.mat`10,489,800B +
  `s01_scalp_SSVEP_0.0.mat`32,937,882B. API의 크기/MD5는 선언값이며 로컬 검증값이 아니다.
- IMU 파일의 공개 다운로드 URL에 대한 HEAD는403/body0이었다. **GET 성공·파일 확보는 미확인**이다.
  논문/저자 code의 OSF 경로도 이번 환경에서 접근 오류가 있었다. 공개 선언과 현재 다운로드 가능성을 구분한다.
- 기존 조사에서 MOABB/NEMAR가 IMU를 EEG로 취급하는 문제를 이미 기록했다.
  이름상 EEG73채널이라는 derivative를 그대로 primary 입력으로 사용하지 않는다.

근거/실패/선택 API projection은 [출처 기록](reports/alternative_metadata_candidate_triage_v1_sources.json)에 있다.
캡차·인증 우회, 새 참가자 자료 요청, 개인정보·유료 접근은 하지 않는다.

## 반증할 기작 가설

**H-M:** 동일한 보정 EEG의 Q와 속도·순서 등 공통 정보를 알고도, 보정 중 head-IMU 특징은
이후 SSVEP 학습에 적절한 공간 규제 행렬을 더 잘 예측한다. 그 prior가 적은 보정량에서
정확도와 실제 목표 정확도 도달 비용을 개선할 수 있다.

1. Source 참가자의 허용 학습 자료로 Q→공간 prior를 학습한다.
2. Q 경로를 고정하고 실제 head-IMU 특징으로 설명되는 작은 잔차만 별도로 학습한다.
3. 새 참가자의 허용 보정 prefix에서 Q와 M을 계산하고 공간필터를 학습한다.
4. Query 이전에 모든 모델을 고정한다. Query IMU/미래 EEG로 prior를 갱신하지 않는다.

M 후보는 raw 축/단위가 확인된 **머리 gyroscope의 회전량 크기 요약**을 우선한다.
축별 해석·절대 물리량 환산·magnetometer 활용은 자동으로 추가하지 않는다.
단위 비의존 특징이라도 실제 채널·clock 대응은 필요하다. 모든 class의 허용 support를 함께
요약해 query class-specific 머리 방향이나 cue를 입력으로 전달하지 않는다.
단순히 몸을 많이 움직였다는 사실이 EEG 품질 저하나 예측 개선을 보장하지는 않는다.

## k=1에서 실제로 작동하는 삽입 위치

TRCA의 반복 쌍 S나 정규화된 단일 예시 가중치는 k1에서 유용한 조절을 못 하므로 사용하지 않는다.
새 후보는 **reference 기반 ridge 공간회귀**다. 각 class c의 허용 support를 Xc,
공통 자극의 sine/cosine reference를 Yc라 두고 다음을 푼다.

```text
(Xc Xcᵀ + λ R) Wc = Xc Ycᵀ
Q:  R = R(Q, common)
QM: R = R(Q, common, measured head motion)
```

R은 양의 정부호이며 양쪽 λ·trace 예산을 맞춘다. Query에서는 Wc로 얻은 출력과 해당
reference의 일치를 점수화한다. 정확한 centering/scoring/수치 floor는 실제 실행 전에 고정한다.
공통 배율이나 가역 변환으로 변화가 상쇄되는지, k1에서 W뿐 아니라 상대 class score까지
달라지는지 생성 검사로 먼저 확인한다. “matrix를 바꿨다”만으로 학습 효과를 주장하지 않는다.

Covariance prior 자체는 기존 검토에도 있었다. 새 발견·최초 one-shot이라고 부르지 않는다.
이번의 별도 후보성은 **임피던스와 다른 측정 + 조건 내 정보 + 단일 trial에서 정의되는
reference 회귀 경로**다. 기존 diagonal-R/pair-S 실패와 negative 결과를 대체하지 않는다.

## 공정한 실험

- Q/Q2/QM/SHAM 모두 속도, 순서/period, channel/task grid, mask, k와 동일한 support를 받는다.
  Subject/session/file ID 자체는 예측 특징으로 주지 않는다. 공통 순서 covariate와 보안상 ID는 구분한다.
- Q는 EEG 잔차의 공간 covariance·진폭·대역 정보 등 충분한 품질 입력을 가진다.
  Q2는 M 없이 추가 적합 용량을 맞춘 대조다. Pooled prior와 단순 조건별 scalar 규제도 비교한다.
- SHAM은 같은 속도/순서·비슷한 보정 비용 안에서 **실제 측정 M의 대응을 끊고 다시 학습**한다.
  Speed만 같은 값끼리 바꾸는 무작동 SHAM을 쓰지 않는다. 실제 consumed feature 변경을 검사한다.
- 참가자 단위 분리·source-only 표준화/모델 선택. 같은 날/속도별 run을 독립 참가자로 세지 않는다.
  속도가 순서와 엮였으므로 속도의 인과효과를 주장하지 않는다.
- k=1/2/3/5는 class당 최소 보정 예시 수다. Chronological prefix가 이를 만족할 때까지 실제
  수집한 모든 trial·경과 시간을 비용에 포함한다. 무작위 class 순서에서 비용을 무조건3k로 계산하지 않는다.
- 고정 query와 무보정/충분한 보정 기준을 유지한다. 정확도뿐 아니라 목표 도달 여부·미도달·harm을 보고한다.
  실제 IMU 장착 시간이 없으면 별도 unknown으로 남겨 “전체 설치시간 절감”을 주장하지 않는다.

## 예산과 중단 기준

이번 triage는 [고정 범위](../configs/analysis/alternative_metadata_candidate_triage_v1.json)의3후보 안에서 종료한다.
Raw 다운로드/숫자 접근/fit/outcome0이며, 사람 실험 실행 예산을 소진했다고 하지 않는다.

**다음 acquisition/schema 단계의 제안 예산:** 공개 EEG–IMU 한 쌍만, decoded 파일 합계64MiB,
정상 공개 GET 최대2파일 각1시도, checksum+schema 검증1회, 새worktree/외부요청/유료/held60은0.
Head가 실패했으므로 GET까지 실패한다고 미리 단정하지 않지만, 또 비공개 권한을 요구하면
이 접근 경로를 종료한다. 축·단위·trigger 대응이나 사전 시간 의존성을 공개 파일/문서로
해결하지 못해도 저자 문의 대기로 되돌리지 않고 후보를 내려놓는다.
이 제안은 아직 실행하지 않았으며 raw 접근 전에 exact URLs/size/hash와 읽을 필드를 별도 고정한다.

적격이면 **사람 효능 후보는1개**, 참가자 분리3outer/2inner, 사전λ3개 이내,
Q/Q2/QM/SHAM×k4의 prior-fit 상한336회(outer48+inner288), CPU시간1시간·출력2GiB를
초기 실험계획 상한으로 삼는다. 데이터/정확한 연산자가 아직 미확인이라 이 숫자는 실행 계약이 아니다.
이를 자동으로 늘리거나 결과 뒤에 feature·하위집단을 바꾸지 않는다. 인위적 예시의 PASS는
사람 성능 근거가 아니며, 실제 Q/Q2/SHAM 이상의 실용 이득과 보정 비용 감소가 없으면 종료한다.
실용 효과의 수치 기준과 uncertainty/harm 판정은 첫 outcome 전에 고정한다.

## 현재 결론

**Choi 문의는 더 기다리지 않는다. Lee2021 실제 head-IMU 후보를 다음 작은 공개 접근 검증 대상으로
선택했다.** 공개 파일 목록을 확인한 것과 새 데이터 확보·실험 성공은 다르다.
이번에 사용한 `academic-research`는 속도만 쓰는 허술한 비교와 공개 배포본 혼동을 걸러냈고,
`coordinate-worktree-changes`에 따라 root만 문서/DB를 작성하고 두 agent는 읽기전용으로 검토했다.
