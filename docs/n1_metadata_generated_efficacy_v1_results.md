# N1 generated metadata efficacy v1 — 결과와 후속 검증 계획

2026-09-10 KST. **등록된 인공 능력 대조는 통과했다. 실제 metadata의 저보정 효용은 여전히 미확립이다.**

유용한 외부 context를 제공한 인공 조건에서 Q 정확도49.28%가 QM65.04%로 올랐다.
같은 EEG에 정보 없는 M을 넣으면 정확도·예측은 변하지 않았다. 기존 제한된 학습 경로도
M을 사용하여 유용한 예측 변화를 만들 수 있다는 증거다. [이전 source39 부정 결과](task_trca_n1_transport_recovery_r1_results.md)를
뒤집거나, 실제 임피던스가 유용하다거나, 보정량이 줄었다고 해석하지 않는다.

[사전 설계](n1_metadata_generated_efficacy_v1_design.md),
[전체 수치 JSON](reports/n1_metadata_generated_efficacy_v1_results.json),
[현재 단계 상태](reports/n1_metadata_generated_efficacy_v1_state.json).

## 쉽게 설명하면

센서8개 중 어느4개가 정답과 경쟁하는 신호를 갖는지 알려 주는 인공 metadata를 만들었다.
지원 EEG만으로는 그 구분을 알 수 없도록 설계했다. 모델은 원래의 작은 조절 범위 안에서
문제가 있는 채널의 기여를 줄이는 방향을 학습했다. 정보 연결을 끊은 metadata로 같은 학습을
하면 그 이득이 사라졌다. **일부러 알려진 답이 있는 상황에서 학습 경로의 능력을 확인한 것**이다.

평가 문제는 두 후보 점수가 매우 가까운 낮은-margin 상황이다. 실제 EEG 전체의 대표적인
잡음·전극 접촉·임피던스 분포를 흉내 냈다는 주장은 하지 않는다. 정답을 잘 맞히는 실제
metadata를 새로 발견한 것도 아니다.

## 사전에 고정한 실행과 실제 수행

- 기작1개·연결/비연결 M2regimes·seed20260916·k3/N256·12classes/5bands/8channels.
- fit8개 pair groups=16records, evaluation32개 pair groups=64records. 두 member는
  counterfactual partners이며 같은 support와 trial 난이도를 공유한다. 독립 사람64명이 아니다.
- 각 regime 원 Q→동결→Q2/QM/SHAM_REFIT4heads×200updates, fixedλ=.001.
  생성1회·2pipelines·8heads·총1600updates·독립 감사1회·재시도0이다.
- 설계/config는 `7d0af63`, 최종 실행 코드는 `17d22c4a134253a4891358eba0ed7fb091add1a6`.
  등록은427개 source/script/test/config/design 핀과 사전 검사 기록을 결합했다.
- UTC16:41:22.044 생성 시작,16:41:57.352174 두 모델 동결,
  16:41:57.352819 query decode,16:42:07.474 평가 완료,16:42:17.817 감사 process 종료.
  프로그램 역할 분리와 기록을 검산했으며 OS sandbox/암호학적 non-access 증명은 아니다.
- 새 사람자료·기존 raw/모델/query 재접근·held60·GPU·외부 요청·유료·설치0.
  원 과학/실행 파일을 바꾸지 않았고 과거 실패·음성 결과를 보존했다.

## 정확도와 공정한 비교

각 regime는 같은3072개 query-record 평가다. 두 regime를6144개의 독립 EEG trial로 세지 않는다.
아래 정수는 감사에서 일치한32개 group 정확도로부터 환산했다.

| 방법 | 연결 M: 정답/3072 (정확도) | 비연결 M: 정답/3072 (정확도) |
|---|---:|---:|
| Q | 1514 (49.2839%) | 1514 (49.2839%) |
| QM | 1998 (65.0391%) | 1514 (49.2839%) |
| Q2: 추가 Q 잔차 | 1514 (49.2839%) | 1514 (49.2839%) |
| SHAM_REFIT | 1538 (50.0651%) | 1514 (49.2839%) |
| PERMUTED | 1508 (49.0885%) | 1514 (49.2839%) |
| STALE | 1998 (65.0391%) | 1514 (49.2839%) |
| MISSING | 1514 (49.2839%) | 1514 (49.2839%) |
| FULL_NATIVE | 1514 (49.2839%) | 1514 (49.2839%) |
| FULL_CENTERED | 1514 (49.2839%) | 1514 (49.2839%) |
| ISO | 1514 (49.2839%) | 1514 (49.2839%) |
| 사전 고정 β=[2,0,0] oracle 경로 | 2048 (66.6667%) | 1534 (49.9349%) |

여기서 oracle은 **같은 제한 안의 고정 기준**이지 가능한 모든 방법의 상한이 아니다.
비연결 regime의 같은 β 경로는 정보 없는 packet을 쓰는 기술적 비교일 뿐 정답을 아는 oracle이 아니다.

| 사전 비교 | 평균 차이 (%p) | 기술적95% 구간 (%p) |
|---|---:|---:|
| 연결 QM−Q | +15.7552 | [13.6921,17.8183] |
| 연결 QM−Q2 | +15.7552 | [13.6921,17.8183] |
| 연결 QM−SHAM_REFIT | +14.9740 | [11.9911,17.9568] |
| 연결 oracle−Q | +17.3828 | [15.2446,19.5210] |
| 비연결 QM−Q | 0 | [0,0] |
| 연결 이득−비연결 이득 | +15.7552 | [13.6921,17.8183] |

모든 유효성/양성 headroom/음성/학습 이득/작동 기준이 통과해
**GENERATED_M_CAPACITY_PASS**로 종료했다. 이미 고정한 기준·seed·λ·δ폭·oracle을
결과에 맞춰 바꾸지 않았다. Q의 두 regime content hash도 완전히 같다.

## 실제로 무엇이 학습됐나

연결 M의 평균 feature 계수는1.36653179, 최대 gradient norm은.026135217이다.
QM의 규제를 포함한 학습 목적값은.70044023→.67840692로 감소했다.
Q 대비 최대 R차이.06077636/F차이1.31254e−7/점수차이.01156246이었고,
3072개 평가에서484개 예측이 달라졌다. 정답 증가도484이므로 모두 오답→정답 변화다.
이를484개의 독립 통계적 성공으로 해석하지 않는다.

비연결 M은 예측 변화0이지만 계수·gradient·R·점수가 정확히0인 것은 아니다.
최대 계수1.66870e−7, gradient4.81211e−6, 점수차이1.41533e−10이다.
M2=0이며 packet3행이 같아 STALE=QM도 설계상 예상한 결과다.

## 독립 검산·자원·재현 정보

독립 감사는 입력 해시 확인 후 raw 파형/RNG z·v·위상·δ/직교성/S/C/Q/M/scalers/donors를
확인하고 NumPy/SciPy로 R/F·10arms와 oracle 점수·argmax·요약을 다시 계산했다.
모든10개 감사 묶음 PASS, 최대 score 차이1.33227e−14, 양성 projector 차이1.45013e−18,
분석 S 공식 차이5.00223e−12였다. 모든 artifact를 반환 직전 다시 결합했다.
Q/M helper는 공유하며 Adam/gradient를 독립 재학습한 것은 아니다. QR noise의 byte-exact
재생성 대신 직교성과 파형 불변량을 검산했다. 외부 실행·코드 핀은 supervisor의 별도 근거다.

Primary46.97198초/peak1,240,636KiB, audit9.64267초/649,620KiB로 각900/600초와8GiB
상한 이내였다. 둘 다 exit0/watchdog 없음/잔류 group 없음이며 reviewer도 지정 PID 부재를
확인했다. 실험 폴더535,484KiB, test snapshot972,892KiB; 합계 약1.44GiB로 예약 범위 안이다.
이는 지정 폴더 snapshot·child별RSS이지 전체 과거누적CPU/호스트 동시peak 측정은 아니다.

최종 통합94PASS6.21초, 전체3692PASS/3CUDA-skips/실패·오류0/67warnings467.22초.
두 test 집합은 겹치므로 독립 검증 횟수로 합산하지 않는다. 감사 unit 첫 회의 basetemp 부모
부재(6PASS/27setupERROR)는 로그와 함께 보존했고 등록 시도를 소비하지 않았다.
테스트 보수적 예산 청구1248.682초<1800초, 새7Python파일 Ruff/format/diff검사 PASS다.
등록 후 source/test 수정·추가 학습·추가 수치감사0이다.

실험 부모: `/home/whwovy/n1-generated-efficacy-v1-gyVfMl`.
검사 부모: `/home/whwovy/n1-generated-tests-v1-WE1oj6`.

| artifact | SHA-256 |
|---|---|
| registration.json | `0cbaf6c1f83811382376ecf51447880bcd3a22b4cd8379bbe58188d1ed5b69a5` |
| data/result.json | `f29ad47f3b8ac9b924972a8c2bda315bd0f68c5a3acd456e88f8cc66fc29828f` |
| data/audit.json | `5c256230f1377aa5b6b05377e4de1f9b7d741b91ee585e155cb44fe740b71d1a` |
| completion.json | `cab749a45599b4716d9d3b5a5eed70291890db7ff0182109a49eaf673b8d111f` |
| preflight.json | `b2d051c653166e11bfd03338f1021a562d558411e90a252309a75cc541b23420` |

Data12파일은0400이며 completion이 해시를 결합한다. Primary/audit 로그와 종료 영수증은
보존·0400이나 로그 bytes 자체는 completion artifact inventory 밖이다. Supervisor 자신의
빈 stdout/stderr는0664로 남는다. 모든 실행 흔적이 불변·완전 인증됐다고 주장하지 않는다.

## 무엇을 결론 내릴 수 없나

이 결과는 좁은 구성에서의 능력 증거다. 실제 M의 정보 부족이 이전 실패의 **원인이라고
확정할 수 없다**. 현실의 측정 시점, 특징 표현, 학습 목표, 신호 강도·margin, 분포 차이는
여전히 분리되지 않았다. Null pair 대칭은 일반적 false-positive 보정력을 보여 주지 않는다.
Donor가 다른 group의 M을 소비하므로 전체 생성법 아래 출력이 무조건 iid도 아니다.
구간은 고정 latent/support/donor/model 조건의 descriptive approximate group-t이며,
정확한 모집단 coverage·동시95%·다중검정 보정·training-seed 변동을 보장하지 않는다.
특히 **k3만 수행했으므로 실제 보정량 감소는 이번에 측정하지 않았다.**

## 다음 검증 계획 — 제안이며 새 사람자료 실행 아님

원 목표는 바꾸지 않는다. 작은 M 경로가 전혀 작동하지 못한다는 설명은 좁혀졌지만,
복잡한 모델·더 강한 규제를 즉시 추가할 근거는 아니다. 이번 고정 단계는 종료한다.

다음의 최소 질문은 **실제 acquisition M이 기준 Q 및 QM 잔차와 용량을 맞춘 Q2 대조군을 넘어, 별도 source block의
채널별 판별 신뢰도를 설명하는가?**이다. 이전 impedance/proxy 복원 이득을 반복 주장하는
것이 아니라 class 신호 구별과 연결된 표적을 검토한다. 예를 들면 학습용 source label로 정의한
채널별 true-class 대 competing-class margin이다. 이는 제안된 표적이며 채택·실행 전 검토가 필요하다.

현재 Q는16계수이며, Q2와 QM의 추가 잔차는 각각3계수다. 후속 단계에서도 비교 대상의
추가 학습 용량을 맞춘다. Metadata 필드·시점·권한·표적·participant 분리·Q/Q2/SHAM·최소 효과·단일
후보·예산·중단 규칙을 새로 고정해야 한다. 최종 query label을 표적 설계나 선택에 쓰지 않는다.
그 조건부 정보가 없으면 같은 M에 모델 복잡도만 올리는 방향을 종료하고, 필요한 새로운
paired acquisition 측정을 명시한다. 정보가 있으면 그 기작을 쓰는 학습 후보1개를 설계하여
고정 k별 정확도와 실제 보정 label 비용을 검증한다. 예측 가능성이 곧 인과효과/보정 절감은 아니다.

Source39는 반복 노출된 개발자료다. 새 사람자료 후보·재사용 권한은 이 generated-only 계약에서
자동 발급되지 않는다. Held60 개봉·외부 요청·유료 자원은 계속 별도 승인 사항이다.
현재 확보한 것은 **인공 능력 통과 경로**이며 **사람 보정 절감의 유망 후보 확보**와는 구분한다.

## 연구 근거와 작업 이력

`academic-research`의 ADEMP 참고는 기작·추정량·방법·평가기준을 사전에 분리하는 데 사용했다.
Morris/White/Crowther2019의 [공식 논문](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8086)은
abstract/선택 HTML 범위로만 참고했고, 실제 결과의 근거는 위 로컬 실험이다. 새 PDF0,
foundation search1회/5records/abstract card1개;429 검색 공백·cutoff2026-09-04를 유지한다.
포괄적 최신문헌 검토나 해당 논문이 본 interval을 보증한다는 주장은 하지 않는다.

Claim `claim:1c18ea450cdfb82d`는5evidence를 갖는 QUALIFIED,
technique `technique:782f50eef1938cc4`를 기록하고 원 gap `gap:6a8254f849b36b2f`는 OPEN으로 유지한다.
원 source39 negative claim도 그대로다. 관찰·기술통계와 일반화 주장을 나눠 근거 DB에 남겼다.

`coordinate-worktree-changes`에 따라 기존40trees 중2개 slot에서 disjoint writer를 운영했다.
Fixture22b9921→12f8039, root runner4f59204, auditor ea85968→17d22c4 순으로 통합했다.
문자 충돌0, root가 API/의미를 검토하고 실제 producer→auditor 경로를 검증했다.
새 worktree0·기존 branch/미추적 산출물 보존·cleanup/push/설치0이다.
**이 한정 단계 완료와 전체 연구목표 완료는 다르다. 전체 goal은 미완료다.**
