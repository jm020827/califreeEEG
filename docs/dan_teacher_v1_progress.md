# 마지막 DAN 후보 — 2026-09-28 진행 상태

> **최종: 07:31 KST COMPLETE_AUDITED.** 전체 학습·단회 평가·saved audit 완료.
> 후보 채택 기준 미달, metadata 보정량 감소 미확인. [최종 결과·인계](dan_teacher_v1_closeout.md).
> 아래 실행 중/준비 중 문구는 보존된 과거 기록이다.

> **최신: 실제 실험 실행 중(02:33 KST 시작).** 전체 생성 검증까지54PASS,
> [실행 전 해시 동결](reports/dan_teacher_v1_qualification.json)을 마쳤다.
> `/home/whwovy/eeg-data/dan-teacher-human-v1-1sfi54y0`에서단회실행,
> PID2620244/tool session87328이다. 09-28 02:33:50 KST 관측에서
> raw39회·metadata4680행 추출완료,
> 72fits/27,000updates 진행,query평가0이다. 자동평가·saved audit·terminal 기록이
> 학습 뒤에 이어진다. 결과효능은 아직 없다. 최신 상태는 출력폴더의`meter.json`,
> `journal.jsonl`,종료시`terminal.json`을 확인한다. 아래 미구현/미시작 내용은 이전 단계 이력이다.

**아래는 실제 실행 전 준비 단계의 기록이다.** 원 계획의 후보1개/고정 설정을 유지하며,
실행 엔진 구성요소 검증을 진행했다. [동결 계획](dan_teacher_v1_plan.md)과
[상태·남은 예산](reports/dan_teacher_v1_state.json)이 기준이다.

## 이번에 연결한 부분

- 정해진535sample prefix만 사용하는 notch/filter bank와1.5초 crop/표준화.
- 독립 수식의 target-only eTRCA와 무보정 FBCCA.
- 4source 통합 pretrain/1source 검증,각5source 개별 fine-tune/검증,
  최적 source-validation checkpoint 선택과 변환 EEG 생성.
- 원 support와 변환 source 결합 → eTRCA,5arm×3band 합성 학습 경로.
- 첫5 source/첫4 scaler-fit 참가자 경계,보정 prefix 복사,실제/섞은 M 대조군.
- 모든 예상 모델 등록 전 평가 금지,완전 동결 후 단회 query 권한.

새 검사6+1+8=15개가 모두 통과했다. 앞선 core33개와 합쳐48개 PASS이나,
이는 **생성 입력에서 각 구성요소를 검증한 총합**이지 전체 human pipeline PASS가 아니다.
독립 NumPy 파형 forward/직접 pairwise covariance/scalar correlation 및 CCA 별도
고유값 수식과도 대조했다. 실제 M 유용성/논문 성능 재현은 아직 미검증이다.

## 계산량

첫 CUDA fit의 초기화 비용까지 반복 외삽하면29.96시간이었다. 별도로 사전 기록한
단일 warm/steady 측정에서 준비0.7277초,정상상태200updates0.2364초였다.
두 배 여유와 고정12.5비율을 적용한 **학습 경로만의** 추산은11.53시간으로24시간
훈련 상한 안이다. 이는 최저 측정치를 고른 반복 탐색이 아니다. 첫 측정도 보존했다.

실제 파일 IO/전체 trace/eTRCA/전체 실행 조율/독립 감사는 포함하지 않아 전체30시간
예산 적합성이나 완료 시각을 보장하지 않는다. 모델 checkpoint 기본량 추산은약2.19GB로,
teacher/decoder/cache 등 추가 산출물까지 합쳐12GiB 한도 안인지 후속 확인이 필요하다.

## 남은 실행 준비

1. Source39만 읽는 단회 reader와 저장 산출물 hash에 결속한 전체 runner.
2. 결과·보정비용 집계 및 별도 saved-result auditor.
3. 파일 저장까지 포함한 전체 생성 fit→freeze→score→audit와 자원 검사.
4. 모두 통과하면 같은 계획으로 실제 단회 학습·평가·검산. 저성능 때문에 튜닝 재개하지 않음.

생성 예산은4/6호출,693/1200updates,11.12/1200초를 사용했다. 수정1/3round.
남은2호출/507updates 안에 통합 검증해야 하며 자동 보충하지 않는다.
실제 raw/manifest/학습/최종평가/held60/외부요청/유료 사용은 모두0이다.
현재 백그라운드 사람 실험 프로세스는 시작하지 않았다. 단계별 재승인을 요구하는 상태는
아니며, 구현·검증 미완료 상태다. 코드만 완성됐다고 human-entry를 허용하지 않는다.

이번 변경은 아직 로컬 작업이며 자동 commit/push하지 않았다. GitHub의
`research/local-state-20260928`는 앞선 core33PASS 시점 snapshot이다.
