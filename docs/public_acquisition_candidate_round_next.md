# 다음 공개 acquisition 정보원 탐색 — DRAFT / 미실행

2026-09-14 KST / 2026-09-13 UTC. MMV 공개 페이지는 보존했지만 inventory/schema는
미확인으로 종료했다. Eye-BCI의 기존 익명 파일403도 유지한다. 현재 새 학습 적격 후보는 없다.
이 문서는 다음 별도 유한 탐색의 제안이며, 종료한 MMV 수집의 남은 슬롯 전용이 아니다.

## 원 질문과 새 탐색 이유

찾아야 할 것은 단순히 더 많은 EEG가 아니라, **저보정 SSVEP의 source/support EEG와
대응되는 외부 acquisition 측정값이 공개돼 있는 자료**다. Q와 이미 알려진 공통 설정을
제외하고도 support template/공간 필터/시간 reference의 불확실성을 설명할 수 있어야 한다.

이미 조사한 논문·실패 주소를 반복하기보다, 접촉 상태·실제 움직임·광학 timing처럼
측정 기작을 명시한 새로운 공개 release/파일 설명이 있는지 확인한다. 이 구분은
데이터가 존재한다는 주장이 아니라 다음 탐색의 선정 기준이다.

## 제안하는 다음 유한 예산

실행 전에 새 계약·signature를 동결하고 academic workspace에서 기존 coverage와
중복/실패 주소를 먼저 확인한다. 다음은 아직 실행하지 않았다.

- 탐색 2개: 접촉/획득 신뢰도와 SSVEP pairing, 독립 movement/optical timing과 SSVEP
  pairing. 각 검색의 정확한 query·branch·이유·source/date 범위를 실행 전에 기록한다.
  제목/abstract만으로 공개 데이터·M 효능을 단정하지 않는다.
- 새 primary official dataset/paper/repository documentation 최대 4개. 기존 보류 주소,
  generic homepage 반복, 사설 API 추측, 계정/DUA/저자 요청/유료는 제외한다.
  공개 링크에서도 실패/요구 조건을 남기고 재시도하지 않는다.
- 문서/metadata만 다루며 raw EEG·사람별 annotation·파일 header·PDF·fit·outcome·held60은 0.
  Source failures와 실제 접근 범위를 기록하고, 검색 결과 수를 정독 수로 세지 않는다.
- 가장 구체적인 후보 **최대 하나만** 다음 exact-release/schema 단계에 추천한다.
  없으면 NONE으로 종료하고 어떤 측정 조건이 부족한지 보고한다. 성공할 때까지 무한한
  query·dataset·모델 탐색을 여는 계획이 아니다.

## 반드시 구별할 항목

1. 실측 acquisition 값인지, EEG-derived Q인지, nominal 공통 사양인지.
2. 동일 사람/세션/블록의 pairing과 support 시점 가용성인지, query 정답이나 미래 정보인지.
3. 실제 field·단위·생성 출처·clock 근거가 있는지, 논문에서 장치를 썼다는 언급뿐인지.
4. 물리적 기작의 후보인지, 모든 사람에게 상수인 dataset ID/장치 이름의 재명명인지.
5. 데이터 공개 선언인지, 실제 파일 접근·버전·라이선스·역할을 검증한 것인지.
6. Few-shot label 수뿐 아니라 버린 support와 장치 setup/재보정까지 비용을 셀 수 있는지.

예를 들어 nominal electrode 좌표가 공통 정보라면 그 자체를 새 M 처리군으로 올리지 않는다.
시선/생리 신호를 접촉 품질로 바꾸어 부르거나 photodiode가 있다는 말만으로 query 입력을
허용하지 않는다. 후보는 새로운 기작·정보권한 근거가 있어야 하며 기존 실패의 계수/채널/
창/참가자 변경 구제와 구분한다.

어떤 후보도 곧바로 학습시키지 않는다. 정확한 최소 공개 파일/schema와 학습에 실제로
작동하는 경로를 검증한 다음 Q/Q2/QM/SHAM/common0/directM·참가자 분리·지원 비용·
악화·중단 기준을 갖춘 별도 학습 예산을 정한다. 전체 목표는 아직 active다.
