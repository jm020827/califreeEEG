# Choi export 후속 조사: 목표는 유지, 사용 시점은 구분

2026-09-10. **공식 근거 일부 추가 / 현재 pre-support 후보 DEFER 유지 / 새 사람 실험 0.**

## 이번에 달라진 판단

원 목표는 **적은 SSVEP 보정자료로 학습할 때 외부 측정정보 M이 EEG 품질 Q보다 추가로 도움이
되는가**다. M이 반드시 보정 EEG보다 먼저 측정되어야 하는 것은 아니다.
첫 query 전에 쓸 수 있는 M이라는 원 조건과, 직전 움직임으로 다음 보정 예시의 신뢰성을
예측한다는 최근의 더 좁은 가설을 구분한다.

| 설계 가지 | M을 얻는 시점·기작 | 현재 판단 |
|---|---|---|
| 최근 pre-support 제안 | 보정 예시 직전 움직임이 다음 EEG의 상태에 지속적으로 영향을 주는지 | 기존 DEFER 유지. 선행 창과 시간 처리 근거가 필요 |
| support-synchronous 대안 | 보정 EEG와 함께 측정한 움직임으로 그 보정자료의 학습 기여를 조절 | 원 목표에 부합하는 별도 제안. 아직 설계·실행 계약 아님 |

두 번째도 **보정 수집을 마친 뒤 모델을 고정하고 query를 평가**해야 한다. 모든 EEG/M 및
필터의 참조 범위가 실제 허용·수집한 보정 prefix 안에 있어야 한다. 전체 40 trial을 처리한 뒤
그중 4개만 골랐다면, 보정 수집량을 4라고 줄여 보고할 수 없다. 미확정 upstream 미래 참조도
cutoff 이름을 바꾼다고 허용되지 않는다. Q/QM/Q2/SHAM은 같은 정보·비용 조건으로 비교한다.

이 구분은 [원 설계](metadata_calibration_efficiency_design.md)의 pre-query 목표와
support-M-only/query-M-only 진단 경로에 근거한다. 현재의 class 내 정규화 가중치만 바꾸는
기작은 여전히 **class당 예시가 1개면 변화가 없다**. One-shot 개선을 주장하려면 다른 기작이 필요하다.
아직 유망 후보를 확보하거나 calibration 절감을 새로 입증한 것이 아니다.

## 공식 문서에서 확인한 것

- [GigaDB 공식 등록](https://gigadb.org/gigadb/api/dataset/get_dataset/?dataset_id=100660)의
  GitHub 링크 필드는 비어 있다. [전체 파일 목록](https://gigadb.org/gigadb/api/dataset/list_dataset_files/?dataset_id=100660&per_page=1000&start=1)은
  TAR, README, questionnaire CSV의 3개다. 이것만으로 코드가 어디에도 없다고 결론 내리지 않는다.
- [논문 HTML](https://academic.oup.com/gigascience/article/8/11/giz133/5641733)은 MATLAB R2013b,
  200 Hz export와 band별 session 번호 의미를 보고한다. 이것이 resample 함수·필터·marker 사건
  시점을 확정하지는 않는다. Table 1도 파일/주파수 목록이지 변환 사양이 아니다.
- 같은 논문의 보충 설명에 head-motion 요약 파일과 별도 publisher ZIP 링크가 있다.
  HEAD에서 크기 **10,814,732 bytes**, range 지원 표시를 확인했다. **ZIP 본문/목록/표 값은 읽지 않았다.**
  논문에 적힌 파일명은 archive에서 확인한 파일명이 아니며, hosting 수정일도 연구자료 개정일로 보지 않는다.
- [현재 공식 센서 사양](https://www.brainproducts.com/solutions/sensors/)은 전압으로 출력되는
  가속도 센서를 설명한다. 따라서 export의 `uV`와 그림의 `g`는 원리상 공존 가능하다.
  **Choi의 실제 모델·변환식이 확인됐다는 뜻은 아니며 vendor 계수를 가져다 쓰지 않는다.**
  [센서 분석 안내](https://pressrelease.brainproducts.com/sensor-data-analysis/)도 1D와 3D 출력을
  구분하지만, 2020년 게시·2026년 갱신된 현재 제품 안내로 2019년 Choi 구성을 특정할 수 없다.

여전히 모르는 것은 `Gyro`의 정확한 대응·축·변환, Temperature의 연결·부위, marker의 사건 시점,
배포 전 1,000→200 Hz 변환의 시간 참조다. 절대 g 단위가 불필요한 특징을 별도로 설계할 수는
있지만, 정규화가 채널 정체성이나 시간 정렬을 증명하지는 않는다.

## 다음 행동과 중단 기준

다음은 새 learner를 만드는 것이 아니라 **찾아낸 publisher ZIP의 파일 목록을 확인하는 별도 유한 단계**다.
설명서·export 코드가 있는지 먼저 확인한다. 현재 계약은 archive 본문을 금지하므로 이번 조사에서
ZIP을 더 읽거나 상한을 늘리지 않았다. 다음 계약은 작은 footer/central-directory 전송과
member 값 열람을 구분하고, 서버가 Range를 무시하거나 형식·크기 제한을 넘으면 중단해야 한다.
참가자 spreadsheet/EEG 값, 코드 실행, 임의 추출은 포함하지 않는다.

README/코드 확장자도 참가자 값을 포함할 수 있으므로, 목록 확인 후 본문 접근 범위를 따로 판단한다.
확인 가능한 공개 경로가 남았으므로 지금 저자 연락만이 유일한 해결책이라고 하지 않는다.
직접 근거를 얻지 못하면 unknown/DEFER를 유지하고 [질문 초안](choi_aux_context_clarification_draft.md)의
필요 항목을 좁힌다. 외부 발송·held60·유료 자원은 계속 별도 승인이다.

## 조사·보존 기록

사전 [config](../configs/analysis/choi_export_evidence_followup_v1.json)를 `7b9118a`에서 동결했다.
표적 검색 3/3, 논리적 문서 대상 9/10(root 5, scout 4), 실패 대상 재시도 1/4에서 종료했다.
이는 전체 HTTP 호출 수가 아니다. 같은 문서의 표/선택 구간과 scout 근거를 root가 재확인했다.
Vendor 페이지 최초 오류는 같은 대상 재시도로 해결했다. ZIP HEAD는 HTTP200 헤더를 반환했지만
2 MiB size guard로 **curl63**이었으므로 성공한 archive 다운로드라고 기록하지 않는다.

선택 HTML/API/vendor 근거만 검토했다. PDF 정독·전체 최신 문헌 조사 완료가 아니다.
새 scholarly CLI 검색/PDF/사람 파형·marker/fit/outcome/데이터셋 다운로드/GPU/발송/유료 0이다.
`academic-research`는 공식 사실·추론·다음 검증을 분리해 저장하는 데 사용했다.
`coordinate-worktree-changes`에 따라 두 agent는 읽기전용, root만 main/연구 DB를 작성했다.
새 worktree/설치/삭제/push 0이며 기존 41 trees와 8 untracked 출력을 보존했다.

학술 workspace에 qualified claim `eb3a685143b50510`/근거 5개를 추가하고 기존 Choi gap은
OPEN으로 갱신했다. Render와 SQLite quick_check를 마쳤다. 전체 기존 DB·갱신 문서까지 포함한
보수적 산출물 snapshot은 약 8.4 MB로 32 MiB 이내다. 새 코드 변경이나 pytest 실행은 없다.

옛 유한 루프 완료와 모든 부정 결과, 직전 인공검사의 배열상한 이탈, Choi DEFER/V2 deny는
바꾸지 않았다. 현재 넓은 `계속 연구` goal은 active다. 이 단계는 근거·범위 검토이지 효능 실험이 아니다.

추출 범위·출처·실패는 [선택 근거 JSON](reports/choi_export_evidence_followup_v1_sources.json),
상태·SHA는 [결과 JSON](reports/choi_export_evidence_followup_v1_state.json)을 따른다.
