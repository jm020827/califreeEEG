# Choi 보충자료 목록 확인: 다음 확인 대상을 두 파일로 좁힘

2026-09-10. **파일목록 확인 성공 / 파일 내용 미개봉 / metadata 효능 미평가.**

실제 publisher ZIP에는 **38개 항목**이 있다. 확장자·이름 규칙으로 분류한 결과 XLSX 36개와
문서·설정 형식 후보 2개다. 등록한 코드 확장자의 항목은 없었다.
이는 **설명서가 없다**거나 **export 코드가 어디에도 없다**는 결론이 아니다.
두 후보는 보충 그림이나 다른 내용일 수도 있으며 아직 이름·정확한 형식·내용을 확정하지 않았다.

출처는 [Choi 논문](https://academic.oup.com/gigascience/article/8/11/giz133/5641733)의 공개 링크가 가리키는
[publisher supplement ZIP](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/gigascience/8/11/10.1093_gigascience_giz133/4/gigascience_8_11_giz133_s18.zip)이다.
공개 링크의 일시적 query는 로그에 저장하지 않았다.

## 실제로 읽은 범위

| 요청 | 확인한 것 | application 본문 수신 |
|---|---|---:|
| Publisher HTML GET | 기존 supplement 링크 | 217,863 bytes |
| ZIP HEAD | 크기 10,814,732 bytes와 동일한 strong ETag | 0 |
| ZIP 끝 22 bytes | comment 없는 표준 ZIP footer | 22 bytes |
| 지정 central directory | 38항목의 목록 metadata | 3,921 bytes |

두 부분 GET은 실제 **HTTP206**, 정확한 Content-Range/Length와 동일 ETag를 반환했다.
ZIP 전체를 받지 않고 metadata **3,943 bytes**만 읽었다. 전체 application 본문 합은
221,806 bytes, 실행은 1회/1.344초였다. 단순 Accept-Ranges 표시만으로 성공이라고 판단하지 않았다.
파일명은 해시로 남겼고 목록 원문, ZIP comments/extra fields, 일시적 URL query는 저장하지 않았다.
Member 본문·local header 읽기, 압축해제, CRC 검증, 전체 ZIP 해시 검증은 **하지 않았다**.
네트워크 버퍼까지 포함한 패킷 수준 수신량을 계측했다는 뜻도 아니다.

## 다음에 확인할 두 후보

| 목록상 파일명 SHA-256의 앞부분 | 선언된 압축/해제 크기 | 목록상 local-header offset |
|---|---:|---:|
| `b557165bd6209f09` | 757,350 / 817,595 bytes | 1,638,324 |
| `24fe2cd55aff4a7a` | 645,869 / 729,188 bytes | 10,164,864 |

전체 해시·CRC 선언값은 [관측 JSON](reports/choi_supplement_directory_v1_observation.json)에 있다.
선언된 해제 크기는 실제 해제 후 검산값이 아니다. 이번 projection은 정확한 확장자도 남기지 않았으므로
크기만 보고 PDF/설명서라고 추정하지 않는다. 다음 별도 범위에서 **이 두 항목의 identity/형식부터
확인**하고, 문서라면 취득·export 설명을 검토한다. XLSX 36개를 여는 경로로 확대하지 않는다.

찾을 내용은 `Gyro`가 실제 어느 센서/Figure 8 신호인지, EEG와 어떻게 정렬되는지,
1,000→200 Hz 변환과 marker/run 경계를 어떻게 처리했는지다. 단순 분석 코드나 그림이 있다고
배포 파일의 생성 경로가 입증된 것은 아니다. 이 문서에서도 답이 없을 때
[미발송 질문](choi_aux_context_clarification_draft.md)을 이 두 핵심으로 좁혀 발송 승인을 요청할 수 있다.
**현재는 공개 문서 후보가 남았으므로 외부 연락만이 유일한 다음 행동은 아니다.**

## 원 연구목표와의 연결

실제 acquisition M인지, 어느 보정 시점까지 사용할 수 있는지 알아야 Q/Q2/SHAM 대비
공정한 metadata 학습 실험을 설계할 수 있다. 그러나 목록 확인 자체는 M의 정보성이나
성능·보정량 감소를 보여주지 않는다. 새 learner/사람 fit/outcome은 0이다.
[직전 범위 정정](choi_export_evidence_followup_v1_results.md)대로 원 조건은 pre-query이며,
보정과 동시 측정한 M을 query 전에 쓰는 별도 가설도 가능하다. 절대 g 단위와 매 support 직전
baseline을 모든 방법의 보편 조건으로 확대하지 않는다. 실제 수집 prefix·비용·필터 의존성은 확인해야 한다.

## 실행·검증·보존

- Config `4ac7c48`에서 범위를 먼저 동결했다. Reader `fae6108` 이후 네트워크를 1회 실행했다.
  요청 4/4, 재시도/redirect 0, directory 3,921/131,072 bytes, 항목 38/256이었다.
- Generated self-test 3호출은 각각 같은 8검사 PASS, fixture 271 bytes다. 서로 다른 24실험으로 세지 않는다.
  범위 무시/잘못된 길이·ETag/unsafe path 등의 거부를 검사했으며 전체 HTTP 보안 테스트는 아니다.
  Ruff/서식 검사 PASS, 전체 저장소 pytest는 실행하지 않았다.
- 독립 읽기전용 검토로 원 directory Base64 제거, critical header만 중복 검사, 전체100초 timer,
  고정 확장자 범주를 반영했다. 모두 실제 source GET 전에 수정했다. 기존 실행 결과를 바꾸거나
  과학적 threshold를 완화한 일이 아니다. 최종 검토 no blocker다.
- `coordinate-worktree-changes`: root만 main·연구 DB 작성, agent는 읽기전용으로 검토했다.
  새 worktree/설치/삭제/push/GPU/held60/사람파형·marker/fit/outcome/외부발송/유료 0.
  기존41trees/8untracked출력과 모든 부정 결과·이전 fixture 상한 이탈·Choi DEFER/V2 deny는 보존한다.
- `academic-research`는 실제 관측과 문서 후보라는 추론의 강도를 구분하고 다음 읽기를 좁히는 데 사용했다.
  새 scholarly 검색/PDF0이며 기존 cutoff2026-09-04를 변경하지 않았다. 최신 연구 전체 조사나 정독 완료가 아니다.

단계는 종료했지만 **전체 `계속 연구` goal은 active**다. 다음은 두 후보의 내용 근거 확인이지
같은 metadata 실험의 무제한 재시도나 인공 PASS 추가가 아니다.

근거는 qualified claim `57a71e00f90eb95b`/3evidence와 기존 OPEN gap에 저장했고,
render/SQLite quick_check를 마쳤다. 현재 상태·원본 보존 SHA는
[기계 판독 상태](reports/choi_supplement_directory_v1_state.json)를 따른다.
