# Choi 보충문서 정독: 온도 제외 이유 확인, motion export는 미해결

2026-09-10. **공개 문서 두 개의 표적 검토 완료. 새 사람 실험은 하지 않았다.**

## 가장 중요한 발견

`Temperature`가 README/헤더에는 있지만 최종 논문에서 확인되지 않았던 이유를 찾았다.
**저자는 온도의 임의 단위·변환 방식에 대한 심사 지적을 받은 뒤, 온도 관련 본문과 그림,
별도 보충 스프레드시트를 제외했다고 답했다.**
[저자 답변 DOCX](/home/whwovy/research-spaces/califree-eeg-experiment-design/sources/choi-supp-docs-mmcr8r/candidate-1.docx)의
XML 문단 P29–P33, 특히 저자 답변 P30/P33이 직접 근거다.
원 출처는 [Choi 논문](https://academic.oup.com/gigascience/article/8/11/giz133/5641733)에 연결된 publisher supplement다.

이것은 단순 검색 실패보다 강한 근거다. 다만 다음은 구분해야 한다.

- **확인:** 온도 결과를 논문·해당 보충자료에서 제외한 저자의 이유.
- **미확인:** 실제 센서 연결·부위·배포 `cnt`에서의 제거 여부·값의 유효성·변환식.
- **현재 판단:** 이를 보정된 체온/환경온도라고 해석하거나 vendor 계수를 임의 적용하지 않는다.
  단위 비의존 학습 자체가 불가능하다는 증명은 아니지만, 현재 바로 쓸 수 있는 검증된 M도 아니다.

## 움직임 센서에 대해서는 무엇을 얻었나

[보충그림 DOCX](/home/whwovy/research-spaces/califree-eeg-experiment-design/sources/choi-supp-docs-mmcr8r/candidate-2.docx)는
호흡·ECG·EMG 두 종류·IMU의 다섯 그림을 포함한다. 온도 그림은 없다.
문단 P31의 `rId15`가 가리키는 원본 PNG와 P32의 Figure 5 설명을 함께 확인했다.

Figure 5는 day별 mean-trial IMU 요약이며 `g` 단위와 부호 있는 축을 사용한다.
**이 그림만으로 배포 `Gyro`의 축·합성 방식·보정식·시간 참조를 확정할 수 없다.**
특히 trial의 요약값은 보정 예시를 얻기 전의 정보라는 보장이 없다.
그림에서 참가자별 수치를 옮기거나, 참가자/band 선택·threshold·새 metric을 정하지 않았다.

두 문서의 검토 범위에는 실제 1,000→200 Hz 필터·smoothing·지연 보정·marker 변환 사양이 없다.
논문 전체나 모든 공개 코드를 조사했다고 확대하지 않으며, 문서에 없는 값은 unknown으로 남긴다.

## 원래 목표와 다음 행동

원 목표는 **외부 acquisition M을 학습에 활용했을 때 Q·공통 정보보다 더 도움이 되고,
실제 필요한 보정량을 줄일 수 있는지**다. 공개 그림의 존재나 배열 이름은 이 효과의 증거가 아니다.
현재 Choi 후보의 DEFER와 이전 실제 부정 결과는 그대로다.

원 pre-query 조건과 선택적 pre-support 가설도 계속 구분한다. 보정과 함께 측정한 motion M을
보정 prefix 종료 뒤 학습에 쓰고 query 전에 모델을 고정하는 방식은 별도 설계할 수 있다.
모든 방식에 절대 g 변환이나 매 trial 직전 baseline을 요구하지 않지만,
실제 허용·수집 구간과 전처리의 참조 범위를 알아야 비용·정보 조건을 공정하게 비교할 수 있다.

**다음은 기존 cnt 한 파일의 처리 이력 존재 여부를 metadata만으로 확인하는 것이다.**
이전 저장 헤더는 `T/clab/fs/x/yUnit`의 선택 기록이지, top-level/`cnt`/attribute 이름의
빠짐없는 census가 아니다. 따라서 아직 로컬 이력이 없다고 결론 내릴 수 없다.
별도 작은 범위에서 키/attribute 이름만 조사하고, 실제 history가 있으면 필요한 설정 항목만
다음 접근 대상으로 고른다. 파형·marker 값이나 다른 참가자 자료를 자동으로 열지 않는다.

그 경로에도 근거가 없다면 [최소 확인 질문 초안](choi_export_minimal_clarification_20260910.md)을
사용할 수 있다. 질문은 ① Gyro–실제 IMU–EEG 대응, ② 실제 export/marker 처리의 두 주제로 좁혔다.
온도나 새 참가자 자료 요청을 자동으로 추가하지 않았으며 **외부 발송은 하지 않았다.**

## 읽은 범위와 실행 증거

- 첫 문서: `responses to the reviewers.docx`, 817,595 bytes, plain-text 문단 P1–P36 검토.
  Metadata/SNR 등 답변을 읽었으나 SNR 수식의 OMML과 무관한 topography TIFF는 해석하지 않았다.
- 둘째: `Supplementary Figures.docx`, 729,188 bytes, plain-text 문단 P1–P32와 Figure 5 원본 PNG 검토.
  Figures 1–4의 이미지 내용은 정독하지 않았다. 문서별 `app.xml`은 4/6 pages라고 보고하지만
  실제 pagination을 렌더링·계측한 값은 아니다. 검토 위치는 페이지가 아닌 XML 문단/figure로 기록한다.
- 실제 형식은 DOCX였다. 정적 ZIP/XML 파싱만 사용해 macro·외부 링크를 실행하지 않았다.
  PDF 스킬 부재와 Poppler fallback을 사전 고지했지만 PDF가 아니어서 그 fallback은 사용하지 않았다.
  PDF 다운로드/전체 논문 정독 완료라는 주장은 없다. 첫 문서 근거는 둘째 다운로드 전에 저장했다.
- 범위는 `07ac8a5`에서 동결했다. Identity reader `8cef350`으로 6요청을 완료했고,
  내용 범위 재검증/ZIP 전용 URL 보강 `2fe981a` 후 member 두 개를 각각 한 번 받았다.
  Strong ETag/부분 GET 범위와 길이·압축해제 상한·EOF·실제 길이·CRC를 확인했다.
- 총 **8/8 HTTP 요청**, 재시도/redirect 0. Identity 2.836초, 문서1 1.619초, 문서2 0.134초.
  Self-test는 3호출(6/6/8검사), 최종 최대 fixture 30 bytes다. Ruff PASS이며 전체 저장소 pytest는 미실행이다.
  독립 source 검토의 payload 시작점 재검증·ZIP 전용 경로 지적은 문서 내용 요청 전에 수리했다.
- 원 ZIP 전체·XLSX 36개·사람 파형/marker·fit/outcome·held60·GPU·발송·유료 접근은 0이다.
  이번에는 문서 member 2개를 읽었으므로 member 읽기 0이라고 보고하지 않는다.
  공개 논문의 그림 검토는 우리 모델의 새 사람 outcome 평가와 구분한다.
- `coordinate-worktree-changes`: root만 main/연구 DB 작성, 두 agent는 읽기전용 검토.
  새 worktree/설치/삭제/push 0, 기존 41trees/8untracked 출력·실패·이전 배열상한 이탈을 보존했다.

`academic-research`에 온도 제외 근거와 IMU 해석 한계를 각각 qualified claim으로 저장했다.
기존 Choi gap은 OPEN이며 최신 문헌 전체 조사로 부르지 않는다. 현재 넓은 연구 goal은 active다.
자세한 SHA·예산·근거 ID는 [상태 JSON](reports/choi_supplement_documents_v1_state.json)을 따른다.
