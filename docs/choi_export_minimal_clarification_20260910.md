# Choi export 최소 확인 질문 — 조건부 미발송 초안

2026-09-10. **발송하지 않았다. 외부 문의는 별도 승인이다.**
아직 기존 cnt 한 파일의 전체 키/attribute 이름 census가 남아 있으므로, 이 문의가 유일한 다음
행동이라고 보지 않는다. 공개/로컬 처리 이력으로 답을 얻으면 질문을 더 줄인다.
[이전 초안](choi_aux_context_clarification_draft.md)은 역사 기록으로 보존한다.

수신 후보는 published supplementary document의 교신저자 Han-Jeong Hwang,
`h2j@kumoh.ac.kr`이다. 문서 P18–P22의 공개 업무 연락처이며 현재 수신 가능성은 확인하지 않았다.
개인정보·새 참가자 자료·새 수집을 요청하지 않는다.

Subject: Exported Gyro and resampling/marker provenance for GigaDB 100660

Dear Professor Hwang,

We are studying whether external acquisition context can improve low-calibration SSVEP learning.
Our intended use is motion measured during permitted calibration, with the model fixed before
held-out prediction. We have read the dataset paper, its reviewer response and supplementary figures.

Could you point us to existing documentation or code for two aspects of the public export?

1. Does `cnt/Gyro` correspond to the head-mounted IMU shown in Figure 8 and Supplementary Figure 5?
   Which sensor output/component does it contain, and how is it aligned with the EEG samples?
2. What actual routine/settings produced the 1,000→200 Hz export and its markers, including any
   smoothing or delay adjustment? Which event does a marker represent, and is processing confined
   to each individual run? An existing export script would be especially helpful.

We are not requesting participant identities, unpublished participant data or new recordings.
We would keep any unresolved aspects explicitly unknown rather than infer a conversion or timing rule.

Thank you for making the dataset available.

---

위 두 항목은 motion 후보의 정체성과 허용 정보 시점을 위한 질문이다. 절대 g 변환계수·정확한
모델명은 선택한 특징에 따라 추가로 필요할 수 있지만 모든 dimensionless 특징의 보편 조건은 아니다.
Temperature는 저자 답변에서 공개 결과 제외 이유가 확인됐으므로 이번 motion 문의의 필수항목에서 뺐다.
특정 session 분할이나 pre-support 가설을 채택할 때만 해당 chronology/baseline 질문을 더한다.
이 초안의 작성은 외부 발송이나 새 사람 실험의 승인이 아니다.
