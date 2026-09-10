# Choi2019 보조센서 확인 질문 — 미발송 초안

2026-09-10. **외부 요청 승인 없음 / 발송하지 않음.** 수신자·연락처도 선택하지 않았다.
다음은 DOI 10.5524/100660 공개 자료의 해석을 위한 최소 질문이다.
개인정보, 비공개 참가자 자료, 재식별 정보, 새 실험 수행은 요청하지 않는다.

Subject: Clarification of auxiliary channel units and timing in dataset 10.5524/100660

We are assessing whether the published auxiliary channels can be used as acquisition context
before SSVEP calibration trials. We would appreciate clarification, or a pointer to existing
acquisition/export documentation, on the following:

1. Does the exported `Gyro` channel correspond to the head-motion signal in Figure 8 of giz133?
   What sensor model, physical quantity, axis or combination of axes, and calibration/conversion
   were used? Figure 8 labels head movement in g, while the global `cnt/yUnit` field in the two
   S1/Day1/LOW files we inspected is `uV`. Is that field generic to the export?
2. Was `Temperature` connected in these recordings? If so, what sensor, body/environment location,
   unit and acquisition procedure does it represent? Are disconnected/missing recordings marked?
3. How were the original 1,000 Hz signals and event markers converted to the released 200 Hz files?
   Were auxiliary channels treated identically? Is code or documentation available for the
   filter, phase/time support, delay compensation, event rounding and any additional smoothing?
4. Which event does `mrk` mark (cue, flicker or another onset), and what do the filename session
   numbers represent in acquisition order? Are pre-onset blank/baseline samples available before
   every trial, including the first trial, and were the LEDs active during those intervals?

We are not requesting participant identities or new data collection. Existing documentation or
export code would be sufficient. We will distinguish published findings from unverified assumptions.

---

답변 전에는 Gyro를 각속도/보정된 가속도라고 확정하지 않는다. Temperature를 환경온도로
가정하지 않는다. 답변이 없어도 현 상태를 모름으로 남길 수 있으며, 무응답을 부정 결과로 세지 않는다.
