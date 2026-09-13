# Reference bank 진단의 합성 구현 완료 — 실제 EEG 효능은 미평가

2026-09-13. 디스크는약295GiB가용. 출처확인77857b4→마감/새합성계약8fe8084.
**새49검사 및 관련 최종260검사 PASS**,ruff/diffcheck PASS. 전체repo suite는실행하지않았다.
상태는`SYNTHETIC_OPERATOR_READY_ONLY`이며실제M효과나보정량절감의성공표시가아니다.

## 이번에 만든 것

[순수 operator](../src/cfeg/mamem_reference_probe_v1.py)는 단일500sample EEG-vector와
전체5class reference bank만 받아 투영 에너지 비율(단일채널 squaredCCA)을 계산한다.
다른API는1trial/class의sample event간격에서고정5Hz를만든다. 주기당2event라는
가정이며물리적주파수/뇌latency추정기가아니다. Querylabel/DIN을scorer에전달하는
인자가없다. 지원자료provenance는이함수만으로보장되지않으므로실제reader계약이필요하다.

향후채널후보E126은저자Dataset-I기본예제에서가져왔다. 그러나무필터2초·h1/h2·
직접argmax조합은우리설계다. **완전한저자CCA재현설정은여전히미확정**이다.
정확한[출처확인결과](mamem_baseline_source_v1_results.md)를덮어쓰지않는다.

## 검증된 범위

- Nominal/명시된shifted bank에서각5개합성sinusoid를해당class로분류하고자기score≈1.
- 두bank가같은합성query의score를변경하므로reference→score전달경로가작동한다.
- 별도least-squares투영식과≤1e−10일치. 상수offset/scale/sign변화와query고정/
  reference기저phase회전에불변. 실제queryphase변경시다른classscore불변을주장하지않는다.
- 전체bank의합집합에직교한합성trace는모든score≈0.
- Shape/NaN/constant/nonorthogonalbank/비정수·비증가events등invalid입력STOP,
  입력배열불변,전체5class API 및exacttie smallest-index를검사했다.

[49개 새검사](../tests/test_mamem_reference_probe_v1.py)는최초0.12초PASS,
두번째는기존211개와함께260개/1.36초PASS. 허용3회중새suite2회,
기존관련회귀suite1회;실패후threshold변경0. Phase문구만첫검사후명확화했으며
기존test도query고정/reference회전이었으므로과학조건변경이아니다.
Root가작성·실행했고별도reviewer는수학/API/출처를읽기전용으로검토했다.
테스트는합성신호이며독립적인실제EEG원자료재구축감사가아니다. 독립정적감사에서
차단오류는발견하지못했다. `reference_rank` 실패분기의직접near-zero입력검사가
없는비차단coverage한계와지원자료provenance한계를보존하며추가반복하지않는다.

## 수행하지 않은 것과 다음 단계

새실제EEG/DIN/NPZ해독0,실제정확도비교0,새gatefit0,다운로드/held60/발송/유료0.
이전v2실제80-fit 부정결과와RETIRE유지. 관련고정producer/보호문서를수정하지않았다.
현재연구goal은active이며metadata효능·보정절감은여전히미확립이다.

다음은[별도개발진단초안](mamem_reference_probe_development_next.md)의정확한계약을
고정하는것이다. S001a지원5trial에서reference를만들어S001b15query를고정2bank로
한번만비교한다는제안이며**아직실행하지않았다**. 단일채널진단통과를강한baseline,
metadata학습성공,새참가자확증으로승격하지않는다. 실제reader/CLI도아직없다.
모든run수·code/test/contract SHA와원test출력은[상태JSON](reports/mamem_reference_probe_synthetic_v1_state.json)에있다.
