# v2 실제 결과 독립 검산 — 실행 전 범위

2026-09-13, source효능 결과를 보기 전 작성. `audit_mamem_source_v2_results.py`는
productionengine을 import하거나회귀를다시fit/solve하지 않는다. 실제시도COMPLETE인
경우에만 고정v2캐시의저장모델JSON·summary·ledger를읽어결정식을독립재계산한다.

검사:10people/20fold/80고유fit·160STARTED/COMPLETE,9source/8priorcontributors,
45source rows/fold, source-only scaler/intercept와ridge normal equation,
800λ scalars, SHAM900rows의jointvector/동일조건cycle·classcentering·evalM일치,
2100saved-score argmax와정확도, 같은query·zero-shot, participantbootstrap/1pp·2pp
판정, support-prefix단위·조건부setup budget·query-ready elapsed UNKNOWN.

원MAT/featurecache/EEG/Q/M 재계산0,fit0,network0. 원신호→PSD/oracle/queryscore의
독립재구성은범위밖이며 audit PASS를그근거로쓰지않는다. Gate계수/λ의변조를잡는
6개생성검사는수동constantmodelfixture이며회귀fit0이다. 결과경로는v2정확한1개,
result<=16MiB,consoleJSON<=64KiB,root가마감14:00전60wall/2GiBAS/30CPU상한으로
한번실행하고표준출력을감사기록으로보존한다. Audit실패시효능승격하지않고원인을기록;
모델재학습/새parameter/새data제외로검사를통과시키지않는다.
