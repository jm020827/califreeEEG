# Source-inspired E126 reference probe v1 — 실제 개발 실행 전 계약

2026-09-13, base d985291/main. 이전 turn은출처확인·합성49/관련260검사로progress.
원저보정SSVEP acquisition-M 목표 유지. 이는완전한authorCCA미확정판정을뒤집는재현이
아니라reference정의의진단이다. 이새계약만이번실제실행범위를정하며이전계약을재개하지않는다.

Problem signature: 단일500sample EEG와전체5class sinusoid bank; 병목은짧은창에서
기록된지원주파수와nominal주파수의검출차이; 허용은고정S001a지원과S001b평가,
목적은기본reference민감도검증,feedback은두arm의15query점수·정답과비용이다.
실패모드는queryDIN정답주파수를reference로주거나한채널성공을M학습효능으로승격하는것이다.

## 고정 자료와 정보 경계

- A: `/home/whwovy/data/mamem_i_v1_20260913/development_first.mat`,137357437bytes,
  SHA256 `57a72c3fde0ff3bc9aaae10721cd7cb450bda63eb5299a96f41704ea696ad10a`.
  Role `development_role.json` SHA `dc26f85099ed667208ec7b992430d257d01f2a4aa727b7b98f8a5195f6d9af18`.
  S001a,header EEG257×117917,DIN4×1966. **A decode는DIN_1/samplingRate만**;
  A EEG는header shape/전체파일hash검사외에decode/수치처리0.
- B: `/home/whwovy/data/mamem_recorded_event_source_v2/S001b.mat`,137631703bytes,
  SHA256 `3beaafa44f7717c690915e7ed85dfa720d96e7da89c7af59113305686c7c9a03`.
  Receipt `S001b_input.json` SHA `4d9f3ab103dd5fba24dd6bb3653c4bb2aaa6f7bd4f791e1a5b01bfa5ee380b59`.
  S001b/member EEG-SSVEP-Part1/S001b.mat. B decode EEG/DIN_1/samplingRate 각한번,
  EEG257×T,0<T≤200000,DIN4×E,0<E≤10000. B전체EEG가decode되지만수치는
  Pythonrow125의15개500sample창만사용. 다른행·257번째행·제외구간수치사용0.
- 두파일각loadmat1회,whosmat1회;hash검사는전체byte읽기이나EEG수치접근과구분한다.
  SamplingRate는numeric scalar250만허용. 기존v2parser/hash고정,23group검증과
  처음8adaptation제외,main15/3perclass,window onset0+250:onset0+750 유지.
- A에서각class첫maintrial하나만선택. 해당창안DINsample index≥4개로
  f=250*(n−1)/(2*(last−first)),한주기2events가정. 모든5개값을class-order로고정하고
  **B가열리기전에bank완성**. A의다른repeat는parser적격성/label검증외feature0.
- B DIN은segmentation/inferredlabel평가에만사용;queryfrequency/M2/descriptor추출0.
  Pure scorer에는한채널EEG와전체5bank만전달. B truth는전체30scores계산이후평가에사용.
  DIN-derived label은독립groundtruth나실제광학주파수검증이아니다.

## 단일 operator와 예산

- operator`src/cfeg/mamem_reference_probe_v1.py` SHA
  `a147f8ac18016d521832f33dc9f45f956ac0f8163aa56df5a5da9f38f32eb7e3`변경0.
  E126(Python125)·250Hz·2초·h1/h2·float64·centering·무filter/CAR/ridge/fit.
  Nominal=(6.66,7.5,8.57,10,12),SAMPLE_SUPPORT=위A5frequencies,두arm고정.
- 단회actualattempt1,2banks×15query=30predictions(150classscores),새gatefit0,
  grid/sweep/parameterrescue/retry0. 실제deadline2026-09-13T15:00:00Z;
  parentwall120s,workerCPU90s/addressspace2GiB/filecap128KiB,총runartifact≤1MiB.
  Free공간reserve8GiB,BLAS/OMP각1thread,GPU0. Artifact는모두exclusivecreate,
  시작/workerclaim/terminal/pins/hash로단회수명검증. 실패해도원출력보존·재실행0.
- 합성integration/canary는실제자료전최대3suite,각60wall초,합성fit0;
  관련기존회귀는최종통합1회/60wall초. 새supportreader와queryscorer·모든핀·auditor를
  실제결과전commit한다. Actualsavedscalar감사1회/60wall/30CPU/2GiB,rawrefit0.
- 다운로드/새문헌검색/PDF/외부사람요청/유료0. S002–S011/c/d/e/held60와기존feature
  NPZ/기존outcomes수치읽기0. v2negative/보호문서는必要시hash만확인한다.

## 비용과 사전 판단

지원5trial의selectedstimulus_duration=25s,analyzed_support_duration=10s,
selected_support_elapsed_prefix=max(selected trial_end0)/250을별도로보고한다.
23group사후적격성검증에는A전체DIN이필요하므로그prefix를실제onboarding시간이라
주장하지않는다. A기록duration=117917/250도함께기록하며query_ready_elapsed/setup는
UNKNOWN. Nominaldecoder자체는support불필요하나이번두arm비교는동일paid-development
collection위에서실행한것이다. 지원주파수arm을zero-calibration이라고부르지않는다.

각arm≥12/15정답이고각class≥1/3일때만그개발기록의`SIGNAL_PRESENT_DEV`로표시한다.
둘다미달이면`NO_ARM_PASSES_STOP`으로probe종료,좋아지는설정까지변경하지않는다.
하나라도통과하면`REFERENCE_DIAGNOSTIC_COMPLETE`일뿐,강한baseline복원/M학습효능/
보정절감성공아니다. Nominal만통과하면지원bank는이진단에서채택하지않는다.
Sample만통과하거나개선되면공통reference후보로기록한다(새M효능예산자동승격0).
두armcounts·classcounts·pairedcorrect/incorrect전이를전부보고,CI/pvalue는이15개
상관trial/개발1인에서확증으로쓰지않는다. 이전v2 RETIRE는어느결과에서도보존한다.

## 동결 출력 계약 — producer와독립auditor 공통

Run=`docs/reports/mamem_reference_probe_development_v1_run/`.
`manifest.json`: schema=`cfeg.mamem-reference-probe-dev-v1.manifest`,created_utc,
deadline_utc,subject=`S001`,attempt_budget=1,fits=0,queries=15,predictions=30,
mat_sha256={a,b},code_sha256={계약/runner/operator/events_v1/events_v2/auditor각relativepath:SHA}.
`started.json`: status=STARTED,attempt=1,parent_pid,started_utc,manifest_sha256.
`worker_claim.json`: attempt=1,pid,parent_pid,started_utc,manifest_sha256.
`terminal.json`: status=COMPLETE 또는STOPPED_NO_RETRY,attempts=1,fits=0,started_utc,
ended_utc,COMPLETE때result_sha256과predictions=30. 실패시error_type/reason보존.

`result.json` schema=`cfeg.mamem-reference-probe-dev-v1.result`,subject=S001,
status=DEVELOPMENT_DIAGNOSTIC_NOT_EFFICACY,mat_sha256={a,b},completed_utc,
sampling_rate_hz=250,channel_index=125,window_samples=500,fits=0,
frequencies_hz={NOMINAL:[5],SAMPLE_SUPPORT:[5]},
support=[5records class순:label,group_index,start0,end0,trial_end0,event_samples:[int]],
queries=[15records time순:group_index,start0,end0,label,
arms={NOMINAL/SAMPLE_SUPPORT:{projection:[5,4],scores:[5],prediction:int,correct:bool}}],
summary={각arm:{correct:int,total:15,per_class_correct:[5],signal_present_dev:bool},
paired:{both_correct,nominal_only,sample_only,both_wrong},decision:위결정문자열},
cost={selected_trials:5,selected_stimulus_seconds:25,analyzed_support_seconds:10,
selected_support_elapsed_prefix_seconds:float,full_support_record_seconds:471.668,
query_ready_elapsed:"UNKNOWN",setup_seconds:"UNKNOWN"},
scope={a_loaded_variables:[DIN_1,samplingRate],b_loaded_variables:[eeg,DIN_1,samplingRate],
a_eeg_numeric_windows:0,b_eeg_numeric_windows:15,b_eeg_numeric_channels:1,
query_metadata_extractions:0,gate_fits:0,source_cohort_reads:0,held60_openings:0},
access_stages=[順序a_loaded,support_bank_frozen,b_loaded,all_scores_frozen,evaluation_done].

projection=U_j.T@(centeredquery/||centeredquery||),raw500vectors는저장하지않는다。
Auditor는producer import없이保存projection sumsq→scores→argmax→accuracy/classcounts/
pairedtransition→decision,지원event式→frequency·cost,範囲·pins·start/claim/terminal/
result bindingを検査する。これは元EEG→projectionを独立再構築した監査ではない。
Float tolerance1e−10,score bounds1e−12。全て未観測実結果前に実装する。

## 所有/隔離/統合

Trackedcheckout34,691,933bytes、rootvenv5.2GiB、free294GiB。監査writinglaneには
新worktree1個、追加≤128MiB(code/test/cache)、dependencyinstall0、元環境read-only
利用(新symlink0)。既存45worktrees/8untracked保護。根writerと同tree同時writer0。

| Lane | Owned paths | Runtime | Integration |
| --- | --- | --- | --- |
| Root/main | 本契約、scripts/analysis/run_mamem_reference_probe_v1.py、tests/test_mamem_reference_probe_runner_v1.py、reports/log/SQLite | 唯一actualrunner/SQLitewriter | 契約→producer→audit→全体test→actual |
| Audit isolated | scripts/analysis/audit_mamem_reference_probe_v1.py、tests/test_mamem_reference_probe_audit_v1.py | 合成testのみ、実データ0、別worktreetmp | commitをrootreview後cherry-pick |
| Protocol/math reviews | 書込みなし、sharedrepo | raw/fit0、staticreview | actual前gate |

新worktree `/home/whwovy/califreeEEG-wt-reference-probe-audit-v1`,branch
`codex/mamem-reference-probe-audit-v1`。依存契約変更はrootだけ、変更時agentへ通知。
全test/実際終端確認はroot。実結果後の再fit/設定変更/既存worktree削除はしない。
