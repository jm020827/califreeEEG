# MAMEM 무지원 공통 reference EEG 진단 v1 — 실행 전 동결

2026-09-14 KST / 2026-09-13 UTC, base main69ea80d. 이전turn은실제DIN구조검사완료로
progress였다. Full잔차A→B RETIRE/M2→PSD v2RETIRE/S00112→14개발관찰보존.
원목표는 학습M의Q·공통정보이상 이득과보정부담감소이며, 이번무지원기준선확인이
그목표를대체하지않는다. 새사용자support를요구하지않는commonbaseline을정확히측정한다.

Problem signature: E126/500sample query와5classreference; 병목은nominal대공통설정의
실제분류차이; 허용은고정된개발10명B15query; feedback은pairedcorrect/perperson·classcounts;
실패는targetDIN frequency누출·posthocchannel/window선택·common개선을개인M효능으로포장.

## 정확한자료·공통bank

- S002–S011b **10MAT만**. 경로/bytes/MAT_SHA/receipt/member는기존
  `configs/governance/mamem_reference_residual_v1_inputs.json`의b행으로고정한다.
  inputspecSHA99a3adfae3765457bb23bf0305519775268007546742b6bff512d33fb3d7135b.
- 이전DIN 결과 `docs/reports/mamem_reference_residual_v1_run/result.json`
  SHA50cef4d3482975efe69d4fc5c145a95d2e4ab3822db30203576ce63dad906389,
  terminalSHA2b2eeb8e1e8b6c61447d6fcc932515962da081cc7b54ded917be4c5df89a0ff4,
  audit `docs/reports/mamem_reference_residual_v1_audit.json`
  SHAf54841c75840ae0f2b0b0d911b34c0489501eb846f315a7cff4344a90ff2c1c4.
  同runのrecord_S002b…S011b.json十個は前result.record_sha256で厳密固定。
  既存20raw/300DINの再実行ではなく保存された統計/窓/推論labelのみ再利用する。
- F[p,c,j]=前結果frequencies_hz[p,1,c,j], shape10×5×3。
  COMMON[p,c]=exp(mean(log(F[o,c,j])) for o≠p,j=0..2).
  NOMINAL=(6.66,7.5,8.57,10,12)。他9名×3repeat/classだけを用い、評価参加者pの
  A/Bのどちらもそのbankに含めない。A値は入力とせずB行だけ選ぶ。
  全10bankをmanifestに**EEG hash/header/load前**に固定。0<f<62.5、class昇順を検証。
- label/window/total_samplesは対応する前recordのhashで固定する。main15groups8..22,
  class当3contiguous、各onset+250:+750窓、trialend=start+1000などの構造を再検証。
  **現MATのDIN/rate再decode0**。rate250は以前同一MAT_SHAで検証済みの値を継承する。
  queryの保存DIN周波数はその本人のbank/scorerに渡さない。querytruthは評価専用。

## EEGアクセスと同一operator

- 各MAT全byteSHA1回、whosmat1回、loadmat1回。loadmat variable_names=[eeg]のみ。
  eeg257×T,double、Tは前recordのtotal_samplesと一致、0<T≤200000。
  samplingRate(1,1)numeric/DIN4×E cellのheaderのみ確認。原eeg全体はdecodeするが
  **Pythonrow125の15×500だけ**数値使用。他row・row257・非選択区間数値処理0。
- `src/cfeg/mamem_reference_probe_v1.py` SHA
  a147f8ac18016d521832f33dc9f45f956ac0f8163aa56df5a5da9f38f32eb7e3不変。
  E126、250Hz、2秒、h1/h2、中心化、無filter/CAR/ridge、float64。scoreは中心化した
  queryの全5class正規直交referenceへのprojection squared norm。最初のargmax。
  E126のOz mapping/最適性/完全なauthorCCA再現は主張しない。
- scorer APIにEEG・label無し窓・全5bankだけ渡す。各participantの30predictionsは
  scores_Sxxx.jsonにlabel無しで保存。全10participantのscoresを保存して
  scores_frozen.jsonを作成後、保存されたlabelsで一度だけ評価する。

## 予算・終了基準

Actual1attempt、10MAT/150EEGwindows/300predictions/1500classscores/6000projections、
learnedmodel fit0。固定2armsのみ、再試行/別channel/window/filter/係数/subject選択0。
deadline2026-09-13T16:00:00Z、parentwall180s、workerCPU120s/AS2GiB、
perfile1MiB/総run8MiB、一般artifactはlogs2MiB+terminal1MiBを予約。free8GiB以上。
BLAS/OMP1thread/GPU0。exclusivecreate・parent/workerclaim・terminal・pinsで一回性を保持。
失敗はSTOPPED_NO_RETRYとpartialartifactを保存、出力を消して再実行しない。

合成suiteはroot1+auditor1+最終統合1の合計最大3、各60wall秒。既存関連回帰は
最終統合に1回のみ。実際の保存projection監査1回/60wall/30CPU/2GiB、raw再読/fit0。
コードと監査をactual前にcommit。ネットワーク/追加文献/PDF/ダウンロード/抽出0。
S001/a/c/d/e/held60/以前のEEGfeatureNPZを開かない。

## 固定された判定と費用

各participant/arm: correct,total15,per_class_correct[5],ready=(correct≥12 and minclass≥1)。
各arm集計: correct,total150,accuracy=correct/150,per_class_correct[5],subjects_ready,
ready=(correct≥120 and subjects_ready≥7)。10人は等しいquery数なので平均も同等重み。

最終decisionは **BOTH_ZERO_SUPPORT_READY_DEV / COMMON_ONLY_READY_DEV /
NOMINAL_ONLY_READY_DEV / NEITHER_BASELINE_READY_STOP** の4通り。
二つreadyなら両方残し、commonが名目より優れると自動宣言しない。
paired全150個 both_correct/nominal_only/common_only/both_wrong、participant別正答差、
common_better/nominal_better/tied人数、common_advantage_pp=100*(C−N)/150を全報告。
平均80%・7人は事前開発運用基準であり統計的有意性/新規独立確認ではない。CI/pvalue0。

両armのtarget_labeled_support_trials=0、target_support_stimulus_seconds=0。
commonのsourceはfold当9人×15eventwindows=135、previously acquired source collection。
query_ready_elapsed/setup_secondsはUNKNOWN。試験queryの記録/分割と校正trialを混同しない。
すでに開いた10人を再使用し、foldはsourceを共有する。独立cohort・online性能ではない。
Commonが80%を満たしても metadata追加による校正削減の証拠とはならない。

## Frozen producer / auditor schema

新run=`docs/reports/mamem_common_reference_v1_run`。
Manifestキー: schema=`cfeg.mamem-common-reference-v1.manifest`,created_utc,deadline_utc,
attempts=1,scope,pins,frequencies_hz={NOMINAL:[5],COMMON:{S002..S011:[5]}}。
pins exact19relativepaths: このcontract,newpuremodule,newrunner,newauditor,immutableprobe,
inputspec,previousresult/terminal/audit,previous10record_B。前述4knownSHA+operatorを固定。

Scope exact: mat_byte_hashes10,mat_headers10,mat_eeg_loads10,din_decodes0,rate_decodes0,
eeg_numeric_windows150,eeg_numeric_channels1,predictions300,class_scores1500,
projection_scalars6000,fits0,target_support_trials0,held60_openings0。

started: status=STARTED,attempt1,parent_pid,started_utc,manifest_sha256。
worker_claim: attempt1,pid,parent_pid,started_utc,manifest_sha256。
attempt_Sxxx: subject,started_utc,mat_sha256,loaded_variables=[eeg]。
scores_Sxxx: subject,mat_sha256,source_record_sha256,completed_utc,
queries[15 {group_index,start0,end0,arms:{NOMINAL/COMMON:{projection:[5,4],scores:[5],prediction:int}}}]。
scores_frozen: created_utc,predictions300,scores_sha256={10subjects:SHA}。

result: schema=`cfeg.mamem-common-reference-v1.result`,completed_utc,scope,
scores_frozen_sha256,participants[10 {subject,queries[15上記+label,arms各+correct:bool],
summary={NOMINAL/COMMON:{correct,total15,per_class_correct[5],ready}}}],
summary={NOMINAL/COMMON:{correct,total150,accuracy,per_class_correct[5],subjects_ready,ready},
paired:{both_correct,nominal_only,common_only,both_wrong},
subject_changes:{common_better,nominal_better,tied},common_advantage_pp,decision},
cost={target_labeled_support_trials:0,target_support_stimulus_seconds:0,
source_subjects_per_fold:9,source_event_windows_per_fold:135,
source_collection:"PREVIOUSLY_ACQUIRED_NOT_TARGET_CALIBRATION",
query_ready_elapsed:"UNKNOWN",setup_seconds:"UNKNOWN"}。

terminal: statusCOMPLETE/STOPPED_NO_RETRY,attempts1,fits0,started_utc,ended_utc,
manifest_sha256,started_sha256,worker_claim_sha256(nullable beforeclaimfailure),
COMPLETEonly result_sha256,失敗時error_type/reason。stdout/stderrは各1MiB以下。

独立auditorはproducer import/rawEEG読取0。前F_B→LOPOcommonbank、保存projection→
score(1e−10tol、scorebounds1e−12)→storedscoresの厳密argmax→label/correct/count/
paired/ready/decision/cost、全pins/入力role/窓/scoresseal→評価/lifetime/hashを検算。
projection再合算topとの差≤2e−10のnear-tieは別count、これを新predとして扱わない。
以前の原DIN→分割と原EEG→projectionの独立再構成・OS I/O tracingではない。
Audit出力はrun外、exclusive、64KiB以下、actual1回、失敗も保存して再実行しない。

## 所有・統合・後続

Root: contract/puremodule `src/cfeg/mamem_common_reference_v1.py`,
runner `scripts/analysis/run_mamem_common_reference_v1.py`,
test `tests/test_mamem_common_reference_v1.py`, actualexecution/reports/SQLite。
Auditagent: **別worktreeのみ** `scripts/analysis/audit_mamem_common_reference_v1.py`と
`tests/test_mamem_common_reference_audit_v1.py`の2ファイル。その他書込禁止、合成test1回。
Read-only protocol/math agentはsharedrepo、raw/fit/write0。

既存clean auditworktree `/home/whwovy/califreeEEG-wt-reference-probe-audit-v1`を
新branch `codex/mamem-common-reference-audit-v1`で本契約commitから再使用する。
旧branch codex/mamem-reference-probe-audit-v1とcommit08281b0は保持する。
checkout36MiB、追加≤64MiB、依存はroot環境read-only、別testtmp、port/GPU/DB使用0。
Free293GiB/既存46worktree/8untrackedを維持し、install/symlink/cleanup0。
契約→rootproducer+隔離auditor並列→rootdiff/semantic review→統合test→commit→freeze→actual。

結果によらずここでこの比較を終了する。両baseline不十分なら条件をいじって救済しない。
Readyでも個人M学習や校正削減成功として昇格しない。次の学習候補は時間的対応と
追加情報の別機作根拠がある場合のみ一つ、新予算・Q/Q2/QM/SHAM/common0/directM対照で
定義する。held60/外部人への依頼/有料資源は別承認。全体goalは未達ならactiveのまま。
