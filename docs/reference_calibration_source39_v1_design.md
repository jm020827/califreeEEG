# Reference-guided source39 v1 — 실행 전 고정 설계

2026-09-07. 권위는 [고정 JSON](../configs/analysis/reference_calibration_source39_v1.json).
이미 노출된 39명의 secondary development 진단이다. 새 확인 실험이나 metadata 효과 검정이 아니다.

## 쉽게 설명

기존 개인 EEG 평균끼리는 정답/오답 모두 높은 상관을 보여 학습이 잘 되지 않았다.
이번에는 **깜빡임 주파수로 만든 기준 사인파**를 함께 써서 개인 EEG에서 class에 관련된
공간 성분을 뽑는 eCCA를 검사한다. 목표인 metadata-assisted low-calibration은 유지한다.
유용한 EEG-only 기준을 먼저 확보해야 나중에 acquisition metadata의 순수 추가 효과를 물을 수 있다.

고정된 두 전처리(no_notch, causal_notch50) 각각에서 다음을 비교한다.

- A0_reference: 현재 query와 사인파만 사용한다.
- ECCA: 위 정보와 class당 1/3/5개의 앞선 labeled support EEG를 함께 사용한다.
- Wrong-label: 개인 EEG는 같지만 template의 class 이름만 11가지 순환 이동한다. 사인파의 class는 고정한다.
- ITCCA: 사인파 없이 개인 template만 사용하는 직전 방법의 별도 진단이다.
- Legacy_A0: 이전 공개 FBCCA 점수의 no_notch 역사적 anchor다. ECCA의 matched 기준으로 대체하지 않는다.

50Hz 처리는 저자 코드에서 발견한 차이를 검사한다. 원인으로 확정한 것이 아니다.
두 arm 모두 trial 시작부터 sample160+N까지 접근할 수 있고, notch arm만 앞선160 samples
(0.64초)를 활용한다. 각 trial/channel에 zero-state causal iirnotch(50Hz,Q35)+lfilter 후
sample160부터 N=125/188/250을 자른다. 미래 sample, 다른 query의 통계는 사용하지 않는다.
공통 7-band zero-phase filter는 잘린 N samples에만 적용한다. 이 차이는 추가 history와
filter dynamics도 포함하므로 물리적 50Hz 잡음의 단독 인과효과나 저자 full-epoch filtfilt의 정확한 재현이 아니다.

## 수식과 증거

[Nakanishi et al. 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703)
PDF p9 Eq14–15의 네 Pearson 항을 사용한다. query X, template Tj, 고정 사인파 Yc에 대해
CCA(X,Yc), CCA(X,Tj), CCA(Tj,Yc)의 공간필터를 구한다. 네 항은 각각
`corr(u1 X,v1 Yc)`, `corr(u2 X,u2 Tj)`, `corr(u1 X,u1 Tj)`, `corr(u3 X,u3 Tj)`다.
각 band에서 `sum(sign(r)*r²)`를 구하고 고정 weight로 선형 평균한다. 두 번 제곱하지 않는다.
Matched Aref는 같은 첫 항 r1²만 쓴다. Ridge singular value를 실제 projected Pearson 대신 쓰지 않는다.
Ridge=1e-8, ITCCA diagnostic ridge=.01. k0 ECCA/wrong은 exact Aref이며 support 항은 존재하지 않는다.
Wrong-label에서는 j=(c-shift)%12를 적용하여 필요한 template/reference 조합을 다시 계산한다.
최종 class score를 단순 roll하는 것은 이 control과 다르므로 금지한다.

공식 [wearable Readme](https://bci.med.tsinghua.edu.cn/upload/zhufangkun/Readme.pdf)와
[stimulation table](https://bci.med.tsinghua.edu.cn/upload/zhufangkun/stimulation_information.pdf)의
axes, dry/wet 순서, 250Hz, class별 frequency/phase와 sample160=(.5+.14)*250을 확인했다.
공개 loader가 float32로 바꾼 뒤 float64로 계산하므로 native precision 보존을 주장하지 않는다.
[SSVEP-DAN 저자 전처리](https://github.com/CECNL/SSVEP-DAN/blob/521cd7a46f69da1ea8c29e476fa74661cc1607ea/data_preprocessing.py#L220)는
50Hz/Q35 full710-sample filtfilt를 사용한다. 저자의 2초/3-band/query6–9 설정과 본 0.5–1초/7-band/query5–9는 다르다.
[Wong 저자 extendedCCA.m](https://github.com/edwin465/SSVEP-MSCCA-MSTRCA/blob/21bf73a023a85e73b4b0c1acd137c7d14505fad7/extendedCCA.m)의
네 항과도 교차 확인했다. 원래 2015 저자의 소프트웨어를 찾거나 논문 정확도를 재현했다는 뜻은 아니다.

문헌 범위는 targeted_fulltext와 code inspection이다. PLOS PDF SHA49a5de310a239cd174a529e520527eb25fe01c8d1972f34cae061cd857baca97,
Readme SHA3a3a3f32607145f4e7a855f6f2a8a288ab1bdba76557a13af41f8a8d9a64a9ca,
stimulation PDF SHAb1f98aec1952c1f6bbf1b168eb406250c89678493b13a795acd8caf097470075.
기존 research workspace에 문서별 카드·읽은 위치·render와 inference 경계를 저장했다.
새 검색의 반복보다 이 operator/전처리 진단이 다음 불확실성을 더 직접 해결한다.

## 고정 평가와 판정

Source39 allowlist, support blocks0–4/query5–9, 12class, dry/wet×3windows를 모두 유지한다.
참가자 순서 modulo3으로 fit26/evaluate13을 세 번 수행한다. 각 pipeline/window/method/budget의
standalone temperature만 31점 grid로 fit하고 모든 fold를 먼저 동결한다. Interface-conditioned head,
fusion lambda, band/pipeline 선택은 없다. 총4,185 objectives와7,956 metric rows다.

각 arm에서 ECCA−Aref early AUC 및 ECCA−wrong early AUC가 모두1pp 이상이고 one-sided95LCB>0이어야 한다.
k1 비악화와 dry/wet safety도 요구한다. 개별 arm의 USEFUL_EEG_CALIBRATION_DEVELOPMENT 또는
AQ_NOT_ESTABLISHED를 모두 보고한다. 전체 terminal DIAGNOSTIC_COMPLETE는 **완료 여부일 뿐 성공 판정이 아니다**.
Paired interaction `[(ECCA−Aref)notch − (ECCA−Aref)no_notch]`도 descriptive로 보고하며 winner 선택을 하지 않는다.
39명은 반복 노출됐고 fold training도 공유하므로 CI는 독립 confirmation이 아니다.

모든7bands×6cells×2arms에서 support 재현성, true-vs-wrong score, 기준 사인파 투영 에너지,
retained crop의 50Hz 투영 에너지를 보고한다. Diagnostics는 decoder input이나 선택 기준이 아니다.
80% 최초 관측 budget, 미달 >5, labeled cost=12k, retained EEG seconds=12k*N/250를 분리한다.
추가 available history=12k*.64초는 별도 보고한다. 실제 전체 보정시간 절감을 주장하지 않는다.

## 동결 API·schema·병렬 ownership

Cache schema `cfeg.reference-calibration-source.features.v1`; axes P=39,A=2,L=6,K=3,F=7,Q=60,C=12.
Header: schema,subject_ids,pipeline_ids,cell_ids,sample_counts,interface_indices,query_labels,
query_block_ids,support_block_ids,calibration_budgets,wrong_label_shifts,filter_weights.
Arrays: ecca_components[P,A,L,K,F,Q,C,4], wrong_ecca_scores[P,A,L,K,11,Q,C],
itcca_band_rho[P,A,L,K,F,Q,C], aref_band_rho[P,A,L,F,Q,C], legacy_a0_scores[P,L,Q,C],
support_correlations[P,A,L,F,C,5,5], reference_projection_fraction[P,A,L,F,10,C,C],
line50_fraction[P,A,L,10,C], raw_file_sha256[P], crop_sha256[P,A,L].
Reference projection uses all candidate references and SVD rank sigma>max(eps,sigma_max*1e-12).

Fit fold: fold_id,fit_subject_ids,evaluation_subject_ids,pipelines[pipeline].windows[N].
Each window: aref_temperature,budgets[k].ecca_temperature/itcca_temperature.
Legacy fit: legacy_a0_temperatures[N]. Temperature record: temperatures,objectives,selected_index,temperature,nll.
Rows add pipeline to prior flat8-metric schema. Wrong k>0 has null posterior hash and11 separate permutation metrics;
k0 uses exact Aref hash and0 permutations. All regular rows have1 permutation.
Summary: status,human_held_unlock,pipeline_summaries[pipeline],paired_pipeline_diagnostics.
Nested summaries retain prior spatial field names AQ_eauc_gain/AQ_k1_gain/delta_vs_A0 etc.,
but AQ=ECCA and A0=matched A0_reference; methods retain their actual names. Nested tables need no pipeline column.
Nested attainment adds history_seconds (12k*.64 for available history in both arms).
Paired diagnostic keys: calibration_gain_interaction,ECCA_eauc,A0_reference_eauc,
wrong_label_support_eauc,ITCCA_k1,ITCCA_k3,ITCCA_k5; each a prior paired-summary record.

| Lane | Owner/path | Runtime | Integration |
|---|---|---|---|
| Contract/audit/docs | Main; plan, this doc, new audit script/tests, research log | Sole SQLite writer and real study launcher | First/final |
| Producer | Isolated reference worktree; new analysis module, runner, producer tests only | Synthetic fixtures only; no human EEG/metadata | Commit reviewed then cherry-pick |
| Independent review | Read-only shared repo/worktree | No human input or writes | Contract then code then outcome review |

Base main880466f, integration main. New worktree `/home/whwovy/califreeEEG-wt-reference-source`,
branch `codex/reference-calibration-source39-v1`, after committed contract. Existing21 worktrees preserved.
Free>300GiB; checkout~12MiB +artifact1GiB +integration temporary1GiB; estimated features440MB/peakmemory2GiB.
CPU4/BLAS1. No dependencies, ports, mutable environment sharing, GPU changes, cleanup or external contact.
Main integration is sequential because producer and independent auditor share a frozen schema, not writing paths.
Real study starts only after clean integrated source and all fixtures pass. Start is exclusive0400/fsync,
then source EEG open, feature publication, all fold freezes, evaluation and result. Existing start consumes attempt.
Only exact4 artifacts under `/home/whwovy/reference-calibration-artifacts/source39-v1` are allowed.
Metadata/manifest, held60, retired1–3, old runner/seals and automatic promotion remain forbidden.
