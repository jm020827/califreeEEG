# Calibration-Efficient SSVEP EEG Decoding

> 2026-09-14 KST / 2026-09-13 UTC **Pre-Gelled 공식 원문 확보·설계 교훈 반영**. 용량290GiB 재확인 후 정확한 IEEE PDF를1GET/HTTP200/3.61MB로 확보해 [표적 정독](docs/pregel_public_fulltext_v1_results.md)했다. 실제 임피던스 측정과 SSVEP 하드웨어 비교는 확인했지만 공개 paired원자료·학습M효능·보정절감은 미확인이다. [추가 설계 규칙](docs/acquisition_contact_measurement_addendum.md)에 측정시점/조정전후/공통조건/총비용 분리를 반영했다. 기존 후보 종료와 held60 봉인 유지, 새raw/fit/발송/유료0. 아래는 시점별 이력이다.

> 2026-09-14 KST / 2026-09-13 UTC **공개 acquisition 탐색 round4 종료: 새 적격 후보 0개**. [결과](docs/public_acquisition_round4_results.md): 검색2회에서 서로 다른 DOI24개를 검토했지만 공식 원문4곳 모두 도구 접근 오류여서 공개 실측 필드/schema를 확인하지 못했다. 데이터가 비공개이거나 metadata가 무용하다는 증거는 아니다. 색인 초록의 Pre-Gelled 전극 연구를 [단일 원문 독서 후보](docs/pregel_public_fulltext_next.md)로만 등록했다. 독립 metadata 검토 PASS, 새 raw/PDF/fit/held60/발송/유료0. 아래는 시점별 이력이다.

> 2026-09-14 KST / 2026-09-13 UTC **최신: MMV 공개 페이지 확보, 파일/schema는 보류**. [결과](docs/mmv_public_capture_v1_results.md): exact landing을 HTTP200/785,384bytes로 보존했고 JSON-LD의 PUBLIC·V3·CC BY 4.0 선언을 확인했다. Canonical 영문 주소는 원주소로302·MIME없음이라 2/4요청에서 사전 중단했다. **실제 파일쌍·acquisition 필드·학습 적격성을 확보한 것은 아니다.** 독립 저장 문서 감사 PASS, 기존 실패·held60 보호 유지; raw/PDF/fit/EEG예측0. [다음 초안](docs/public_acquisition_candidate_round_next.md)은 동일 보류 주소 재시도가 아닌 다른 공개 실측 acquisition 정보원의 제한된 탐색이며 미실행이다. 원 연구목표는 미완료/active, 아래는 시점별 이력이다.

> 2026-09-14 KST / 2026-09-13 UTC **최신: 공통 reference의 개선 확인, 고정 baseline 비교 종료**. [결과](docs/mamem_common_reference_v1_results.md): 같은 개발 10명·150 EEG창에서 명목 32.67%→source 공통 58.00%(+25.33%p), 8명 개선·2명 악화였다. 준비 인원은 0→3명으로 사전 80%·7명 기준에 두 arm 모두 미달, **NEITHER_BASELINE_READY_STOP**이다. 1attempt/300predictions/0 learned-model fits, 독립 저장 수치 감사·최종 관련428tests PASS. **공통 정보의 효용이지 개인 M 학습·보정량 감소의 증거는 아니다.** 이전 부정 결과·held60 보호 유지. [다음 초안](docs/mamem_post_common_reference_next.md)은 새 공개 paired 정보원의 inventory/schema 자격 확인이며 미실행이다. 전체 연구목표는 미완료/active, 아래는 시점별 이력이다.

> 2026-09-13 UTC **최신: 공통값 대비 full잔차 전달 경로 종료**. 용량293GiB 재확인 뒤 기존10명×2기록의 DIN300창을 단회 분석했다. [결과](docs/mamem_reference_residual_v1_results.md): 반복변동을 넘는 차이는 관측됐지만, A잔차를 B에 그대로 전달하면 공통값보다 event-frequency MSE가 **87.58% 증가**, 개선3/10명으로 사전기준 미통과였다. 새 EEG decode/분류/학습0; 저장 산술 검산PASS, 새22+기존305 관련검사 통과. **모든metadata무효나 EEG정확도87.58%하락이란 뜻은 아니다.** 기존v2RETIRE·S001개발관찰·held60보호 유지. [다음 메모](docs/mamem_post_residual_decision_next.md)는 무지원공통bank EEG기준선 확인과 새학습후보 진입조건이며미실행이다. 아래는시점별이력이다.

> 2026-09-13 **최신: S001 실제 reference 진단 12/15 → 14/15**. [결과](docs/mamem_reference_probe_development_v1_results.md): 같은15query에서 지원 기록 bank가 추가2개를 맞혔고 새오답은0이었다. 1attempt/30predictions/0fit, 독립 저장 수치 감사·관련366tests PASS. **개발1인의 reference 비교이며 학습된M 효과·보정량 감소는 미입증**이다. Nominal도 무지원80% 관문을 통과했고, 지원5trial의선택prefix445.94초와전체기록471.668초를 실제준비시간으로바꾸지않는다. v2RETIRE/held60보호 유지. [다음 초안](docs/mamem_reference_residual_learning_next.md)은 공통reference와잔여M정보 분리이며미실행이다. 아래는시점별이력이다.

> 2026-09-13 **최신: 용량295GiB 재확인·reference 진단 합성 구현 완료**. [출처 확인](docs/mamem_baseline_source_v1_results.md)에서 MAMEM-I의 완전한 저자 CCA 설정은 미확정으로 종료했다. 별도 단일채널 reference-sensitivity [합성 구현](docs/mamem_reference_probe_synthetic_v1_results.md)은 새49/관련260검사·독립 정적감사를 통과했다. 이번 새 실제 EEG 평가·학습0, metadata 효능·보정량 감소는 미확립이며 v2 부정 결과 유지. [다음 개발 진단](docs/mamem_reference_probe_development_next.md)은 별도 실제 계약을 고정하기 전인 미실행 초안이다. 아래는 시점별 이력이다.

> 2026-09-13 **최신: 개발용15창 신호 진단 완료, 공통 baseline 점검 필요**. [S001 결과](docs/mamem_signal_validity_v1_results.md): timestamp/sample 간격 계산은 서로 부합하지만 일부 nominal reference와 크게 어긋났고, 기존 CCA는 시간순열 후에도 최대 score 평균0.604였다. 실제15창/0학습, 저장 산술·절차 감사 PASS. 이는 주파수 수정의 효능·metadata 이득이나 물리적 오류를 입증한 것은 아니다. 아래 v2 80-fit 부정 결과 유지. [다음 계획](docs/mamem_common_baseline_qualification_next.md)은 단일 공통 baseline의 출처 확인과 support-only reference 비교이며 미실행이다.

> 2026-09-13 **최신: 실제 MAMEM metadata 학습 80회 완료, 후보 종료**. [v2 결과](docs/mamem_recorded_event_source_v2_results.md): 10명 one-shot Q/Q2/QM/SHAM 평균 모두 25.33%, metadata 추가 이득과 보정량 감소 기준 미통과. M은 혼합 계수·일부 예측을 바꿨으나 정확도를 높이지 못했다. [독립 저장 산술 감사](docs/reports/mamem_recorded_event_source_v2_audit.json)와 관련 272tests PASS. 실제 query-ready 시간 절감은 미확인이다. 여유 약 296GiB, 기존 실패·held60 보호 유지. [다음 초안](docs/mamem_signal_validity_next.md)은 개발용 S001a의 시간축/reference/기본 검출 진단이며 아직 미실행이다. 아래의 “현재” 표기는 해당 과거 시점의 기록이다.

> 2026-09-13 **현재: 실제256행 입력과 one-shot M 생성 학습 관문 완료**. [S001 고정2초 입력](docs/mamem_i_256row_adapter_v1_results.md)은 전체 decode 후 선택256×500만 QC해 finite/nonconstant를 확인했다. [새 후보의12-fit 생성 결과](docs/recorded_event_shrinkage_generated_v1_results.md)는 M→PSD template 혼합→score 작동과 null/schedule/SHAM 대조27개를 통과했다. 관련147tests 및 재fit 없는 독립 수치 감사 PASS. **실제 강한Q 이상의 M 효능·보정량 감소는 아직 미검증**이며 기존 부정 결과/held60 보호를 유지한다. [다음 실제 source 연결 초안](docs/mamem_recorded_event_source_learning_v1_draft.md)은 아직 실행하지 않았다. 아래는 이력이다.

> 2026-09-13 **현재: 용량 확보·MAMEM scalar 관문 완료**. 가용 약 323GB를 재확인했고, S001a에서 samplingRate 하나만 단회 확인해 250Hz/MATCH를 얻었다. [결과](docs/mamem_i_scalar_adapter_v1_results.md), [상태](docs/reports/mamem_i_scalar_adapter_v1_state.json). 관련 77tests PASS. MOABB는 마지막 행을 명시적으로 stim으로 처리한다. 이를 우리 EEG 입력에 포함하지 않도록 하되 원본 row257의 물리적 의미를 확인한 척하지 않는다. 미정의 DIN descriptor는 읽지 않았고 새 EEG파형/fit/held60/발송/유료 0. [다음 초안](docs/mamem_i_256row_adapter_v1_draft.md)은 공통 256행 입력 격리이며 아직 미실행이다. 아래는 시점별 이력이다.

> 2026-09-13 **현재: MAMEM 시간 metadata의 해석 관문 완료**. [결과](docs/mamem_i_timing_identifiability_v1_results.md), [상태](docs/reports/mamem_i_timing_identifiability_v1_state.json). 배포PDF로S001–S011 namespace를확인했으며S013동일인alias/257번째행은미확정이다. Cedrus의ST100–EGI polarity공지와3개생성상황검사로DIN만의국소clock일치≠physicaljitter검증을구분했다. `기록된 event 규칙성`의추가예측가치는열려있고새fit/EEG값/held60/발송/유료0이다. 관련43tests·독립생성검산PASS,PDFqueue-before-download이탈은보존했다. [후속초안](docs/mamem_i_recorded_event_metadata_v1_draft.md)은채널/이벤트정의와boundedS001metadata확인부터진행한다. 아래는시점별이력이다.

> 2026-09-13 **현재: MAMEM I 원자료6.59GB 확보·첫 DIN 구조 점검 완료**. [결과](docs/mamem_i_acquisition_v1_results.md), [상태](docs/reports/mamem_i_acquisition_v1_state.json). 공개v1 압축본2개의 checksum을 검산하고48MAT/파일명11ID 목록을 확인했다. S001a 한 기록만 추출·검산했다. 첫72이벤트의 국소 timestamp↔sample 일관성을 확인했지만 physical jitter·M 효능·보정절감은 미검증이다. S001 전체는개발용, EEG파형/새fit/held60/발송/유료0. S011↔S013 파일명 차이와257행 channel mapping을 임의 수정하지 않는다. [다음 관문 초안](docs/mamem_i_timing_identifiability_v1_draft.md)은 공통 보정·격자·기록오차와 잔여M을 구분한다. 아래는 시점별 이력이다.

> 2026-09-12 **현재: 공개 provenance 일부 해소, 실제 학습은 아직 미실행**. [후속 결과](docs/public_provenance_resolution_v1_results.md), [상태](docs/reports/public_provenance_resolution_v1_state.json). MMV DataCite 일반GET200으로 V3·CC BY4.0·정확한 ScienceDB 주소를 확인했다(실제 파일 접근/validity는 미확인). MAMEM 단일 상자 실험의 DIN이 광센서+StimTracker에서 왔다는 저자 보고도 확보했다. Query DIN 누출은 여전하며 DatasetII/.flash/정확한 파일 연결은 별도다. 다음 우선 확인은 MAMEM I 실제 DIN–EEG schema 연결과 support-only 잔여 timing 정보다. 새 raw/fit/held60/외부요청/유료0, 기존 부정 결과 보존. 아래는 각 시점의 이력이다.

> 2026-09-12 **현재: MMV·MAMEM 병렬 preflight 완료, 두 경로 보류**. [결과](docs/paired_metadata_parallel_preflight_v1_results.md), [학습 전 관문](docs/paired_metadata_learning_gate_v1.md). MMV loader의 stream gap은 광학 validity/EEG 품질의 증거가 아니며, MAMEM I loader는 DIN 이벤트 간격에서 label을 계산한다. 실제 MMV 배포·offline validity와 MAMEM 이벤트 생성 출처는 미확인이다. One-shot 정규화 가중평균 상쇄·고정 순서 누출·추가 장치 보정비용을 설계에 반영했다. 읽기 전용3agent+root 단독 통합, 새 raw/fit/held60/요청/유료0. 새 공개 provenance 없이 동일 실패 경로나 모델 튜닝으로 자동 연장하지 않는다. 아래는 시점별 이력이다.

> 2026-09-12 **현재: 공개 acquisition-M 2차 조사 종료, 학습 가능 후보 미확정**. [결과·한계](docs/public_paired_acquisition_round2_v1_results.md), [상태](docs/reports/public_paired_acquisition_round2_v1_state.json). MMV의 보정구간 원래 tracking-validity와 MAMEM I/II의 flash 출처·정답누출 여부를 다음 확인 대상으로 좁혔다. 장시간 PERCLOS를 M로 대체하지 않으며 SpiralE는 raw 요청 방식으로 보류한다. [MMV 정확한 배포 metadata 확인 초안](docs/mmv_public_metadata_preflight_v1_draft.md)은 미실행이다. 새 raw EEG/fit/held60/발송/유료0; 시간 목표 초과·기존 수치행 우발 검색을 기록해 전체 절차 PASS로 표시하지 않았다. 원목표와 기존 부정 결과는 유지한다. 아래는 시점별 이력이다.

> 2026-09-12 **최신: 다른 acquisition-M 후보의 공개자료 조사 종료, 새 사람 실험 없음**. [Eye-BCI](docs/post_gyro_candidate_feasibility_v1_results.md)는 정확한 EEG–Tobii 파일쌍 목록까지 확인했으나 익명 파일 GET403으로 보류했다. [XR timing 원문 검토](docs/xr_timing_candidate_feasibility_v1_results.md)는 reference 보정의 기작을 확인했지만 공개 paired raw 경로·support-only 조건을 확보하지 못해 실행 후보로 넘기지 않았다. 단순 보정과 학습된 M의 추가 이득을 분리한다. 아래 gyro 부정 결과·held60 보호 유지, 새로운 raw EEG/fit/외부 요청/유료0. 후속은 다른 공개 paired acquisition 측정의 새 유한 feasibility 계획이며 이번 소진 예산의 자동 연장은 없다. 아래는 시점별 이력이다.

> 2026-09-11 **최신: SAFE 실제 EEG 144회 완료, gyro source-routing 후보 종료**. [결과·해석](docs/block_scaled_router_human_v1_results.md), [상태v8](docs/reports/mobilebci_research_progress_v8_state.json). M가 gate·예측에 실제로 작동했지만 저보정 Q68.77%/QM68.61%, Q2보다 미우위, 보정 trial 절감0%였다. 원 저보정 목표는 유지하되 같은16명에서 이 구조의 optimizer 재시도는 중단한다. held60·비공개 요청·유료 사용 없음. 아래는 각 시점의 이력이다.

> 2026-09-11 **실제형상 생성통합24회 완료**: [결과·한계](docs/block_scaled_router_integration_v1_results.md). 유익한인공M에서SAFE Q4.17%→QM94.44%,독립M조건에서는이득없음으로수치관문을통과했다. 기존학습기의양성QM100%와SAFE의축소수락0회도보존하므로새학습법우월성은아니다. 원역할보고덮어쓰기결함은원본보존후96행사후재구성으로보완했다. 실제gyro효능·보정절감미확립,새사람fit0. [후속사람검증초안](docs/block_scaled_router_human_v1_draft.md)은실행전이다.

> 2026-09-11 **수치 진단12회 완료·위상/metadata 설계 재검토**: [수치 결과](docs/router_saturation_numerical_v1_results.md), [개념 설명·후보 우선순위](docs/ssvep_phase_metadata_design_review_20260911.md). 인공적으로 섞기가 최선인 문제에서59열 합산은self100%로 포화됐고블록평균은알려진최적값근처로수렴했다. 실제gyro효능이나사람실험원인의확정은아니다. 원저보정목표·기존부정결과유지,새사람fit0. 측정조건을반영하는작은적응경로를우선하고Neural ODE는보류한다. 아래는시점별이력이다.

> 2026-09-11 **Source-borrowing 실제144회 학습 종료**: [결과·포화 해석](docs/mobilebci_source_borrowing_human_v1_results.md). Q/Q2/QM/SHAM 저보정 정확도 모두67.85%, 보정trial절감0%였다. 모델은사실상self만선택했고M는내부logit을바꿨지만최종예측을바꾸지못했다.32/144fits의학습loss악화도보존하므로metadata정보자체가없다고결론내리지않는다. 이번후보예산종료·자동재시도0·held60보호, 아래는이전단계이력이다.

> 2026-09-11 **새 source-borrowing 기작 합성 관문 통과**: [결과·문헌 재검토·다음 계획](docs/mobilebci_source_borrowing_generated_v1_results.md). 단회4fits/400updates와 독립57항목 검산을 완료했다. 인공적으로 유익하게 만든 M에서 Q33.33%→QM100%, 엇갈린 M50%였다. **실제 gyro 효능·보정량 감소 증거는 아니며 새 사람 학습0**이다. [144-fit 사람 검증 초안](docs/mobilebci_source_borrowing_human_v1_draft.md)은 아직 실행 전이다. 직전336-fit 부정 결과·Choi보류·held60보호를 유지한다.

> 2026-09-11 **실제 336회 metadata 학습·평가 완료, 현재 후보 종료**: [결과와 후속 판단](docs/mobilebci_prior_efficacy_v1_results.md). 16명에서 Q 66.52% / QM 66.56%, Q2·SHAM 미우위, 보정 trial 절감 0%. M 자체가 prior/점수를 바꾸는 경로는 확인했지만 실용 이득은 없었다. 독립 검산 PASS, 336/336 예산 종료. 기존 부정 결과·Choi 보류·held60 보호는 유지하며 아래는 시점별 이력이다.

> 2026-09-11 공개 paired 코호트16명·48run·96파일(2.20GB) 확보 완료: [당시 검증 결과](docs/mobilebci_cohort_acquisition_v1_results.md). 아래 기록의 미평가·다음 단계는 해당 시점의 상태다.

> 2026-09-11 **공개 MobileBCI EEG–IMU 한 쌍 확보 성공, 시간 파서 수리 필요**: [실제 관측·다음 실행](docs/mobilebci_public_pair_results.md). 두 파일43.4MB의 크기/MD5·머리 gyro 채널을 확인했다. 이어진 event/t 검사는 첫 파일의 미지원 내부 형식에서 중단됐으며 자동 재시도0이다. 시간 불일치나 가설 실패로 해석하지 않는다. 최종 논문과 MAT 배포본의 구간 차이도 발견했다. 다음은 합성 MAT로 파서 수리 후 유한 시간 대응 검증, 새 파형 분석·효능 평가는0이다. 아래는 시점별 이력이다.

> 2026-09-10 **사용자 지시에 따라 Choi 문의 경로를 보류하고 새 후보 선택**: [Lee2021 head-IMU one-shot 후보](docs/mobilebci_head_motion_candidate_v1.md). 속도·순서는 Q에도 주고 실제 머리 움직임의 추가 정보만 시험한다. Public Figshare 파일 목록은 확인했지만18명 버전/최종논문 차이와 HEAD403을 남겼으며 raw 확보·효능0이다. 다음은 작은 공개 파일 한 쌍의 접근/schema 검증이다. 또 사적 접근이 필요하면 기다리지 않고 제외한다.

> 2026-09-10 **Choi 한 파일 구조 점검 종료**: [결과·재개 조건](docs/choi_header_census_v1_results.md). Root/cnt의 전체 직접 키·attribute 이름에 명시적인 sensor/export 처리 이력은 없었다. 조사 범위에 한정된 결과이며 파일 전체의 부재 증명은 아니다. 움직임 후보 DEFER·기존 부정 결과 유지, 새 효능 실험0. 다음은 센서–EEG 대응과 실제 export/marker 처리의 두 주제에 대한 저자 확인을 제안하며 외부 발송은 별도 승인이다. 아래는 시점별 이력이다.

> 2026-09-10 **Choi 보충문서에서 온도 제외 이유 확인**: [정독 결과·다음 행동](docs/choi_supplement_documents_v1_results.md). 저자는 온도 단위/스케일 심사 지적 뒤 온도 관련 공개 결과를 제외했다고 설명했다. 센서 미연결이나 raw 채널 삭제로 단정하지 않는다. IMU 보충그림의 g 표시는 확인했지만 실제 Gyro 대응·export 시간 처리는 미확정이다. 다음은 기존 cnt의 처리 이력 이름만 확인하는 작은 metadata census이며 새 사람 실험은 없다.

> 2026-09-10 **Choi 보충 ZIP 목록 확인**: [결과·다음 두 후보](docs/choi_supplement_directory_v1_results.md). 실제 부분 GET으로 38항목(XLSX36/문서·설정 형식 후보2)을 확인했다. Member 내용은 열지 않았고 문서의 정체·내용은 아직 미확정이다. 다음은 이 두 후보의 취득/export 설명 확인이며 새 효능 실험은 없다. 기존 부정 결과·Choi DEFER·held60 보호를 유지하고 전체 연구는 계속한다.

> 2026-09-10 **Choi 공식 export 후속 조사**: [결과·설계 범위 정정](docs/choi_export_evidence_followup_v1_results.md). 원 목표는 pre-query이며, 보정 EEG와 동시 측정한 M을 첫 query 전 학습에 쓰는 별도 설계도 가능하다. 최근 pre-support 후보의 DEFER는 유지한다. 공식 목록·vendor 안내로 실제 센서/시간 처리를 확정하지 못했지만 publisher 보충 ZIP 경로를 찾았다. 다음은 별도 유한 파일목록 조사이며 ZIP 본문·사람 실험은 이번에 0이다. 현재 넓은 연구 goal은 active, 아래 기록은 시점별 이력이다.

> 2026-09-10 **후속 시간경계·실제 수집비용 구현**: [결과와 한계](docs/acquisition_boundary_generated_v1_results.md). 별도 generated-only 경로의 최종34tests와 독립 검토를 마쳤다. 금지된 미래/다른 clock·trial의 혼입과 수집 전 baseline 비용 누락을 검사한다. 기존 회귀 fixture의 배열상한 이탈은 기록했으므로 전체 예산 준수 PASS는 아니다. 사람 학습·효능 평가0, Choi 후보 DEFER 유지. 현재 새 `계속 연구` goal은 active이며 아래의 완료는 이전 유한 루프의 상태다.

> 2026-09-10 **후속 Choi 보조센서 검토 완료 — 새 실험은 보류**: [근거·쉬운 설명·다음 설계](docs/choi_aux_context_feasibility_v1_results.md). 논문은 실제 머리 IMU 기록을 보고하지만 `Gyro`의 물리량/변환·사전 시간경계는 미확정이다. 온도도 연결·부위가 확인되지 않았다. 기존 전체-run 양방향 전처리와 선택 label 수를 실제 수집 비용으로 보는 해석을 새 설계에서 분리한다. 아래 유한 루프 완료/부정 결과는 유지하며 새 학습·outcome·held60 접근0이다. 확인 질문은 미발송 초안이다.

> 2026-09-10 **유한 연구 루프 통합 완료 — 실제 metadata 효용은 미확립**: [전체 결과·요구사항별 완료 감사·후속 검증 계획](docs/metadata_research_loop_closeout_20260910.md). 원 후보2개와 별도 승인된 N1/복구/인공 능력/채널 진단 단계가 모두 종료됐다. N1-R는 공정한 실제 정확도·관측 보정량 비교를 완료했고 절감0이었다. 따라서 연구 작업의 완료와 양성 가설 입증을 구분한다. 시간·compute 상한 전부 소진이나 metadata 일반의 무용성을 주장하지 않는다. 원본 JSON21개 hash 일치·독립 읽기전용 검토, 신규 실험0; 기존 실패와 held60 보호 유지. 아래 ‘전체 goal 미완료’는 당시 단계의 기록이며 현재 전체 상태는 이 통합 보고를 따른다.

> 2026-09-10 **실제 source39 채널 판별력 probe 종료**: [결과·후속 판단](docs/source39_channel_margin_probe_v1_results.md). 승인된120ridge fits/독립120재구성/단회 실행을 마쳤고 감사PASS다. QM의 예측 MSE는 Q보다0.197% 증가, SHAM보다0.598% 감소로 사전2% 기준 미달이다. Q2보다13.13% 낮지만 Q2 자체가 Q보다 나빴으므로 metadata 이득으로 승격하지 않는다. 유효성 관문은 모두 통과했으며 이 한정 경로를 부정 결과로 종료한다. 최종 관련486testsPASS, held60/기존query6–9/외부요청/유료/GPU0. 정확도·보정 비용은 이번 단계에서 미측정이며 원 저보정 목표는 미완료다. 아래는 과거 단계 이력이다.

> 2026-09-10 **후속 기작 진단 설계·표적 구현 완료**: [source39 채널 판별력 probe](docs/source39_channel_margin_probe_v1_design.md), [구현/권한 상태](docs/reports/source39_channel_margin_probe_v1_state.json). Metadata가 별도 source block의 상대 채널 판별력을 Q·비선형 Q2·SHAM 이상으로 예측하는지, k3/1초·참가자 분리·최대120ridge fits·유한 중단 규칙을 정했다. 순수 표적/관련 회귀검사165PASS이며 **실제 데이터 읽기·새 학습0**이다. Reader/전체 학습·감사는 아직 미구현, source39 재사용 범위 승인 전이다. 정식 조건부 독립 검정이나 보정 절감 실험으로 부르지 않는다. 원 목표·이전 부정 결과·held60 보호는 유지한다. 아래는 단계별 이력이다.

> 2026-09-10 **인공 metadata 능력 대조 완료**: [결과와 후속 검증 계획](docs/n1_metadata_generated_efficacy_v1_results.md). 사전 고정한 생성1회·1600updates·독립 감사1회를 완료했다. 유용한 M 조건에서 Q49.28%→QM65.04%(+15.76%p), 비연결 M에서는 예측·정확도 변화0이다. **제한된 학습 경로의 인공 능력 증거이며 실제 EEG 효능·보정량 절감 증거는 아니다.** 이전 source39 부정 결과를 유지한다. 전체3692PASS/3CUDA-skips, 새 사람자료·held60·GPU·외부요청·유료0. 이번 단계는 종료, 전체 연구목표는 미완료이며 다음 실제 M의 조건부 정보 검증은 제안 상태다. 아래는 이전 단계 이력이다.

> 2026-09-10 **복구 실험·독립 검산 완료, metadata 이득 미확립**: [결과와 후속 판단](docs/task_trca_n1_transport_recovery_r1_results.md). 출력 연결 오류를 고친 뒤 같은 과학설정으로24,000updates·전체 평가를 완료했다. QM3−Q3는0%p, QM5−Q5는−0.006677%p(정답1개 감소),312개 조건의 보정량은 전부 같아 절감0이다. Metadata는 학습·필터·점수에 반영됐지만 유용한 예측 변화로 이어지지 않았다. 이번 후보는 유효한 개발자료상 부정 결과로 종료하며, 원 연구목표는 미완료다. 이전8,800update 실패 보존·held60/외부요청/유료/GPU0. [복구·쉬운 실험 설명](docs/task_trca_n1_transport_recovery_r1.md). 아래는 이전 단계 이력이다.

> 2026-09-09 **새 N1 source39 실행 실패·감사 완료**: [결과·원인·후속 복구 제안](docs/task_trca_n1_source39_v1_results.md). 전체3575tests·자원·생성24,000updates/cold를 통과했으나 실자료 실행은8,800/24,000updates 뒤 진행 로그 출력의 `BrokenPipeError`로 종료했다. 독립 실패 감사PASS, 평가query0으로 **성능·보정량은 미평가**다. 가설 실패나 수치 오류로 단정하지 않는다. 기존/이번 실패를 보존하며 재실행0·held60·외부요청·유료·GPU0이다. 다음은 과학설정을 유지한 출력/실행수명 복구의 별도 승인 제안이며 전체 연구goal은 미완료다. 아래는 이전 완료 이력이다.

> 2026-09-09 **N1 인공 학습기 통합 검증 완료**: [결과·해석·다음 제안](docs/task_trca_n1_integration_v1_results.md). 생성9개 ID/24,000updates를 실행하고, 네 학습 경로 작동·전체10arms·36평가조건·모든 모델의 query 전 동결을 독립 검산했다. 전체3300tests PASS. **인공자료 공학 검증이며 실제 metadata 효능·보정량 절감은 미평가**다. 사람 EEG·held60·GPU 접근0, 기존 실패·부정 결과 보존. 다음은 별도 새 후보·입력권한/모델개봉예산의 사람자료 개발 검증 제안이다. 원 goal은 미완료이며 아래는 시점별 이력이다.

> 2026-09-09 **별도 인공 수치 안정화 검증 완료**: [결과와 후속 계획](docs/numerical_stability_v1_results.md). 두 방법 모두1152개 행렬·209개 방향미분 사례·9개 잘못된 입력의 고정 기준과 독립80/120자리 검산을 통과했다. 사전 규칙대로 N1을 다음 통합 후보로 선택한다. 전체3002tests PASS. **실제 학습기 통합·metadata 효능·보정량 절감은 아직 미검증**이며, 기존2후보 재개·새 사람자료/held60 접근0이다. 다음은 별도 범위/승인의 인공 학습기 통합 검증이다. 아래 상태들은 시점별 이력이다.

> 2026-09-09 상태 정정: **전체 연구 goal은 미완료·추가 승인 대기**다. 아래의 고정2후보 프로그램 종료는 전체 목표 완료가 아니다. 새 유효한 성능·보정량 비교는 아직 없으며, 다음 수치 검증 단계를 현재 계약 밖에서 자동 실행하지 않는다. [정정 근거](docs/metadata_learning_program_v1_completion_audit.md#전체-thread-goal-상태-정정).

> 2026-09-09 **고정 2후보 프로그램 종료**: [종합 결과와 후속 검증 계획](docs/metadata_learning_program_v1_results.md). C1은 source39의24,000updates 뒤 최종 수치 검증 실패, C2는 균일 대조군의 사전 인공 수치 검증 실패로 종료했다. 독립 실패 감사·전체2857tests PASS를 완료했지만 **효능·보정량 감소는 미평가이며 유망 후보는 없다.** 원 저보정 목표와 연구 질문은 유지한다. 사람 attempt1/query모델 개봉1/recovery0; held60·외부 요청·유료 자원0. 다음 수치 검증 연구는 별도 계획/승인 대상으로 남기며 세 번째 후보나 실패 재실행을 자동 추가하지 않는다. 아래는 이전 단계 이력이다.

> 2026-09-09 [Temporal Q/QM 학습기 통합 검증 완료](docs/task_trca_temporal_v1_engineering.md): 같은 새 점수의 Q/QM/Q2/SHAM 및 원 FULL_NATIVE/별도 FULL_CENTERED 대조 설계를 고정했다. 실제 Q15·Q 동결·잔차·nested 학습을 생성6명에서 연결했고, 네 head CPU/CUDA 일치·독립 cold 감사·전체2376tests PASS를 확인했다. **사람 EEG 효능·metadata 보정량 절감은 미평가**다. 이번 source archive/실제 M/기존 실패 진단/사람 query/held60 접근0. 다음은 새 역할 제한 reader·완료 경로 감사·실행 manifest의 별도 구현·생성 검증이며, 기존 실패 후보 재개나 source39 자동 실행은 없다. 아래는 이전 단계 이력이다.

> 2026-09-08 [필터 부호 독립 수학·인공 검증 완료](docs/task_trca_shape_signfree_v1_engineering.md): 새 C-normalized projector와 성분별 시간중심 점수의32tests·CPU/CUDA·독립 SciPy 검산을 통과했다. 보존된 실패 행렬 한 지점에서도 계산·1차 미분을 확인했다. **Native 점수와 일반적으로 다른 별도 후보이며 metadata 효과·보정량 절감은 미평가**다. 기존 실제 후보는 종료 유지, 새 사람 학습·query·held60 접근0(이미 노출된 작은 진단 JSON만 재검산). 다음은 동일 새 scorer의 Q/QM 및 native/centered FULL 대조 설계·실제 learner 결합 검증이다. 자동 사람 재실행은 없다. 아래는 이전 단계 이력이다.

> 2026-09-08 [task-shape 실제 실행 종료](docs/task_trca_shape_source39_v1_results.md): `VALIDITY_FAILURE`. 첫 외부 분할10pipelines는 완료·독립 검산했으나 두 번째 분할의 SHAM 학습이 native 기준 정렬 하한을 위반했다. 같은 실패 지점의 CPU/GPU 재현으로 확인했으며 최종 query·held60 접근0이다. **Metadata 효과 없음이 아니라 효능 미평가**다. 연구목표는 유지하고 다음은 별도 필터 방향·부호 규칙의 수학/인공 검증 설계다. 원 후보 재실행·실패 대조군 제외·threshold 완화·held60 자동 개봉은 없다. 아래는 이전 단계 이력이다.

> 2026-09-08 학습기·인공 검증 완료: [task-shape 구현 결과](docs/task_trca_shape_engineering_v1_results.md). Q 분류 학습·동결 후 M/Q2/SHAM 잔차, bounded projector, mask-only Q와 참가자 분리 선택을 구현했다. 신규180/전체2057tests PASS, CPU 학습·CUDA forward/gradient 일치·실제 native 인공 bridge·인공6명 중첩8000steps를 확인했다. **사람 실험/metadata 효능 확인은 아니다.** 다음은 실제 역할 제한 reader와 독립 auditor/cold CLI이며 held60은 열지 않았다. 아래는 이전 단계 기록이다.

> 2026-09-08 새 후보 설계 완료: [분류 목표로 학습하는 Q+metadata](docs/task_aligned_trca_shape_v1_design.md), [구현 명세](docs/task_aligned_trca_shape_v1_implementation.md). 원 저보정 목표를 유지하고 source 분류 오차로 Q를 학습·동결한 뒤 작은 M 잔차를 비교한다. 공통 규제의 모든 방향 분모 증가≤10%, Q2/SHAM 대조, 참가자 nested 선택·보정비용·harm·유한 종료 규칙을 명세했다. **설계만 완료했으며 새 학습/사람 실험은0이다.** 다음은 인공 gradient/native/누수 검증을 포함한 구현이며 기존 후보 종료·held60 미개봉을 유지한다. 아래는 이전 단계 기록이다.

> 2026-09-08 후속 기하 진단 완료: [규제가 실제 support 필터를 얼마나 바꿨나](docs/trca_support_geometry_v1_results.md). Metadata를 넣기 전 고정γ=.1만으로 필터 방향이 중앙값 약44.4°/42.3° 바뀌었다. 큰 연산자 변화는 확인했지만 정확도 손실의 인과 원인이나 새 metadata 이득은 입증하지 않았다. 원 후보 종료·held60 보호는 유지하며 새 gamma 탐색/분류 실행은 없다. 아래는 기존 효능 결과와 이전 단계 기록이다.

> 2026-09-08 실제 실험 종료: [Source39 metadata-prior 결과](docs/metadata_prior_source39_v1_results.md). 39명×8조건의 단일 후보 학습·평가와 독립 검산을 완료했다. QM3−Q3 −0.006677%p, QM5−Q5 0%p이고312개 사람×조건의80% 최초 도달 단계가 모두 같아 보정량 절감은0이다. **이 구현의 metadata 이득 미확립으로 종료**한다. 공통 규제부터 native FULL보다 낮았다는 한계, 실행 복구와 정수 count 보고 정정은 결과 문서에 보존한다. 연구목표는 유지하되 새 후보 튜닝·held60 자동 개봉은 없다. 아래는 이전 단계 기록이다.

> 2026-09-08 검증 설계 수리 완료: [M-blind Q 학습·필터 전달성 검사](docs/metadata_prior_validation_repair_results.md). 참가자 분리 nested Q 선택, proxy 수준/채널 모양 분리, 식으로 만든 배열의 필터→점수→선택 경로를 구현·검증했다. 새 M 효능 실험은 아니며 이전 합성 v1의 효과 미확립 판정은 유지한다. 다음은 M을 보지 않는 별도 난이도 검증 설계이며 실제 EEG/held60은 이번에도 접근하지 않았다. 아래는 이전 단계 기록이다.

> 2026-09-08 합성 단계 완료: [M-prior v1 실제 합성 결과](docs/metadata_trca_prior_synthetic_v1_results.md). Native-compatible operator와 Q/QM·대조군 구현/검산은 완료했지만, 고정 screen은 **효과 미확립**이다. 주요3시나리오 Q 정확도100%로 분류 ceiling이 있었고, 반복 불일치 예측도 Q만 추가 적합한 Q2가 QM보다 좋았다. 실패 start와 같은 suite의 복구 결과를 모두 보존했다. 새 사람 데이터/held60 접근0이며 난이도·Q 적합의 검증 설계가 다음 검토 대상이다. 결과를 보고 설정을 바꾸거나 사람 실험으로 자동 승격하지 않는다. 아래는 이전 단계 기록이다.

> 2026-09-08 설계 검토 완료: [Metadata의 보정학습 삽입 재검토](docs/metadata_learning_covariance_design_review.md). 목표는 그대로이며, native TRCA에 동일 총량의 Q/Q+M 채널 규제를 주는 후보 하나를 제안했다. V1·context-template도 이미 학습단계 M을 사용했으므로 ‘최초 metadata 학습’이 아니다. 관련 공개 PDF2편 선택 정독·인공 대수 검산 완료, 새 사람 데이터 접근·효능 결과는 0이다. 다음은 proxy·대조군 명세와 순수 합성 구현 검증이며, source-only 입력 준비·실제 평가는 별도 단계다. 아래 완료 결과와 종료 경계는 그대로 보존한다.

> 2026-09-08 현재 완료: [Headroom cold-r1 진단](docs/native_subset_headroom_cold_r1_results.md)과 독립 수치 검산을 마쳤다. 실제 Q k3/k5=36.99/44.21%, 정답을 아는 사후 상한=46.67/56.06%이며, 이상적 선택도312개 사람×조건 중217개는 관측 grid에서80% 미도달이다. **선택 개선 여지는 있으나 M 효과·실제 보정량 절감을 입증한 것은 아니다.** 전체1561tests PASS, 실패한 이전 start 보존·held60 미개봉. [현재 자료 우선 계획](docs/current_data_first_research_plan.md)의 외부 paired-M 선택적 보강 원칙을 유지한다. 새 learner/추가 조건 탐색은 자동 실행하지 않는다. 아래는 과거 단계 이력이다.

> 2026-09-08 현재 완료: [Known-zero 단일 수정 실제 결과](docs/native_subset_known_zero_source39_results.md)를 개발39명에서 단회 평가하고 독립 전체 검산을 통과했다. 수정 QM−Q는 k3 **0pp**, k5 **+0.01068pp**이며 80% 최초 도달 보정량은312개 사람×조건 모두 같았다. 수정 자체도 k5 정답을 Q2개/QM1개 줄였다. **추가 M 보정비용 절감 미확립으로 이 구현 탐색을 종료한다.** 직접 M 평가1회+기작수정 평가1회 예산을 사용했으며 새 후보/재학습/held60 자동 개봉은 없다. 연구목표는 유지하고 다음 공백은 독립 paired-acquisition 자료와 사전 가설이다. 전체1520tests PASS, 새 독립M자료·외부요청0. 아래는 보존한 이전 단계의 이력이다.

> 현재 결과: [envelope-r1 직접 M 평가](docs/native_subset_m_envelope_r1_results.md)를 개발39명에서 완료했고 독립 검산을 통과했다. 입력 연결만 복구했으며 과학설정·SHAM배정은 불변이다. **Q 대비 임피던스 추가 이득은 k3 0pp, k5 +0.00534pp(정답1개)**였고, 80% 최초도달 보정량은 전부 같았다. Q 자체의 선택 개선은 있지만 M의 추가 보정비용 절감은 미확립이다. 전체1345tests PASS; held60 미개봉. 아래는 보존한 이전 단계의 기록이며 당시 ‘최신/다음/미승인’은 현재 상태가 아니다.

> 최신 실행 상태: [직접 acquisition-M v1 입력 오류](docs/native_subset_m_source39_v1_results.md). 전체1294tests 뒤 시작했으나 실제 metadata의11-key 저장 형식을6개로 가정한 연결 오류로 중단했다. 기존 metadata는 읽었지만 EEG·학습·효능 평가는 시작되지 않았다. M 가설 실패가 아니며 start-only 기록을 보존했다. 연구목표는 유지하고 새 입력-adapter 실행은 별도 authority가 필요하다. 아래 native eTRCA는 직전 완료된 성능 결과다.

> 최신 실제 결과: [source39 native eTRCA·두 bridge](docs/author_etrca_source39_v1_results.md) 39명 단회 완료·독립 검산 PASS. Wet0.5초 A0/3예시/5예시=33.42/43.72/51.50%, 하지만 전8조건 ETRCA5−A0=+2.24pp(CI −0.59~+5.07),17명 평균 악화다. 호환성 sprint는 종료하고 matched Q 대 추가 acquisition-M 비교로 돌아간다. 새 M 효능·독립 확인은 아직 없다. Dry/wet은 native preset에 이미 사용됐으며 impedance/held60은 열지 않았다.

> 2026-09-07 최신 판단: [기존 판단 재검토·새 연구 루프](docs/research_decision_reaudit_20260907.md). 목표는 유지하되 ‘Q가 먼저 평균적으로 A0를 이겨야만 M을 시험한다’는 필수 순서를 철회했다. 기존 저자 구현 하나의 호환성 확인 → 직접 M 가설1개 → 원인이 특정된 수정 최대1회 뒤 프로그램 점검으로 바꾼다. 기존 reference 두 arm의 AQ_NOT_ESTABLISHED와 모든 종료 결과는 그대로다. Source39는 개발용이며 held60은 열지 않는다. 아래 V1/V2 설명은 보존한 과거 이력이다.

이 저장소의 상위 목표는 **처음 보는 사용자가 쓸 만한 closed-set SSVEP 성능에 도달하는 데 필요한 labeled target calibration을 최소화하는 것**이다. `k=0`은 calibration-free anchor이고 `k=1/3/5`는 명시적인 low-calibration 자원점이다. Metadata는 연구목표가 아니라 이 부담을 줄이기 위한 수단이며, 현재 직접 시험하는 metadata도 개인정보나 dataset ID가 아니라 query 전에 관측되는 wet/dry interface와 block별 채널 impedance다.

완료된 `metadata-calibration-efficiency-v1`은 같은 완전한 calibration block과 같은 고정 query에서, EEG·구조·signal-derived QC만 쓰는 `A_Q`보다 pre-query acquisition context를 추가한 `A_QM`이 `k=0/1/3` early-budget curve를 개선하는지 물었다. Source 39명·24 jobs에서 eAUC 차이는 `−0.004843`, correct−shuffle은 `0`이었고 기본 A_Q k=5도 `0.476282<0.50`여서 `development_no_go`로 종료했다. 사전 규칙대로 held 60명은 열지 않았다. 전체 설계는 [metadata-assisted low-calibration 설계](docs/metadata_calibration_efficiency_design.md), 수치·해석·후속 결정은 [source 결과](docs/metadata_calibration_efficiency_results.md), 현재 상태는 [연구 프로토콜 현재 상태](docs/research_protocol_status.md), 결정 이력은 [append-only 연구일지](docs/research_log.md)에 있다.

후속 V2는 stronger FBCCA anchor, 안전한 support update와 pairing-aware acquisition context를 새 synthetic lockbox에서 먼저 검증하도록 동결했다. V11 단회 실행은 claim과 미래 NIST beacon을 정상 소비했지만 efficacy participant 생성 전 evidence validator의 `RecursionError`로 끝났다. 따라서 과학적 PASS/FAIL이 아니라 `infrastructure inconclusive`이며 같은 lockbox를 재실행하지 않는다. BETA/Dong, Choi와 wearable held 60명의 V2 outcome은 전혀 실행하지 않았다. 자세한 경계는 [V2 설계](docs/metadata_calibration_efficiency_v2_design.md)와 [V2 terminal 결과](docs/metadata_calibration_efficiency_v2_results.md)에 있다.

기존 `query-reliability-spatial-v1`은 사람 EEG outcome을 열지 않은 채 primary에서 내려와 frozen query-only baseline 후보로 남는다. 그 구현·hash·CUDA forward 증거는 보존하지만 기존 BETA 35/20 outcome plan은 실행하지 않는다. `physical_hybrid_v1` no-go, S1–S3 추가 outcome 금지와 `reliability-spatial-v1` Stage-0 terminal failure도 그대로 유지한다.

이전 physical-hybrid 계약은 역사 문서에 보존한다. 현 metadata-calibration 실험은 사전 할당 source 39명의 성능을 한 번 공개했으며, 나머지 held 60명의 support label, query prediction·outcome은 계속 미개봉이다. 현 후보의 threshold나 방법을 바꿔 같은 source/held를 재실행하지 않는다.

> **P0 revision 경계:** 배포 Readme의 impedance-axis 문구는 실제 수치·원 논문 Figure 9의 조건 평균과 모순된다. 수치상 axis 0/1은 각각 dry/wet(`261.67/19.63 kΩ`)이므로 코드가 이 signature를 fail-closed로 검증한다. 반대로 결합된 기존 `wearable_v2`와 그 A2 결과는 무효이며, raw-to-processed deep audit를 통과한 `wearable_v3`만 Protocol 0.4-dev 실행에 사용한다.

## Kubernetes quickstart

~~~bash
git clone https://github.com/jm020827/califreeEEG.git
# private repository 또는 SSH key를 쓰면:
# git clone git@github.com:jm020827/califreeEEG.git
cd califreeEEG

# jm020827 interns cluster: exact NVMe/DDN paths
source scripts/env_k8s_interns.sh

# Existing legacy cache/data: inspect first, then migrate once.
bash scripts/migrate_server_storage.sh
bash scripts/migrate_server_storage.sh --apply

bash scripts/cfeg.sh setup
bash scripts/cfeg.sh assets synthetic
bash scripts/cfeg.sh smoke
bash scripts/cfeg.sh help
~~~

Setup은 의존성만 준비하고 데이터와 weight를 받지 않는다. 프로젝트 전용 Python 3.10 환경에 NumPy 1.26.4와 Torch 2.2.2+cu121을 설치하고 실제 CUDA forward/backward probe가 실패하면 즉시 중단한다. Ambient/system Torch는 상속하지 않는다.

## 서버 저장 경로

`scripts/env_k8s_interns.sh`는 현재 interns Kubernetes mount를 다음처럼 고정한다.

| 용도 | 경로 |
|---|---|
| Hugging Face 상위 설정 | `/mnt/nvme/cache/interns/hf` |
| 실제 Hub model/dataset cache | `/mnt/nvme/cache/interns/hf/hub` |
| EEG raw/processed/MNE | `/mnt/ddn/prod-runs/interns/jm020827/califreeEEG/storage/eeg_data` |
| W&B 지속 로그 | `/mnt/ddn/prod-runs/interns/jm020827/califreeEEG/storage/wandb` |
| 임시 파일 | `/mnt/nvme/cache/interns/tmp/jm020827/califreeEEG` |
| pip cache | `/mnt/nvme/cache/interns/pip/jm020827/califreeEEG` |

`HF_HOME`은 Hugging Face 전체 상위 경로이고 `HF_HUB_CACHE=$HF_HOME/hub`가
`models--*`, `datasets--*`, `.locks`의 실제 위치다. 예전 코드가 만든 빈
`eeg_models/`는 사용하지 않는다. 기존 HF 루트의 REVE cache와 DDN의
`.local/eeg_data`는 `migrate_server_storage.sh`가 대상 덮어쓰기나 파일시스템 간
이동 없이 정리한다. 기본 실행은 dry-run이고 `--apply`에서만 `mv`한다.

일반 PVC 환경은 서버 프로필 대신 직접 지정한다.

~~~bash
export HF_HOME=/mnt/pvc/hf
export HF_HUB_CACHE=/mnt/pvc/hf/hub
export CFEG_HF_ROOT=/mnt/pvc/hf
export EEG_DATA_ROOT=/mnt/pvc/eeg
export WANDB_DIR=/mnt/pvc/wandb
~~~

## HF와 W&B

Secret은 Pod 환경변수로 주입한다.

~~~bash
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
export WANDB_MODE=online
export WANDB_PROJECT=calibration-free-eeg
export WANDB_ENTITY=jm020827
~~~

WANDB_API_KEY가 있으면 scripts/cfeg.sh는 online logging을 켜고, 없으면 disabled다. WANDB_MODE=offline도 지원한다.

~~~bash
kubectl -n <namespace> create secret generic califree-credentials \
  --from-literal=HF_TOKEN='<token>' \
  --from-literal=WANDB_API_KEY='<key>'
~~~

Pod spec에는 secretRef로 연결한다. Token은 Git에 저장하지 않는다.

## Asset

REVE gated access 승인 후:

~~~bash
bash scripts/cfeg.sh assets reve
bash scripts/cfeg.sh assets beta

# Wang은 MOABB adapter의 S35 누락을 피하기 위해 공개 Zenodo MOABB re-upload를 직접 사용한다.
bash scripts/cfeg.sh assets wang

# Wang/BETA와 exact 40-class이면서 8-channel semi-dry acquisition인 Dong2023
bash scripts/cfeg.sh assets dong2023
~~~

`scripts/cfeg.sh assets`는 full-cohort 준비와 검증만 수행한다. 소수 subject 개발용 raw가 필요하면 별도 pilot directory에 `scripts/fetch_dataset.py --subjects ...`를 직접 실행한다. 대량 공개자료는 `CFEG_FETCH_WORKERS=8`로 파일 단위 병렬 다운로드할 수 있다. BETA는 Figshare v3, Wearable은 Figshare v4, Wang은 Zenodo record 14865172, Dong2023은 Zenodo record 18847318에 고정하며 각 배포 API가 제시한 size와 MD5를 검증한다.

Wearable은 Figshare에서 subject 파일과 Impedance·공식 설명 파일을 선택 다운로드하며 크기와 MD5를 검증한다.

~~~bash
# 원격 파일 목록만 확인
python scripts/fetch_dataset.py --dataset wearable --subjects 1,2,3 --probe-remote
# 3명 raw pilot은 full cohort와 분리
python scripts/fetch_dataset.py --dataset wearable --subjects 1,2,3 \
  --raw-dir "$EEG_DATA_ROOT/raw/wearable_pilot_s001_s003"
# full 102명 준비·검증
bash scripts/cfeg.sh assets wearable
python scripts/audit_wearable_processed.py \
  --raw-dir "$EEG_DATA_ROOT/raw/wearable" \
  --processed-dir "$EEG_DATA_ROOT/processed/wearable_v3" \
  --expected-subjects 102
~~~

전용 parser가 `[channel,time,electrode,block,target]`, dry/wet, block, 8채널 impedance, headband 착용순서, 공식 8채널과 9.25–14.75Hz 12개 target을 읽는다. EEG axis는 배포 Readme와 독립 공개 SSVEP-DAN pipeline으로 `[dry, wet]`을 교차확인했다. Impedance axis는 Readme 문구의 오류를 실제 수치와 논문 조건 평균으로 교정해 `[dry, wet]`으로 읽는다. Deep audit는 Figshare v4 파일 MD5, 전체 raw EEG→HDF5/query-QC, raw impedance→채널 vector/scalar, headband order, exact trial grid를 대조하고 receipt를 남긴다. 기존 `wearable_v2`와 그 A2 결과는 제외한다. `window_start_sec=0.64`는 시작 offset이고 실제 window는 2.0초다.

Wang/BETA 전처리는 알려진 원본 tensor schema, target/block 수, 유한값과 공식 channel order를 검증한다. schema가 다르면 heuristic으로 축을 추측하지 않고 중단한다. Wang/BETA의 electrode/cap 재질은 공개 근거가 없어 `unknown`으로 기록한다.

Dong2023은 versioned Zenodo mirror의 size/MD5를 검증하고 `[8,1250,40,4]`를 canonical 8.0–15.8 Hz 40-class로 전처리한다. NEMAR 배포 license는 CC BY-NC 4.0이다.

## 기존 Wang/BETA label 변환

예전 processed asset에서 `Class ... conflicting frequencies` 오류가 나면 신호를 다시
다운로드하거나 전처리하지 않는다. 주파수 기준으로 manifest, `signals.h5/y`,
`class_map.json`만 변환한다. 원래 label metadata와 y는 processed 폴더 안의
`.label-alignment-backup-*`에 보존된다.

~~~bash
bash scripts/cfeg.sh migrate-labels          # dry-run
bash scripts/cfeg.sh migrate-labels --apply  # 실제 변환, 한 번만
~~~

## Train

~~~bash
CFEG_BACKBONE=tiny_transformer bash scripts/cfeg.sh train wang-to-beta
CFEG_BACKBONE=reve WANDB_MODE=online bash scripts/cfeg.sh train wang-to-beta
~~~

지원하는 mutable train preset은 `wang-to-beta`, `beta-to-wang`, `joint`, `synthetic`뿐이다. Wearable development/LOSO/dry↔wet preset과 `controls`/`research` shortcut은 폐기했으며 호출하면 fail-closed다. 첫 prompt-based grid와 두 번째 physical six-role grid는 모두 완료된 역사적 S1–S3 개발 실험이다. Physical reveal #2는 `DEC-20260901-004`와 clean annotated source tag에서 실행·소비됐으므로 다시 실행하지 않는다. Governed wearable은 전용 manifest orchestrator만 사용한다.

Legacy/외부 데이터 run은 `split.csv`, source-validation checkpoint, held-out metric을 저장한다. Governed wearable run은 학습 중 test loader나 `metrics_test.json`을 만들지 않는다. `metadata-calibration-efficiency-v1`은 전용 one-shot lifecycle로 source 24 jobs를 완료했고 development gate FAIL로 종료했다. 같은 후보를 재실행하거나 generic entrypoint로 held를 우회하는 것은 거부한다.

Wang과 BETA label은 raw index가 아니라 stimulus frequency로 canonical 40-class 8.0, 8.2, ..., 15.8Hz에 정렬된다. 학습 strong view는 8/4/2채널 subset을 명시적으로 포함한다. Source validation만 checkpoint 선택에 쓰며 target은 test-only다.

## Evaluate, robustness, calibration, inference

~~~bash
bash scripts/cfeg.sh eval wang-to-beta outputs/research/wang_to_beta/best.pt
bash scripts/cfeg.sh eval beta-to-wang outputs/research/beta_to_wang/best.pt

bash scripts/cfeg.sh channel-stress outputs/research/wang_to_beta/best.pt \
  "$EEG_DATA_ROOT/processed/beta_v1"
bash scripts/cfeg.sh robustness outputs/research/wang_to_beta/best.pt \
  "$EEG_DATA_ROOT/processed/beta_v1"

# Outcome-gated physical S1-S3 checkpoint는 generic predict가 아니라
# run_physical_mechanism_loso.py의 manifest-bound atomic reveal만 사용한다.
~~~

Robustness는 external metadata 결측 25/50/75/100%, acquisition-block 단위 derangement shuffle, global/channel/query-QC 분리 제거, downsample, re-reference, broadband/band-limited noise와 복합 4채널 조건을 평가한다. External missing/shuffle은 channel ID/mask, sampling/time grid, query QC를 바꾸지 않는다. Shuffle은 donor block 하나의 12개 label을 label/window 정렬로 함께 교환하고 donor mapping CSV·mapping hash·실제 metadata 변경률을 저장하므로 불가능한 row mosaic와 상수-metadata no-op을 구분한다. Signal perturbation 뒤에는 저장된 query QC를 invalid 처리한다. checkpoint 옆 `split.csv`의 held-out test ID 또는 명시적 target filter가 없으면 실행을 거부한다.

기존 generic calibration 유틸리티는 피험자·class별 k=0/1/3/5와 fixed query를 지원하지만 label별 sample을 따로 고르므로 서로 다른 block을 섞을 수 있다. 현 source 실험은 전용 runner에서 각 interface의 block 1–5를 nested support, block 6–10을 모든 역할·budget에서 동일한 query로 두고 complete block만 선택해 완료했다. Held 60명은 source FAIL 뒤 계속 거부된다. S1–S3를 다시 calibration 개발자료로 쓰지 않는다.

Training-free CCA/FBCCA baseline은 실제 canonical correlation과 sub-band filter bank를 계산한다. 현 source bundle에서 strict FBCCA k=0 `0.677564`, SAME3 component k=1 `0.560043`을 포함한 동결 baseline 15 jobs를 완료했다. S1–S3를 추가 실행하거나 held 60명을 여는 action은 없다.

## Historical physical ablation과 전체 suite

~~~bash
bash scripts/cfeg.sh ablation
bash scripts/cfeg.sh ablation A0_eeg_only,A4_full_latent
python scripts/run_ablation.py --include-optional --continue-on-error
# 과거 prompt v1 manifest/result는 immutable historical artifact다.
# run_development_loso.py prepare와 mutable run_ablation 재생성은 명시적으로 거부된다.

# physical reveal #2는 완료·소비됐으며 재실행하지 않는다.
# 공개 결과: outputs/development-loso/physical-mechanism-v2/reveal-bundle/analysis/revealed_summary.json

# 과거 physical 6개 역할의 outcome-free contract regression만 허용
python scripts/run_ablation.py \
  --config configs/train/wearable_physical_mechanism.yaml \
  --output-root outputs/development/wearable_v3/physical_contract_dryrun \
  --only A0_eeg_only,A2_structured_condition_prompt,M1_global_only,M2_channel_only,M3_full_shuffle_train,M4_metadata_only \
  --dry-run
# REVE는 snapshot-byte binding과 serial-layout 성능 gate 전까지 exploratory secondary다.
# 완료된 S1-S3 prompt grid를 cfeg.sh research로 다시 실행하지 않는다.
~~~

`run_ablation.py`는 `--config`, `--base-config`, `--output-root`, 반복 가능한 `--override KEY=VALUE`를 지원한다. 당시 physical primary A0/A2는 `physical_hybrid_v1`의 전체 module graph와 parameter 수, protocol/fairness, parameter schema, 초기 trainable state, split, vocabulary, exact asset bytes, source tree, 실행환경 hash가 모두 같아야 한 pair로 완료됐다. 안정적인 artifact ID `A2_structured_condition_prompt`는 역사적으로 남지만 새 primary가 아니다. Development control family도 이 계약을 공유하며, dynamic elapsed/peak-memory 값은 동일성 조건이 아니라 별도 `runtime_metrics.json` 증거로 남긴다. Metadata-only는 EEG·structure·query-QC를 우회하는 shortcut 진단이고, 자연 missingness pattern이 하나뿐인 missingness-only는 invalid assay였다. Shuffle-train과 wet↔dry counterfactual은 사전 계약대로 평가됐지만 correct metadata reliance gate를 통과하지 못했다. Retired physical 39명 training과 60명 lockbox는 no-go이며 새 후보는 별도 계약을 요구한다.

`analyze_ood_coverage.py`는 당시 physical 후보의 한-seed 진단용 비교다. CSV sidecar, checkpoint, split, analysis manifest, source revision이 모두 연결된 prediction bundle만 받는다. 당시 최종 primary 계약은 A0/A2×seeds `[42,43,44]`의 6 jobs를 모아 lockbox subject 안에서 seed BA를 먼저 평균한 N=60 표였으며 현재 실행하지 않는다.

~~~bash
python scripts/analyze_ood_coverage.py \
  --baseline outputs/a0/predictions.csv \
  --candidate outputs/a2/predictions.csv \
  --processed-dir "$EEG_DATA_ROOT/processed/wearable_v3" \
  --success-threshold <preregistered-BA> \
  --out-prefix outputs/analysis/a0_vs_a2

# physical development 결과는 이 공개 bundle에서 읽는다. Reveal #2는 이미 소비됐다.
jq '.mean_metrics, .predeclared_gate_evaluation' \
  outputs/development-loso/physical-mechanism-v2/reveal-bundle/analysis/revealed_summary.json

# 이 명령은 retired physical_hybrid_v1의 historical confirmatory 경로이며 금지된다.
# 새 metadata 후보에는 별도 candidate, runner, freeze와 owner execution decision이 필요하다.
~~~

집계기는 누락·중복 run, 60명 lockbox drift, 변조된 CSV/sidecar/checkpoint/train-metric/split/execution manifest/reveal receipt, 현재 plan/role/source와 다른 contract를 거부한다. Plan은 canonical manifest와 run root를 하나만 허용하고 manifest는 create-exclusive다. 여섯 completion의 exact epoch/runtime/resume 계약이 맞기 전에는 lockbox가 열리지 않으며, generic evaluation과 canonical hidden staging 밖 개별 prediction은 reveal 이후에도 dataset 생성 전에 실패한다. 6개 prediction은 staging과 공개 후 final 위치에서 모두 검증된다. 이 보호는 같은 OS 사용자의 raw-file 읽기까지 막는 암호학적 봉인이 아니라 application/procedural gate와 hash provenance다. Physical margin과 `development_gate_only`는 이미 동결됐고 사전 substantive gate가 실패했다. Confirmatory 값과 source tag를 채우는 것만으로는 현 no-go를 해제할 수 없다. Seed는 독립 피험자로 세지 않으며 S1–S3와 39명 training participant는 primary table에 포함하지 않는다.

## 규칙

- Label, stimulus frequency/phase, dataset/hardware/subject/session/trial ID와 source file은 primary conditioner 입력이 아니다. headband order, condition period, reattach와 경과시간은 분석 covariate/confound로만 보존한다.
- Dataset-ID-only ablation은 continuous/channel metadata도 사용하지 않는다. Structured-without-ID ablation은 dataset_id도 제거한다.
- Frequency overlap이 없으면 unrelated class id 비교를 거부한다.
- k=0은 calibration-free anchor이고 k=1/3/5는 low-calibration 결과다. k>0을 calibration-free라고 부르지 않는다.
- Wearable primary에서 k는 participant·electrode condition당 완전한 labeled block 수이며 한 block은 12 trial이다.
- Primary metadata contrast는 동일 Q·support·query·adaptation 권한에서 `A_QM−A_Q`다. Query-derived QC를 external metadata로 세지 않는다.
- `dataset_id`, opaque hardware ID, constant reference/cap은 새 primary의 external context `M`이 아니다. 현재 식별 가능한 `M`은 wet/dry interface와 block 전 채널별 impedance/availability뿐이다.
- 새 primary `A_QM−A_Q`는 하나의 composite source checkpoint에서 bounded metadata residual만 on/off한다. Common backbone과 Q-only calibration estimator는 M을 fit하기 전에 고정하며, all-missing M은 동일한 frozen `A_Q` 계산으로 정확히 돌아가야 한다.
- 평균 성능 개선과 OOD coverage 확장을 구분한다. `learned > forgotten` 및 worst-group 개선 전에는 “적용 범위가 확장됐다”고 쓰지 않는다.
- Raw/processed EEG, REVE weight, checkpoint, token, W&B log는 Git 제외다.
- Download는 명시적인 assets 명령에서만 일어난다.
- Frozen REVE는 checkpoint에 복제하지 않고 `HF_HUB_CACHE`에서 다시 읽는다.

현재 설계와 실행 차단조건은 [metadata-assisted calibration-efficient 설계](docs/metadata_calibration_efficiency_design.md)와 [프로토콜 상태](docs/research_protocol_status.md)를 따른다. 과거 physical strict-k0 구현 기록은 [역사적 체크리스트](calibration_free_eeg_codex_implementation_plan.md)에만 보존한다.
