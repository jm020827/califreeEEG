# N1 source39 복구 실험 결과

완료: 2026-09-10 00:22:08 KST (2026-09-09T15:22:08Z).

## 결론

**출력 연결 오류는 복구했고 실험·독립 검산을 끝냈다. 하지만 이번 구현에서는 metadata의
추가 정확도 이득과 보정량 감소가 없었다.** 최종 판정은
`METADATA_INCREMENT_NOT_ESTABLISHED`다. 이전의 실행 실패와 달리, 이번에는 유효한
개발자료상 부정 결과를 얻었다. 이 N1-R 후보는 종료하며 같은 자료에서 설정을 바꿔 다시
실행하지 않는다. 원 저보정 SSVEP 연구 질문 자체가 반증됐다는 뜻은 아니다.

연구 질문과 Q/QM/대조군은 [쉬운 설명](task_trca_n1_transport_recovery_r1.md#쉽게-보는-이번-실험),
실행 권한과 복구 검증은 [복구 기록](task_trca_n1_transport_recovery_r1.md),
최종 수치는 [원 결과](/home/whwovy/task-trca-n1-source39-recovery-r1-1i9cau/task-trca-n1-source39-primary1/result.json)의
`summary`에 있다. [독립 검산](/home/whwovy/task-trca-n1-source39-recovery-r1-1i9cau/task-trca-n1-source39-primary1/cold_audit.json)이
그 결과 SHA를 결합한다.
[기계 판독 요약](reports/task_trca_n1_transport_recovery_r1_results.json)에는 모든 비교 구간,
판정 조건, 작동 지표와 자원·원본 SHA를 함께 보존했다.

## 정확도와 보정량

39명 × dry/wet × 4개 시간창을 모두 포함한다. 각 보정량에서 312개 사람·조건,
14,976개 query 판정이다. 서로 다른 시간창이 같은 trial을 재사용하므로 14,976개 독립 trial이나
312명으로 해석하지 않는다. 3/5 blocks는 각각 12개 자극당 3/5개, 총 36/60개 보정 trial이다.

| 방법 | 36개 보정 trial 정확도 | 60개 보정 trial 정확도 |
|---|---:|---:|
| Q: EEG 특징 기반 학습 | 34.32826% (5,141 정답) | 40.94551% (6,132 정답) |
| QM: Q 고정 + 실제 임피던스 | 34.32826% (5,141) | 40.93884% (6,131) |
| Q2: 같은 크기의 EEG 추가 모듈 | 34.32826% (5,141) | 40.94551% (6,132) |
| SHAM: 조건을 맞춘 대체 임피던스로 학습 | 34.32826% (5,141) | 40.94551% (6,132) |
| FULL_NATIVE: 기존 분류기 | 34.41506% (5,154) | 41.03900% (6,146) |
| FULL_CENTERED: 같은 기존 필터, 새 점수 | 34.41506% (5,154) | 41.04567% (6,147) |

실제 무보정 기준 A0는 **39.22276% (5,874 정답)**다. ISO/PERMUTED/STALE/MISSING도
원 결과에 모두 보존했다. MISSING은 Q와 정확히 일치하며, 누락 조건을 제외하지 않았다.

- 주요 비교 QM3−Q3, QM3−Q2_3, QM3−SHAM3는 모두 **0%p**다. 참가자별 기술통계 구간도
  모두 [0, 0]이다. 이는 이 관측 자료에서의 동일 결과이지, 모집단 효과가 정확히 0이라는 확증이 아니다.
- QM5−Q5는 **−0.006677%p**, 기술통계 구간 [−0.020195, +0.006840]%p다.
  예측 하나가 정답에서 오답으로 바뀌었다. 39명 중 38명 동률, 1명 정답 1개 감소다.
- QM3−Q5는 **−6.61725%p**, 구간 [−8.20588, −5.02863]%p다. 보정 trial을 60개에서
  36개로 줄이는 손실을 metadata가 메우지 못했다.
- QM3−A0는 **−4.89450%p**, 구간 [−7.98068, −1.80831]%p다. QM3 평균 정확도 80%
  기준도 미달이다. FULL_NATIVE/FULL_CENTERED에 대한 −1%p 허용폭 검사는 통과했지만,
  이것만으로 실용적인 보정량 절감 후보가 되지는 않는다.

0·36·60개 보정 trial 중 48개 평가 trial에서 39개 이상을 맞힌 첫 지점은 **Q와 QM이
312개 조건 모두 같았다.** 50개 조건은 0개, 15개는 36개, 12개는 60개에서 처음 도달했고,
235개는 관측 지점 어디에서도 도달하지 않았다. 양쪽 모두 도달한 77개 조건의 보정량 평균은
각각 16.36364개, 평균·합계 절감은 **0개**, 신규 도달·도달 상실은 각각 **0개**다.
미도달에는 임의 비용을 부여하지 않는다. 이는 관측한 trial 수의 비교이지, 실제 적응형 중단
정책이나 전극 준비·임피던스 측정을 포함한 전체 시간의 비교가 아니다.

## Metadata가 사용되지 않았던 것인가?

**아니다. 학습과 점수에는 반영됐지만 선택한 정답을 개선하지 못했다.**

QM의 기록된 최대 gradient norm은 0.00020356624, 최대 계수 절댓값은 0.90696619였다.
이 수치는 30개 학습 pipeline의 경로 작동 기록이며 모든 최종 모델에 같은 값이 있다는 뜻이나
독립 Adam 재현 증거는 아니다. 세 outer 모델의 Q-only 선택 lambda는 모두 0.001이었다.

36개 보정 trial에서는 Q→QM의 채널 규제 R, 필터 projector F, 합성 J, 점수가 312개 조건
모두에서 달라졌다. 점수 최대 절대 차이는 0.0008855488이지만 **argmax 변화는 0개**였다.
60개에서는 최대 점수 차이 0.0008820429, argmax 변화 1개이며 그 변화는 오답 방향이었다.
대체 임피던스도 78개 사람·interface 단위 전부에서 특징을 바꿨다(coverage 100%).
따라서 metadata 누락이나 대조군 값 미교체로 이번 0 이득을 설명할 수는 없다.

다만 이 결과만으로 “규제가 너무 약해서 실패했다” 또는 “임피던스에는 쓸 만한 정보가 없다”를
구분하지는 못한다. 정보의 부족, 특징 요약의 손실, 제한된 학습 모듈, 점수로 전달되는 변화의
크기 등이 미확인 설명으로 남는다. 단순히 규제 강도·계수·시간창을 사후 변경할 근거로 쓰지 않는다.
`structural_no_actuation=false`이며, 예측이 같다는 이유로 구조적 무작동이나 실행 실패로 재분류하지 않는다.

## 실행·독립 검산·자원

이번 새 primary 1회는 30 pipelines / 120 heads / **24,000 updates**를 완료했다.
세 outer 모델을 모두 고정한 뒤 query에 접근했다. fit 접근 2,184회, 전체 접근 4,213행,
query-bearing 호출 1,248회, event 182행으로 고정 계약과 일치한다. Global barrier의 실제 접근
행은 seq2184이고 그 직후 event가 기록한 next-access counter는2185다. 첫 query-bearing은2445다.

Supervisor가 complete cold를 **1회** 실행했다. Primary와 audit는 모두 exit0, watchdog 없음,
stderr 0 bytes이며 추가 실행·수동 감사는 없다. Cold는 세 분할 각각 208개 평가 case, 열 개 arm의
argmax·정수 정답 수, MISSING=Q와 두 FULL의 동일 native 필터를 확인했다. 최대 score 재구성
오차는 약 1.057e−12, projector는 약 5.795e−11, inner Q 선택 CE는 약 1.288e−14다.

이 감사는 **저장된 Q/S/C/prefix M/Gram/native 필터**에서의 독립 계산과 접근 계보 검증이다.
원 EEG 전처리, Q15 계산, native fitting, Adam을 처음부터 독립 재실행한 것이 아니다.
`result.cold_audit_status=COLD_AUDIT_PENDING`은 감사 전 원본이므로 수정하지 않는다.
별도 cold receipt의 `COLD_INDEPENDENT_AUDIT_PASS`가 최종 감사 결과다.

| 자원 | 실제 사용 | 한도 |
|---|---:|---:|
| Primary 외부 경과 시간 | 10,168.491초 (약 2시간 49분 28초) | 21,600초 |
| Primary peak RSS, wait4 | 2,804,920KiB (약 2.675GiB) | 16GiB |
| Cold 외부 경과 시간 | 89.748초 | 1,800초 |
| Cold peak RSS, wait4 | 1,768,576KiB (약 1.687GiB) | 기록용 |
| 복구 parent 최종 할당량 | 2,911,412KiB | 전체 새 출력 16GiB 내 |
| 복구 tests/prepare parent 최종 할당량 | 54,288KiB | 검사 한도 128MiB 내 |

Primary 내부 시간은 10,163.624초, cold 내부 시간은 89.259초다. wait4 RSS는 각 child의
peak이며 supervisor 전체 process tree의 동시 peak를 측정한 것은 아니다. Tests는 사전
301 PASS 및 최종 새26 PASS와 전호출45.07초의 기존 기록을 재사용했고, 종료 뒤 다시 돌리지 않았다.
기타 검사20초 예약까지 300초 이내다. 원 전체3575 tests도 이번에 재실행한 것으로 세지 않는다.

원36 science/runtime pins, 새 config/launcher/tests/preflight/manifest, 옛 failure·terminal과
새 completion·cold·result의 SHA 연결을 재확인했다. 다음 원본은 모두 별도 경로로 보존한다.
Primary의 정확한55개 파일과 supervisor의10개 JSON·primary/audit 로그는0400·단일 링크다.
Supervisor 자신의 빈 stdout/stderr 두 파일은0664이며 completion의 해시 결합 대상이 아니다.
따라서 supervisor 폴더 전체가 불변이라고 주장하지 않으며, 해당 파일도 변경하지 않았다.

| 원본 | SHA-256 |
|---|---|
| 이번 result | `4eae8d40f99fe3c8dda083d02c4717f497d74ad0853f1da9e3e7edbc67adba1a` |
| 이번 cold | `5df15979142d358c47f009ef9b195ea4edf168db25cf2979ed54c22ea014148f` |
| Supervisor completion | `3f702ea159595e892a2b5b12f2a3d1d529e824233580a74e51b8b747043f9359` |
| Recovery authority | `ed8dfe176d418a3280a96c2a4a3d34d66fbe67a0cdfb3ebf64fc906e7c81d179` |
| 이전 BrokenPipe failure | `e4ebf797526580fd3a163038832e2f0d09dd3fd639140929ee8144bf96f2a107` |
| 이전 실패 감사 | `3e92cd6b2328c5f20cdba27507e860ee683644a73e8f50b8a0e088c6c76a92a9` |

이번 runtime revision은 `91b728229a3e3871fb0403f48a7a50b6ea3fd7de`다.
옛 실자료 시도 8,800 updates/query reveal0와 이번 24,000/reveal1은 분리해서 기록한다.
이는 새 과학 후보가 아니라 동일 후보의 별도 승인 인프라 복구이며, 옛 부분 모델 재사용은 없다.
Held60·원 raw MAT/full M·GPU·외부 요청·유료 자원·패키지 설치·삭제는 모두0이다.

## 결정과 다음 검증 제안

1. **유지:** 원 저보정 SSVEP 목표, 유효한 대조군, N1 수치 구현·독립 검산·파일 로그 실행기.
2. **종료:** 이 고정 N1-R metadata 삽입 후보. 유망 후보0이며 부정 결과를 보존한다.
   단일 복구 예산을 소진했으므로 같은 source39에서 강도·특징·seed·시간창을 바꾸지 않는다.
3. **다음 제안(미착수):** 사람 query를 다시 쓰기 전에, Q로 충분히 설명되지 않는 acquisition
   정보가 정답에 도움이 되도록 생성 기작을 명시한 인공 양성 대조와 정보 없는 음성 대조를
   하나의 고정 시험으로 설계한다. 질문은 “이 제한된 경로가 유용한 M을 받았을 때 실제 선택을
   개선할 능력이 있는가?”다. 기존 생성24k 검증은 실행 가능성 검사였지 이런 효능 검사가 아니었다.
   관계·기준·방법 수·seed·시간 예산을 먼저 고정하고, 실패하면 사람 재학습 없이 해당 경로를
   종료한다. 통과해도 실제 EEG에서 M이 유용하다는 증명으로 삼지 않는다.
4. **그 이후의 조건:** 새로운 실자료 후보에는 앞 시험이 드러낸 기작 차이와 기존 음성을
   설명하는 반증 가능한 이유가 필요하다. 단순한 강도 확대나 양성이 나올 때까지의 탐색은 하지
   않는다. 새 연구 범위 승인을 받은 뒤에만 구현·실행하며, held60·외부 자료 요청·유료 사용은
   각각 별도 승인 대상으로 유지한다.

Source39는 반복 노출 개발자료다. 여기서의 부정 결과를 metadata 전체의 무용성으로 확대하거나,
추후 양성 결과를 독립 확증으로 부르지 않는다. 이번 유한 시도는 끝났지만 **전체 연구 goal은
미완료**다. `academic-research`에는 이전 실패와 분리된 결과·구현 카드 및 열린 증거 공백으로
기록한다. 새 문헌 검색/PDF 읽기/자료 수집은0이며 기존 문헌 cutoff 2026-09-04의 공백도 유지한다.

근거맵은 claim `23d98ce865902ae8`(QUALIFIED, 5개 evidence), technique `6e9b67c65620cd0d`,
열린 gap `6a8254f849b36b2f`로 갱신했다. 이전 실패 claim `ffe23763f1e4081f`는 그대로 보존한다.
Render 완료/SQLite quick_check 정상. 누적1060 papers·80 searches·41 cards·16 techniques·53 claims·
124 evidence·38 gaps·36 deep-read items·5 analogies이며, 새 검색이나 정독을 한 것으로 세지 않는다.
