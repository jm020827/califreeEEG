# 기존 구현 호환성 준비 — toolbox eTRCA

> 후속 완료: [source39 native·두 bridge 결과](author_etrca_source39_v1_results.md)가 이제 실제39명 단회 실행/독립 검산까지 포함한다. 아래는 그 이전의 합성 API 준비 기록이며 원논문 전체cohort 재현이나 M 효능 입증으로 승격하지 않는다.

2026-09-07. 현재 **로컬 합성 fixture/API 확인 PASS**. 실제 source39 성능 확인,
원 TRCA 논문의 MATLAB 실행, 논문 수치 재현, M 효능 실험은 아직 아니다.
[판단 재검토·유한 루프](research_decision_reaudit_20260907.md)의 A단계 준비 작업이다.

## 선정과 소유권

하나의 baseline은 [SSVEP Analysis Toolbox](https://github.com/pikipity/SSVEP-Analysis-Toolbox/tree/3344bd199daf78888e364d9db00ae7d8128d2b5f)의
**수정하지 않은 ETRCA**다. Toolbox 논문 저자 구현이며2018 TRCA 원저자 구현과 구분한다.
README는 CC BY-NC-SA4.0 및2024 toolbox 논문 인용을 명시한다. 원본 attribution/license를
checkout에 보존하고 본 저장소에 원본 코드를 복사하거나 재라이선스하지 않았다.
상업적 사용·배포 권리가 있다고 주장하지 않는다.

Revision `3344bd199daf78888e364d9db00ae7d8128d2b5f`를 `gh repo clone` 후 detached checkout했다.
경로 `/home/whwovy/ssvep-author-compatibility-20260907/upstream`, clean, 약25MiB.
원 TRCA MATLAB 저장소는 MIT지만 MATLAB/Octave가 현재 PATH에 없어 이번 실행 대상으로
선정하지 않았다. 해당 저장소의 human sample.mat는 가져오거나 열지 않았다.

Main만 [얇은 fixture adapter](../scripts/check_author_etrca.py)와 tests/docs를 작성했다.
공유 원본 수치 코드, main .venv, CUDA, 기존 후보 runner/결과는 수정하지 않았다.
읽기 전용 에이전트2개를 사용했으며 새 Git worktree나 동시 writer는 만들지 않았다.

## 기존 실험과의 차이

| 요소 | 이번에 확인하는 native toolbox 경로 | 직전 reference 실험 |
| --- | --- | --- |
| 방법 | Ensemble TRCA; k3/5만, 원본 fit/predict | 네 항 ECCA와 ITCCA |
| 처리 구간 | sample125:160+N을 처리한 뒤 앞35samples 제거 | crop160:160+N, notch arm만 prefix0부터 처리 |
| Notch | Cheby1 order4/ripple2,47–53Hz bandstop | 없음 또는 iirnotch50Hz/Q35 causal |
| Filterbank | 5bands,9/18/27/36/45–90Hz; ripple1 | 7bands,6/14/22/30/38/46/54–90Hz; ripple0.5 |
| 정규화 | 구간 내 linear detrend, channel별 ddof1 표준화 | 동일 전처리가 아님 |
| 최종 score | 원본 선형 band-weighted Pearson r | signed-square 네 항 합 |
| 공개 평가 | Toolbox의 leave-one-block-out,9support/1query | chronological 첫k support/마지막5query |

Native 경계는 [BaseDataset.get_data_single_trial](https://github.com/pikipity/SSVEP-Analysis-Toolbox/blob/3344bd199daf78888e364d9db00ae7d8128d2b5f/SSVEPAnalysisToolbox/datasets/basedataset.py#L415)와
[wearablepreprocess.py](https://github.com/pikipity/SSVEP-Analysis-Toolbox/blob/3344bd199daf78888e364d9db00ae7d8128d2b5f/SSVEPAnalysisToolbox/utils/wearablepreprocess.py)를 읽고 옮겼다.
N125/188/250은0.5/0.752/1초이며 native helper는 추가35samples(0.14초)를 처리한다.
Query 종료 뒤 sample은 쓰지 않지만 구간 내 filtfilt이고, 배포 자료의 upstream downsampling까지
완전 causal임을 증명하지 않는다. 여러 요소가 달라 결과 차이를 notch 하나의 효과로 돌릴 수 없다.

Interface별 공개 band weights는 native 호환성 확인용이다. 실제 M 순증분 연구에서 쓴다면
Q와 M+Q 양쪽에 같은 설정을 주고 interface 정보 자체를 impedance 이득으로 세지 않는다.
또 이 preset은 source39-only fit이 아니다. 전체 코호트에서 조정된 공개 설정의 집계 tuning
노출 가능성을 명시하며, 그대로 쓰는 held60 평가를 완전히 독립적인 confirmation으로 부르지 않는다.

## 환경과 실행

기존 main NumPy1.26.4에서는 원본 `from numpy import object`가 호환되지 않는다.
원본을 수정하는 대신 별도 Python3.9.21 환경을 만들었다. numerical/package versions는
[고정 requirements](../configs/environments/ssvep_toolbox_compatibility_requirements.txt)를 따른다.
NumPy1.23.4/SciPy1.13.0/joblib1.4.2/sklearn1.3.0/mat73.63은 upstream environment.yml과
맞추고 h5py3.11.0/threadpoolctl3.5.0도 고정했다. 약230MiB, 예약1GiB 이내다.
Python 배포와 다운로드 cache는 이 크기에 포함하지 않으며 기존 환경을 교체하지 않았다.
setup.py의 다른 상한으로 재해결하지 않고 clean checkout을 직접 import한다.

~~~bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/whwovy/ssvep-author-compatibility-20260907/.venv/bin/python \
  scripts/check_author_etrca.py \
  --upstream /home/whwovy/ssvep-author-compatibility-20260907/upstream
~~~

CPU1/BLAS1, fixture seed20260907. 최초 실행2.125초,12cases PASS.
재확인2.171초의 [원본 stdout report](reports/author_etrca_fixture_20260907.json)를 보존한다
(SHA256 `5261d9a612ef647694f268da40c3af41a9550959c477497bf0070fadd2019f9a`).
읽기 전용 reviewer도 같은 명령을 독립 실행해12cases PASS/2.240초를 확인했다.
이 반복은 개발 fixture 검산이지 새로운 미관측 과학적 실험이 아니다.
Main 회귀17tests PASS 뒤 commit `7855b46482ac32a0bd46fa49948d5d1f5fff75a8`에서
전체1114tests PASS/116.77초/기존Torch warnings68개를 확인했다.
12cases는2interface preset×3windows×k3/5이고 독립 참가자12명이 아니다.
합성 정답12/12는 fixture가 잘 연결됐다는 진단일 뿐 사람 정확도나 좋은 DGP의 증거가 아니다.

확인한 것은 원본 public API, template 평균, native 선형 score/argmax,
독립 projected-Pearson 계산(maxabs8.88e−16), singleton/batch 일치, predict의 spatial-filter 불변,
label 순환 equivariance, 허용 crop 밖 sample 변경 불변이다. Fit 자체의 generalized-eigen
해를 독립 재현하거나 원 논문의 수치와 대조한 것은 아니다.

`datasets`/`evaluator`는 import하지 않고 `SimpleNamespace(srate=250)`만 전달한다.
전체102명/metadata를 자동 다운로드하는 dataset constructor와 성능순위 participant selector를
호출하지 않는다. Script는 human-data 인자를 받지 않는다. Python audit hook는 data 확장자
open, 네트워크·추가 process, dataset/evaluator import를 거부한다. 이는 검토된 fixture의
방어 장치이지 OS 수준 보안 sandbox가 아니다. 실행 후 import된 source hash도 출력한다.

## 아직 해야 할 실제 호환성 확인

고정 tree에 Wearable 성능 그림은 있지만 직접 비교할 numeric result 파일은 확인하지 못했다.
다음 source39-only 단계는 **author-implementation-compatible 개발 분석**으로 명명하고,
정확한 전체 논문 재현을 주장하지 않는다. Native9support LOBO와 우리 chronological
k3/5의 차이를 먼저 명시해야 한다. source39를 읽기 전에 허용39개 파일·숫자/축·분할·window·
필터 경계·A0 comparator·출력·자원 예산을 별도 실행 명세로 고정하고 입력 불변 시험을 한다.
Fixture adapter를 그대로 human runner로 호출할 수 없도록 데이터 접근은 분리한다.

이 준비 단계에서는 새 human EEG/impedance/held60/retired1–3 접근0,
새 independent M 자료0, 저자 연락0이다. 별도 M 결합식과 평가 명세도 아직 고정·실행하지 않았다.
이번 fixture PASS를 M 실험의 성공 gate로 세지 않는다.
