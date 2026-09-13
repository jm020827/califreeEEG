# MAMEM baseline 출처 확인 v1 — 완전한 CCA 재현 설정은 미확정

2026-09-13, source 계획 77857b4 이후. 디스크 여유 약295GiB를 재확인했다.
**판정: MAMEM-I의 완전한 author CCA baseline qualification은 NOT MET.**
따라서 이 계획 아래 S001a/b 실제 두-reference 비교와 새 metadata 학습은 실행하지 않는다.
이것은 연구 전체의 불가능 판정이 아니라, 서로 다른 예제를 한 구현으로 오인하지 않는 경계다.

## 확인된 사실과 확인되지 않은 연결

고정 [저자 revision](https://github.com/MAMEM/eeg-processing-toolbox/tree/5a03abe2a6a874e9adaceea29a52c2fce35d8a03)의 tree1회와 code/docs8파일을 읽었다.
아래 line은 GitHub JSON의 base64를 decode한 원본 기준이다.

| 출처 | 직접 확인 | 적용 한계 |
| --- | --- | --- |
| README L28–55 | Dataset I는250Hz·HydroCel256·단일 자극, nominal5개; 예제별 dataset 구분 | 채널의 실제 후두 위치나 완전한 CCA 설정 아님 |
| CCA.m L14–54,125–137 | caller가 frequencies/channels/fs/harmonics를 지정; 전체 reference bank를 trial loop 전에 만들고 canoncorr; label은 feature 뒤에 처리 | CCA 클래스 내부에서 query label로 reference를 만들지 않지만 모든 미열람 caller의 안전성을 보장하지 않음 |
| CCA.m L73–123 | 1e-8 covariance ridge가 있는 대체 함수 전체가 주석 | 이 ridge를 활성 저자 CCA의 설정으로 인용하면 오류 |
| exampleDefault L10–34 | Dataset-I 예제는 E126,1250samples,PWelch,filter,SVM | E126을 고른 근거는 있으나 E126 CCA/무필터2초/argmax의 저자 재현 근거는 아님 |
| SampleSelection L9–12,25–39 | 기본 channel126, 실제 선택은 원 signal의 MATLAB행 번호 | E126=Oz라는 해부학 매핑을 입증하지 않음 |
| exampleOptimal L10–43 | 별도 PWelch/AMUSE/SVD/SVM 설정,channel138 | 이름의 optimal을 우리 CCA의 성능보장으로 사용하지 않음 |
| exampleEPOCCCASVM L5,13–38 | Dataset III,고정 nominal5bank,128Hz,4harmonics,원채널6:9,filter,SVM | Dataset I 256행에서 같은 번호를 쓰면 다른 전극; allFeatures=1이며 최대CCA argmax classifier도 아님 |
| Experimenter L56–140 | preprocessing→feature extraction→classifier/evaluation 연결 | generic wrapper만으로 Dataset-I CCA recipe가 확정되지 않음 |
| LICENSE L1–201 | Apache2.0,CERTH2016 표시 | 원본 보존만 했고 MATLAB 코드를 실행/복제하여 배포하지 않음 |

보존 MOABB adapter(blob eef0203f06c86f18f6db53f7cc67cbd4a6af685b)의
L157–170은 첫256행을 E1–E256,마지막행을 stim으로 두고 GSN-HydroCel-256
montage를 붙인다. 후두 subset·CAR·5–48Hz filter·CCA를 실행하지 않는다.
기술 metadata의 전처리 목록을 실행 연산으로 오인하지 않는다. DIN sample을 그대로
Python index로 쓰므로 우리의 sample1−1과4ms 차이가 있으며 기본 epoch도[1,4]로
다르다. 우리는 label convention만 채택했지 MOABB 전체 재현은 아니다.

[MathWorks canoncorr 문서](https://www.mathworks.com/help/stats/canoncorr.html)의
Algorithms는 centered QR/SVD를,Output Arguments는 rank-deficient 처리를 설명한다.
[SciPy1.15.3 orth](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.linalg.orth.html)는
SVD의 eps×max(shape) 수치 cutoff를 사용한다. 수치 rank 판정을 통계적 noise 제거로
해석하지 않는다. 로컬 sklearn1.7.2 CCA에도 ridge/shrinkage 인자는 없다.
LW/OAS를 곧바로 covariance에 대체하는 것은 원 CCA 재현이 아니고, 원 cross term을
그대로 유지하면 score가1을 넘을 수도 있다. 이번에는 새로운 shrinkage도 선택하지 않는다.

## 보존 및 읽기 예산

Source root: `/home/whwovy/research-spaces/califree-eeg-experiment-design/sources/`.
`mamem_baseline_code_{0..7}_20260913.json`은 순서대로 README,CCA,Default,Optimal,
SampleSelection,EPOCCCASVM,LICENSE,Experimenter의 GitHub wrapper다.
Decoded 합계 **51,492bytes**, tree152entries/nontruncated; 8개 Git blob hash를
로컬 재계산하여 wrapper SHA와 일치 확인했다. 자세한 hash는
[source 상태](reports/mamem_baseline_source_v1_state.json)에 있다.

추가 공식 문서는 위2페이지만 읽었다. web 도구가 반환한 선택 본문만 확인했으며 원 HTML을
다운로드·hash한 척하지 않는다. **계획의 문서별30초/1MiB 제한을 web 호출에 기술적으로
강제하지 못했다**는 절차 이탈을 보존한다. GitHub 요청은 timeout30초를 적용했고
wrapper/decoded 크기는 수신 후 검사했다(사전 streaming 차단은 아님).
소스 실패/재시도0,새 scholarly discovery/PDF0. 이 분기를 위해 새 EEG/MAT/NPZ decode,
feature cache/outcome 읽기·fit·accuracy 계산0. S001b의 경로/hash receipt만 읽었다.

Root가 main과 SQLite를 단독 수정했다. 3개 reviewer는 shared repo 읽기 전용,
새worktree0/동시writer0. 45기존worktrees와8untracked pytest디렉터리 보존,
install/symlink/push/삭제/held60/사람자료요청/유료0이다.

## 다음은 별도의 작은 진단, 연구 기여의 축소 포장이 아님

남은 질문은 '고차원 공간필터를 제거하면, support에서 만든 전체 reference bank가
nominal bank와 어떻게 다른가?'다. E126 하나를 사용하는 **source-inspired diagnostic**은
설계할 수 있다. E126은 결과를 보고 고른 행이 아니라 저자 Dataset-I 기본 예제의 행이고,
한 채널은 공간 가중치의 고차원 적합 자유도를 없앤다. 후두부 위치를 추측할 필요도 없다.
하지만 무필터·2초·2harmonics·argmax의 조합은 우리의 새로운 선택이다.

현재 source qualification 실패를 성공으로 바꾸거나 같은 실제 비교를 강행하지 않는다.
별도 [합성 전용 계약](mamem_reference_probe_synthetic_v1_contract.md)으로 수학/API의
구현만 검증한다. 실제 자료 실행은 이 단계에서0. 그 이후 명시적인 새 개발 계약에서만
E126 sensitivity probe를 검토할 수 있으며, 통과해도 강한 common baseline 복원이나
학습된M 효과를 입증하지 않는다. 실패해도 metadata 일반의 무용성을 뜻하지 않는다.

Saturation: 이번 pass는 inactive ridge/잘못된 dataset 전용 montage 전용을 배제했고
미확정 recipe를 식별했다. 검색을 더 늘려 좋은 성능을 보장하려 하지 않는다. 다음 정보는
합성 수학검사와 별도 저용량 개발 진단에서 얻는다. 실제 M 효능 비교에는 모든
Q/Q2/QM/SHAM에 공통 정보를 제공하는 강한 대조와 보정량/시간 평가가 여전히 필요하다.
v2 RETIRE와 이전 부정 결과 유지, 전체 연구 goal은 active다.
