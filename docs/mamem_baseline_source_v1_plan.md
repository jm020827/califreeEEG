# MAMEM 공통 baseline 출처 확인 v1 — source 읽기 전 범위

2026-09-13, base11997ce/main. 이전turn은 S001 실제15창 진단/검산으로 progress다.
원 저보정 SSVEP acquisition-M 학습목표 유지, v2 80-fit 부정과 S001 진단 원본 보존.

Problem signature: 표현은 고정2초 multichannel SSVEP reference bank; 병목은 nominal
주파수/채널/정규화의 source-backed 정의와 query-label 정보경계. 허용은 공개 primary
code/docs 및 기존 저장자료 읽기, 목적은 다음 개발 비교의 단일 공통 baseline 확정이다.
자원은 아래 유한 읽기 예산, 피드백은 정확한 코드 연결/가정/누출·부족한 근거의 식별이며,
실패모드는 query DIN 정답을 predictor에 넘기거나 소스명만으로 montage를 추측하는 것이다.

## 읽기 예산

- Deadline2026-09-13T14:15:00Z. GitHub MAMEM/eeg-processing-toolbox의 고정revision
  5a03abe2a6a874e9adaceea29a52c2fce35d8a03 tree조회1회, 필요한code/docs최대8파일.
  Tree/API wrapper각1MiB cap, decoded code/docs합계512KiB, request별30초,재시도0.
- 별도 official documentation 최대2건; 페이지별1MiB/30초상한. 기존 보존된
  Session/Trial/MOABB/code/article선택본문은 먼저 재사용한다. 새 scholarly검색/PDF0.
- 파일 경로는 tree에서 실제 확인한 것만 선택하고 exactblob/hash/revision/읽은line을
  기록한다. 소스 내용은 research space의 sources에 root가보존한다. 참조코드 실행0.
- 이 단계의 EEG/MAT/NPZ/feature cache/새outcome/학습0. S001b도이단계에서열지않는다.
  held60/사람자료요청/유료0. 비공개/실패source는한번보존후다른합법경로가없으면보류.

Root가source조회/공유문서/SQLite·통합을단독소유한다. 에이전트는 읽기전용으로
기존MOABB설정/저자frequency정보흐름과기본CCA수학을독립검토한다. 새worktree0,
기존45worktrees/8untracked·약295GiBfree 보존. 같은workingtree 동시writer0.

## 선택 규칙과 다음 경계

1. Authorfrequency→classifier API와caller를연결하고 querytruth유래인지확인한다.
2. Channel·reference·filter·covariance변환을source에명시된범위만해석한다.
3. 단일 공통baseline을근거와함께정할수있으면별도개발계약을먼저고정한다:
   nominal5bank vs S001a first1support/class sampleclock5bank, 동일15query/단일baseline.
   모든query에전체bank를주고queryDIN주파수로reference를만드는shortcut은금지한다.
4. Geometry가불명확하면posteriorchannel번호를추측하지않는다. 공통baseline qualification이
   안되면실제개발비교를강행하지않으며,새학습이나parameter rescue로예산을늘리지않는다.
5. 성공해도S001은개발용,같은S002–S011재평가는development reuse다. 공통reference
   수정의이득은학습된M기여가아니다. 새M후보/효능예산은별도사전고정한다.
