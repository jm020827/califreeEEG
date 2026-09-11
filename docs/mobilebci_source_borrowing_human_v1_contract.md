# Source borrowing 실제 개발 실험 실행 계약 v1

2026-09-11, base71939a4. 사용자 “응 시작”에 따른 새 한정 실행이다.
[이전 초안](mobilebci_source_borrowing_human_v1_draft.md)의 과학설정을 채택하고 아래
구현·판정 세부사항을 사람 cache 수치/새 outcome을 보기 전에 확정한다.
초안은 역사문서로 보존하며 이 계약과 SHA고정 코드/config가 실제 실행을 관할한다.

## 범위·역할·유효성

- 고정16명/48run, 기존 hash고정 cache1개만1checksum/1numeric load한다. Raw0.
  Cache 로딩은 전 참가자의 저장 통계를 물리적으로 읽지만 각 fit의 sourcebank/scaler/
  supervised loss는 허용 training 사람만 사용한다. 이를 물리적 heldout 파일 미개봉으로
  오해하지 않는다. 데이터는 이미 노출된 development cohort이지 독립 확인 자료가 아니다.
- Q123/Q2·QM·SHAM125,source4people×3speeds+self13,λ=.1,100AdamWsteps,lr=.05,
  weight_decay=.01,zero initialization,float64CPU1. 생성기작 재실행·새 feature 탐색0.
  Q의 compatibility 평균/std/min은 self도 실제 값이며, 기존 generated helper의 self를
  zero로 만든 보조열을 재사용하지 않는다. Pairwise self differences만0이다.
- 실제 support covariance/cross는 평균 통계다. Class W solve는
  `(C+.1*max(trace(C)/9,1e-12)*I)^−1 XY`; query는비정규화합과그에맞는Gram을쓴다.
  Coherent support compatibility에는500samples를사용한다. 각 bank source는k5,
  self는현재k. Source의모든속도는recipient사람과함께제외한다.
- SHAM은4source사람의sortedID를한칸순환,각speed의M5만대체. TargetMk/EEG/Q/common불변.
  각speed의M2벡터쌍multiset동일·IDfixedpoint0을필수검사한다. 모든 episode에서
  실제pairwise aux 최대변화>0이어야contrast유효. 수치가같으면VALIDITY_INCONCLUSIVE로
  종료하며 donor변경/재시도0. SHAM을chance/조건부독립검정이라고하지않는다.
- Loss는source각run내3class동일가중NLL→세속도/사람동일가중. Standardization은
  sourceepisodes×13expert행의populationmean/std, stdfloor1e−12다. 각episode동일13행으로
  source사람·speed균등가중과일치한다. Query M/labels/indices는router입력0.
- Source4명선택은초안의고정SHA순위,bank행은선택ID사전순×speed수치오름차순,self첫행.
  모든role/subject/row/실제bank/donor목록을저장한다. 예측동점은최저class index.

## 잠금·예산

96innerfits/9600updates를먼저완료하고outer3×arm4의policy를전부저장/fsync/hash고정한다.
Inner2의OOF predictions를모아사람×세속도에같은가중으로평균한값이 **≥0.80**인
가장작은k를선택한다. 없으면k5+source_target_unmet. λgrid선택은없다.
그후48outer-sourcefits를수행한다. 각head는자신의outerquery점수/정답평가전에동결한다.
전체144fits/14400updates/576uniqueclassridge solves상한,실제1attempt/retry0.
이전합성capacity출력은재생성하지않는다.

600초전체실행상한은pin/checksum/load,expert score,fit,평가,bootstrap,직렬화를포함한다.
종료/오류보고를durably저장하는최종정리는timer밖일수있으며그시간도별도보고한다.
산출물128MiB상한:result+models+fitjournal+choices+독립감사용generated/실제episode자료.
Independent saved-artifact audit는추가fit/raw/cacheload0,120초/보고2MiB상한으로별도다.
기존zeroCCA는이전hash고정result의동일subject/speed/query결과를참조하며재적합0.
학습/진단중간로그는파일에flush/fsync하며stdout이끊겨도계산을중단시키지않는다.
첫예상밖오류는실제부분계수와실패를보존해STOPPED_NO_RETRY로종료한다.

Generated adapter suite≤3calls,trainer/CLI integration suite≤3calls;각입력fixture
≤200000numeric elements/2MiB. CPU1,테스트총600초/출력8MiB상한,새의존성0.
Tiny generated CLI smoke는9generatedpeople/3speeds/4k/4arms×2updates로
144mockfits/288updates 이하를사용하고실제효능실험횟수로세지않는다.
Mock bank13·Q123/125·queryclassbalancedloss·globalpolicylock/JSON로그경로를그대로검사한다.
실제cohort는16명이어야하며CLI에서mockbudget을선택할수없다.

## 유지 기준·진단

초안의10개실용criterion을그대로적용하고이전계약의zeroCCA우위도추가필수로유지한다:
QM−Qlowk≥.02,QM−Q2>0,QM−SHAM>0;QM−Q95%CI하한>−.02;
QMpolicyBAcc≥.80,policyQM−QCI하한>−.02;prefixcost비율감소≥.10;
policyreachQM≥Q;highkQM−Q≥−.01;lowk>5pp손실사람≤3;QMpolicy>기존zeroCCA.
CI는participantpairedbootstrap2000,seed20260911,고정crossfitpredictions조건부다.
Q·QM의target-only/균일차용대비손익도보고하되유리한비교로대체하지않는다.

FrozenQM의source입력과해당outer입력에서정해진SHAM으로M만바꿔aux/centeredlogit/
gate/classmargin최대변화를기록한다. 구조/SHAM검증통과후유지하려면각QMfit의source
gate·margin변화>1e−12가필요하다. 작동이0이면자동구현실패로재시도하지않고
NO_DEMONSTRATED_M_ACTUATION이라는후보종료근거로구분한다. Outer진단은해석용,
결과보고feature·학습률·중단기준을고르지않는다.

모두통과하면RETAIN_FOR_INDEPENDENT_VALIDATION,아니면CLOSED_NEGATIVE또는
VALIDITY_INCONCLUSIVE로한정후보를닫는다. Source39/기존336fit/ChoiPARKED/held60은
변경하지않는다. Held60/새외부연락/유료는별도승인,현재0이다.

## 작성 소유권

Main=rootsolewriter(코드/계약/SQLite/런타임파일),3reviewer는읽기전용.
41기존worktrees와8untrackedpytest출력을보존한다. 가용41426812KiB/98%로새tree/
대규모복사/설치를피하고공유API가긴밀한adapter/trainer는순차구현한다. 독립검토는병렬이다.
