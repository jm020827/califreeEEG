# Pre-Gelled 연구: 단일 공개 원문 자격 확인 — DRAFT / 미실행

2026-09-14 KST / 2026-09-13 UTC. **새 학습/데이터 후보가 아니라 독서 후보**다.
Round4의2검색/4primary URL 예산은 종료했고 다른 키워드 검색을 보충하지 않는다.

정확한 문헌: *A Pre-Gelled EEG Electrode and Its Application in SSVEP-Based BCI*,
DOI `10.1109/tnsre.2022.3161989`, OpenAlex `W4220855852`.
현재 근거는 색인된 author abstract뿐이다. DOI open은 도구 safe-open 오류였으며
실제 저널의 공개/비공개·계정 요구를 확인한 것은 아니다. 성능·공개 실측 필드·보정량은
원문 확인 전에는 주장하지 않는다.

## 왜 이 논문 한 개인가

초록에서 접촉 임피던스 비교와 SSVEP 과제가 함께 언급된다. 그러나 다음 중 어느 경우인지는
모른다: (a) 실제 pairing 가능한 측정값이 공개됨, (b) aggregate impedance/하드웨어 비교만
보고됨, (c) 데이터가 요청 의존임. 이를 구분하면 다음 schema 단계의 진입 여부가 달라진다.
기존 negative result를 구제하는 계수/채널/참가자 선택은 아니다.

## 다음 별도 실행 계약에 고정할 범위

- 새 broad search 대신 **정확한 논문 한 개의 합법적 공개 원문 해석/접근 경로 확인 1회**.
  현재 DOI/실패 publisher endpoint를 같은 방식으로 재시도하지 않는다. 공개 저자 원고나
  공식 저장소 사본을 lawful resolver로 확인하는 별도 작업으로 구분한다.
- 실행 전에 기존 resolver의 실제 source routing/자동 다운로드/크기·시간 제한을 읽고,
  허용된 공개 범위를 보장할 수 있을 때만 사용한다. 도구 call 수를 underlying HTTP 수로
  바꾸어 세지 않는다. 인증·유료·저자 요청·접근 우회가 필요하면 즉시 PARK하고 멈춘다.
- PDF가 합법적으로 확보될 경우에만, **이미 등록한 deep-read queue**의 질문에 답하는
  Methods/Data Availability/관련 한계 부분을 읽는다. 실제 PDF hash·revision·페이지/
  section을 기록하고 표/그림에 의존하면 렌더링한다. 공개 PDF를 얻기 전에는 읽었다고 하지 않는다.
  현재 세션에 PDF 전용 skill이 없으면 그 한계를 알리고 사용 가능한 문서 추출·시각 확인
  도구로 대체하되 evidence 범위를 지킨다. 세부 byte/시간 한도는 새 실행 전 계약에서 고정한다.
- 논문의 raw/annotation/실제 파일 header·수치·fit·EEG예측·held60·발송·유료 0.

답할 질문:

1. 실제 impedance와 SSVEP EEG가 공개돼 있는가, 아니면 평균/기준만 보고했는가?
2. 정확한 release/file·사람/session/block/channel의 연결, 단위·측정 주파수·측정 시점은?
3. Support 전에 얻는 정보와 query/future/common electrode 조건을 분리할 수 있는가?
4. 접촉/재습윤/전극 교체/setup·label 획득 비용을 셀 수 있는가?
5. 외부 M을 학습에 넣는 것이 Q/common·직접 보정 이상의 이득을 낼 근거가 있는가?

결과는 DOCUMENTED_SCHEMA_NEXT_CANDIDATE / NO_RELEASE_OR_RELEVANT_M_DOCUMENTED /
PARK_PUBLIC_FULLTEXT_UNRESOLVED 중 하나이며, 어느 경우에도 직접 학습으로 넘어가지 않는다.
원문이 불충분하면 다음 논문을 무한히 추가하지 않고 부족한 근거와 이번 종료를 보고한다.
