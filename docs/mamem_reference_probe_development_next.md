# 다음: source-inspired E126 reference 민감도 개발 진단

2026-09-13 **미실행 초안**. 합성 operator 통과와 실제 EEG 검증은 별개다.
완전한 author MAMEM-I CCA baseline qualification은 실패한 상태로 보존한다.
그 계획을 다시 실행하지 않으며, 아래의 더 좁은 질문을 위한 새로운 실제 계약이 필요하다.
사용자가 부여한 유한 루프 내 설계 자율성으로 다음 단계에서 계약을 고정할 수 있으나,
이 초안이나 합성 통과만으로 실제 raw 실행을 허용하는 것은 아니다.

## 질문과 선택 이유

공간필터를 적합하지 않는 E126 하나에서 S001a support로 만든 전체5reference bank가
nominal bank보다 S001b의 알려진 자극을 잘 구별하는가? 이 비교는 reference 민감도와
기본 신호 검출의 개발 진단이다. 학습된 acquisition-M이나 실용 보정량 절감 시험이 아니다.
E126은 author Default/SampleSelection에서 결과 독립 선택했다. Oz·최적전극이라 부르지 않는다.

## 제안되는 유한 범위

- 실제 입력은 기존 개발용S001a/b만; a/b각 최대1decode. source S002–S011 및
  c/d/e/held60 접근0. 신규 다운로드/외부사람요청/유료0.
- S001a는 DIN_1/samplingRate만 decode하여 첫 main support1trial/class에서
  선택2초창 event sample-index로5Hz를 만든다. 5지원trial의 비용과 마지막 지원
  trial까지 elapsed prefix를 세며 zero-calibration이라고 부르지 않는다.
  나머지지원trial정보와a EEG는사용하지않는다. 기존parser의23group검증/label계산은
  모두 읽게 됨을 명시해야 한다(지원 trial선택과 의미검증용,예측 feature아님).
- S001b EEG는E126·고정15창[1,3)만 수치사용한다. B DIN은window/평가label만;
  query DIN 주파수는bank나score에제공하지않는다. 모든query에같은각5classbank.
- 단일 operator·2banks·15queries·총30predictions·0fits·단회실행. 임의채널/정규화/
  filter/window/clock/seed변경금지. S001은이미개발에사용돼독립효능확증이아니다.
- 계약에서 실제 input path/bytes/hash/역할·samplingRate·실행deadline·CPU/RAM/출력cap,
  30scores-vector의schema/실행핀,독립저장산술감사와누출검사를먼저고정한다.

관측 전 정할 방향 기준 제안: 각 bank의15개 중≥12정답 및 각class≥1/3정답이면
단지 '해당개발기록에서명확한검출'로기록한다. 둘다미달이면이probe를종료하고
좋아지는설정까지sweep하지않는다. 어느조건이통과해도1인의소규모개발진단이며
통계적일반화·강한baseline확보·학습된M효용으로승격하지않는다.

Reference 변경 이득이 있어도 이를 모든 Q/Q2/QM/SHAM의 공통 입력으로 처리한다.
잔여metadata를학습할새기작이있는지,공통정보를넘는효과를볼강한baseline과
독립cohort가있는지별도판단한다. 이전v2 RETIRE는취소하지않는다.
