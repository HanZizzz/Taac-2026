# Paper Research Agent Report

## 목적
2026 KDD Cup의 알려진 조건(행동 시퀀스 + 비순차 피처를 함께 다루는 추천/CTR 계열 문제 가능성, **no-ensemble**, **latency 제약**, **재현 가능한 코드 제출**)을 기준으로 아래 3편을 실전 관점에서 검토한다.

- [1] **InterFormer**: *Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction* (CIKM 2025)
- [2] **OneTrans**: *Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender* (arXiv 2025)
- [3] **HyFormer**: *Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction* (arXiv 2026)

이 문서는 논문 자체의 novelty 요약보다, **대회에서 실제로 가져다 쓸 수 있는가**에 초점을 둔다.

---

## 1. 먼저 결론

### 한 줄 판단
- **InterFormer**: 가장 보수적이고 현실적인 차용 후보. **아이디어 차용 가치 높음**, 풀모델 구현 부담은 중간.
- **OneTrans**: 이번 KDD Cup 문제 해석과 가장 잘 맞는 **정공법 후보**. 다만 구현 난도와 latency 최적화 부담이 크다.
- **HyFormer**: 연구적으로는 가장 공격적이고 흥미롭지만, **대회 단기전/재현성/엔지니어링 안정성 기준으로는 가장 위험**하다.

### 실전 추천
1. **가장 먼저 빌려올 아이디어**: InterFormer의 *이질 정보 간 양방향 상호작용* 관점
2. **주력 실험 후보**: OneTrans류의 *단일 backbone unified tokenization* 
3. **선별 차용 후보**: HyFormer의 *global query / query boosting* 개념 일부
4. **피해야 할 것**: HyFormer 전체 구조를 대회 초반 주전으로 바로 채택하는 것

### 대회 적합도 순위
- **1위: OneTrans**
- **2위: InterFormer**
- **3위: HyFormer**

단, “논문 완성도”가 아니라 **이번 대회의 no-ensemble + latency + reproducibility 제약 하에서의 실전 적합도** 기준이다.

---

## 2. 왜 이 3편이 이번 대회와 직접 연결되는가

현재 프로젝트 문맥상 이번 KDD Cup은 단순 tabular CTR보다 아래 성격에 더 가깝게 해석된다.

- 유저 행동 시퀀스가 중요할 가능성 높음
- user/item/context 같은 비순차 피처도 강하게 중요함
- 단일 모델로 높은 점수를 내야 함
- latency가 tie-break 혹은 핵심 제약이 될 수 있음
- 코드 제출/재현성 검증이 필요함

즉 핵심 질문은 다음이다.

> **“시퀀스 모델링과 feature interaction을 어떻게 단일 backbone 또는 최소한 강하게 결합된 구조로 처리할 것인가?”**

세 논문은 모두 정확히 이 질문을 다룬다. 차이는 결합 방식이다.

- **InterFormer**: 두 모듈을 유지하되, 상호작용을 양방향으로 강화
- **OneTrans**: 아예 하나의 Transformer backbone으로 통합
- **HyFormer**: unified 쪽으로 가되, 긴 시퀀스를 직접 다루기 위해 global query 기반 hybrid 구조를 설계

---

## 3. 논문별 리뷰

## 3.1 InterFormer

### 3.1.1 핵심 아이디어
InterFormer는 CTR 모델에서 흔한 두 문제를 정면으로 지적한다.

1. **비순차 피처 → 시퀀스** 방향의 정보 흐름만 있고,
   **시퀀스 → 비순차 피처** 역방향 흐름은 약하다.
2. 긴 시퀀스와 다수의 비순차 피처를 다루기 어렵기 때문에,
   너무 이른 단계에서 sum/pool/concat으로 요약해 **정보 손실**이 크다.

이를 해결하기 위해 InterFormer는:
- **Interaction Arch**: 비순차 피처 상호작용 학습
- **Sequence Arch**: 행동 시퀀스 학습
- **Cross Arch**: 둘 사이 정보를 선택적으로 교환

이라는 3부분 구조를 두고, **interleaving style**로 양방향 정보 교환을 수행한다.

핵심은 “둘을 완전히 하나로 합치기”보다, **각 모달리티의 구조는 보존하면서 교차 학습을 세게 만든다**는 점이다.

### 3.1.2 아키텍처 요약
- dense / sparse non-sequential feature 임베딩
- sequence feature 임베딩
- 비순차 피처는 DCNv2, DHEN 같은 interaction module과 결합 가능
- 시퀀스는 MHA 기반으로 context-aware sequence embedding 학습
- Cross Arch가 양쪽 representation을 연결하며 selective summarization 수행
- 토큰 단위 정보를 가능한 오래 유지하고, 너무 빠른 pooling을 피함

### 3.1.3 강점
- **직관적이다.** 기존 CTR 파이프라인에서 업그레이드 포인트가 분명하다.
- **모듈형이다.** 기존 interaction network나 sequence encoder를 완전히 버리지 않고 붙일 수 있다.
- **양방향 상호작용**을 명시적으로 도입했다는 점이 실전적으로 유효하다.
- “완전 통합 backbone”이 부담스러운 팀에게 **중간 단계 전략**으로 좋다.
- 논문상 Meta Ads 배포 결과까지 제시되어 실무 설득력이 있다.

### 3.1.4 약점
- 구조가 깔끔한 single-backbone은 아니다.
- 모듈이 여러 개라서 **엔지니어링 복잡도**가 생각보다 있다.
- unified transformer류에 비해 scaling story가 덜 강하다.
- 대회가 정말로 “하나의 backbone으로 sequence와 feature를 공동 최적화하는 설계”를 높게 평가한다면, 다소 보수적인 해법일 수 있다.

### 3.1.5 compute / latency 관점
- 완전 통합형보다 **추론 경로가 분리**되어 있어 최적화 포인트가 늘어난다.
- 하지만 HyFormer/OneTrans처럼 매우 큰 unified backbone을 돌리는 것보다는 상대적으로 통제 가능하다.
- 초기에 sequence를 과도하게 압축하지 않으므로, naïve 구현 시 비용이 올라갈 수 있다.
- 다만 모듈별 pruning / caching / 경량화가 가능해 **실전 latency 튜닝 여지는 넓다.**

### 3.1.6 이번 대회와의 관련성
**관련성 높음.** 특히 다음 이유에서 그렇다.

- sequence + non-sequence를 함께 다루는 문제 설정과 정확히 맞닿아 있다.
- no-ensemble 상황에서, 기존 strong baseline을 크게 버리지 않고 단일 모델을 강화하기 좋다.
- 완전한 SOTA unified model로 바로 가기 전 **안전한 bridge architecture** 역할을 할 수 있다.

### 3.1.7 대회용 판단
- **빌려올 가치**: 높음
- **바로 주전으로 쓸 가치**: 중간~높음
- **리스크**: 중간

실전에서는 “InterFormer full reproduction”보다도,
**양방향 cross interaction + 늦은 요약 + selective summarization**만 뽑아내도 충분히 가치가 있다.

---

## 3.2 OneTrans

### 3.2.1 핵심 아이디어
OneTrans는 기존 산업 추천 모델의 전형적인 구조인:

- 시퀀스 인코더 따로
- feature interaction 모듈 따로
- 마지막에 붙이는 late fusion

이 설계 자체를 문제로 본다.

대신,
**시퀀스 토큰과 비순차 토큰을 하나의 token sequence로 만들고,
하나의 Transformer backbone 안에서 동시에 처리**한다.

즉 핵심은:
- unified tokenizer
- unified Transformer stack
- mixed parameterization
- serving 효율화를 위한 KV cache / pyramid compression

이다.

이번 대회가 “유저 행동 시퀀스 + heterogeneous features를 단일 backbone으로 통합”하는 방향이라면, OneTrans는 가장 직접적인 참고 논문이다.

### 3.2.2 아키텍처 요약
- non-sequential features → NS tokens
- multi-behavior sequences → S tokens
- 둘을 하나의 token sequence로 합침
- causal mask를 적용한 Transformer stack 처리
- **S-token은 파라미터 공유**, **NS-token은 token-specific parameter** 사용
- pyramid stack으로 깊어질수록 query token 수를 줄여 compute 절감
- cross-request KV caching으로 request 단위 재사용

핵심은 LLM식 엔지니어링 최적화를 ranking/CTR backbone에 가져왔다는 점이다.

### 3.2.3 강점
- **이번 대회 문제 정의와 가장 잘 맞는다.**
- single-backbone이라 **no-ensemble 제약**에 매우 잘 맞는다.
- 구조가 통합되어 있어 “논문화 가능한 스토리”가 좋다.
- scaling 관점이 명확하다. depth/width 확장이 비교적 자연스럽다.
- KV cache, mixed precision, FlashAttention 같은 **현대적 시스템 최적화**를 붙이기 좋다.
- 시퀀스와 피처 상호작용이 초반 레이어부터 jointly optimized 된다.

### 3.2.4 약점
- **구현 난도가 높다.**
- tokenization 설계 실패 시 오히려 성능/latency 둘 다 애매해질 수 있다.
- unified Transformer가 항상 CTR 특화 interaction module보다 잘 먹힌다는 보장은 없다.
- 데이터 공개 직후 빠르게 baseline을 안정화해야 하는 대회 환경에서는, 첫 버전이 불안정할 가능성이 있다.
- 논문에 있는 cross-request KV caching, pyramid schedule, mixed parameterization까지 모두 재현하려면 엔지니어링 비용이 크다.

### 3.2.5 compute / latency 관점
- 기본형은 무겁다. 토큰 수가 늘어나면 비용이 바로 커진다.
- 그러나 이 논문의 진짜 강점은 **최적화 수단을 구조 안에 포함했다는 것**이다.
  - causal attention
  - pyramid compression
  - cross-request KV caching
  - FlashAttention / mixed precision 친화성
- 즉 **잘 만들면 latency-friendly**, 못 만들면 그냥 비싼 큰 Transformer가 된다.

대회 관점에서 보면, 이 논문은 단순히 accuracy paper가 아니라 **accuracy-latency 동시 설계 논문**에 가깝다.

### 3.2.6 이번 대회와의 관련성
**매우 높음.**

이번 프로젝트에서 이미 정리된 대회 해석은 다음이었다.
- 단일 모델 중심
- 시퀀스와 비순차 피처 통합 중요
- latency 제약 중요
- 재현 가능한 구조 중요

이 조건에 OneTrans가 가장 직접적으로 들어맞는다.

특히 아래 점이 중요하다.
- 앙상블 금지면, 내부적으로 표현력을 올리는 unified backbone이 유리하다.
- latency tie-break가 있으면, 추후 cache-friendly한 구조가 매우 중요하다.
- 논문/워크숍까지 본다면 architectural story가 좋다.

### 3.2.7 대회용 판단
- **빌려올 가치**: 매우 높음
- **주전 후보 가치**: 매우 높음
- **리스크**: 중간~높음

결론적으로 OneTrans는 **가장 강한 주전 후보**다. 다만 조건이 있다.

> “논문 전체를 그대로 구현”이 아니라,
> **대회 시간 안에 돌릴 수 있는 축소판 OneTrans**를 먼저 만들고,
> latency/재현성 확인 후 확장해야 한다.

---

## 3.3 HyFormer

### 3.3.1 핵심 아이디어
HyFormer는 OneTrans보다 한 단계 더 나아가,
“그냥 다 합친 Transformer”만으로는 산업 LRM에서 충분치 않다고 본다.

논문의 문제의식은 다음과 같다.
- 긴 시퀀스를 압축할 때 query token이 너무 단순하다.
- late fusion이라 sequence-compressed token과 heterogeneous token의 상호작용이 얕다.
- unified transformer를 무작정 쓰면 serving 효율이 급격히 떨어질 수 있다.

그래서 HyFormer는 **global tokens**를 중심 인터페이스로 두고,
두 과정을 반복한다.

1. **Query Decoding**
   - non-sequential/global 정보로 query를 만들고
   - long sequence의 layer-wise K/V를 cross-attention으로 읽어냄
2. **Query Boosting**
   - decoded query와 non-sequential token을 token mixing으로 섞어
   - 다음 레이어에서 더 강한 query로 다시 sequence를 읽음

즉 “시퀀스를 읽는 쿼리 자체를 점점 강화한다”는 발상이다.

### 3.3.2 아키텍처 요약
- semantic grouping 기반 tokenization
- non-sequential features + pooled sequence summary로 global query 생성
- long sequence는 별도 sequence encoding 경로로 K/V 생성
  - full transformer / LONGER-style efficient encoding / decoder-style lightweight encoding 중 선택 가능
- query decoding으로 global query가 sequence를 읽음
- query boosting으로 query와 NS feature 사이 상호작용 강화
- multi-sequence를 단순 merge하지 않고, sequence별로 따로 decoding한 뒤 query-level에서 상호작용

### 3.3.3 강점
- **긴 시퀀스 처리 문제를 더 진지하게 다룬다.**
- 단순 unified transformer보다 산업 serving을 더 의식한 설계다.
- multi-sequence 상황에서 “무조건 한 줄로 merge”하지 않는 점이 실용적이다.
- compute budget 대비 성능 향상, scaling behavior를 강하게 주장한다.
- query token을 semantic interface로 보는 관점이 흥미롭고 응용 여지가 크다.

### 3.3.4 약점
- 세 논문 중 **가장 구조가 복잡하다.**
- 논문 아이디어는 설득력 있지만, 대회 코드 수준에서 안정적으로 재현하기가 쉽지 않다.
- query generation / decoding / boosting / sequence encoder 옵션이 많아 **하이퍼파라미터 공간이 넓다.**
- 잘못 구현하면 “복잡한데 별로 안 좋은 모델”이 되기 쉽다.
- 논문이 주장하는 장점 상당수가 대규모 산업 로그 + 강한 시스템 최적화 환경에 기대고 있어, 대회 데이터 규모에서 그대로 재현될지는 불확실하다.

### 3.3.5 compute / latency 관점
- 이론적으로는 long sequence를 효율적으로 다루기 위한 구조다.
- 하지만 실제로는 query-decoding과 boosting의 반복으로 인해 구조가 꽤 무겁다.
- multi-sequence를 독립 처리하면 의미론은 보존되지만, 구현/서빙 복잡도는 올라간다.
- GPU pooling, async all-reduce 같은 시스템 최적화까지 전제하는 부분은 **대회 환경에서 그대로 활용하기 어렵다.**

즉 논문 관점에선 효율적일 수 있어도,
**대회 참가자 관점에선 구현 복잡도가 latency보다 먼저 문제**가 될 가능성이 있다.

### 3.3.6 이번 대회와의 관련성
**관련성은 높지만, 직접 채택 위험도도 높다.**

- sequence + feature interaction을 통합한다는 점에서 분명 관련 있다.
- 특히 긴 multi-sequence가 중요할 경우 강한 힌트를 준다.
- 하지만 no-ensemble, 재현성, 빠른 반복 실험이 중요한 대회에서는 “첫 주전 모델”로는 무겁다.

### 3.3.7 대회용 판단
- **빌려올 가치**: 중간~높음
- **주전 후보 가치**: 중간
- **리스크**: 높음

실전에서는 HyFormer 전체보다,
다음만 부분 차용하는 편이 낫다.
- global query 개념
- layer-wise query refinement
- multi-sequence를 무작정 merge하지 않는 설계

---

## 4. 세 논문 비교

## 4.1 핵심 비교표

| 항목 | InterFormer | OneTrans | HyFormer |
|---|---|---|---|
| 기본 철학 | 분리된 모듈 간 양방향 상호작용 강화 | 시퀀스+피처를 하나의 Transformer로 완전 통합 | long sequence와 heterogeneous interaction을 global query 기반 hybrid로 통합 |
| 구조 복잡도 | 중간 | 중간~높음 | 높음 |
| 구현 난도 | 중간 | 높음 | 매우 높음 |
| latency 최적화 잠재력 | 중간 | 높음 | 중간~높음 |
| 재현성 | 비교적 좋음 | 중간 | 낮은 편 |
| no-ensemble 적합성 | 높음 | 매우 높음 | 높음 |
| 대회 초반 안정성 | 높음 | 중간 | 낮음 |
| 연구/논문화 스토리 | 중간~높음 | 높음 | 매우 높음 |
| 이번 KDD Cup 적합도 | 높음 | 매우 높음 | 중간 |

---

## 4.2 no-ensemble 관점
이번 대회는 앙상블 없이도 강해야 한다는 전제가 중요하다.

- **InterFormer**: 기존 strong baseline을 한 단계 진화시키는 단일 모델로 적합
- **OneTrans**: single-backbone 자체가 강력해서 가장 이상적
- **HyFormer**: 단일 모델로는 강할 수 있으나, 안정적 수렴까지 비용이 큼

=> **no-ensemble 관점 최우선은 OneTrans**다.

---

## 4.3 latency 관점
- **InterFormer**: 최적화 여지는 많지만 구조가 다소 분리형이라 end-to-end serving 최적화는 OneTrans보다 약함
- **OneTrans**: KV cache / causal mask / pyramid stack 등으로 가장 체계적인 latency 설계
- **HyFormer**: long sequence 효율화 의도는 강하지만 구조 복잡도가 높아 실제 대회 구현에서는 불리할 수 있음

=> **latency까지 포함하면 OneTrans 우세**.

---

## 4.4 reproducibility 관점
- **InterFormer**: reproduction 난도가 가장 낮다.
- **OneTrans**: 핵심 아이디어는 재현 가능하지만, 논문 full stack 재현은 쉽지 않다.
- **HyFormer**: 재현 부담이 가장 크다.

=> **재현성만 보면 InterFormer 우세**.

---

## 4.5 데이터 미공개 상황에서의 준비 가치
지금은 데이터셋이 공개되지 않았기 때문에, “어떤 구조를 미리 준비할 가치가 있는가”가 중요하다.

- **InterFormer**: 거의 어떤 CTR/recsys 데이터에도 적용 가능한 일반 해법
- **OneTrans**: 데이터가 sequence-heavy일수록 준비 가치가 매우 큼
- **HyFormer**: 정말 긴 multi-sequence 문제가 아니면 과투자 가능성 있음

=> 데이터 공개 전에는 **InterFormer 아이디어 + OneTrans 축소판** 준비가 가장 합리적이다.

---

## 5. 우리 팀에 대한 실전 권고

## 5.1 무엇을 빌려와야 하나

### A. 반드시 테스트할 것
#### 1) OneTrans식 unified tokenization
- sequential / non-sequential feature를 하나의 token space로 넣는 버전
- 단, 처음에는 full 논문 구현 말고 **작은 backbone + 간단 tokenizer**로 시작
- 목표: “single unified backbone이 baseline보다 의미 있게 이기는지” 확인

#### 2) InterFormer식 양방향 정보 흐름
- non-seq → seq 뿐 아니라 seq → non-seq 업데이트를 넣는 구조
- 기존 baseline에 추가하기 쉬운 실전 개선 포인트
- unified backbone이 불안정할 때 좋은 fallback

### B. 선택적으로 테스트할 것
#### 3) OneTrans의 pyramid / KV cache 아이디어
- 대회 추론 제약이 빡셀 경우 우선순위 상승
- accuracy보다 latency가 중요해지는 순간 매우 유효

#### 4) HyFormer의 global query 아이디어
- multi-sequence가 매우 길고, 단순 merge가 성능을 해치면 고려
- 전체 HyFormer 대신 “compressed global token set” 정도만 차용하는 게 현실적

### C. 피하거나 뒤로 미룰 것
#### 5) HyFormer full reproduction
- 대회 초반에는 비추천
- 시간 대비 리턴이 불확실
- 재현성과 디버깅 비용이 너무 큼

---

## 5.2 추천 실험 순서

### 1단계: 안전한 strong baseline
- tabular/context feature interaction baseline
- sequence encoder + late fusion baseline
- strict CV / latency profiler / inference pipeline 먼저 완성

### 2단계: InterFormer-lite
- 기존 baseline 위에 양방향 cross interaction 추가
- 너무 이른 pooling 제거
- selective summarization 실험

### 3단계: OneTrans-small
- unified tokenizer
- 작은 Transformer backbone
- sequence length / NS token 수 / mask 구조 ablation
- latency 측정 병행

### 4단계: OneTrans-efficient
- pyramid compression
- candidate/request caching 가능성 검토
- mixed parameterization 단순화 버전 실험

### 5단계: 필요 시 HyFormer 아이디어 부분 차용
- global query token 수 실험
- multi-sequence 분리 처리 후 query-level fusion 실험

---

## 5.3 최종 추천 전략

### 추천 주전 방향
가장 합리적인 주전 방향은 다음이다.

> **OneTrans를 목표 아키텍처로 삼되,
> InterFormer를 안전한 중간 다리로 활용하고,
> HyFormer는 부분 아이디어만 차용한다.**

이유는 단순하다.

- OneTrans는 이번 대회의 “single backbone + sequence + feature interaction + latency” 요구와 가장 잘 맞는다.
- 하지만 바로 full-size로 가면 실패 확률이 높다.
- InterFormer는 성능과 안정성 사이에서 좋은 중간 단계다.
- HyFormer는 흥미롭지만, 지금 우리에게는 full adoption보다 selective borrowing이 훨씬 낫다.

---

## 6. 논문별 최종 판정

### InterFormer
- **판정**: 적극 차용
- **권장 역할**: 안정적인 주전 보조축 / strong baseline upgrade
- **한줄 코멘트**: 가장 실전적이고 덜 위험한 개선안

### OneTrans
- **판정**: 최우선 실험
- **권장 역할**: 최종 주전 후보 1순위
- **한줄 코멘트**: 이번 KDD Cup 방향성과 가장 잘 맞는 단일 backbone 설계

### HyFormer
- **판정**: 부분 차용
- **권장 역할**: 후반부 고급 아이디어 저장소
- **한줄 코멘트**: 연구적으로는 매력적이지만 대회 주전으로는 부담이 큼

---

## 7. 최종 요약

이번 3편을 대회 관점으로 다시 압축하면 다음과 같다.

- **InterFormer**는 “기존 분리형 파이프라인을 어떻게 더 똑똑하게 연결할 것인가”에 대한 답이다.
- **OneTrans**는 “애초에 왜 분리하느냐, 하나의 backbone으로 가자”는 답이다.
- **HyFormer**는 “하나로 가는 건 맞지만, 긴 시퀀스와 serving 효율까지 생각하면 query 중심 hybrid가 더 낫다”는 답이다.

이번 KDD Cup의 제약을 고려하면, 우리 팀의 최적 행동은 다음이다.

> **초기에는 InterFormer-lite로 안전하게 올라가고,
> 주전은 OneTrans-small/efficient 계열로 밀고,
> HyFormer는 global query 등 일부 아이디어만 선별 도입한다.**

이 조합이 **성능 / latency / 재현성 / 개발 속도**의 균형이 가장 좋다.

---

## 참고 메모
본 보고서는 각 논문의 공개 초록/본문 설명을 기반으로, 프로젝트 내 기존 KDD Cup 해석 문서(대회 제약, unified backbone 필요성, single-model/latency/reproducibility 우선 전략)와 연결하여 작성했다.