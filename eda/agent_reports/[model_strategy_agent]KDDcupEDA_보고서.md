# 모델 전략 에이전트 보고서

## 목적
이 문서는 `tasks/KDDcupEDA/analysis_agent_report.md`의 규정/절차 분석을 전제로, **이번 KDD Cup에서 실제로 제출 가능한 합법적 모델링 전략**을 다시 정리한 것이다. 

핵심 전제는 다음 네 가지다.
- **앙상블 금지**: 제출물은 반드시 **single model + single inference pipeline** 이어야 한다.
- **코드 검증/재현성 필수**: 최고 점수 제출물의 학습·추론 코드가 공식 환경에서 재현 가능해야 한다.
- **latency 제약 실질 반영**: 점수가 높아도 latency 제한을 넘으면 무효다.
- **문제 성격 가정**: 대회명과 규정 문구상, 이번 과제는 **대규모 추천/CVR 계열**이며, **sequence modeling + feature interaction**을 함께 다루는 형태일 가능성이 높다.

따라서 이 보고서는 일반적인 대회형 “여러 모델을 섞어서 최고 점수 만들기” 전략이 아니라, **단일 모델을 점진적으로 강화하는 전략**에 초점을 둔다.

---

## 1. 핵심 결론 요약

1. 이번 대회에서 가장 중요한 전략 원칙은 **"복잡한 앙상블"이 아니라 "강한 단일 모델"** 이다.
2. 모델 선택의 기준은 단순 AUC 최대화가 아니라, **AUC + latency + 재현성 + 코드 검증 통과 가능성**의 동시 만족이다.
3. 추천/CVR 계열 가능성이 높으므로, 출발점은 **범주형/행동 로그 feature interaction을 잘 처리하는 단일 backbone** 이어야 한다.
4. sequence 정보가 중요할 가능성이 크므로, 최종 지향점은 **"행동 시퀀스 요약 + feature interaction"을 한 모델 안에서 통합**하는 구조가 유력하다.
5. Round 1부터 이미 **최고 제출물 기준 코드 검증**을 염두에 둬야 하므로, 실험 단계에서도 **best submission commit / config / seed / inference 경로 고정**이 필요하다.
6. 특히 Industrial Track은 latency 제약이 더 엄격할 가능성이 높으므로, **과도하게 깊은 구조나 무거운 추론 경로는 후반에 오히려 발목을 잡을 수 있다.**

한 줄로 정리하면:

> **이번 대회 최적 전략은 “단일 모델을 끝까지 고도화하되, sequence와 feature interaction을 한 inference graph 안에서 처리하고, latency와 재현성을 처음부터 설계에 포함시키는 것”이다.**

---

## 2. 규정 기반 모델링 원칙

`analysis_agent_report.md`에 따라, 모델 전략은 아래 원칙을 벗어나면 안 된다.

### 2.1 반드시 지켜야 하는 원칙
- 제출은 **단일 모델(single model)** 이어야 한다.
- 추론은 **단일 파이프라인(single inference pipeline)** 이어야 한다.
- 최고 점수 제출물의 코드를 공식 환경에서 재현할 수 있어야 한다.
- latency 제한은 점수와 별개인 참고값이 아니라 **유효성 게이트**다.
- tie-break도 낮은 latency가 유리하므로, 상위권일수록 속도 최적화가 중요하다.

### 2.2 전략적으로 의미하는 바
즉, 이번 대회에서는 아래 사고방식이 필요하다.
- “여러 모델을 섞어서 최고 점수” → **불가 또는 매우 위험**
- “단일 백본을 더 잘 설계해서 최고 점수” → **정석**
- “학습은 복잡해도 추론은 단순” → **가능하지만 재현 가능해야 함**
- “후처리로 여러 checkpoint/variant 예측 결합” → **금지 가능성 높음**

---

## 3. 무엇이 허용되고, 무엇이 위험하며, 무엇이 사실상 금지인가

이번 섹션은 실제 실험 운영에서 가장 중요하다.

## 3.1 명확히 허용되는 것
아래는 규정상 비교적 안전한 범주다.

### A. 단일 모델 내부의 구조적 복잡성
- 하나의 모델 안에 여러 블록이 들어가는 것
- 예: embedding block + sequence encoder + MLP interaction head
- 예: DIN/DIEN류 attention block + dense tower + categorical embeddings
- 예: Transformer encoder 위에 CTR prediction head를 붙인 단일 네트워크

핵심은 **파라미터 세트가 하나이고, 최종 예측이 하나의 모델 forward path에서 생성되는가**다.

### B. 단일 학습본을 이용한 추론
- 하나의 최종 checkpoint를 선택해 추론
- quantization/pruning 등 단일 모델 경량화
- feature hashing, embedding compression, layer reduction
- knowledge distillation 결과로 얻은 **학생 단일 모델** 제출

즉, 학습 과정에서 여러 시도를 하더라도, **최종 제출물 자체는 하나의 모델**이면 된다.

### C. 단일 파이프라인 내 합법적 feature engineering
- user/item/context count, frequency, recency, time-gap features
- sequence pooling 결과를 입력 feature로 사용
- groupby 통계, cross feature, normalized numerical features
- offline precomputed feature table를 inference pipeline의 일부로 사용하는 것

단, 해당 feature 생성 경로도 **재현 가능**해야 하고, official environment에서 다시 만들 수 있어야 안전하다.

## 3.2 위험한 것
아래는 규정 문구상 해석에 따라 문제될 수 있으므로 매우 보수적으로 접근해야 한다.

### A. multiple checkpoints 활용
- epoch별 checkpoint 평균 예측
- best checkpoint A와 B를 상황에 따라 선택
- 여러 checkpoint를 score-based로 route해서 예측

규정상 **multiple checkpoints combined prediction**이 금지 예시로 언급돼 있으므로 매우 위험하다.

### B. multiple model variants
- 같은 backbone이라도 hidden size, feature set, input window가 다른 여러 버전의 예측 결합
- variant A는 short sequence, variant B는 long sequence를 보고 평균

이 역시 규정상 **multiple model variants combined prediction**으로 볼 가능성이 높다.

### C. test-time augmentation에 가까운 반복 추론 결합
- sequence crop 여러 개를 뽑아 평균
- window 여러 개를 태워 평균
- dropout inference 여러 번 돌려 평균

기술적으로 TTA처럼 보일 수 있지만, 운영진 입장에서는 **사실상 다중 예측 결합**으로 해석될 여지가 크다.

### D. 2-stage 중 “실질적으로 서로 독립 학습된 두 모델” 구조
- candidate generation 모델 따로 학습
- ranking 모델 따로 학습
- 최종 제출물이 두 모델 결합 없이는 동작하지 않음

일반 추천 시스템에서는 흔한 구조지만, 이번 규정의 **single model + single inference pipeline** 문구를 보수적으로 해석하면 위험하다. 
특히 “독립 학습된 복수 모델”로 보이면 방어가 어렵다.

## 3.3 사실상 금지로 보는 것이 안전한 것
- averaging / weighted averaging
- voting
- stacking / blending
- seed ensemble
- fold ensemble
- independently trained multiple models 결합
- 서로 다른 backbone 결합
- 후보 모델 + 재랭커 모델의 점수 결합이 명백한 경우
- multi-checkpoint average
- 여러 모델의 logits/probability 결합

즉, 기존 Kaggle식 상위권 전략의 상당 부분은 이번 대회에서 그대로 쓰면 안 된다.

---

## 4. 문제 유형 가정: 왜 추천/CVR + sequence + feature interaction으로 보는가

규정 분석 보고서에 나온 공식 대회명은 다음을 강하게 시사한다.

> Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendation

여기서 실무적으로 읽어야 할 포인트는 세 가지다.

### 4.1 추천/광고/CTR/CVR 계열 가능성
- user-item-context 삼자 구조가 있을 가능성이 높다.
- label은 binary click / conversion / engagement 계열일 확률이 높다.
- metric이 ROC-AUC인 점도 CTR/CVR 스타일과 잘 맞는다.

### 4.2 sequence modeling 필요성
- 과거 행동 이력, 세션 순서, 최근성, 노출/클릭 전이 등이 중요할 가능성이 높다.
- 단순 tabular만으로는 ceiling이 낮을 수 있다.

### 4.3 feature interaction 필요성
- user 특성 × item 특성 × context 특성의 교차 효과가 중요할 가능성이 높다.
- wide & deep, DCN, xDeepFM, AutoInt, DIN/DIEN류 사고가 유효할 수 있다.

따라서 초기 baseline은 tabular CTR 모델에서 시작하되, 최종 경쟁 모델은 **sequence와 interaction을 통합한 단일 backbone** 으로 가는 흐름이 합리적이다.

---

## 5. 권장 단일 모델 발전 경로: baseline → contender

이번 대회에서는 앙상블이 아니라 **단일 모델을 점진적으로 강화하는 progression** 이 중요하다.

## 5.1 Stage 0: 재현 가능한 베이스라인 파이프라인 고정
가장 먼저 할 일은 최고 점수 모델이 아니라 **재현 가능한 최소 제출 파이프라인**을 만드는 것이다.

필수 구성:
- 데이터 로딩 스크립트
- feature generation 스크립트
- 학습 스크립트
- 단일 checkpoint 저장 규칙
- inference script / service entrypoint
- seed / config / dependency 고정
- latency 측정 스크립트

이 단계의 목적은 두 가지다.
1. Round 1부터 검증 대응 가능한 구조를 만든다.
2. 이후 모델이 복잡해져도 “합법적 제출 경로”를 잃지 않는다.

## 5.2 Stage 1: 강한 표준 baseline
초기 baseline은 **복잡한 시퀀스 모델**보다 먼저, **빠르고 튼튼한 CTR/CVR형 단일 모델**로 잡는 것이 좋다.

추천 후보:
- **LightGBM / CatBoost 단일 모델**
- 또는 **단순 DNN 기반 CTR 모델(embedding + MLP)**

### 왜 이런 순서인가
- baseline은 빠르게 학습/검증 가능해야 한다.
- latency 추정이 쉽다.
- feature importance와 오류 분석이 쉽다.
- sequence block이 정말 필요한지 비교 기준점이 생긴다.

### baseline feature 권장
- user/item/context 기본 ID embedding 또는 encoded categorical
- count / frequency
- recency / time gap
- 최근 행동 개수
- item popularity / user activity level
- group statistics

이 단계에서는 **“단일 모델로도 꽤 강한 점수”**를 먼저 확보해야 한다.

## 5.3 Stage 2: feature interaction 강화
baseline 다음 단계는 대개 sequence보다 먼저 **interaction modeling 강화**가 효율적일 수 있다.

추천 방향:
- **DeepFM / xDeepFM / DCN-V2 / AutoInt** 계열 검토
- 범주형 feature 조합을 잘 흡수하는 구조 사용
- sparse + dense 혼합 입력 처리

### 기대 효과
- user-item-context 조합 효과를 더 잘 잡는다.
- GBDT에서 손으로 만들던 cross feature 일부를 네트워크가 내재적으로 흡수한다.
- 여전히 추론 구조는 단일 모델로 유지 가능하다.

### 주의점
- embedding 차원이 커질수록 latency와 메모리 비용이 증가한다.
- 고카디널리티 ID가 많으면 embedding table 최적화가 중요하다.

## 5.4 Stage 3: sequence 신호 통합
대회명상 가장 유력한 승부처는 여기다. 
다만 이 단계도 **독립 sequence 모델 추가**가 아니라, **기존 단일 모델 내부에 sequence encoder를 붙이는 방식**으로 가야 한다.

추천 방향:
- 최근 N개 행동을 입력으로 사용
- sequence pooling / attention pooling / Transformer block / GRU block 중 하나 선택
- sequence representation을 interaction head와 결합

구체적으로는 아래 형태가 안전하다.
- categorical embeddings
- dense/context features
- behavior sequence encoder
- concatenation / gated fusion
- single prediction head

즉, **하나의 그래프 안에서 sequence와 interaction을 함께 처리하는 구조**가 이번 대회 취지와도 가장 잘 맞는다.

### 현실적인 후보 구조
1. **DIN 계열**
   - 타깃 item과 과거 행동 간 attention
   - CTR/CVR 문제와 잘 맞음
   - 비교적 해석이 쉽고 실전성이 높음

2. **DIEN/GRU 기반**
   - sequence order를 좀 더 직접 반영
   - 단, latency와 구현 복잡도 증가 가능

3. **얕은 Transformer + interaction head**
   - sequence modeling 유연성 높음
   - 긴 시퀀스에서는 latency 부담이 커질 수 있음

4. **SASRec/BERT4Rec류를 단일 예측 모델로 변형**
   - 추천 시퀀스에는 강할 수 있음
   - 그러나 과제 target이 단일 AUC 기반 CTR/CVR라면 과설계일 수 있음

## 5.5 Stage 4: contender 모델 선택
최종 후보는 아래 셋 중 하나로 좁히는 것이 현실적이다.

### 옵션 A. 가벼운 interaction 중심 모델
- DeepFM / DCN-V2 / AutoInt 계열
- 장점: 빠르고 안정적
- 적합: latency가 매우 빡세고 sequence 이득이 제한적일 때

### 옵션 B. DIN류 sequence-aware CTR 모델
- 장점: 추천/CTR 문맥에 매우 실전적
- 적합: 최근 행동 이력이 확실히 강한 신호일 때

### 옵션 C. 얕은 Transformer 기반 unified model
- 장점: sequence + interaction 통합 표현력이 좋음
- 적합: 데이터 규모가 크고 sequence 구조가 핵심일 때
- 단점: latency/재현성/튜닝 복잡도 관리가 더 어렵다.

실전적으로는 **Round 1은 B 혹은 A**, **Round 2 후보로 C를 검토**하는 흐름이 가장 안전해 보인다.

---

## 6. latency 제약을 반영한 아키텍처 원칙

이번 대회에서 latency는 부가 조건이 아니라 본선 진입과 최종 순위를 결정하는 핵심 요소다.

## 6.1 latency 친화적 설계 원칙
- sequence 길이를 무작정 늘리지 않는다.
- embedding table를 과도하게 키우지 않는다.
- attention block 수를 최소화한다.
- online feature dependency를 줄인다.
- inference path branching을 최소화한다.
- CPU/GPU 어느 환경에서든 재현 가능한 속도를 목표로 한다.

## 6.2 sequence 관련 실무 가이드
- 먼저 **최근 10~50개 행동** 정도의 짧은 윈도우에서 시작
- 성능 이득이 크지 않으면 긴 history를 포기
- self-attention은 layer 수와 head 수를 얕게 유지
- mean pooling / attention pooling 같은 경량 요약도 충분히 강할 수 있음

## 6.3 feature 관련 실무 가이드
- online join이 많은 feature는 위험하다.
- inference 시 복잡한 lookup 체인을 만들지 않는다.
- 가능하면 학습/추론 모두에서 동일한 전처리 경로를 사용한다.
- 고카디널리티 categorical은 pruning / hashing / shared embedding 등을 검토한다.

## 6.4 트랙별 시사점
- **Academic Track**: 상대적으로 모델 자유도가 조금 더 있을 수 있음
- **Industrial Track**: latency가 더 엄격하므로, 너무 무거운 sequence 모델은 초반부터 배제할 필요가 있다.

즉 Industrial이라면, **DIN급 경량 모델 또는 interaction 중심 모델**이 실제 승률이 더 높을 수 있다.

---

## 7. validation 전략도 규정에 맞게 바꿔야 한다

이번 대회는 submission budget이 작고, latency invalid 가능성이 있으며, public/private drift 가능성도 있다. 따라서 validation도 일반 대회처럼 느슨하게 하면 안 된다.

## 7.1 기본 원칙
- **AUC와 latency를 함께 관리**한다.
- submission 3회/일 제한을 감안해, 로컬에서 충분히 걸러낸 뒤 제출한다.
- sequence/recommendation 문제라면 **시간순 split**을 우선 고려한다.
- user/session leakage를 막기 위해 group-aware split이 필요하다.

## 7.2 추천 validation 구조
1. **Main split**: 모델 선택용
2. **Stress split**: 시간 drift / cold-start 대응력 확인용
3. **Latency benchmark**: 후보 모델별 추론 속도 체크

이번 대회에서는 “CV 점수 조금 상승”보다, **CV 점수 유지 + latency 통과 + 재현성 확보**가 더 가치 있다.

## 7.3 모델 선택 기준
최종 선택은 아래처럼 보는 것이 합리적이다.

- 1순위: latency 제한 내에서 가장 높은 AUC
- 2순위: 재현이 가장 안정적인 구조
- 3순위: Round 2 확장 데이터에서도 버틸 수 있는 메모리/학습 안정성

---

## 8. 재현성 중심 실험 운영 원칙

규정상 상위 제출물은 코드 검증 대상이므로, 실험 관리 자체가 모델 전략의 일부다.

## 8.1 반드시 남겨야 할 것
- best run의 git commit 또는 코드 snapshot
- config 파일
- random seed
- train/valid split version
- feature version
- 학습 시작/종료 시간, 데이터 범위
- 최종 checkpoint 경로
- 제출에 사용한 정확한 inference command

## 8.2 제출용 규격 관점에서 좋은 구조
- `train.py`: end-to-end 학습 진입점
- `infer.py`: 단일 checkpoint 기반 추론 진입점
- `config.yaml`: 실험 설정 고정
- `prepare_features.py`: feature 재생성 가능
- `measure_latency.py`: 동일 입력 배치 기준 속도 측정

## 8.3 피해야 할 운영 습관
- notebook 수작업 전처리
- 임시 feature csv를 수동 덮어쓰기
- checkpoint를 여러 개 두고 그때그때 골라 쓰기
- 실험별 seed가 기록되지 않는 상태
- 제출물과 실제 코드 버전이 어긋나는 상태

이번 대회는 “좋은 모델”만으로는 부족하고, **검증 가능한 모델**이어야 한다.

---

## 9. 단일 모델 기준의 구체적 추천안

## 9.1 가장 현실적인 기본 추천
데이터 공개 전 현재 시점에서 가장 실전적인 기본안은 아래다.

### 기본안
- **CTR/CVR용 단일 DNN 또는 GBDT baseline** 구축
- 이후 **DIN/DCN-V2/DeepFM 계열** 중심으로 확장
- sequence가 강하면 **DIN 또는 경량 Transformer 통합형**으로 수렴

### 이유
- 추천/CVR 문맥에 잘 맞는다.
- feature interaction을 잘 다룬다.
- sequence를 한 모델 내부에서 통합하기 쉽다.
- 앙상블 없이도 경쟁력 있는 단일 모델을 만들 수 있다.

## 9.2 아키텍처 우선순위 제안
아래 순서대로 실험하는 것이 좋다.

### 1순위: DCN-V2 / DeepFM 계열
- 가장 안정적인 단일 모델 출발점
- interaction 효과 확인에 좋음
- latency 관리가 비교적 수월

### 2순위: DIN 계열
- user behavior sequence가 중요할 때 강력
- CTR/CVR에 직접적
- sequence와 target item 상호작용을 자연스럽게 반영

### 3순위: 경량 Transformer 통합형
- 데이터가 충분히 크고 시퀀스가 핵심이면 도전 가치 있음
- 단, Round 1보다는 Round 2 확장안에 가깝다.

## 9.3 GBDT의 역할
이번 대회에서 GBDT는 최종 제출용보다 아래 용도로 특히 유용하다.
- 빠른 baseline
- feature sanity check
- leakage 탐지
- sequence 모델 대비 성능 기준점 확보

단, 최종 제출물은 규정·문제 성격을 감안하면 **sequence + interaction 통합형 단일 신경망** 쪽이 더 자연스럽다.

---

## 10. Round 1과 Round 2를 나눠 본 전략

## 10.1 Round 1 전략
Round 1의 목표는 단순히 public score를 높이는 것이 아니라, **합법적 contender 구조를 빠르게 확정**하는 것이다.

권장 흐름:
1. 재현 가능한 baseline 파이프라인 완성
2. GBDT 또는 단순 DNN baseline 확보
3. interaction 모델(DCN/DeepFM)로 확장
4. sequence block 통합(DIN 등)
5. latency 측정 후 제출 후보 고정

Round 1에서는 지나치게 야심찬 구조보다, **검증 가능한 강한 단일 모델**이 더 중요하다.

## 10.2 Round 2 전략
Round 2는 데이터가 약 10배로 커지고 latency도 더 엄격해진다.

따라서 Round 2에서는 다음이 중요하다.
- 더 깊은 모델보다 **스케일 안정성**
- feature table 크기와 embedding 메모리 관리
- 긴 sequence보다 **짧고 강한 sequence summary**
- 논문 스토리와 연결되는 **unified block 설계**

Round 2용 이상적 방향은:
- 하나의 반복 가능한 unified block을 정의하고
- block depth / sequence length / embedding width에 대한 scaling law를 실험하며
- 최종적으로 latency 내에서 가장 효율적인 단일 모델을 선택하는 것이다.

이건 규정 분석 보고서의 **paper review 기준(unified, stackable block / scaling laws)** 과도 잘 맞는다.

---

## 11. 이번 대회에서 버려야 할 사고방식

기존 Kaggle 스타일 습관 중 이번 대회에서 버려야 할 것들을 분명히 적어둘 필요가 있다.

### 버려야 할 것
- fold 평균 제출
- seed 평균 제출
- 여러 모델 가중 평균
- stage별 독립 모델 점수 결합을 당연시하는 사고
- public leaderboard 보고 조합 바꾸기
- best checkpoint 여러 개를 섞는 운영

### 대신 가져가야 할 것
- 한 모델 안에서 성능을 끌어올리는 설계
- speed-aware architecture search
- feature engineering과 sequence encoding의 통합
- 제출물 수준의 재현성 관리
- latency까지 포함한 validation

즉, 이번 대회는 **“리더보드용 조합 기술”보다 “제출 가능한 시스템 설계 능력”이 더 중요하다.**

---

## 12. 최종 제안

현재까지의 규정 분석을 반영한 최종 제안은 아래와 같다.

### 최우선 전략
- **single model + single inference pipeline** 원칙을 절대 기준으로 둔다.
- baseline부터 최종 후보까지 모두 이 원칙 안에서만 발전시킨다.

### 모델 발전 경로
1. **GBDT 또는 단순 CTR DNN baseline**
2. **feature interaction 강화: DeepFM / DCN-V2 / AutoInt**
3. **sequence 통합: DIN 또는 경량 Transformer block 추가**
4. **latency/재현성 기준으로 최종 단일 모델 1개 고정**

### 가장 현실적인 제출 후보
- **Round 1**: DCN-V2 또는 DIN 계열의 경량 단일 모델
- **Round 2**: sequence + interaction unified block을 갖춘 단일 모델
  - 단, block 수와 sequence 길이는 latency 내에서 제한

### 운영 원칙
- AUC만 보지 말고 latency와 재현성을 함께 본다.
- submission budget이 적으므로 로컬 검증 체계를 강하게 만든다.
- 최고 제출물은 언제든 코드 검증 가능한 상태로 유지한다.

마지막으로 가장 중요한 문장을 다시 적으면:

> **이번 KDD Cup의 승부처는 앙상블이 아니라, 추천/CVR 문맥에서 sequence와 feature interaction을 하나의 단일 모델 안에 얼마나 효율적으로 통합하느냐이다.**

> **좋은 전략은 “가장 복잡한 모델”이 아니라, “latency를 넘지 않고 재현 가능하며 규정상 안전한 가장 강한 단일 모델”을 만드는 것이다.**
