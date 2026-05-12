# KDD Cup EDA/Modeling PM 최종 보고서

## 문서 목적
이 문서는 현재까지 취합된 4개 축의 산출물(규정 분석, EDA 전략, 모델 전략, 논문 리서치 보강)을 바탕으로, **이번 KDD Cup을 실제로 어떻게 운영할지** PM 관점에서 정리한 최종 실행 문서다.

핵심 전제는 하나다.

> **이번 대회는 single model + single inference pipeline 원칙을 강하게 요구하며, ensemble은 공식적으로 금지된다.**

따라서 기존 Kaggle식 다중 모델 평균, seed 평균, fold 평균, stacking/blending, multi-checkpoint averaging 전제 전략은 이번 대회의 기본 전략으로 채택할 수 없다.

---

## 1. 최상위 결론

### 1.1 우리가 따라야 할 대회 운영 원칙
1. **단일 모델 원칙 준수**
   - 모든 제출은 single model + single inference pipeline로 생산해야 한다.
   - averaging, weighted averaging, voting, stacking/blending, multiple checkpoints, multiple model variants 결합은 금지 범주로 본다.

2. **점수보다 먼저 유효 제출(valid submission)**
   - ROC-AUC가 핵심 지표지만, 트랙/라운드별 latency 제한을 넘으면 점수 자체가 무효다.
   - 따라서 “AUC 최대화”가 아니라 **AUC/latency/reproducibility 동시 최적화**가 목표다.

3. **Round 1부터 재현성 중심으로 개발**
   - 상위 제출은 코드 검증 대상이다.
   - 최고 점수 제출물을 공식 환경에서 재현하지 못하면 진출/수상 자격이 사라질 수 있다.

4. **EDA의 중심은 시각화가 아니라 split/leakage/shift 진단**
   - KDD Cup류 데이터는 시간, 그룹, 세션, 로그 구조가 얽힐 가능성이 크다.
   - 잘못된 CV는 이후 모든 모델 의사결정을 오염시킨다.

5. **논문 전략까지 고려한 모델 선택 필요**
   - 이 대회는 leaderboard-only competition이 아니다.
   - 최종적으로는 unified block, sequence modeling + feature interaction 통합, scaling law 관찰 같은 서사가 중요하다.

---

## 2. 에이전트 역할 재정의 반영 요약

이번 revised workflow 기준에서 각 에이전트 역할은 아래처럼 정리된다.

### 2.1 Analysis Agent
역할:
- 공식 규정, 일정, 제출 제한, 코드 검증, latency 제약, no-ensemble 정책을 확정하는 역할

이번 산출물의 핵심 기여:
- **공식 no-ensemble 규정 확정**
- **single model + single inference pipeline 원문 확인**
- **Round 1/2 코드 검증 및 재현성 의무 확인**
- **ROC-AUC + latency validity gate 구조 확인**

PM 판단:
- 이 에이전트의 결론은 전략 제약조건이 아니라 **최상위 하드 룰**이다.
- 모델링/실험/제출 파이프라인은 모두 이 문서에 종속돼야 한다.

### 2.2 EDA Strategy Agent
역할:
- 데이터 공개 직후, 가장 빠르게 leakage/shift/split 구조를 판별하고 로컬 검증 체계를 세우는 역할

이번 산출물의 핵심 기여:
- Day-0 EDA 체크리스트 제시
- leakage 중심 EDA 원칙 정리
- random/group/time/purged split 후보 비교 프레임 제시
- adversarial validation과 duplicate/entity overlap 점검 체계 제시

PM 판단:
- 이번 대회에서 EDA는 “인사이트 탐색”보다 **CV를 망치지 않기 위한 방어선 구축**에 가깝다.
- 모델 고도화보다 먼저 strict validation 설계를 끝내야 한다.

### 2.3 Model Strategy Agent
역할:
- 데이터 유형별 baseline → strong baseline → contender 흐름을 설계하는 역할

이번 산출물의 핵심 기여:
- GBDT 중심 strong baseline 철학 제시
- tabular / ranking / graph / sequence / time-series 분기 전략 정리
- feature engineering 우선순위 및 validation 중심 운영 원칙 제시

중요 보정:
- 원래 모델 전략 보고서에는 Kaggle 일반론에 따라 ensemble, seed ensemble, fold ensemble, stacking 등이 포함돼 있다.
- **하지만 본 대회 공식 규정상 이 부분은 최종 실행 전략에서 제외해야 한다.**

PM 판단:
- 모델 전략 보고서는 “문제 유형별 판단 프레임”으로는 매우 유효하다.
- 다만 최종 대회용 운영안에서는 **ensemble 관련 절을 제거한 단일모델 전략**으로 해석해야 한다.

### 2.4 Paper Research Agent
역할:
- 대회 주제와 논문 스토리라인을 연결하고, 어떤 구조를 borrow/test/avoid 할지 제안하는 역할

현재 상태:
- 지정 경로의 `paper_research_agent_report.md`는 작업 시점 기준 누락되어 있었다.
- 대신 지정된 3개 논문(InterFormer, OneTrans, HyFormer)의 공개 정보 기준으로 PM 차원에서 핵심만 직접 보강 반영했다.

보강 결론:
- 세 논문 모두 공통적으로 **sequence modeling과 heterogeneous feature interaction을 하나의 backbone 안에서 통합**하려는 흐름에 있다.
- 특히 이번 KDD Cup의 대회명과 규정 방향(unified, stackable blocks, latency-aware single pipeline)과 잘 맞는다.

PM 판단:
- 논문축은 “무조건 최신 구조를 그대로 복제”가 아니라,
  1) unified block 관점,
  2) sequence + feature interaction 통합,
  3) layer stacking/scaling 관찰,
  4) latency-aware simplification
  의 4요소를 설계 원칙으로 흡수하는 용도로 쓰는 것이 적절하다.

---

## 3. 공식 no-ensemble rule 반영 후 최종 전략

## 3.1 금지로 해석할 항목
공식 규정 및 보수적 해석 기준으로 아래는 모두 금지 또는 매우 위험으로 본다.

- 서로 다른 모델 2개 이상 평균/가중평균
- 같은 구조라도 독립 학습한 여러 모델 결합
- fold별 예측 평균
- seed별 예측 평균
- checkpoint averaging / checkpoint selection 후 결합 예측
- stacking / blending / voting
- variant A/B/C 예측을 후처리로 합치는 구조

즉, 일반적인 Kaggle 상위권 레시피의 절반 이상이 이번 대회에서는 사용할 수 없다.

## 3.2 허용 전략의 방향
우리가 가야 할 방향은 아래다.

### 허용/권장 축
- **하나의 backbone 안에서 성능을 끌어올리는 구조 개선**
- **하나의 모델 내부에서의 feature interaction 강화**
- **하나의 모델 내부에서의 sequence modeling 강화**
- **latency를 해치지 않는 범위의 block stacking**
- **reproducible training/inference 파이프라인 고정**
- **strict CV 기반 단일 모델 선택**

### 실전 해석
이번 대회에서 승부는 “여러 모델을 섞는 능력”이 아니라,

> **하나의 모델을 얼마나 잘 설계하고, 얼마나 정확하게 검증하고, 얼마나 안정적으로 재현하느냐**

로 좁혀진다.

---

## 4. 논문 리서치 보강 요약과 적용 방향

## 4.1 InterFormer
핵심 아이디어:
- heterogeneous data를 interleaving style로 상호작용시키며, 모달/피처 간 bidirectional 정보 흐름을 강화하는 구조

장점:
- 이질적 feature 간 interaction을 강화
- 지나친 조기 압축을 줄여 정보 손실을 완화
- 산업 환경에서 성능 및 QPS 개선 사례가 보고됨

리스크:
- 구조가 복잡해질수록 구현/디버깅 부담 증가
- 실제 대회 latency budget에서 block 수나 token 처리량이 민감할 수 있음

대회 적합성 판단:
- **중상**
- heterogeneous interaction 설계 원칙은 유용하나, 구현 단순화가 필요하다.

## 4.2 OneTrans
핵심 아이디어:
- sequence feature와 non-sequential feature를 하나의 token sequence로 통합하고, 단일 transformer backbone으로 처리

장점:
- 이번 대회가 요구하는 “unifying sequence modeling and feature interaction”에 가장 직접적으로 부합
- single backbone, stackable block, LLM/Transformer 최적화 활용 가능
- caching/causal attention 등 serving 효율 관점의 아이디어가 있음

리스크:
- full transformer류는 latency와 메모리 리스크가 큼
- 데이터 공개 후 실제 token 길이/feature 규모가 크면 비용이 빠르게 상승할 수 있음

대회 적합성 판단:
- **상**
- 이번 대회의 메인 설계 철학과 가장 잘 맞는 참고축 중 하나다.

## 4.3 HyFormer
핵심 아이디어:
- 긴 sequence modeling과 heterogeneous feature interaction을 global token 기반으로 결합해, decoupled pipeline의 한계를 줄이는 hybrid transformer

장점:
- long sequence와 dense/non-sequential feature를 함께 다루는 구조적 설계가 강점
- sequence 압축과 interaction 사이의 병목을 줄이는 관점이 유용
- scaling behavior 측면에서도 논문 스토리와 연결하기 좋음

리스크:
- 긴 시퀀스를 다루는 만큼 latency budget 초과 위험 존재
- 구현과 실험 비용이 높을 수 있음

대회 적합성 판단:
- **상~중상**
- sequence가 긴 데이터일수록 유효성이 커진다.

## 4.4 PM 권고: borrow / test / avoid

### 적극 borrow
- unified backbone 철학
- sequence + static/context feature를 한 스택 안에서 상호작용시키는 설계
- stackable blocks 기반 확장성 관점
- latency-aware simplification

### 우선 test
- lightweight token mixer 또는 compressed sequence block
- global token / query token 기반 sequence summarization
- shallow-to-medium depth transformer/hybrid block
- GBDT를 버리기 전, GBDT strong baseline 대비 단일 backbone 개선 폭 측정

### 보수적 접근 또는 avoid
- 너무 깊은 full transformer
- 긴 시퀀스를 거의 그대로 다 태우는 구조
- latency 희생형 대형 모델
- “나중에 앙상블하면 된다” 전제의 모델 다변화 전략

---

## 5. 최종 모델링 방향: single-model competition에 맞춘 수정안

## 5.1 공개 전 기본 원칙
데이터 공개 전에는 다음을 준비한다.

1. **EDA 자동화 틀**
   - schema, missing, unique, duplicate, shift, split diagnostics

2. **validation 유틸**
   - random / group / time / strict split 템플릿

3. **단일모델 baseline 템플릿**
   - LightGBM
   - CatBoost
   - XGBoost
   - 필요 시 simple sequence / transformer baseline

4. **reproducibility 템플릿**
   - config 고정
   - seed 고정
   - feature version 고정
   - train/infer script 분리
   - best submission commit/tag 관리

## 5.2 데이터 공개 직후 우선순위

### Step 1. 문제 구조 판별
아래 중 어디에 가까운지 먼저 판별한다.
- tabular classification/regression
- sequence/session CTR류
- ranking/recommendation
- graph/interaction
- time-aware log prediction

### Step 2. 강한 단일 baseline 확보
**주의: baseline도 단일 모델이어야 한다.**

권장 초기 선택:
- tabular이면 CatBoost 또는 LightGBM 중 하나를 strong single baseline으로 확정
- sequence/log 구조가 강하면, 먼저 통계/recency feature + GBDT 단일모델을 둔 뒤 transformer 계열을 비교
- ranking 구조여도 제출은 single inference pipeline이어야 하므로, 복잡한 multi-stage를 쓸 경우 규정 해석을 매우 보수적으로 해야 한다

### Step 3. CV 확정 후 모델 고도화
- leakage watchlist 확정
- strict split 확정
- adversarial validation으로 shift 확인
- 그 다음에만 backbone 개량 시작

## 5.3 추천 모델 의사결정

### 경우 A. 데이터가 전형적 tabular CTR에 가까울 때
가장 현실적인 시작점:
- **CatBoost strong single model** 또는 **LightGBM strong single model**

이유:
- 빠르고 재현 쉽고 latency 관리가 비교적 수월함
- feature interaction을 꽤 잘 처리
- baseline 안정성이 높음

다음 단계:
- group/time-safe feature engineering 강화
- count, recency, aggregation, interaction 파생
- strict OOF 기반 target-aware feature 검토

### 경우 B. sequence 정보가 매우 중요한 CTR/recsys 구조일 때
권장 방향:
- **single unified/hybrid backbone** 탐색
- OneTrans/HyFormer 계열 철학을 간소화해 적용

실전 해석:
- 긴 시퀀스를 모두 태우기보다,
  - compressed sequence summary
  - recent window
  - token-efficient interaction
  - lightweight stack
  중심으로 설계하는 편이 안전하다.

### 경우 C. ranking/recommendation 구조가 강할 때
주의점:
- 일반적인 retrieve→rank→ensemble 구조는 이번 대회 규정상 해석 리스크가 있다.
- 제출물이 여러 독립 모델 결합처럼 보이지 않도록 매우 조심해야 한다.

권장:
- 단일 backbone 안에서 candidate scoring까지 처리 가능한지 우선 검토
- 또는 주최측 포럼에 명확한 허용 범위를 확인하기 전까지는 conservative하게 간다.

---

## 6. EDA 운영안

## 6.1 Day-0 핵심 목표
- target, ID, time, group 컬럼 후보 파악
- duplicate / near-duplicate 탐지
- leakage watchlist 작성
- split 후보 2~4개 정의
- adversarial validation baseline 수행

## 6.2 꼭 만들어야 할 산출물
- `schema_summary.csv`
- `missing_summary.csv`
- `unique_summary.csv`
- `duplicate_report.md`
- `shift_report.md`
- `split_hypotheses.md`
- `leakage_watchlist.md`

## 6.3 PM 판단
이 대회에서 EDA 성공 기준은 예쁜 그래프가 아니라 아래다.
- 어떤 split이 가장 test-like한지 설명 가능해야 함
- 어떤 컬럼이 leakage 후보인지 팀 전체가 공유해야 함
- 어떤 drift가 있는지 수치로 말할 수 있어야 함

---

## 7. 운영 리스크와 방지책

## 7.1 가장 큰 리스크
1. random CV를 믿고 잘못된 모델 선택
2. latency 측정 실패로 invalid submission 발생
3. no-ensemble 규정 위반 소지가 있는 설계 채택
4. 최고 점수 제출물 재현 실패
5. paper/leaderboard 방향이 분리되어 후반 정리 실패

## 7.2 방지책
- 모델 실험 체크리스트에 **“single-model compliance”** 항목 추가
- 제출 전 latency smoke test 의무화
- best submission commit/tag 및 config 고정
- OOF, holdout, public LB를 분리 기록
- 논문용 실험은 early stage부터 표준화해 남기기

---

## 8. PM 최종 실행 권고안

## 8.1 대회 전체 전략 한 줄

> **이번 대회는 앙상블 대회가 아니라, strict validation 위에 단일 backbone을 올려 AUC·latency·재현성을 동시에 맞추는 대회로 운영해야 한다.**

## 8.2 실전 우선순위

### 1순위
- 규정 준수 체계 확정
- EDA 자동화 및 strict CV 확정
- strong single-model baseline 확보

### 2순위
- feature engineering 고도화
- latency-aware backbone 개선
- unified block / sequence-feature interaction 실험

### 3순위
- scaling law 관찰
- 논문용 ablation 및 스토리 정리
- Round 2용 end-to-end reproducibility 정리

## 8.3 가장 추천하는 초기 노선
현 시점 기준 가장 안전한 노선은 다음 둘 중 하나다.

1. **CatBoost/LightGBM 중심 strong single model + feature engineering 고도화**
2. **데이터가 sequence-heavy일 경우 lightweight unified transformer/hybrid single model 탐색**

단, 어떤 경우에도:
- fold averaging 금지
- seed averaging 금지
- model blending 금지
- checkpoint mixing 금지
를 기본 원칙으로 둔다.

---

## 9. 최종 결론
이번 KDD Cup의 승부 포인트는 일반 대회와 다르다.

- ROC-AUC만 높다고 끝나지 않는다.
- latency를 넘으면 무효다.
- 재현이 안 되면 진출/수상이 막힌다.
- 앙상블은 공식적으로 금지다.
- 논문까지 고려하면 unified block와 scaling 관점의 설계가 유리하다.

따라서 PM 관점의 최종 전략은 아래로 정리된다.

> **EDA는 split/leakage/shift 중심으로 빠르게 고정하고, 모델은 single-model 원칙 아래 strong baseline에서 출발해 unified backbone 방향으로 확장하며, 모든 실험은 latency와 reproducibility를 함께 관리한다.**

---

## 참고 메모
- `tasks/KDDcupEDA/paper_research_agent_report.md`는 작업 시점 기준 존재하지 않아, 지정된 3개 논문의 공개 정보 기준으로 PM 레벨 보강 요약을 반영했다.
- 향후 해당 보고서가 추가되면, 논문 비교/권고 섹션만 업데이트하면 된다.
