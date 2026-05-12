# EDA 전략 보고서 (규정 반영 v2)

## 목적
이번 KDD Cup의 EDA는 일반적인 탐색 분석보다 훨씬 운영 제약 중심으로 설계해야 한다. `analysis_agent_report.md` 기준으로 이번 대회는 다음 조건이 강하다.

- **앙상블 금지**: `single model + single inference pipeline` 원칙
- **latency gate 존재**: AUC가 높아도 latency 제한을 넘으면 제출 무효
- **코드 검증 / 재현성 필수**: 최고 점수 제출물이 공식 환경에서 재현되어야 함
- **제출 횟수 제한**: 하루 3회 수준이라 로컬 검증 신뢰도가 중요
- **추천/CTR/CVR/시퀀스 모델링 계열 가능성 높음**: user-item-context-time 구조, group/time leakage 위험 큼

따라서 이번 EDA의 목적은 단순히 분포를 보는 것이 아니라, 아래 4가지를 빠르게 확정하는 데 있다.

1. **무효 제출을 만드는 구조적 리스크**를 초기에 제거한다.  
   - leakage
   - 잘못된 validation
   - latency를 폭증시키는 feature/materialization 구조
2. **단일 모델로 끝까지 밀 수 있는 feature space**를 찾는다.  
   - 앙상블 없이도 견딜 수 있는 강한 단일 입력/표현 체계
3. **공식 환경 재현이 쉬운 데이터 처리 흐름**을 만든다.  
   - deterministic feature generation
   - split 재현 가능성
   - train/infer 동일 파이프라인
4. **리더보드와 로컬 CV의 정렬 가능성**을 빠르게 판단한다.  
   - public/private drift 가능성
   - time/group shift
   - latency-AUC tradeoff

---

## 1. 이번 대회에서 EDA의 역할이 왜 달라지는가

일반 대회에서는 EDA가 “어떤 피처가 좋아 보이는가”에 치우치기 쉽다. 하지만 이번 대회에서는 규정상 그보다 먼저 확인해야 할 것이 있다.

### 1.1 no-ensemble 제약의 의미
앙상블이 금지되어 있으므로, 다음과 같은 접근은 초반부터 배제해야 한다.

- 서로 다른 전처리 버전 여러 개를 따로 학습해 합치는 전략
- 여러 checkpoint 평균/선택을 전제로 한 전략
- heavy model + light model 혼합 전제 전략
- fold별 모델 예측 결합을 실전 제출 구조로 상정하는 전략

즉 EDA도 **“후반에 여러 모델을 조합해서 해결”**을 전제로 하면 안 된다. 처음부터 아래 질문을 봐야 한다.

- 단일 모델에서 가장 안정적으로 먹히는 핵심 신호는 무엇인가?
- feature engineering이 inference 시점에도 단일 파이프라인으로 계산 가능한가?
- 성능은 조금 좋아도 재현/배포/latency 리스크가 큰 피처는 아닌가?

### 1.2 latency gate의 의미
이번 대회에서 latency는 참고 지표가 아니라 **제출 유효성 조건**이다. 따라서 EDA 단계에서도 feature의 예측력만 볼 것이 아니라 아래를 함께 봐야 한다.

- 이 피처가 **online / evaluation inference** 시 계산 가능한가?
- 고-cardinality join, 긴 sequence aggregation, 복잡한 cross feature가 latency 병목이 되는가?
- sparse explosion, huge embedding table, 과도한 lookup 수를 유발하는가?
- train에서는 편하지만 inference에서는 비싼 계산을 요구하는가?

즉 이번 EDA는 **AUC-oriented EDA**가 아니라 **AUC + latency-feasibility EDA**여야 한다.

### 1.3 재현성 검증의 의미
상위권 점수는 코드 검증 대상이다. 따라서 EDA 결과도 단순 노트북 메모 수준이면 부족하다.

반드시 남겨야 할 것은 다음이다.

- 어떤 컬럼을 썼고 왜 썼는지
- 어떤 split을 썼고 왜 그 split이 타당한지
- 어떤 컬럼을 leakage 위험으로 제외했는지
- train/infer 모두 동일하게 생성 가능한 feature 목록
- 데이터 버전, seed, 집계 기준, 시간 cutoff

이번 대회에서 좋은 EDA는 “인사이트가 많은 EDA”보다 **나중에 코드 검증 문서로 전환 가능한 EDA**다.

---

## 2. 데이터 공개 직후 가장 먼저 확인할 것

규정 제약상, 데이터 공개 직후 초반 2~6시간의 EDA 우선순위가 중요하다. 예쁜 그래프보다 먼저 아래를 확인해야 한다.

## 2.1 1순위: 제출 단위와 추론 단위 파악
가장 먼저 확인할 질문:

- 한 row가 무엇을 의미하는가?
  - user-item impression?
  - session step?
  - sequence prefix?
  - candidate ranking sample?
- 예측 대상은 click / conversion / purchase / like / dwell / next-item 중 무엇인가?
- 하나의 user나 session이 여러 row로 반복되는가?
- test에서 주어지는 입력은 train과 동일 구조인가?
- inference는 row-wise인가, group-wise ranking인가?

이건 단순 이해용이 아니라 **latency와 validation 구조를 같이 결정**한다.  
예를 들어 row-wise binary classification처럼 보이지만 실제로는 session/group ranking 구조라면, random split 기반 CV는 빠르게 망가질 수 있다.

## 2.2 2순위: ID / group / time 후보 컬럼 파악
추천/CTR/CVR 계열에서 가장 위험한 부분은 group/time leakage다. 가장 먼저 후보를 식별한다.

- `user_id`, `device_id`, `member_id`
- `item_id`, `author_id`, `shop_id`, `ad_id`
- `session_id`, `request_id`, `query_id`, `impression_id`
- `timestamp`, `event_time`, `request_time`, `exposure_time`
- platform/source/domain/channel/country/device type

즉시 봐야 할 것:

- entity별 row count 분포
- 같은 entity가 train/test에 모두 등장하는지
- 시간축이 존재하는지
- sequence order가 존재하는지
- item/user long-tail이 심한지

## 2.3 3순위: leakage 후보 컬럼 탐색
규정상 최고 점수 제출 코드 검증이 들어오므로, 초반 high-score leakage를 그대로 믿고 가면 나중에 치명적이다.

우선 탐지할 패턴:

- 사후 상태를 담는 컬럼
- 후행 이벤트 존재 여부를 암시하는 컬럼
- label과 거의 동치인 count/stat column
- 전체 기간 집계처럼 보이는 컬럼
- test 시점에는 계산 불가능해 보이는 누적 통계
- 지나치게 완성된 profile/embedding/summary 컬럼

특히 추천 데이터에서는 아래가 흔한 함정이다.

- user-item 과거 상호작용 요약처럼 보이지만 사실 미래 포함 집계
- conversion 이후에만 생성되는 action count
- full-history 기반 popularity / target rate proxy
- session 종료 후에만 계산 가능한 feature

## 2.4 4순위: train-test shift와 source mismatch
public/private 분리와 시간 드리프트가 있을 가능성이 크므로, early EDA에서 다음을 본다.

- train vs test feature schema mismatch
- unseen category 비율
- 주요 수치형 quantile 차이
- source/domain/device/month 비중 차이
- adversarial validation AUC
- test가 미래 구간처럼 보이는지 여부

이 단계에서 shift가 크면, 이후 모델 성능보다 먼저 **CV 전략부터 다시 잡아야 한다.**

## 2.5 5순위: feature cost inventory
이번 대회는 latency-sensitive이므로, 데이터 공개 직후부터 각 컬럼/피처를 아래 3분류로 관리하는 것이 좋다.

### A. cheap features
- 원본 수치형 / 범주형
- 간단한 clip/log/binning
- low-cost lookup

### B. medium-cost features
- 제한된 윈도우 count/stat
- 짧은 sequence summary
- 적은 수의 cross feature

### C. expensive / risky features
- 긴 시퀀스 전부를 쓰는 피처
- 고-cardinality pairwise cross 폭증
- 많은 join / lookup / dynamic aggregation
- inference 시 매 row 큰 상태를 재구성해야 하는 피처

이 분류는 모델링 직전이 아니라 **EDA 단계에서** 만들어야 한다.

---

## 3. 이번 대회용 EDA 핵심 원칙

## 3.1 split보다 분포 그래프가 먼저가 아니다
이번 대회는 제출 횟수 제한이 있어 로컬 CV가 흔들리면 실험 효율이 크게 떨어진다. 따라서 초반 EDA의 최우선은 아래다.

1. **올바른 split 가설 2~4개 세우기**
2. **각 split에서 baseline 성능 비교**
3. **AUC뿐 아니라 latency proxy도 같이 기록**
4. **가장 보수적인 main CV와 stress CV 확정**

권장 CV 체계:

- `sanity CV`: stratified/random baseline
- `main CV`: 가장 그럴듯한 production-like split
- `stress CV`: 더 엄격한 time/group split

이번 대회에서는 random CV 하나만 믿는 것은 특히 위험하다.

## 3.2 feature importance보다 suspicious feature triage가 먼저다
초반 베이스라인에서 중요한 것은 “무슨 피처가 강한가”보다 다음이다.

- 비정상적으로 강한 단일 피처가 있는가?
- strict split으로 가면 바로 무너지는 피처가 있는가?
- adversarial validation에서 train/test 분리의 핵심인 컬럼이 무엇인가?
- latency 비용 대비 gain이 너무 낮은 피처가 있는가?

즉 EDA 결과물도 `top features`보다 먼저 `watchlist / blacklist / expensive-but-weak list`가 나와야 한다.

## 3.3 EDA 결과는 모델 설계 제약 문서여야 한다
이번 대회에서는 EDA가 모델링 아이디어를 넓히는 문서라기보다, **무엇을 하면 안 되는지 / 무엇이 안전한지**를 정하는 문서여야 한다.

예:

- 사용 가능: raw ID + short history + cheap stat
- 보류: full-history aggregated user-item stat
- 금지: test 시점 재현 불가능한 future-aware aggregate
- 주의: 정확도는 높지만 latency spike가 큰 long-sequence feature

---

## 4. 추천/CTR/CVR형 데이터라고 가정할 때의 우선 EDA

공식 대회명이 recommendation / sequence modeling / feature interaction을 암시하므로, 데이터가 공개되면 아래를 먼저 본다.

## 4.1 user-item-context 구조 확인
확인 질문:

- user, item, context가 분리돼 있는가?
- 하나의 request에 candidate가 여러 개 붙는 구조인가?
- exposure → click → conversion의 다단계 이벤트인가?
- sequence prefix 기반 예측인가?
- negative sampling이 이미 적용된 데이터인가?

이 구조를 알아야 아래를 판단할 수 있다.

- binary classification으로 볼지
- ranking/group metric 보조 해석이 필요한지
- split을 user/session/time 기준으로 할지
- inference latency가 per-row인지 per-request인지

## 4.2 long-tail 구조 확인
추천 데이터에서는 long-tail이 거의 핵심이다.

반드시 볼 것:

- user frequency distribution
- item frequency distribution
- session length distribution
- rare user / rare item 비율
- cold-like sample 비율
- test에서 unseen user/item 비율

시사점:

- high-cardinality ID를 어떻게 다룰지
- embedding or hashing이 필요한지
- rare bucket 정책이 필요한지
- heavy cross feature가 latency 대비 가치가 있는지

## 4.3 sequence 길이와 비용 확인
시퀀스 모델링 대회라면, sequence length는 성능뿐 아니라 latency를 좌우한다.

초기 EDA에서 꼭 측정할 것:

- sequence length quantile (p50, p90, p95, p99)
- 유저/세션별 history length
- 긴 sequence가 label과 얼마나 관련 있는지
- truncation 시 정보 손실 가능성
- 최근 n개만 써도 충분한지

이 분석은 매우 중요하다. 이유는 다음과 같다.

- 긴 sequence 전체를 쓰는 모델이 AUC는 조금 더 좋을 수 있다.
- 하지만 latency gate 때문에 실제 제출 불가능할 수 있다.
- 따라서 EDA 단계에서부터 **“최근 20 / 50 / 100개면 충분한가?”**를 진단해야 한다.

## 4.4 label delay / attribution 문제 확인
CVR류 데이터는 label delay가 자주 문제다.

확인할 것:

- conversion이 exposure 직후가 아니라 일정 시간 뒤 발생하는가?
- train 마지막 구간의 label completeness가 떨어지는가?
- 특정 기간은 아직 정답이 덜 성숙한 상태인가?
- target 생성 창(window)이 명시돼 있는가?

이건 단순 EDA가 아니라 validation 신뢰도와 직결된다.  
잘못 보면 최근 데이터가 성능을 낮추는 것처럼 보여도 사실은 **label maturity 문제**일 수 있다.

---

## 5. leakage 중심 EDA를 더 엄격하게 해야 하는 이유

이번 대회는 no-ensemble이라, leakage를 나중에 다른 모델 조합으로 상쇄하는 우회가 없다. 게다가 코드 검증이 있기 때문에 초반에 leakage를 밟고 얻은 인사이트는 대부분 독이 된다.

## 5.1 반드시 수행할 leakage 점검

### A. 단일 피처 점수 테스트
- 각 피처 단독 baseline 점수
- 비정상적으로 강한 피처 자동 경고
- random split과 strict split에서 성능 차이 비교

### B. group/time split 민감도 테스트
- random stratified
- GroupKFold / StratifiedGroupKFold
- time split
- group + time custom split

### C. duplicate / near-duplicate 점검
- exact duplicate
- key-subset duplicate
- same user-item repeated sample
- train-valid / train-test overlap

### D. aggregate feature 생성 가능성 감사
질문:
- 이 통계는 inference 시점에도 계산 가능한가?
- 공식 평가 환경에서 동일하게 재현 가능한가?
- 미래 포함 위험은 없는가?
- fold 밖 정보가 섞이지 않는가?

## 5.2 추천 데이터에서 특히 위험한 누수 패턴

- 전체 기간 CTR/CVR aggregate
- user-item pair historical conversion rate가 미래 포함 집계인 경우
- full-session summary를 prefix 예측에 사용하는 경우
- exposure 이후 행동 로그를 feature로 사용하는 경우
- item popularity를 test까지 합쳐 전처리한 경우
- shared preprocessing table이 full dataset 기준으로 생성된 경우

## 5.3 suspicious feature 판정 기준
아래 중 2개 이상이면 강한 의심 대상으로 둔다.

- 단일 피처로 점수가 비정상적으로 높다
- random CV에서만 강하고 strict CV에서는 붕괴한다
- train/test 분리 모델에서 중요도가 매우 높다
- 사후 이벤트 / 전체 집계 느낌의 이름이나 분포를 가진다
- inference cost가 큰데 gain이 불안정하다

---

## 6. validation reliability를 위한 EDA 전략

이번 대회는 제출 예산이 적으므로, 오프라인 검증이 흔들리면 실험 운영 전체가 비효율적이 된다.

## 6.1 split 후보를 동시에 운영해야 한다
권장:

### main CV
가장 실제 test 조건과 가깝다고 판단되는 split

### stress CV
더 엄격한 split
- 미래 구간 holdout
- unseen group 비율 증가
- leakage 가능성 최소화

### sanity CV
빠른 iteration용 단순 split
- feature sanity check
- 코드 버그 탐지

핵심은 **모델 순위가 CV 간 얼마나 안정적인지** 보는 것이다.

## 6.2 drift-aware validation이 필요하다
EDA 단계에서 다음을 정량화한다.

- fold별 기간 범위
- fold별 user/item overlap ratio
- fold별 target rate 차이
- fold별 unseen category ratio
- adversarial validation AUC

이렇게 해야 단순 평균 CV score 대신, **“이 CV가 얼마나 믿을 만한가”**를 같이 판단할 수 있다.

## 6.3 latency-aware validation 로그도 같이 남겨야 한다
오프라인 검증에서 metric만 남기면 부족하다. 각 실험마다 최소한 아래를 기록한다.

- feature count
- categorical cardinality 총량
- sequence truncation length
- non-zero / density summary
- batch inference wall-clock proxy
- per-row feature generation time proxy

공식 latency 환경과 정확히 같지는 않아도, 상대 비교 지표는 초반 의사결정에 충분히 유용하다.

---

## 7. single-model 설계를 위한 EDA 산출물

이번 대회의 EDA 산출물은 “분석 리포트”를 넘어서, 단일 모델 설계 입력으로 바로 넘어가야 한다.

## 7.1 반드시 필요한 산출물

### 1) schema summary
- 컬럼명, dtype, unique count, missing rate
- ID/time/group 후보 태깅
- cheap/medium/expensive cost 태그

### 2) split diagnostics
- candidate split 정의
- fold별 entity overlap
- fold별 time coverage
- fold별 target distribution
- split별 baseline AUC 비교

### 3) leakage watchlist
- 의심 컬럼
- 의심 이유
- 제외/보류/재검토 상태
- 필요한 추가 검증

### 4) shift report
- train/test 주요 차이
- adversarial validation 결과
- drift-driving features
- validation에 반영해야 할 포인트

### 5) latency risk memo
- 비싼 feature 목록
- 비싼 이유
- 대체 가능한 cheap proxy
- truncation/quantization/hash 후보

### 6) reproducibility memo
- 데이터 버전
- split seed / split rule
- aggregate cutoff rule
- feature generation deterministic 여부
- notebook이 아닌 script화 필요 항목

## 7.2 특히 중요한 출력: feature triage table
이번 대회에서는 아래 같은 표가 매우 중요하다.

| feature / feature family | 예측력 | leakage 위험 | latency 비용 | 재현성 위험 | 조치 |
|---|---:|---:|---:|---:|---|
| raw user_id | 중~높음 | 낮음 | 낮음 | 낮음 | 사용 |
| long user history full aggregate | 높음 가능 | 중~높음 | 높음 | 높음 | 보류 |
| recent 20-step summary | 중~높음 | 중간 | 중간 | 중간 | 후보 |
| post-event action count | 높음 가능 | 매우 높음 | 낮음 | 높음 | 금지 |
| user-item full CTR stat | 높음 가능 | 매우 높음 | 중~높음 | 높음 | 엄격 재검토 |

이 표가 있어야 모델링이 과열되지 않는다.

---

## 8. 재현성 검증을 염두에 둔 EDA 운영 원칙

코드 검증이 예정되어 있으므로, EDA 단계에서부터 다음 원칙을 지켜야 한다.

## 8.1 notebook-only 결론을 만들지 않는다
노트북에서 본 현상을 실제 파이프라인으로 옮길 수 있어야 한다.

- feature logic을 script/function으로 분리
- split 규칙을 코드로 고정
- seed와 cutoff를 파일로 저장
- artifact를 csv/json/md로 남김

## 8.2 aggregate는 생성 규칙까지 문서화한다
특히 추천 데이터에서는 aggregate feature가 강력하지만 위험하다. 반드시 기록한다.

- 어떤 key로 묶었는가
- 어떤 시간 범위까지 사용했는가
- min_count / smoothing 여부
- train/valid/test에 동일 규칙 적용 가능한가
- OOF 방식이 필요한가

## 8.3 best submission 후보는 EDA 단계부터 lineage를 남긴다
나중에 코드 검증 대응을 쉽게 하려면 EDA 결과도 lineage가 있어야 한다.

- dataset snapshot ID
- split version
- feature set version
- exclusion list version
- latency budget memo version

---

## 9. Day-0 / Day-1 실전 체크리스트 (규정 반영형)

## 공개 후 30분 내
- 파일 구조, row/col 수, target, sample submission 확인
- ID / group / time / sequence 후보 컬럼 태깅
- schema summary 생성
- 결측률 / 유니크율 / 상수 컬럼 리포트 생성
- feature cost 1차 분류

## 공개 후 2시간 내
- duplicate / entity repeat 점검
- train/test shift 1차 분석
- adversarial validation baseline
- split 후보 2~4개 작성
- suspicious column 1차 목록 작성

## 공개 후 반나절 내
- baseline 모델을 split별로 비교
- single-feature suspicious score 테스트
- sequence length / history length 분포 분석
- latency risk feature 정리
- main CV / stress CV 초안 확정

## 공개 후 1일 내
- leakage watchlist 확정
- cheap vs expensive feature family 우선순위 정리
- 재현 가능한 aggregate 규칙 문서화
- 모델 에이전트로 넘길 단일 모델용 feature shortlist 작성
- EDA Day-1 보고서 작성

---

## 10. 모델링에 바로 넘겨야 할 EDA 결론 형식

EDA가 끝나면 모델링 팀/에이전트에 아래 형식으로 넘겨야 한다.

## 10.1 사용 권장 feature family
- latency와 재현성 면에서 안전
- strict CV에서도 안정적
- train/test drift가 관리 가능

## 10.2 보류 feature family
- 점수는 나올 수 있으나
- leakage / drift / latency / reproducibility 리스크가 있음
- 추가 검증 필요

## 10.3 금지 feature family
- 미래 정보 포함 가능성 큼
- 공식 환경에서 동일 계산 어려움
- 단일 inference pipeline 원칙과 충돌 가능

## 10.4 권장 validation protocol
- main CV 정의
- stress CV 정의
- 비교 시 반드시 같이 볼 latency proxy
- feature 추가/삭제 승인 기준

이 포맷이 중요하다. 이번 대회는 점수 좋은 아이디어를 많이 쌓는 것보다, **위험한 아이디어를 빨리 버리는 것**이 더 중요할 수 있다.

---

## 11. 이번 대회에서 특히 중요한 EDA 출력만 요약

아래 출력은 이번 규정에 직접 연결되므로 우선순위가 높다.

### single-model 설계에 중요한 출력
- feature triage table
- cheap/medium/expensive feature inventory
- sequence truncation 분석
- user/item/session long-tail 분석
- strict split에서 안정적인 feature family 목록

### reproducibility에 중요한 출력
- split diagnostics
- leakage watchlist
- aggregate generation rule memo
- artifact 기반 schema/target/missing/shift 요약
- seed / cutoff / version 기록

### latency에 중요한 출력
- feature cost inventory
- cardinality / embedding size 추정
- sequence length quantile
- per-feature-family inference risk memo
- 성능 대비 비용이 낮은 feature 후보군

### validation reliability에 중요한 출력
- random vs group vs time split 비교
- adversarial validation 결과
- fold overlap / unseen ratio 리포트
- label delay 가능성 메모
- public/private drift 가설 문서

---

## 12. 한 줄 운영 원칙

이번 KDD Cup에서 EDA는 “무슨 피처가 좋아 보이는지 찾는 작업”이 아니라,

> **단일 모델로 재현 가능하게 제출할 수 있으면서, latency 제한 안에서 살아남고, 로컬 검증이 리더보드를 속이지 않도록 만드는 작업**

으로 설계해야 한다.

조금 더 실무적으로 말하면 다음과 같다.

- **점수보다 먼저 split을 고른다.**
- **강한 피처보다 먼저 위험한 피처를 분류한다.**
- **복잡한 feature보다 먼저 inference 가능한 feature를 고른다.**
- **노트북 인사이트보다 먼저 재현 가능한 artifact를 남긴다.**
- **모델 성능표와 함께 latency proxy와 leakage 상태를 같이 본다.**

이 원칙을 지키면, 데이터 공개 직후 흔히 빠지는 함정인
- random CV 착시
- 미래 누수
- expensive feature 과투자
- 재현 불가능한 전처리
- leaderboard chasing

를 상당 부분 줄일 수 있다.