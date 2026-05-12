# KDD Cup 2026 규정/절차 분석 보고서

## 범위
- 이 문서는 **공식 규정/운영 절차**만 정리한다.
- 모델 아이디어, EDA, 논문 레퍼런스 확장은 제외한다.
- 출처는 주로 **algo.qq.com 공식 사이트의 Challenge Rules 본문**이며, JS 번들에서 렌더링되는 규정 텍스트를 직접 확인했다.

---

## 1. 핵심 결론 요약

1. 이번 대회는 **Academic / Industrial 2개 트랙**으로 운영된다.
2. **1인 1팀, 1인 1트랙**만 허용된다.
3. 팀은 **1~3명**이며, **2026-04-23 23:59:59 AOE 이후 팀 변경 금지**다.
4. **Round 1 → Round 2 진출 컷**은 기본적으로 Academic 상위 50팀, Industrial 상위 20팀이다.
5. 리더보드 핵심 지표는 **ROC-AUC**다.
6. 하지만 점수만으로 끝나지 않고, 모든 제출은 **트랙/라운드별 latency 제한**을 만족해야 유효하다.
7. **앙상블은 엄격히 금지**된다. 공식 문구상 **multiple independently trained models**, **averaging**, **weighted averaging**, **voting**, **stacking/blending**, **multiple checkpoints**, **multiple model variants**를 이용한 결합 예측 모두 불가다.
8. 상위권 팀은 **최고 점수 제출물의 코드 검증**을 받아야 하며, **공식 평가 환경에서 재현 가능**해야 한다.
9. 재현 실패, 코드 미제출, 앙상블 은닉/난독화는 모두 실격 또는 점수 0 처리로 이어질 수 있다.
10. Round 2 종료 후에는 **complete end-to-end training and inference code**와 **KDD Cup workshop paper**를 제출해야 한다.

---

## 2. 공식 확인 사실

### 2.1 대회명 / 문제 설정
공식 규정의 개요 섹션에는 다음이 나온다.
- Competition Title: **Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendation (Tencent UNI-REC Challenge)**

운영 규정 관점에서 중요한 부분은, 이 대회가 단순 점수 경쟁이 아니라:
- 단일 backbone 기반 설계
- stackable unified blocks
- 실서비스형 latency 제약
- 코드 검증과 재현성
을 함께 요구한다는 점이다.

---

## 3. 일정

공식 규정의 “Competition Period & Key Dates” 기준:

- **Official Website Launch:** Mar. 5, 2026 (23:59:59 AOE)
- **Release Demo Dataset:** Mar. 15, 2026 (23:59:59 AOE)
- **Global Registration:** Mar. 19, 2026 – Apr. 23, 2026 (23:59:59 AOE)
- **Round 1:** Apr. 24, 2026 – May 23, 2026 (23:59:59 AOE)
- **Round 2:** May 25, 2026 – Jun. 24, 2026 (23:59:59 AOE)
- **Winner Notification:** Jul. 15, 2026
- **Public Announcement:** Aug. 9, 2026

추가 공식 문구:
- Organizers may adjust dates, submission limits, or evaluation details if necessary.
- 변경 시 official website/platform 공지.

### 실무 함의
- **AOE(Anywhere on Earth)** 기준 마감이므로, 한국 시간으로는 보통 다음 날 오후까지 걸친다. 내부 일정표는 KST로 환산해 별도 관리해야 한다.
- Registration 마감과 팀 고정 시점이 사실상 같다. 즉 **팀 확정, 트랙 확정, 신원 검증**을 늦추면 그대로 리스크가 된다.

---

## 4. 참가 자격 / 신원 확인

### 4.1 기본 참가 자격
공식 규정상 참가자는 다음을 만족해야 한다.
- 각 지역 법규상 참가 가능해야 함
- **official competition platform의 real-name verification** 완료
- 규정 또는 추가 공지에서 제외되지 않아야 함

또한 제재 국가/수출통제 관련 제한이 길게 명시되어 있다.

### 4.2 Tencent 관련 제외 조건
공식 규정은 다음을 명시한다.
- **2026-04-23 23:59:59 AOE 시점에 Tencent와 active employment / internship / labor dispatch / outsourcing / joint training / other service relationship이 있으면 양 트랙 모두 참가 불가**

### 4.3 검증 권한
공식 문구:
- Organizers may request reasonable documentation or information to confirm eligibility at any time.
- Failure to provide requested documentation may result in disqualification, leaderboard removal, forfeiture of advancement, and/or ineligibility for prizes.

### 실무 함의
- **신분/소속 증빙은 나중에 모으는 게 아니라 초기에 확보**해야 한다.
- 특히 Academic Track은 학생 여부 검증이 엄격해서, 사후에 증빙이 꼬이면 점수를 따도 탈락 가능하다.

---

## 5. 트랙 구조

### 5.1 트랙 종류
공식 규정:
- The Competition offers **two tracks**.
- Each track has its own **dataset, inference-latency requirements, leaderboard, and prize/award pool**.

즉 트랙별로:
- 데이터셋이 다를 수 있고
- latency 기준이 다르며
- 리더보드와 상금도 분리된다.

### 5.2 Academic Track
공식 규정 핵심:
- 팀원 **전원이** university or college affiliation이어야 함
- 등록 마감 시점 기준 **full-time enrolled** 상태여야 함
- part-time students, working professionals enrolled in study programs, adult continuing education, 아직 공식 등록 전 admitted applicant, graduates는 **불가**

또한 포럼/검증 페이지 문자열까지 보면 postdoc도 Academic 측 설명에 포함되지만, 규정 본문은 full-time enrolled institution 기준으로 쓰여 있으므로 **증빙 기준을 본문 중심으로 해석**하는 게 안전하다.

### 5.3 Industrial Track
공식 규정 핵심:
- Academic Track 자격이 없는 참가자 대상
- professionals, faculty, researchers, postdoctoral fellows, part-time students, graduates 등 포함
- **stricter inference-latency requirements** 적용

### 5.4 트랙 선택 및 변경
공식 규정:
- Teams must select their track at registration.
- ineligible track 판정 시 organizers may move the team or disqualify it.
- track change rules는 별도 공지 가능

### 실무 함의
- **Industrial은 latency가 더 빡세다**는 점이 규정에 명시돼 있다.
- Academic으로 들어갈 수 있다면 latency 측면에서 상대적으로 유리할 수 있지만, 자격 증빙 리스크가 크면 오히려 위험하다.

---

## 6. 팀 규칙

### 6.1 팀 크기
공식 규정:
- Each team must consist of **one (1) to three (3)** participants.

### 6.2 중복 참가 금지
공식 규정:
- Each individual may join **only one team** for the duration of the Competition.
- Each participant may participate in **only one track**.
- multiple accounts, multiple teams, circumvention은 실격 가능

### 6.3 팀 고정 시점
공식 규정:
- Team membership is frozen after **April 23, 2026, 23:59:59 (AOE)**
- no additions, removals, or substitutions are permitted after this deadline.

### 실무 함의
- 팀원 구성 변경을 late-stage에 유연하게 할 수 없다고 봐야 한다.
- 코드 담당, 논문 담당, 제출 담당 역할을 **4/23 이전**에 확정해야 한다.

---

## 7. 플랫폼 / 데이터 접근 / 학습 환경

공식 규정:
- Model training and evaluation are conducted **exclusively on Tencent’s in-house AngelML platform**.
- training/test datasets released during competition
- test ground-truth labels are not disclosed

### 실무 함의
- 로컬 자유 학습 대회가 아니라 **공식 플랫폼 중심 운영**이다.
- 환경 차이로 인한 reproducibility issue를 줄이기 위해서라도 **공식 환경 기준 코드 구조**가 필요하다.
- 제출 artifact는 단순 prediction file만이 아니라, 라운드/트랙에 따라 **containerized inference service or pipeline**이 요구될 수 있다.

---

## 8. Round 1 운영 규칙

공식 규정 기준:
- Round 1은 모든 등록팀에게 개방
- 각 트랙별로 training set, test set, baseline/starter kit 공개
- 팀은 공식 플랫폼에서 학습하고, **inference code**를 제출해 평가받음
- 트랙별 public leaderboard 운영

### 8.1 anti-overfitting 조치
공식 규정:
- organizers may apply anti-overfitting measures
- examples include **submission frequency limits (e.g., up to three submissions per day)** and leaderboard policies

### 8.2 리더보드 표시 최소 조건
공식 규정:
- **ROC-AUC ≤ 0.50** submissions will not be displayed on the leaderboard

### 8.3 Round 2 진출
공식 규정:
- leaderboard freeze 후
- **Academic top 50 teams**
- **Industrial top 20 teams**
- 또는 official materials에 지정된 다른 cutoff
- 단, 규정 준수 및 required verification 통과가 전제

### 실무 함의
- Round 1 상위권이어도 **검증 실패 시 Round 2 진출 취소**가 가능하다.
- 즉 Round 1부터 이미 “점수 + 코드 적법성”을 동시에 맞춰야 한다.

---

## 9. Capacity-aware admission

공식 규정에 매우 중요한 운영 예외가 있다.

- 동시 지원 가능 팀 수는 **up to 5,000 teams**
- 전체 등록팀이 5,000을 넘으면 추가 규칙으로
  1. 누가 competition entry eligible 인지
  2. eligible team이 언제 들어갈 수 있는지
  를 정할 수 있음
- guiding principle: **earlier registration gets higher priority** within each track
- inactive team 제거 권한도 보유

### 실무 함의
- 이번 대회는 단순히 등록만 해두면 다 같은 조건으로 시작한다고 보장되지 않는다.
- **빠른 등록 자체가 실질적 advantage**가 될 수 있다.
- submission이 거의 없고 랭킹이 매우 낮은 팀은 resource allocation 차원에서 밀릴 수 있다.

---

## 10. Round 2 운영 규칙

공식 규정 핵심:
- Round 2는 **robustness and scalability** 강조
- 양 트랙 모두 **약 10배 큰 데이터셋** 사용
- **stricter inference latency constraints** 적용
- 팀당 약 **eight high-performance GPUs** provisioned
- 단, platform policy / fair-use / resource availability에 따름

Round 2 종료 후 필수 제출:
1. **complete end-to-end training and inference code**
2. **KDD Cup workshop paper**

### 실무 함의
- Round 2는 점수만 잘 나오면 끝이 아니라 **코드 제출과 논문 제출이 모두 필수**다.
- 학습 코드가 지저분하거나, inference만 겨우 돌아가는 구조면 후반에 크게 불리하다.

---

## 11. 리더보드 갱신 정책

공식 규정:
- daily submissions before **16:00 AOE**를 기준으로
- historical best score descending 정렬
- leaderboard updated at **19:00 AOE same day**

### 실무 함의
- 실시간 반영이 아니다.
- 하루 단위로 배치 업데이트되므로, 제출 타이밍과 실험 cadence를 이에 맞춰야 한다.

---

## 12. No-ensemble 정책

이 대회에서 가장 중요한 규정 중 하나다.

### 12.1 공식 원문 핵심
공식 규정은 다음 취지로 명시한다.

> “To ensure fairness and comparability, ensemble methods are strictly prohibited.”

또한 아래를 ensemble로 간주한다고 적었다.
- **averaging**
- **weighted averaging**
- **voting**
- **stacking/blending**
- **multiple independently trained models**의 결합
- **multiple checkpoints**를 이용한 combined prediction
- **multiple model variants**를 이용한 combined prediction

그리고:
> “Each submission must be produced by a single model and a single inference pipeline.”

### 12.2 해석상 중요한 포인트
규정상 금지 범위는 상당히 넓다.
- 서로 다른 모델 여러 개 결합: 당연히 금지
- 같은 구조라도 독립 학습본 여러 개 평균: 금지
- 여러 checkpoint average / selection / merge prediction: 금지
- variant별 예측을 후처리 결합: 금지

### 12.3 경계 사례
공식 규정은 마지막에 이렇게 적는다.
- whether a method constitutes an ensemble is determined by organizers in their sole discretion
- unsure하면 official forum에 사전 문의하라

### 실무 함의
- “이건 기술적으로 ensemble이 아닌 것 같은데?”라는 자기 해석은 위험하다.
- **single model + single inference pipeline**를 보수적으로 지키는 쪽이 맞다.
- multi-checkpoint test-time average 류도 규정상 매우 위험하다.

---

## 13. 코드 검증 / 재현성 / 집행 방식

### 13.1 Round 1 → Round 2 검증
공식 규정:
- advancement-selected teams must submit code used for their **highest-scoring Round 1 submission**
- ensemble detected 시 **Round 2 disqualification**
- vacant slot은 next eligible team으로 대체

### 13.2 Round 2 수상 검증
공식 규정:
- prize-eligible teams must submit code used for their **highest-scoring Round 2 submission**
- ensemble detected 시 **Round 2 score = zero**
- leaderboard prizes and innovation awards 모두 불가

### 13.3 재현성 실패 / 코드 미제출
공식 규정:
- refusal to provide code
- inability to reproduce submitted results in official evaluation environment
- attempts to obfuscate or conceal ensemble behavior
→ 모두 violation으로 취급 가능

### 13.4 Submission Requirements 섹션의 추가 문구
공식 규정:
- After Round 2 concludes, each team must submit **complete, end-to-end training and inference code**
- submitted code must reproduce the **best-scoring result** under official environment/settings
- teams that fail code review or cannot reproduce best result will be **ineligible for any awards**

### 실무 함의
- 이 대회에서 재현성은 부가 요소가 아니라 **수상 자격 자체의 게이트**다.
- 내부적으로는 최소한 아래가 필요하다.
  - best submission commit/tag 고정
  - config / seed / dependency lock
  - train → inference end-to-end 재실행 스크립트
  - official environment 가정 문서화

---

## 14. 평가 지표와 유효성 제약

### 14.1 리더보드 지표
공식 규정:
- primary metric: **ROC-AUC**
- public leaderboard는 designated public test set
- final ranking은 separate held-out private test set

### 14.2 latency는 단순 참고값이 아니라 validity gate
공식 규정:
- 모든 제출은 **track- and round-specific inference latency limits** 만족 필요
- 이를 초과하면 **invalid** 처리되며 ranked 되지 않음

### 14.3 동점 처리
공식 규정:
1. lower measured inference latency
2. earlier submission timestamp
3. 또는 Sponsor가 공지한 deterministic rule
- **random/chance-based method는 사용하지 않음**

### 실무 함의
- 모델이 아무리 AUC가 높아도 latency gate를 넘으면 무효다.
- tie-break도 latency 우선이므로, 상위권에서는 미세한 latency 차이가 실제 순위를 바꿀 수 있다.

---

## 15. 제출 제한

공식 규정:
- Each team may submit up to **three submissions per day**
- organizers may apply additional throttling/cooldown
- fairness / platform stability를 위해 조정 가능

### 실무 함의
- 하루 3회 기준으로 보면, 실험 운영은 “많이 던져보기”보다 “오프라인 검증 후 선별 제출” 구조가 맞다.

---

## 16. 논문 평가 기준

공식 규정의 Workshop Paper Review Criteria:
- review will emphasize **novelty and insight**
- focus on:
  1. **unified, stackable block** design
  2. **scaling laws** exploration
- high AUC alone does **not** guarantee strong paper evaluation

### 실무 함의
- 리더보드 전략과 paper 전략은 일부 겹치지만 완전히 같지 않다.
- 최종 수상 포트폴리오를 노리면, 단순 score chasing만으로는 부족하다.

---

## 17. 확인 사실 vs 해석

### 17.1 확인 사실
- 2개 트랙 운영
- 팀 1~3명
- 1인 1팀 / 1인 1트랙
- real-name verification 필요
- 4/23 AOE 이후 팀 변경 금지
- Round 1: Academic 50 / Industrial 20 진출
- Round 2 데이터 약 10배
- 트랙/라운드별 latency 제한 존재
- metric은 ROC-AUC
- tie-break는 latency, then timestamp
- 하루 3회 제출 제한
- no-ensemble 엄격 금지
- best submission code 검증 필요
- official environment에서 재현 필요
- Round 2 후 code + workshop paper 제출 필요

### 17.2 합리적 해석
- **single model 원칙을 매우 보수적으로 적용**해야 한다.
- multi-checkpoint inference average도 위험하다.
- Round 1부터 재현 가능한 코드 체계가 없으면 후반에 점수 보전이 어렵다.
- Industrial Track은 latency 부담이 더 크므로 모델 설계 자유도가 낮을 수 있다.
- 조기 등록은 단순 형식 절차가 아니라 실제 입장 우선순위와 연결될 수 있다.

---

## 18. 후속 에이전트용 실무 메모

### EDA 에이전트가 반드시 반영할 점
- public/private split drift 가능성 고려
- offline validation은 **AUC + latency** 동시 관리 필요
- submission budget이 적으므로 검증 셋 설계가 중요

### 모델 에이전트가 반드시 반영할 점
- 앙상블 전제 전략 금지
- single model / single inference pipeline 유지
- latency-aware architecture 필요
- 재현 가능한 train/infer full pipeline 설계 필요

### 논문/리서치 에이전트가 반드시 반영할 점
- paper review는 leaderboard-only가 아님
- unified block / scaling law를 중심 스토리로 잡아야 함

---

## 19. 한 줄 결론

이번 KDD Cup은 **ROC-AUC 리더보드 대회**이면서 동시에 **latency 제약, 단일모델 원칙, 코드 검증, 재현성, 논문 제출**까지 강하게 묶여 있는 대회다. 따라서 가장 안전한 전략은 처음부터 **“single model + reproducible pipeline + latency budget 준수”**를 기준선으로 삼는 것이다.

---

## 출처 메모
- algo.qq.com 공식 사이트 프런트엔드 번들 내 `Challenge Rules` 렌더링 텍스트 직접 확인
- 확인된 대표 공식 문구:
  - “To ensure fairness and comparability, ensemble methods are strictly prohibited.”
  - “Each submission must be produced by a single model and a single inference pipeline.”
  - “Track-specific inference latency limits will apply and will be enforced during evaluation.”
  - “The submitted code must be able to reproduce the team’s best-scoring result under the official evaluation environment and settings.”
