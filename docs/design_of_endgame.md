# CLAUDE.md — 프로젝트 설계 컨텍스트

> 이 문서는 본 연구의 설계 결정을 담은 기준 문서다. Claude Code는 코드를 수정하기 전에
> 이 문서를 먼저 읽고, **기존 코드베이스의 구조를 파악한 뒤** 여기 정의된 설계를 그 위에
> 매핑한다. 처음부터 새로 짜지 말고, 기존 모듈을 최대한 재사용·확장한다.

---

## 0. 한 줄 요약

지상 모바일 로봇의 **mm급 정밀 도킹**을, VLA 계열의 **imitation learning + diffusion
policy**로 푸는 연구. 핵심 아이디어는 **역할 분담**이다 — 학습 정책은 강건한 *광역 접근*을,
**2D LiDAR 기하 정합**은 *종단 mm 정밀*을 책임진다. 이 분담 덕분에 학습 정책은 정밀
시연 없이 적은 시연으로 학습되며, 이것이 본 연구의 1순위 novelty(시연 개수 효율)다.

---

## 1. 시스템 구성 (센서)

- **Camera (RGB)**: 방향성(directionality) 신호 담당. DINO feature 공간에서 동작.
- **2D LiDAR (planar scan)**: metric 기하 담당. 지상 도킹은 본질적으로 SE(2) 문제라
  2D로 충분. 도킹 타겟은 **LiDAR로 구별 가능한 기하**(돌출/홈/비대칭)를 가짐 → scan
  matching / 알려진 형상 ICP로 종단 mm pose 직접 산출 가능.
- **Wheel encoder (odometry)**: metric 변위 담당. 단, 슬립 누적오차 있으므로 절대
  기준(LiDAR/vision) 없이 인코더만 믿는 구간은 짧게.

**역할 분담 원칙**: vision = "어느 방향으로 어긋났나", LiDAR/encoder = "그게 몇 mm인가".
단안 up-to-scale 모호성은 LiDAR/encoder의 metric 신호로 해소된다.

---

## 2. 핵심 아키텍처

### 2.1 두 영역(two-regime) 구조 — 절대 타협하지 말 것

```
[광역~중간 구간]  학습 정책 (diffusion policy, DINO feature subgoal 추종)
       │          - 넓은 수렴 영역, 외관/조명 강건, 멀티모달 접근 경로
       │          - 목표: LiDAR 정합이 동작 가능한 영역까지 안정적으로 진입
       ▼
   [핸드오프]      트리거 조건 = LiDAR가 타겟 기하를 충분히 관측 + 대략 정렬
       │
       ▼
[종단 mm 구간]    2D LiDAR 기하 정합 (scan matching / known-shape ICP)
                  - metric, textureless에 강함, mm 정밀
```

- **종단 mm는 학습 정책이 배우지 않는다.** LiDAR가 책임진다. 이것이 데이터 효율의 원천.
- 핸드오프 트리거 로직(언제 LiDAR로 넘길지)은 명시적 모듈로 구현하고 디버깅 가능하게 둘 것.

### 2.2 학습 정책 내부

- **백본**: diffusion policy (NoMaD/LaWAM 계열 참고). 골 이미지 대신 **골 feature**로 조건화.
- **표현**: DINO feature map을 subgoal로 사용. 픽셀 생성 X (LaWAM식 latent subgoal).
- **폐루프화**: 액션 청크를 **짧게** 끊고 **갱신 주파수를 높여** open-loop 실행 구간을 최소화.
  (청크 단위 BC는 open-loop라 종단 반응성이 약하다는 알려진 한계를 회피)

### 2.3 골 feature 조건화 + 손실 (검증된 조합부터)

- **A (백본, 필수)**: 골 feature를 **조건 입력**으로 넣고 행동은 imitation/denoising 손실로 학습.
  (거의 모든 강한 goal-conditioned diffusion policy의 기본형: NoMaD, BESO, D-GCBC 등)
- **B (보조 손실, 그다음)**: **예측된 subgoal feature**를 실제 미래 feature에 맞추는 손실
  (LaWAM의 `L_wm`, MDT의 self-supervised 손실 계열). **작은 가중치**로만. feature 재구성
  과적합 주의.
- **B-outcome (선택, 차별점 후보)**: "실행 결과 feature가 골 feature에 가까워지도록" 하는
  손실은 **LaWM을 미분가능 forward model로 쓸 때만** 가능. 단, LaWM feature 해상도가 거칠어
  outcome 손실의 정밀도 상한도 같이 제한됨 → mm는 LiDAR에 맡기는 원칙과 일관되게,
  이건 광역 표현 정형화 용도로만.

### 2.4 멀티모달 융합

- 권장: **역할 분리형 융합** (vision=방향 / LiDAR=metric / encoder=변위). late fusion보다
  실패 모드가 투명하고 도킹 디버깅에 유리.
- **extrinsic calibration (LiDAR–camera)** 정밀도가 mm 도킹 정밀도의 상한을 직접 결정.
  캘리브레이션 품질을 항상 의심하고, 관련 파라미터는 한 곳에 모아 관리.
- 시간 동기화 필수. 모달리티 신뢰도 기반 동적 가중을 고려.

---

## 3. Novelty & 평가 (가장 중요)

### 우선순위
1. **A — 시연 trajectory 개수 효율 (1순위, 논문 중심)**
2. **C — 학습/추론 연산량 효율 (보조, A의 부산물로 한 문단 처리)**

### A를 정당화하는 논리 (귀속을 명확히)
"종단 mm를 LiDAR가 책임지므로 학습 정책은 정밀 보정 시연이 불필요 → 거친 광역 접근만
배우면 되고, 이는 적은 시연으로 커버된다." → 데이터 효율은 **DINO의 공로가 아니라 역할
분담 아키텍처의 공로**다.

### 메인 실험: 시연 개수 축 ablation
- x축 = 시연 개수 N ∈ {5, 10, 20, 50, ...}, y축 = 성공률 **및 수렴 영역**.
- 비교: (ours: LiDAR 핸드오프 O) vs (baseline: 정책이 종단까지 책임, LiDAR 핸드오프 X / NoMaD류).
- 기대: ours는 적은 N에서 빨리 포화, baseline은 N 키워도 천천히 오르다 포화(ResiP 현상).

### 귀속 분리 ablation 2종 (필수)
1. **DINO feature vs scratch feature** (같은 N) → 데이터 효율 중 DINO 몫(C) 분리.
2. **LiDAR 핸드오프 on/off** (같은 N, 같은 DINO) → 시연 개수 효율 중 아키텍처 몫(A) 분리.

### 평가 지표 주의
- 성공률만 보지 말 것. **"다양한 초기 자세에서 LiDAR 정합 영역 진입에 성공한 비율"**을
  별도 측정. 적은 시연의 진짜 한계는 성공률보다 **수렴 영역 축소**로 먼저 드러난다.

---

## 4. 위험 / 함정 (코드 작성 시 계속 의식)

- **시연에 없는 행동은 정책이 못 배운다.** mm 정밀을 정책에 기대지 말 것(→ LiDAR).
- **B 손실 과다 → feature 재구성 과적합.** 가중치 보수적으로.
- **핸드오프 미트리거**: 적은 시연으로 광역이 약하면 LiDAR 정합 영역 진입 자체 실패. 트리거
  조건과 진입률을 항상 모니터링.
- **LiDAR 기하 aliasing**: 타겟이 구별 기하를 가졌다는 전제가 깨지면(반복/대칭) 종단이 흔들림.
  타겟 형상 가정을 코드 주석/설정에 명시.
- **extrinsic 오차가 mm 상한**: 센서가 정밀해도 캘리브레이션이 cm급이면 mm 도킹 불가.
- **occupancy raster ≠ mm 정밀**: grid 양자화로 mm 보장 불가. 종단 pose는 raw-point
  known-shape ICP가 책임. occupancy 정합도는 핸드오프 트리거/ICP 수렴검증/aliasing 방어
  보조 신호로만.

---

## 5. Claude Code 작업 지침

1. **먼저 기존 코드베이스를 파악한다.** 정책 백본(diffusion policy 구현 위치), 관측 인코더,
   골 조건화 경로, 액션 헤드, 데이터 로더, 학습 루프, 배포/제어 루프를 찾아 매핑한 뒤
   작업을 시작한다. (NoMaD/LaWAM 포크라면 해당 모듈을 우선 식별)
2. **새로 짜기 전에 재사용한다.** 위 설계를 기존 모듈의 최소 수정으로 얹는 방향을 우선 제안.
3. **두 영역 분리를 코드 구조에 반영한다.** 학습 정책 모듈과 LiDAR 종단 정합 모듈, 그리고
   둘을 잇는 핸드오프 모듈을 명확히 분리.
4. **변경 시 이 문서의 설계 결정과 충돌하는지 확인**하고, 충돌하면 먼저 알린다.
5. **평가 코드는 novelty(A)에 정렬한다**: 시연 개수 ablation과 귀속 분리 ablation 2종,
   그리고 수렴 영역(정합 영역 진입률) 지표를 측정할 수 있도록 구성.

---

## 6. 합의된 설계 결정 로그 (요약)

- [결정] 종단 mm = 2D LiDAR 기하 정합, 광역 = 학습 정책. (역할 분담)
- [결정] DINO feature subgoal 추종, 픽셀 생성 안 함.
- [결정] 골 feature는 A(조건 입력) 백본 + B(예측 subgoal 감독) 보조 손실.
- [결정] B-outcome은 LaWM forward model 사용 시에만, 광역 정형화 용도.
- [결정] 폐루프화 = 짧은 청크 + 높은 갱신 주파수.
- [결정] novelty 1순위 = 시연 개수 효율(A), 핸드오프 구조가 그 정당화.
- [결정] 데이터 효율 귀속 분리를 위해 DINO on/off, LiDAR 핸드오프 on/off ablation 필수.
- [확인됨] 센서 = RGB camera + 2D LiDAR + wheel encoder (멀티모달).
- [확인됨] 도킹 타겟은 LiDAR로 구별 가능한 기하를 가짐.
- [결정] occupancy 정합도는 mm 산출 수단이 아님 — 핸드오프 트리거 / ICP 수렴검증 / aliasing 방어 보조 신호로만.
