# 마스터 플랜 — 단일 모델(Option A) 정밀 도킹

> 기준일: 2026-06-23 · 브랜치 `vla`
> **결정**: 하나의 diffusion(→flow matching) 정책이 접근~정밀을 끝까지 담당.
> **지금(feasibility, 단순·빠름)**: 단일 도킹-goal + **ICP aux 포즈**로 정밀/도착 판정, **raw LiDAR 점** 입력.
> **나중(확장)**: 멀티 sub-goal DINO 정합("전부 정합=성공") + anomaly 탐지. → §5.
> ICP는 **오프라인 교사(라벨)+baseline**로만 (런타임에 ICP 알고리즘 없음).

---

## 1. 지금 아키텍처 (feasibility)

```
관측(접근~정밀 동일 경로):
  room1/room2 RGB ─► DINO feature ┐
  encoder velocity ───────────────┤
  raw LiDAR 점(군집 crop) ─────────┤─► SensorFusion 조건망 ─► 조건벡터
  goal feature(도킹 프레임 1장) ───┘                          │
                                                              ▼
                          DiT/flow 디노이저 ─► 미래 속도 궤적 [H,2]   (주 출력)
                                       └─────► aux: ICP 도크 포즈 [x,y,θ]  (정밀/도착 판정)
```

- **주 출력**: 속도 궤적. 시연이 mm로 도킹했으므로 입력만 충분하면 정밀도 BC로 학습.
- **goal**: 도킹 프레임 1장의 DINO feature(=방향 안내). NoMaD 마스킹(조건/무지향).
- **aux head**: 근접(ICP-reliable) 프레임에서 **ICP 도크 포즈 회귀**(교사=ICP 라벨). → "현재 state가 goal인가"의 **도착 판정**(예측 포즈 ≤ 1cm). (σ/anomaly는 §5로 보류.)
- **LiDAR = raw 점**: BEV 래스터 X. 이유 — 정밀(양자화 손실 없음)·효율(점~190개로 작음)·온라인 일관(들어오는 raw 그대로). 입력=점집합 → **point-set 인코더**(PointNet/소형 set-transformer). crop=**가장 가까운 군집 주변**(오프라인=온라인 동일).
- 핸드오프·런타임 ICP·멀티모델 **없음**.

### 추론 시 goal 도달 방식 (개념)
goal-conditioned 정책은 *(현재 관측, goal)* → "goal로 가는 다음 행동"을 내도록 학습됨 → 폐루프로 자연히 도달. **도착/정지 판정만** aux ICP 포즈(≤1cm)가 담당.

---

## 2. 코드 맵 + 정리(cleanup)  — ✅ 완료
- **삭제**: 런타임 2-regime(`endgame/handoff.py`,`orchestrator.py`,`scripts/demo_endgame.py`), 독립 2-모델 학습(`scripts/train_precision_vision.py`), 중복문서.
- **유지**: 기존 diffusion 파이프라인, 오프라인 ICP 라벨도구(`label_subgoals`,`build_dock_template`), ICP 코어(`endgame/{icp_matcher,se2,target_model,config,occupancy}`+`assets`), `precision_docking/`(인코더 컴포넌트 재사용).
- **외부 절대 불가침**: `plugins/`, `rc_server/`.

---

## 3. 순서 계획 (각 단계 = md 1개)

| # | 단계 | md | 상태 |
|---|---|---|---|
| 0 | 정리 + 플랜 | `00_overview.md` | ✅ |
| 1 | 데이터+센서퓨전 설계 | `01_dataset_sensor_fusion.md` | ✅ |
| 2 | **Preprocessing — raw 점(+군집 crop) + ICP 포즈 라벨** | `02_preprocessing.md` | 🔄 |
| 3 | **학습** — 조건망 point 브랜치 + 단일 goal + ICP aux head | `03_training.md` | ⬜ |
| 4 | **결과 확인** (정밀도 mm / 성공률 1cm) | `04_results.md` | ⬜ |
| 5 | **결과 Plot** | `05_plots.md` | ⬜ |

## 4. 완료 기준 (DoD)
- 2: raw 점(+mask)·dock_pose·reliable h5에 정합(소수 에피소드 검증).
- 3: smoke loss↓ + 본 학습 체크포인트 + aux 포즈 오차(mm).
- 4: held-out 정밀도(mm)·성공률(≤1cm).
- 5: 재현 가능한 plot.

---

## 5. 나중 계획 (확장 — 지금은 안 함)

### 5.1 멀티 sub-goal DINO 정합 (★ 사용자 핵심 아이디어, 복원 예정)
- **조건화**: goal = 도킹 1장이 아니라 **현재 위치의 다음 sub-goal**(라벨러의 0.9/0.7/0.55/0.45m+docked 프레임) DINO feature.
- **달성 판정 (coarse→fine)**:
  - 원/중거리: `cosine(현재 DINO feat, sub-goal feat) ≥ τ` (τ=성공 시연 분포 캘리브) → 다음 sub-goal 진행.
  - 최종: aux ICP 포즈 ≤ 1cm.
  - **모든 sub-goal 순서대로 정합 = 도킹 성공** (사용자 정의).
- 역할 분담: DINO feature는 멀리서 영역 판정(가까이선 포화), 마지막 mm만 ICP. → "DINO 정합 + ICP 통합".

### 5.2 anomaly 탐지
- 1차: aux에 σ(heteroscedastic) 헤드 → 기초 불확실성 신호.
- 2차: σ 임계 대응 로직(감속/정지/재시도) + 제대로 된 이상탐지(one-class/recon/OOD).

### 5.3 기타
- **BEV ablation**: raw 점 vs BEV 비교(`--lidar_format bev` 보존).
- **단안/Orbbec 깊이**: `dataset/new/record_46`의 orbbec 깊이를 깊이 브랜치로.
- **flow matching**: diffusion feasibility 후 백본 교체.
