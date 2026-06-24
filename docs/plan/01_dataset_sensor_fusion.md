# Step 1 — 데이터셋 분석 + 센서 퓨전 기법 + 단일모델 입력 설계

> 선행: [00_overview.md](00_overview.md) · 결정: 단일 모델(Option A)
> 목적: after_0328을 입력으로 어떻게 받고, **다른 브랜치의 퓨전 기법을 참조**해 단일 모델의 입력/융합을 확정.

---

## 1. after_0328 데이터셋 (학습에 쓸 것만)

| 모달리티 | 파일 | 형태 | 단일모델에서의 역할 |
|---|---|---|---|
| RGB room1/2 | `image/room1\|room2/*.jpg` (640×480) | 시간 히스토리(희소 Tv=5) | 방향/외관 (DINO feature) |
| encoder velocity | `encoder.csv` (vx, wz) | [Tm=30, 2] | 자기 운동·metric 변위 |
| LiDAR | `lidar.jsonl` (raw x,y, ~190점, 12.5Hz) | → **BEV 이미지** | metric 기하(정밀 보조) |
| (라벨) ICP 도크 포즈 | `icp_labels/<ep>.npz` (우리가 생성) | per-frame [x,y,θ] | **정밀 aux 헤드 교사** |
| (라벨) goal | 에피소드 도킹 프레임 | 단일 프레임 | goal feature 조건 (Loss A) |

- 전부 같은 물리 도크, 실패 시연 없음, 성공 허용오차 1cm. 자세한 분석은 [../design_of_endgame.md](../design_of_endgame.md) §1.

---

## 2. 센서 퓨전 기법 조사 (다른 브랜치 참조)

### 2.1 공통 패턴 — "모달리티별 토큰 → 공유 Transformer (late/token fusion)"
현재 `cleandiffuser/nn_condition/sensor_fusion_condition.py`(vla):
- 각 카메라: DINO feature [B,Tv,196,768] → `vision_proj` → **PerceiverResampler**(16 latents) → +time/slot/modality emb → 토큰.
- velocity → MLP → 토큰. 모든 토큰 + readout을 **Transformer로 융합**, readout 토큰을 조건벡터로.
- modality emb: 0=img1, 1=img2, 2=vel, 3=readout. (goal은 우리가 4·null 토큰으로 추가 — Loss A.)

### 2.2 LiDAR 퓨전 — `dinoless_add_lidar` 브랜치 (★ 채택 레퍼런스)
- **`vision/lidar_map_encoder.py` `LidarMapEncoder`**: BEV occupancy [B,C,H,W] → CNN stem(Conv×4, GroupNorm, GELU) → `AdaptiveAvgPool(14,14)` → 1×1 conv → **[B,196,768] 패치 토큰** (DINO 패치와 동일 포맷).
- **조건망 lidar 브랜치**: `lidar_patch_encoder` → `lidar_proj` → `lidar_resampler`(Perceiver 16) → +lidar_time/slot emb + **modality=4**. 입력 `lidar_map [B,Tv,C,S,S]` (BEV in [0,1], vision과 같은 Tv).
- 즉 **LiDAR를 "또 하나의 카메라"처럼** 토큰화해 같은 융합 경로에 합류. → 우리 단일 모델에 그대로 이식.

### 2.3 (참고) 대안 — global-feature 융합
우리 PoC `precision_docking/model.py`는 모달리티별 1벡터(128d) → concat → MLP. 더 단순하지만 **공간정보 손실** → 정밀엔 토큰 융합(2.2)이 유리. → 단일 모델은 2.2 채택, precision_docking 인코더는 참고/aux로만.

---

## 3. 단일 모델 입력/출력 설계 (확정)

```
조건망(SensorFusionConditionNetwork 확장):
  DINO(room1/2) ─► img 토큰 (modality 0,1)
  velocity ───────► vel 토큰 (modality 2)
  LiDAR-BEV ──────► LidarMapEncoder→resampler→ lidar 토큰 (modality 4)   ← 신규(이식)
  goal feature ───► goal 토큰 (+null 마스킹)                              ← Loss A(완료)
        └─ Transformer 융합 → 조건벡터 [B, d_model]
                       │
         ┌─────────────┴───────────────┐
   DiT/flow → 속도 궤적 [H,2] (주)   aux head → ICP 포즈[x,y,θ] + anomaly(σ)  ← 신규
                                       (근접 프레임에서만 손실, ICP 라벨 교사)
```

### 입력 텐서 스펙
| 키 | 형태 | 비고 |
|---|---|---|
| `dino_feat1/2` | [B,Tv,196,768] | 기존 |
| `velocity` | [B,Tm,2] | 기존 |
| `lidar_map` | [B,Tv,C,S,S] | **신규** BEV (C=2: occupied+log_density) |
| `goal_feat1/2`,`goal_mask` | [B,1,196,768],[B] | Loss A |
| `act`(타깃) | [B,H,2] | 속도 궤적 |
| `dock_pose`(aux 타깃) | [B,3] | ICP 라벨, 근접 프레임 |

### BEV 파라미터 결정 (정밀 지향)
- `dinoless_add_lidar`는 navigation용 **range 6m / res 0.05m**. 정밀 도킹은 도크가 ~0.4–0.6m이므로 **range 2.0m / res 0.02m (S≈100)**, C=2(occupied+log density). (정밀 상한은 occupancy가 아니라 **aux의 ICP 라벨**이 정함 — 아래 4.)

---

## 4. 설계 결정 / 정합성 메모
- **occupancy = 입력 보조, 정밀 supervision은 raw-point ICP 라벨.** (CLAUDE.md §4/§6의 "occupancy로 mm 내지 말 것"과 충돌 안 함 — BEV는 *입력 feature*일 뿐, 정밀 정답은 ICP raw-point 포즈.) 모델은 BEV+RGB+학습 prior를 합쳐 occupancy 양자화 한계를 넘을 여지.
- **퓨전 = late/token fusion** (모달리티별 토큰 → Transformer). 투명·모달리티 ablation 용이(§2.4 정신).
- **goal 마스킹(NoMaD)** 유지 → 조건/무지향 동시 학습.
- **aux head 위치**: 조건벡터(readout)에서 분기. 근접(ICP-reliable) 프레임에서만 aux 손실(가중치 작게) → 표현 정밀화 + anomaly(σ).
- **diffusion ↔ flow matching**: 백본 교체는 직교. 먼저 diffusion으로 feasibility, 이후 flow matching.

---

## 5. Step 1 산출 = 입력/퓨전 확정 → 다음 Step 2(Preprocessing)에서 이 스펙대로 패킹
- preprocessing이 만들 것: room1/2 이미지, encoder, **lidar BEV(또는 raw→on-the-fly BEV)**, episode_ends, **goal 프레임 인덱스**, **per-frame ICP dock_pose 라벨(+reliable 마스크)**.
- 포맷 결정(h5 vs 직접읽기 vs 패킹)은 Step 2에서.
