# 정밀 도킹 비전 모델 — 학습 파이프라인 / 센서 퓨전 / 입출력

> 기준일: 2026-06-23
> 근거 코드: `precision_docking/model.py`, `precision_docking/dataset.py`, `scripts/train_precision_vision.py`
> 방향: 종단 mm를 **런타임 알고리즘(ICP)이 아니라 학습 비전 모델**로. ICP는 **오프라인 교사(라벨 생성)+비교 baseline**으로만. (관련: [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md), [icp_background.md](icp_background.md))

---

## 0. 한 문장

**RGB(room1/2) + LiDAR-BEV 이미지**를 각각 인코딩해 **feature 레벨에서 융합**하고, 그 융합 표현에서 **도크 포즈(x,y,θ)** 와 **anomaly(불확실성)** 를 출력한다. 정답(교사)은 우리가 ICP로 미리 만든 도크 포즈 라벨이며, **추론 시엔 비전만** 돈다(알고리즘 없음).

---

## 1. 전체 학습 파이프라인

```
[데이터: after_0328]                         [오프라인 교사: ICP]
 room1/room2 jpg (640×480) ─┐                scripts/label_subgoals.py
 lidar.jsonl (raw 점)   ────┤                = ICP로 프레임별 도크 포즈 라벨
                            │                  → subgoal_labels/<ep>.npz
                            ▼                       │ (pose, reliable, ts)
        PrecisionDockingFrameDataset ◄─────────────┘  ※ reliable 프레임만 = 정밀 구간
        한 샘플 = (room1, room2, BEV) → ICP 포즈 라벨
                            │
                            ▼
        ┌──────────── PrecisionDockingNet ────────────┐
        │ room1 ─[RGB enc]─┐                           │
        │ room2 ─[RGB enc]─┤ concat → MLP(fuse) → 256d │→ pose [x,y,sinθ,cosθ]
        │ BEV   ─[BEV enc]─┘   (feature-level fusion)  │→ logvar (anomaly)
        └──────────────────────────────────────────────┘
                            │
                  heteroscedastic NLL loss  (pose 회귀 + 분산=anomaly)
                            │  (정답 = ICP 포즈)
                            ▼
                     AdamW 역전파 → 체크포인트
```

- **h5 불필요**: 데이터셋이 jpg 프레임 + ICP 라벨 npz를 **직접** 읽는다.
- 학습 샘플은 **ICP-reliable 프레임만** = 도크가 LiDAR로 잡히는 근접(정밀) 구간 = 우리가 학습시키려는 바로 그 구간.

---

## 2. 입력 (INPUT) — 무엇이, 어떤 형태로

| 모달리티 | 출처 | 전처리 | 텐서 형태 |
|---|---|---|---|
| **room1 RGB** | 온보드 카메라 jpg (라벨 프레임 ts에 가장 가까운 것) | resize→[0,1] | `[3, 96, 128]` |
| **room2 RGB** | 카메라 2 | 동일 | `[3, 96, 128]` |
| **LiDAR-BEV** | 그 프레임의 raw 스캔 → BEV occupancy 래스터 (로봇 중심, range 2.0 m, res 0.02 m) | `points_to_bev` | `[1, 100, 100]` |
| (선택) **단안 깊이** | RGB→깊이 (사전학습) | resize | `[1, 96, 128]` *(현재 off)* |

- BEV 래스터 컨벤션: 로봇 중심, +x(전방)→위, +y(좌)→좌 (occupancy와 동일).
- **타깃 정규화**: 도크 포즈 `(x,y)`는 데이터셋 통계로 정규화 `(x−mean)/std` (예: mean≈[0.04, −0.63], std≈[0.29, 0.08] — 센서 프레임). 각도 θ는 `(sinθ, cosθ)`로.

> 라벨은 **LiDAR ICP 프레임** 인덱스로 정의되고, 이미지는 **타임스탬프 최근접**으로 정렬해 페어링한다.

---

## 3. 센서 퓨전 — 어떻게 합치나 (핵심)

**Feature-level(late) fusion**: 각 모달리티를 *독립 인코더*로 1차 벡터(128d)로 만든 뒤 **concat → MLP**로 합친다. (early 채널-concat이나 attention fusion 대신.)

```
room1 [3,96,128] ─► SmallConvEncoder(RGB) ─► 128d ┐
room2 [3,96,128] ─► SmallConvEncoder(RGB) ─► 128d │  ← RGB 인코더는 두 방 공유(weight share)
 BEV  [1,100,100]─► SmallConvEncoder(BEV) ─► 128d ┤
(depth ───────────► SmallConvEncoder(Dep) ─► 128d)┘
                         concat → [384] (깊이 포함 시 512)
                              │
                         fuse MLP: 384 → 256 → 256   (GELU)
                              │
                  ┌───────────┴───────────┐
             pose_head(256→4)        logvar_head(256→1)
```

- **각 SmallConvEncoder**: Conv(stride2)×4 + GroupNorm + GELU → AdaptiveAvgPool → Linear(128). 사전학습 가중치 불요(오프라인 안전).
- **왜 late fusion**: 모달리티별 기여가 투명하고(실패 모드 분리, CLAUDE.md §2.4 정신), **모달리티 ablation(예: BEV 빼기, depth 넣기)이 쉽다.** 융합 위치/방식은 나중에 attention으로 교체 가능.
- RGB는 *방향/외관*, BEV는 *metric 기하*를 담아 **둘이 합쳐 정밀도를 함께 끌어올린다** (단안 모호성을 BEV가 보완).

---

## 4. 출력 (OUTPUT)

| 출력 | 형태 | 의미 | 역변환 |
|---|---|---|---|
| **pose** | `[4] = (x̂, ŷ, sinθ̂, cosθ̂)` | 도크의 SE(2) 포즈(센서 프레임), x,y는 정규화 | `x = x̂·std+mean`, `θ = atan2(sin,cos)` → 미터/라디안 |
| **logvar** | `[1]` | 예측 분산의 로그 = **anomaly/confidence** | `σ = exp(0.5·logvar)` (클수록 불확실/이상) |

- pose의 `(sin,cos)`는 forward에서 **단위원으로 정규화**(`F.normalize`)해 유효한 각도를 보장.
- 도크 포즈를 얻으면 이후 서보는 기존 오케스트레이터와 동일 — 단 **포즈가 ICP가 아니라 비전에서** 나온다.

---

## 5. 지도신호 / 손실 — ICP distillation

- **정답(교사)** = 그 프레임의 **ICP 포즈**(우리가 `label_subgoals.py`로 생성). 즉 ICP가 *오프라인에서만* 정답을 만든다.
- **손실** = heteroscedastic Gaussian NLL (`pose_nll_loss`):

```
L = 0.5 · exp(−s) · ‖pose − target‖²  +  0.5 · D · s        (s = logvar, D = 4)
```

- 분산 헤드 `s`가 *읽기 어려운(=비전으로 포즈가 불확실한)* 프레임에서 커지도록 학습 → 그게 곧 **정밀 도킹 중 anomaly 신호**. (값이 잘 맞는 곳은 σ↓, 가림/모호하면 σ↑.)
- 추후 one-class/reconstruction 헤드로 anomaly를 강화 가능(현재는 분산 기반 confidence).

---

## 6. 학습 설정 / 실행

```bash
python scripts/train_precision_vision.py --smoke              # 배선 확인(4 ep, 40 step)
python scripts/train_precision_vision.py --steps 20000 \
       --batch_size 64 --max_episodes 40 --lr 3e-4            # 본 학습(GPU)
```

- optimizer AdamW(lr 3e-4, wd 1e-4), 모델 ~0.68M 파라미터(의도적으로 소형).
- 로깅: loss + **pose 오차(mm / deg)** + anomaly σ.
- 체크포인트에 `xy_mean/xy_std` 동봉(추론 역정규화용).

**현재 수렴 (40 ep, train-set 기준, 진행 중):**
| step | pose 오차 | anomaly σ |
|---|---|---|
| ~40 (smoke) | 100–300 mm / 10° | 0.8 |
| ~1200 | ~30 mm / 1.8° | 0.11 |
| ~3400 | **~28–35 mm / 1–2°** | ~0.09 |

> 주의: 위는 **train 오차**다. 일반화는 held-out로 따로 측정해야 하고, 현재 ~3 cm는 소형 모델·BEV 2 cm·깊이 off 기준 PoC다. 데이터↑·step↑·깊이 추가·BEV 해상도↑로 더 내려갈 여지.

---

## 7. 추론 시 사용 (배포 관점)

- 추론: `(room1, room2, BEV) → PrecisionDockingNet → pose + σ`. **ICP/알고리즘 런타임에 없음.**
- ai-control 플러그인 `inference_fn`에 끼우면(같은 `CommandStep` 인터페이스) **비전 정밀 도킹**이 그대로 배포된다 (로봇 브리지 무수정). σ가 높으면 감속/정지/재시도 같은 안전 정책에 연결.
- 관련 배포 흐름은 [ai_server_robot_pipeline.md](ai_server_robot_pipeline.md), 추론 I/O 일반은 [inference_io.md](inference_io.md).

---

## 8. ICP(알고리즘)와의 관계 / 남은 것

- **ICP = 오프라인 교사 + baseline**: 정답 라벨을 만들고, "학습 정밀 vs ICP 정밀"을 같은 N에서 비교하는 §3 실험의 기준선.
- **남은 것**: (1) 단안 깊이 브랜치 켜기(가중치 다운로드 필요), (2) held-out 일반화·시연-개수 ablation, (3) anomaly 헤드 강화(one-class), (4) BEV 해상도/누적 스캔으로 정밀도 향상, (5) 플러그인 `inference_fn` 연결.
