# 추론 시 입력/출력 구조 (Inference I/O)

> 기준일: 2026-06-23 · 브랜치 `vla` · 근거: [CLAUDE.md](CLAUDE.md), [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md)
> 실제 코드 기준: `scripts/inference_ema.py`, `endgame/orchestrator.py`, `endgame/icp_matcher.py`

추론은 **두 영역**으로 나뉜다. 광역 접근은 학습 정책(diffusion)이, 종단 mm는 LiDAR ICP가 담당하며, 둘은 핸드오프로 연결된다.

```
매 제어 tick:
  ┌─────────────────── APPROACH ───────────────────┐   ┌──────── ENGAGED ────────┐
  관측 → DINO feature ─┐                              │   raw LiDAR 점 → ICP →    │
  encoder velocity ────┼─→ 조건망 → DiT 샘플링 →     │   mm SE(2) pose → 서보 →  │
  (goal feature) ──────┘    속도 궤적 [60,2]          │   속도 (v,ω)              │
                            첫 스텝으로 제어            │   1cm 이내면 DONE         │
  └────────────────────────────────────────────────┘   └─────────────────────────┘
        (핸드오프: ICP가 신뢰 가능 + 대략 정렬되면 ENGAGED로 전환)
```

---

## A. 학습 정책 추론 (diffusion policy)

`scripts/inference_ema.py` (EMA 평활) / `scripts/inference_rtc.py` (랭킹 선택). 매 스텝 **open-loop로 미래 속도 궤적을 생성**하고, 첫 스텝(들)을 제어에 쓴 뒤 갱신한다 (CLAUDE.md §2.2: 짧은 청크 + 높은 갱신율).

### A.1 입력 (INPUT) — 매 스텝 조건(condition) 구성

| 항목 | 출처 | 텐서 형태 | 설명 |
|---|---|---|---|
| `image_room1` | room1 카메라 히스토리, `::vision_stride`로 희소화 | `[B, Tv, 3, H, W]` | Tv=5 (30/6) |
| `image_room2` | room2 카메라 히스토리 | `[B, Tv, 3, H, W]` | |
| → `dino_feat1/2` | frozen DINO (`DinoBatchDetector`) | `[B, Tv, 196, 768]` | 196 패치 × 768 |
| `velocity` | encoder 속도 히스토리 (정규화 v,ω) | `[B, Tm, 2]` | Tm=30 |
| **`goal_feat1/2`** ※ | **도킹(목표) 프레임의 DINO feature** | `[B, 1, 196, 768]` | Loss A 경로 (아래 A.4) |
| **`goal_mask`** ※ | 1=목표 조건화 (추론 시 보통 1) | `[B]` | |
| `prior` | 가우시안 노이즈 (디퓨전 시작점) | `[n_samples, horizon, 2]` | horizon=60 |

샘플링 하이퍼파라미터(config): `solver=ode_dpmsolver++_2M`, `inference_sampling_steps=100`, `w_cfg`(guidance weight), `num_samples`(궤적 표본 수).

> 조건은 `n_samples`만큼 복제되어 한 번에 여러 궤적을 샘플한다:
> `context = {dino_feat1, dino_feat2, velocity}` → 각 키 `.repeat(n_samples, ...)`.

### A.2 파이프라인

```
obs(이미지·속도) ─→ frozen DINO ─→ 조건망(SensorFusionConditionNetwork)
                                      이미지1/2 토큰 + 속도 토큰 (+ goal 토큰)
                                      → Transformer → readout = 조건벡터 [B, d_model=384]
                                                                │
prior 노이즈 [N,60,2] ──→ DiT1d 디노이저 (조건벡터로 adaLN) ──→ DPM-Solver++ (100 step)
                                                                │
                                                        정규화 속도 궤적 [N,60,2]
```

호출부:
```python
sample_out = nn_diffusion.sample(
    solver=args.solver, w_cfg=args.w_cfg, prior=prior,
    condition_cfg=context, n_samples=n_samples,
    sample_steps=args.inference_sampling_steps)
```

### A.3 출력 (OUTPUT)

| 단계 | 내용 | 형태 |
|---|---|---|
| 원시 샘플 | `n_samples`개 정규화 속도 궤적 | `[N, 60, 2]` |
| 역정규화 | `denormalize(out, act_scale, act_min)` → 실제 (v,ω) | `[N, 60, 2]` |
| 표본 집계 | N개 평균 (또는 RTC 랭킹) → 단일 궤적 | `[60, 2]` |
| 시간 평활 | `apply_trajectory_ema` (직전 예측과 EMA, α) | `[60, 2]` |
| 제어 | 궤적 첫 스텝 (v,ω)를 명령으로 사용, 다음 tick 재생성 | `[2]` |
| 시각화/궤적 | `reconstruct_pose_rk4` (속도→포즈, RK4 적분) | `[61, 3] = (x,y,θ)` |

- 실시간 배포: `socket_path.send_pyobj(local_predicted_path)` 로 ZMQ 스트리밍.
- **출력 = 미래 속도 궤적 [60, 2] (선속도, 각속도)**; 불확실성은 `n_samples`의 평균·표준편차로 표현.

### A.4 ⚠ goal 조건화의 추론 배선 상태

학습(`train.py`)에는 goal-feature(Loss A)가 배선됐지만, **추론 스크립트(`inference_ema.py`)는 아직 obs-only**다 (조건망을 `use_goal` 없이 생성, context에 `goal_feat` 미포함). goal 조건 추론을 켜려면:

1. 조건망을 `use_goal=True`로 생성 (학습과 동일 구조).
2. 목표 프레임(도킹 프레임)을 DINO 인코딩 → `goal_feat1/2 [B,1,196,768]`.
3. `context["goal_feat1/2"]`, `context["goal_mask"]=ones` 추가.
4. `w_cfg`로 guidance 강도 조절(무지향 대비 목표 방향 강조).

> 참고: 갓 학습된/EMA 체크포인트에선 adaLN-Zero 때문에 조건 효과가 0으로 보일 수 있다 → 충분히 학습 후, 또는 `w_cfg`↑/조건벡터 직접 확인으로 검증.

---

## B. 종단 ICP 추론 (endgame)

`endgame/icp_matcher.py` — 학습 없음, 순수 기하.

### B.1 입력 (INPUT)
| 항목 | 형태 | 설명 |
|---|---|---|
| `scan_pts` | `[N, 2]` | raw LiDAR 점 (로봇/센서 프레임, extrinsic 적용) |
| 템플릿 | `[M, 2]` | 공식 도크 형상 `make_template("real_dock")` (1640점) |
| `init_pose` | `[3] = (x,y,θ)` | 코스 초기 추정 (정책/핸드오프가 제공) |

### B.2 출력 (OUTPUT) — `ICPResult`
| 필드 | 의미 |
|---|---|
| `pose` `[x,y,θ]` | **도크의 mm SE(2) 포즈** (로봇 프레임) |
| `rms_residual_m`, `inlier_ratio` | 수렴 진단 |
| `ambiguous` | §4 aliasing 플래그 (≥2 basin) |
| `is_trustworthy()` | 수렴 ∧ 비-aliased → mm 권한 부여 가능 |

```python
res = ICPMatcher(make_template("real_dock"), ICPConfig.for_real_dock()).match(scan_pts, init_pose)
# res.pose → 도크 포즈, res.is_trustworthy() → 신뢰 여부
```

---

## C. 두 영역 오케스트레이터 추론 (per-tick)

`endgame/orchestrator.py` 의 `TwoRegimeController.step(scan_points, policy_obs)` — 한 제어 tick 전체.

### C.1 입력
| 항목 | 형태 | 설명 |
|---|---|---|
| `scan_points` | `[N, 2]` | LiDAR 점 (extrinsic 자동 적용) |
| `policy_obs` | (불투명) | APPROACH에서 주입된 `policy_fn`에 그대로 전달 (위 A의 obs) |

### C.2 출력 (dict)
| 키 | 의미 |
|---|---|
| `v`, `w` | 명령 속도 (선·각속도) |
| `mode` | `approach` / `engaged` / `done` |
| `done` | 도킹 완료 여부 (1cm 이내) |
| `handoff` / `icp` | 디버그용 핸드오프 결정 / ICP 결과 |
| `translation_err_m`, `rotation_err_rad` | ENGAGED에서 남은 오차 |

### C.3 모드별 동작
- **APPROACH**: `v,w = policy_fn(policy_obs)` (위 A 정책). 매 tick 핸드오프 평가; 조건(기하 관측 ∧ 대략 정렬 ∧ 비-aliased) 충족이 K프레임 지속되면 ENGAGED.
- **ENGAGED**: ICP `match` → 도크 포즈 → 유니사이클 go-to-pose 서보 → `(v,w)`. 신뢰 불가 시 정책으로 폴백.
- **DONE**: 잔여 오차가 허용오차 이내(서보 조준 3mm, 수용 `success_translation_tol_m=1cm`)면 정지.

---

## D. 형태(shape) 빠른 참조

| 기호 | 값 | 의미 |
|---|---|---|
| `B` | 1 (추론) | 배치 |
| `Tv` | 5 | 희소 vision 프레임 (30/`vision_stride`=6) |
| `Tm` | 30 | encoder 속도 히스토리 (`obs_horizon`) |
| `horizon` | 60 | 미래 속도 궤적 길이 |
| `d_model` | 384 | 조건/DiT 은닉 차원 |
| `n_samples` | 8–10 | 궤적 표본 수 |
| DINO feature | `[*, 196, 768]` | 패치 196 × 768 |
| 정책 출력 | `[horizon, 2]` | (v_linear, v_angular) |
| ICP 출력 | `[3]` | (x, y, θ) mm 포즈 |

---

## E. 요약: 한 문장씩

- **정책 추론**: (이미지→DINO feature) + encoder velocity (+ goal feature) → DiT가 노이즈에서 **미래 속도 궤적 [60,2]** 생성 → EMA 평활 → 첫 스텝으로 제어 (open-loop chunk, 높은 갱신율).
- **ICP 추론**: raw LiDAR 점 + 공식 도크 템플릿 + 코스 초기추정 → **mm SE(2) 도크 포즈** → 유니사이클 서보 속도.
- **오케스트레이터**: tick마다 `scan + policy_obs` 입력 → `{v, w, mode, done, 진단}` 출력. APPROACH(정책)↔ENGAGED(ICP) 전환, 1cm 이내 DONE.

### 미배선/주의
- 추론 스크립트의 **goal 조건화 미배선** (A.4) — 학습엔 있으나 `inference_ema.py`엔 추가 필요.
- **실제 LiDAR 런타임 미연결** — `TwoRegimeController`에 실제 정책 `policy_fn` 연결 + extrinsic yaw 확정 + 스캔 누적 필요.
- 현재 정책 추론은 **데이터셋 리플레이**(h5)에서 동작; after_0328 학습엔 h5 전처리 선행.
