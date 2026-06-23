# 구현 진행 기록 (Implementation Log)

> 작성 기준일: 2026-06-23 · 브랜치: `vla`
> 근거 설계 문서: [CLAUDE.md](CLAUDE.md)
> 이 문서는 지금까지 구현·검증한 내용을 정리한 것이다. 모든 수치는 실제 실행 결과다.

---

## 0. 한눈에 보기

CLAUDE.md의 **두 영역(two-regime) 도킹** 설계를 코드로 내려, 다음 3개 블록을 구현·검증했다.

```
   [학습 쪽: 광역 접근]                         [기하 쪽: 종단 mm]
   diffusion policy ───────── 핸드오프 ─────────► raw-point ICP
   (DINO + goal feature 조건화)   (수렴영역 진입)    (known-shape, mm)
        Loss A 배선 ✅             라벨러로 정량화 ✅      엔진+템플릿 ✅
```

| 블록 | 상태 | 한 줄 |
|---|---|---|
| **후보 1** — 종단 LiDAR ICP 엔진 + 공식 도크 템플릿 | ✅ 완료 | 실데이터에서 sub-mm 검증 |
| **후보 2** — 오프라인 ICP sub-goal 자동 라벨러 | ✅ 완료 | marker_pose 없이 155개 시연 학습 가능화 |
| **Loss A** — goal-DINO-feature 조건화 (NoMaD 마스킹) | ✅ 코드 배선·단위검증 | 실학습은 h5 전처리 필요 |

---

## 1. 데이터셋 분석 (`dataset/after_0328`)

- **실데이터**: `dock/` 에 **155 에피소드** (전부 같은 물리적 도크, 실패 시연 없음 — 사용자 확인).
- 빈 폴더 `episode_postech`, `episode_saung` 는 무시.
- 에피소드당 모달리티: `command.csv`(지령 v), `encoder.csv`(측정 v), `room1/room2`(mp4+jpg, **640×480**), `lidar.jsonl`, `marker_pose.csv`, jitter csv.
- **LiDAR**: 이미 Cartesian (x,y) 점, 360° FOV, **~190점/스캔, 12.5 Hz**, range 0.5–4.3 m.
  - 각 해상도 ~1.3°, **점 간격 @0.3 m = 6.7 mm / @0.5 m = 11 mm**.
- **이미지 해상도 불일치**: 신규 데이터는 640×480 인데 기존 config는 240×320 / `image_top·bottom` → 전처리 시 정합 필요.

### 핵심 통찰 — "점 간격 ≠ 정밀도 바닥"
점당 LiDAR 노이즈는 ~11 mm 이지만, **알려진 형상에 수십 점을 맞추면 포즈는 √N 으로 평균화되어 sub-mm** 가 나온다. 이것이 ICP가 mm를 내는 원리이며, 래스터(이미지화)는 각 점을 셀에 양자화해 이 평균화를 버리므로 mm용으로 부적합하다 (CLAUDE.md §4/§6 일치).

---

## 2. 후보 1 — 종단 LiDAR ICP 엔진 + 공식 도크 템플릿

### 2.1 `endgame/` 패키지 (정책과 완전 분리)
학습 정책은 `policy_fn`으로 주입만 하고 torch 스택과 분리 (CLAUDE.md 작업지침 §3).

| 파일 | 역할 |
|---|---|
| `endgame/se2.py` | SE(2) 포즈 유틸 (numpy 전용) |
| `endgame/config.py` | **모든 캘리브레이션 한 곳에** — extrinsic(=mm 상한), 타겟 형상, ICP/핸드오프 파라미터 |
| `endgame/target_model.py` | 알려진 도크 형상 템플릿 + 대칭/aliasing 메타데이터 |
| `endgame/icp_matcher.py` | **raw-point known-shape ICP → mm SE(2) pose** (point-to-line) + 수렴진단 + aliasing 탐지 |
| `endgame/occupancy.py` | occupancy 래스터 — **보조 신호 전용**(포즈 산출 안 함) |
| `endgame/handoff.py` | 명시적·히스테리시스 핸드오프 상태기계 (모든 조건 로깅) |
| `endgame/orchestrator.py` | APPROACH(정책)→ENGAGED(ICP 서보)→DONE |
| `scripts/demo_endgame.py` | 합성 스캔 스모크 테스트 (3종 전부 PASS) |

### 2.2 디버깅으로 얻은 핵심 결정
- **point-to-point ICP는 sparse polyline에서 3–6 mm 국소최소에 갇힘** → **point-to-line ICP**(법선 기반 Gauss-Newton)로 교체해 mm 달성.
- **Aliasing 탐지**는 ICP를 **template 중심 기준 제자리 회전**으로 재시작해야 동작(원점 기준은 발산). ≥2 basin이면 `trustworthy=False`.

### 2.3 실데이터 검증 (`scripts/icp_real_data.py`)
marker_pose가 (당시) 불가라 GT 없이 **반복정밀도 + 알려진 오프셋 복원**으로 측정:

| 측정 | 결과 |
|---|---|
| 단일 스캔 복원 | ~3–5 mm |
| **10-스캔 누적 복원** | **~0.8–1.1 mm (100% < 2 mm)** |
| 정지 반복정밀도 | ~2–5 mm / 0.25° |

→ **실데이터에서 mm 확정.** 도크에서 로봇이 정지(|v|~1e-4)하므로 스캔 누적은 공짜.

### 2.4 도크 정체 — 사용자가 직접 표시
- 도크 = **U자 홈(notch) + 양쪽 어깨** 구조. 로봇이 홈에 들어가 도킹.
- **홈 바닥만(반경 25cm) 쓰면 거의 대칭 → aliased → 복원 34 mm 실패.** 어깨까지 포함(반경 ~0.45 m)해야 비대칭이 생겨 mm.
- LiDAR sensor 프레임이 로봇 heading과 **~90° 어긋남** (서보 방향에만 영향, ICP 기하엔 무관).
- room1 카메라는 **로봇 탑재 온보드**.

### 2.5 공식 도크 템플릿 자산 (`scripts/build_dock_template.py`)
- 155개 중 **154개 에피소드를 ICP로 정렬·누적** → `endgame/assets/dock_template_real.npy` (1640점, +meta json). 정렬 RMS 3.6 mm.
- 좁은 restart band 고정(180° flip 방지): `ICPConfig.for_real_dock()` (0, ±5, ±10°).
- `make_template("real_dock")` 로 endgame에서 로드.
- **홀드아웃 검증** (baseline 보정): ICP 정밀도 **중앙값 0–2 mm, 4/5 에피소드 100% < 2 mm**.
- 보너스: 에피소드별 도킹 위치 차이(P0) 2.7–26 mm = ICP 오차 아닌 **실제 시연 변동**.

---

## 3. 후보 2 — 오프라인 ICP sub-goal 자동 라벨러

`scripts/label_subgoals.py` → `dataset/after_0328/subgoal_labels/*.{json,npz}` (155개) + `_index.json`

### 3.1 메커니즘
- **도킹 프레임(ICP 확실)에서 시간 역방향으로 프레임마다 ICP 추적**(이전 포즈로 seed; 단일 restart로 5× 가속).
- 프레임별 도크 상대 포즈/거리 복원 → **marker_pose 대체**.
- 도크 거리는 **최근접 도크 표면점**까지로 정의 (템플릿 중심은 벽 팔에 편향).

### 3.2 산출 라벨
- **sub-goal**: 도크 표면 {0.9, 0.7, 0.55, 0.45 m, docked} 에서, 각 지점을 가장 가까운 room1/room2 카메라 프레임에 앵커 (골 feature 소스).
- **핸드오프 onset 거리** = ICP가 안정적으로 잠기는 최원거리 = 정책이 도달해야 할 **수렴영역 경계** (§3 지표).
- **GT 없는 성공 라벨** (ICP 확인 도크 도달).

### 3.3 집계 (155 시연)
| 지표 | 값 |
|---|---|
| 성공 | **155/155 (100%)** |
| 핸드오프 onset (수렴영역 경계) | **중앙값 85 cm** (p10–p90 53–119, 범위 42–147) |
| docked clearance | 중앙값 42 cm, std **0.7 cm** |
| ICP 커버리지 | 중앙값 90% 프레임 |
| sub-goal/에피소드 | 5개 |

→ 정책은 **약 85 cm + 대략 정렬**까지 데려가면 ICP가 인계받는다 = §3 데이터-효율 주장의 정량 근거.

---

## 4. marker_pose 재평가 — 독립 GT 확보 (사용자 정정)

marker_pose(카메라 ArUco)는 **off-angle에선 불가**하지만 **도크를 향한/근접 구간에선 신뢰 가능**.

검증 (ICP 라벨과 대조):
- marker 거리가 **접근 전 구간에서 ICP 거리와 거의 완벽히 추적**.
- **카메라↔LiDAR 장착 오프셋 +2.8 cm로 전 에피소드 일관** (실제 extrinsic 측정값).
- 오프셋 제거 후 **두 독립 센서 일치 ~10 mm 중앙값** → ICP가 반복정밀할 뿐 아니라 **정확함을 교차검증**.

→ 그동안 없던 **절대 GT 확보.** 도크 근처 성공 판정/검증 앵커로 사용 가능.

**도킹 성공 허용오차 = 1 cm** (사용자 지정) → `EndgameConfig.success_translation_tol_m = 0.01`. 마침 marker↔ICP 일치(~1cm)·시연 변동(~1–2.6cm)과 일관.

---

## 5. Loss A — goal-DINO-feature 조건화 (CLAUDE.md §2.3)

sub-goal을 diffusion policy 학습에 연결. **goal = 도킹 프레임의 DINO feature**(픽셀 아님), NoMaD 마스킹으로 한 정책이 조건화/무지향 동시 학습. 모두 하위호환(`use_goal=false`면 기존 동작).

| 파일 | 변경 |
|---|---|
| `cleandiffuser/nn_condition/sensor_fusion_condition.py` | goal 경로(전용 resampler/slot/modality emb + null 토큰), `_build_goal_tokens`, 마스킹(1=조건화, 0=무지향). +2.4M 파라미터 |
| `dataset/docking_dataset.py` | `with_goal`, `goal_mask_prob`, `ep_end_map` → 도킹 프레임을 goal 이미지로 반환 + goal_mask |
| `scripts/train.py` | goal 프레임을 frozen DINO로 인코딩 → `goal_feat1/2`+`goal_mask`를 condition에 추가 |
| `utils/setups.py`, `configs/robot/smr.yaml` | 배선 + `use_goal: true`, `goal_mask_prob: 0.5`, `num_goal_latents: 16` |

**검증**: 조건망에 랜덤 feature 입력 시 goal_active vs 무지향 조건벡터가 **1.10 차이**(per-sample ~0.53) → goal이 실제로 조건에 반영됨. (짧은 체크포인트 궤적이 아니라 *조건 인코더 출력*을 직접 검증 — adaLN-zero gotcha 회피.)

---

## 6. 검증 상태 요약

| 항목 | 검증 방법 | 결과 |
|---|---|---|
| ICP mm (합성) | `scripts/demo_endgame.py` | 1.1 mm, 핸드오프 도킹 2.4 mm, aliasing 거부 — 3/3 PASS |
| ICP mm (실데이터) | `scripts/icp_real_data.py` | 10-스캔 누적 0.8–1.1 mm |
| 공식 템플릿 (홀드아웃) | `scripts/build_dock_template.py` | 중앙값 0–2 mm, 4/5 100%<2mm |
| ICP vs marker GT | 교차검증 | 일치 ~10 mm, 오프셋 +2.8 cm |
| sub-goal 라벨러 | 155 에피소드 집계 | 성공 155/155, onset 85 cm |
| goal 조건화 | 조건망 단위 테스트 | 조건벡터 차이 1.10 |

---

## 7. 파일 인벤토리

### 신규
```
endgame/                     se2, config, target_model, occupancy,
                             icp_matcher, handoff, orchestrator, __init__
endgame/assets/              dock_template_real.npy, dock_template_real.json
scripts/                     demo_endgame.py, icp_real_data.py,
                             build_dock_template.py, label_subgoals.py
dataset/after_0328/subgoal_labels/   155 × {json, npz} + _index.json
outputs/icp_probe/           검증/시각화 PNG들
docs/IMPLEMENTATION_LOG.md   (이 문서)
```

### 변경 (Loss A 배선 + 성공 허용오차)
```
cleandiffuser/nn_condition/sensor_fusion_condition.py
dataset/docking_dataset.py
scripts/train.py
utils/setups.py
configs/robot/smr.yaml
endgame/config.py            (success_translation_tol_m = 0.01)
```

---

## 8. 남은 것 / 다음 길목

1. **after_0328 → h5 전처리 (다음 길목).** 로더가 기대하는 h5(`image_top/image_bottom/encoder/episode_ends`)로 155개를 변환해야 실제 학습 가능. (현 config의 `validation_dataset1.h5` 부재). 640×480 해상도·room 네이밍 정합 포함.
2. **Loss B** (예측 subgoal feature 감독) — 모델이 feature를 출력해야 하는 더 큰 변경. CLAUDE.md대로 작은 가중치로 나중에.
3. **§3 평가/ablation 하베스트**: 시연-개수 N∈{5,10,20,50} 스윕, DINO on/off, LiDAR 핸드오프 on/off, 그리고 **수렴영역 진입률**(라벨러의 onset 분포 활용).
4. **실제 LiDAR 런타임 배선**: extrinsic yaw 확정 + 스캔 누적을 endgame 서보에 내장 + `TwoRegimeController`에 실제 정책 연결.

---

## 부록 — CLAUDE.md 설계 결정과의 정합

- [✓] 종단 mm = raw-point ICP, 광역 = 학습 정책 (역할 분담) — 실데이터로 입증.
- [✓] DINO feature subgoal 추종, 픽셀 생성 안 함 — Loss A는 goal *feature* 조건화.
- [✓] occupancy = mm 산출 수단 아님, 보조 신호로만 — `endgame/occupancy.py` 도크스트링에 강제.
- [⚠ 충돌 기록] `dinoless_add_lidar` 브랜치는 LiDAR occupancy를 정책 condition 입력으로 사용 → 본 설계(분리된 ICP)와 충돌. 새 `endgame/`은 분리 유지.
- [✓] 데이터 효율 귀속 분리 ablation(DINO on/off, 핸드오프 on/off) — 라벨러 onset 분포 + `use_goal` 플래그로 측정 준비됨.
