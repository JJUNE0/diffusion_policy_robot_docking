# Step 3 — 단일 모델 학습 (diffusion + raw LiDAR + goal + ICP aux)

> 선행: [02_preprocessing.md](02_preprocessing.md) · 코드: `scripts/train.py`, `utils/setups.py`, `utils/docking_dataset.py`, `cleandiffuser/nn_condition/sensor_fusion_condition.py`, `configs/robot/smr.yaml`

---

## 1. 무엇을 학습하나 (단일 모델)
```
입력(조건):  DINO(room1/2) + encoder velocity + raw LiDAR 점(point 인코더) + goal feature
출력:        (주) 미래 속도 궤적 [H,2]  ─ denoising 손실
            (aux) ICP 도크 포즈 [x,y,sin,cos] ─ reliable 프레임만, masked 손실  → 정밀/도착 판정
```
- 조건망에 **point-set 브랜치**(`_build_lidar_tokens`: 점→MLP→마스킹 Perceiver→토큰)와 **aux head**(readout→4) 추가.
- 학습 루프: `denoise_loss = nn_diffusion.loss(...)`(조건망 forward가 aux 예측 캐시) → `total = denoise + aux_weight·masked_pose_loss` → 단일 backward(EMA 포함).

## 2. ⚠ RAM 경량화 (15GB 머신 필수)
원인: dataset가 샘플마다 **30프레임×2카메라 float32**(~55MB) 반환 → `num_workers` 프리페치로 RAM 폭증 → OOM/머신 다운.
수정: `sparse_vision: true` → dataset가 **5프레임 uint8만** 반환(~24× 절감, GPU에서 float 변환). 학습 프로세스 호스트 RAM ~140MB.
> inference 스크립트는 `sparse_vision_uint8=False`(기본) 유지 → 영향 없음.

## 3. 관련 config 플래그 (`configs/robot/smr.yaml`)
| 키 | 값 | 의미 |
|---|---|---|
| `use_lidar_points` | true | raw 점 브랜치 |
| `num_lidar_latents` | 16 | point Perceiver latent 수 |
| `use_aux_pose` | true | ICP 포즈 aux head |
| `aux_weight` | 1.0 | aux 손실 가중치 |
| `use_goal` | true | 도킹 프레임 goal feature (Loss A) |
| `sparse_vision` | true | RAM 경량(5프레임 uint8) |
| `train_data_path` | `…/after_0328_train.h5` | Step2 산출 h5 |

---

## 4. 전체 학습 CLI (처음부터 끝까지)

```bash
cd ~/ws/diffusion_policy_robot_docking
# (환경) wandb 끄고, HF DINO는 캐시 사용
export WANDB_MODE=disabled HF_HUB_OFFLINE=1

# 1) ICP 라벨 생성 (한 번만; 이미 있으면 생략)  → dataset/after_0328/icp_labels/
python scripts/label_subgoals.py --all

# 2) 전처리 → 학습용 h5 (raw 점 + ICP 라벨)
python utils/preprocessing.py \
  --data_root dataset/after_0328 --save_path dataset/after_0328_train.h5 \
  --use_lidar --with_labels --lidar_format points --lidar_crop_r 0.8 --lidar_max_points 256
#   (디스크/속도: 전체 155대신 일부만 → 위에 --max_episodes 30)

# 3) 학습 (15GB RAM 안전 설정)
python scripts/train.py \
  experiment_name=single_model_feasibility \
  diffusion_gradient_steps=3000 batch_size=8 num_workers=2 \
  log_interval=50 save_interval=1500 \
  train_data_path=$PWD/dataset/after_0328_train.h5
```

- **RAM 부족하면**: `batch_size=4 num_workers=0` 로 더 낮춤 (가장 안전).
- **백그라운드 + 세션/터미널 종료에도 생존**: 앞에 `setsid nohup` , 뒤에 `> results/train_single.log 2>&1 < /dev/null &`.
- 체크포인트: `results/single_model_feasibility/<timestamp>/checkpoint_step_*.pt` (action_min/scale 동봉).
- 재개: `... resume_path=results/single_model_feasibility/<timestamp>`.

### 로그 보는 법
```
Step 1500 | Loss: 0.42 | aux 0.08 | dock 31.5mm
            └ denoising  └ 정밀 aux 손실  └ 도크 포즈 오차(작아질수록 정밀)
```

---

## 5. 검증 (스모크)
- `[8,5,3,240,320]` sparse vision 확인, denoising+aux+dock mm 로깅 정상, **호스트 RAM ~140MB**(다운 X).
- 초기(미학습) dock ~117–296mm → 학습하며 감소(standalone PoC는 ~3cm까지 내려감). 본 학습 결과는 [04_results.md](04_results.md)에.

## 6. 다음
- Step4: held-out 정밀도(dock mm)·denoising 수렴·성공률(≤1cm) 확인.
- (나중 §5) 멀티 sub-goal DINO 정합, anomaly.
