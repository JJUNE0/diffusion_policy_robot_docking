# Step 2 — Preprocessing (after_0328 → 학습용 h5, **raw LiDAR 점**)

> 선행: [01_dataset_sensor_fusion.md](01_dataset_sensor_fusion.md) · 코드: `utils/preprocessing.py`
> 결정 반영: LiDAR는 **BEV 이미지 대신 raw 점**(정밀·효율·온라인 일관). BEV는 `--lidar_format bev`로 ablation 보존.

---

## 1. 왜 h5 + 왜 raw 점
- **h5**: 기존 diffusion 파이프라인 재사용 + 이미지 로딩 빠름.
- **raw 점(BEV 아님)**: ① 양자화 손실 없음(정밀, ICP 논리와 동일) ② 점~190개로 작아 효율(BEV 100×100×2=2만값 → 점 256×2 패딩, 실제 ~80점) ③ 온라인에 들어오는 raw 그대로(래스터화 단계 X).

## 2. 산출 h5 스키마
| dataset | shape | dtype | 설명 |
|---|---|---|---|
| `image_top` / `image_bottom` | [N,3,240,320] | u1 | room1 / room2 (resize) |
| `encoder` | [N,2] | f4 | (vx, wz) 보간 |
| `episode_ends` | [E] | i8 | 경계 (goal=ep_end−1 도킹 프레임) |
| **`lidar_points`** | [N,256,2] | f4 | **raw 점**(로봇 프레임), 군집 crop, zero-padding |
| **`lidar_npoints`** | [N] | i4 | 프레임별 유효 점 수(패딩 마스크용) |
| **`dock_pose`** | [N,3] | f4 | **ICP 도크 포즈[x,y,θ]**; 비신뢰=NaN |
| **`reliable`** | [N] | u1 | 1=유효 ICP 라벨 → **정밀 aux 손실 마스크** |

(ablation: `--lidar_format bev` → `lidar_map [N,2,S,S]`)

## 3. 파이프라인 (sync 로직)
`StrictSyncRoboticsDataset`:
1. `dock/episode_*`마다 encoder/room1/room2/lidar 타임스탬프 수집.
2. `target_hz`(30) 격자 `target_ts`.
3. 각 ts에서 최근접 room1/room2 이미지, encoder 선형보간, 최근접 lidar(`lidar_idx`).
4. **raw 점 crop**(`_make_lidar_points`): 스캔에서 **가장 가까운 점 기준 `crop_r`(0.8m) 이내** 점만 → M=256 패딩 + 유효수. (결정적·ICP 불필요 → 온라인 동일 재현)
5. **라벨**: `lidar_idx`로 `icp_labels/<ep>.npz`의 `pose`,`reliable` 조회 → `dock_pose`,`reliable`. (`ep_name`의 `dock/` 접두사는 basename 처리)
6. h5 기록 + `episode_ends`.

## 4. 실행
```bash
# 전체(155)  — raw 점(기본)
python utils/preprocessing.py \
  --data_root dataset/after_0328 --save_path dataset/after_0328_train.h5 \
  --use_lidar --with_labels --lidar_format points --lidar_crop_r 0.8 --lidar_max_points 256

# 검증: 앞 N개 (... --max_episodes 30)
# ablation: --lidar_format bev --lidar_range_m 2.0 --lidar_resolution 0.02
```

## 5. 검증 결과 (3 에피소드)
- 3357 프레임, 키 = image_top/bottom, encoder, episode_ends, **lidar_points[3357,256,2]**, lidar_npoints, dock_pose, reliable.
- `lidar_npoints`: min 20, **mean 80**, max 101 (crop 0.8m). raw 점 자체는 ~7MB(초소형).
- reliable 88% (라벨러 일치). 점들이 dock_pose 주변에 정확히 위치(예: pts y[-1.0,-0.47] vs dock y≈-0.64).
- h5 크기는 **이미지가 지배**(3 ep ≈ 960MB). raw vs BEV의 효율 이득은 lidar 필드(7MB vs ~96MB)에 있음.

## 6. 메모 / 다음
- **goal**: 별도 저장 불필요 — 도킹 프레임=`ep_end−1`, 기존 `docking_dataset.ep_end_map`이 goal로 사용.
- **이미지 저장이 용량 지배** → 155 전체는 ~50GB. 본격화 시 이미지 다운샘플/jpeg-encode 고려(지금 feasibility엔 30 ep로 충분).
- **`dataset/new/record_46`**(orbbec 깊이)는 미래 깊이 브랜치 소스.
- 다음 Step3: `docking_dataset`가 `lidar_points`/`npoints`/`dock_pose`/`reliable` 반환 → 조건망에 **point 인코더 브랜치** + **ICP aux head**.
