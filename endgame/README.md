# `endgame/` — 오프라인 ICP (도크 포즈 라벨 생성)

> 한 줄: **2D LiDAR raw-point ICP로 "도크가 로봇 기준 어디에 있나(x,y,θ)"를 mm급으로 계산**하는 모듈.
> 단, **런타임엔 안 돕니다.** 단일 모델(Option A)이 이 ICP 출력을 *오프라인 교사 라벨*로 distill하고, 추론 땐 비전만 동작.
> ICP 개념·레퍼런스는 [../docs/icp_background.md](../docs/icp_background.md).

---

## 이 디렉토리의 역할

- **무엇**: 알려진 도크 형상(template)을 LiDAR 스캔에 정합(ICP)해 도크의 정밀 SE(2) 포즈를 산출.
- **왜 오프라인만**: 정밀 도킹을 *학습 모델*이 하기로 결정(Option A). ICP는 시연 데이터에 한 번 돌려 **프레임별 도크 포즈 라벨**(= 단일 모델 aux head의 정답)과 **공식 도크 템플릿**을 만드는 데만 쓰임. 런타임 핸드오프/서보 코드는 제거됨(`docs/plan/00_overview.md` §2).
- **baseline**: "학습 정밀 vs ICP 정밀" 비교의 기준선이기도 함.

## 파일

| 파일 | 역할 |
|---|---|
| `icp_matcher.py` | **핵심.** raw-point known-shape ICP (point-to-line, Gauss-Newton). 멀티-restart aliasing 가드, 수렴 진단(rms/inlier). `ICPMatcher.match(scan, init) → ICPResult(pose, …)` |
| `target_model.py` | 도크 형상 템플릿. `make_template("real_dock")` = 실데이터로 만든 공식 도크, `make_template("l_notch_dock"/"symmetric_rect")` = 합성 테스트용 |
| `se2.py` | SE(2) 포즈 유틸 (compose/inverse/transform_points/pose_distance) — numpy 전용 |
| `config.py` | `ICPConfig` (대응거리 anneal·수렴 임계·restart 각도) + `ICPConfig.for_real_dock()` (좁은 restart band) |
| `assets/dock_template_real.npy` | **공식 도크 템플릿** (155 에피소드를 ICP 정렬·누적, 1640점) + `.json` 메타 |
| `__init__.py` | `ICPMatcher, ICPResult, make_template, TargetTemplate, ICPConfig` 등 export |

## ICP가 하는 일 (요약)
```
초기 추측 → ① 템플릿 각 점 ↔ 최근접 스캔 점 짝짓기 → ② 최적 회전+이동 계산
         → ③ 적용 후 반복(수렴까지) → 도크 포즈 (x,y,θ)
```
점 하나는 노이즈가 있어도 *아는 모양*에 수십 점을 동시에 맞추면 평균화돼 **mm급** 포즈가 나옴(실데이터 검증: 단일 스캔 ~3–5mm, 10-스캔 누적 ~0.8mm).

## 누가 쓰나 (전부 오프라인 스크립트)
| 스크립트 | 용도 |
|---|---|
| `scripts/label_subgoals.py` | 시연을 역추적하며 ICP → **프레임별 도크 포즈 라벨** 생성 → `dataset/after_0328/icp_labels/<ep>.npz` |
| `scripts/build_dock_template.py` | 여러 에피소드 정합·누적 → `assets/dock_template_real.npy` |
| `scripts/icp_real_data.py` | 실데이터 ICP 정밀도/aliasing 검증 |

## 주의 (실데이터 도크 특성)
- 도크 = U자 홈 + 양쪽 어깨. **홈 바닥만 쓰면 거의 대칭 → aliasing** → 어깨까지 포함해야 mm.
- 도크가 borderline 180° 대칭이라 **restart band는 핸드오프 정렬 불확실성에 맞춰 좁게**(`for_real_dock`). 전체 360° 탐색은 flip 유발.
- 라벨은 LiDAR 프레임 인덱스 기준. (`docs/plan/02_preprocessing.md` §3 라벨 정합 참고)
