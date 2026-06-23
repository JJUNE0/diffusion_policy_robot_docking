# Step 4 — 결과 확인 (feasibility)

> 선행: [03_training.md](03_training.md) · 로그: `results/train_single.log` · 플롯: [05_plots.md](05_plots.md)
> 학습 설정: 30 에피소드 h5, 3000 step, batch 8, sparse-uint8 vision (15GB RAM 안전).

---

## 1. 핵심 결과 — feasibility 확인 ✅

**하나의 모델이 (주)속도 정책과 (aux)ICP 도크 포즈를 동시에 학습한다.**

| step | denoising loss | dock 포즈 오차 (train) |
|---|---|---|
| 0 | 0.95 | 117 mm |
| 1150 | 0.10 | 47 mm |
| 2350 | 0.13 | 32 mm |
| 2950 | **0.07** | **21 mm** |

- **정책(denoising)**: 0.95 → ~0.07 로 수렴 → goal-조건 속도 궤적을 학습 중.
- **정밀 aux(ICP distill)**: 117 → ~21 mm (train, 평균) → 비전+raw점만으로 도크 포즈를 **~2–3 cm** 로 회귀. = "현재 state가 goal인가"의 도착 판정을 ICP 없이 모델이 흉내냄.
- **RAM**: 학습 프로세스 ~140 MB (sparse-uint8). 머신 다운 없음.

→ Option A(단일 모델, 정밀 내재화, 런타임 ICP 없음)가 **돌아간다는 것**이 확인됨.

## 1b. Held-out 일반화 (학습 안 한 8 에피소드, 11,834 프레임) ✅
`scripts/eval_heldout.py`, step-1500 체크포인트, 학습 정규화 통계로 denorm.

| 지표 (held-out) | 값 |
|---|---|
| denoising loss | **0.095** (train 0.07–0.15와 동등 → 과적합 아님) |
| dock 포즈 오차 | 중앙값 **31 mm**, 평균 44 mm, p90 95 mm |
| ≤1 cm / ≤3 cm | 9% / 49% |

→ **처음 보는 에피소드에서도 정책·정밀이 train과 비슷하게 전이** = 진짜 일반화. (최종 3000-step은 train 21mm였으니 held-out도 더 낮을 여지; 저장된 건 1500.)

## 2. 정직한 한계 (아직 아닌 것)
- **~3 cm, 아직 1 cm 아님**: feasibility 규모(30 ep·3000 step·깊이 off·소형). p90 ~9.5cm(일부 프레임 큼). 더 내려갈 여지: step↑·데이터↑·단안/Orbbec 깊이·점 인코더 튜닝·스캔 누적·최종 체크포인트 저장.
- **성공률 9%@1cm**: 정밀 목표(1cm) 미달 — 위 개선축 필요.
- **정책 롤아웃 도킹 미검증**(현재는 라벨 회귀·denoising 수렴까지; 실제 닫힌루프 도킹은 다음).

## 3. 이게 증명하는 것 (연구 관점)
- "ICP 라벨을 비전 모델에 distill → 추론은 비전만"이 **학습 가능**하다는 것.
- 단일 모델 안에서 **접근 정책 + 정밀 포즈 판정**이 공존 가능 → 핸드오프·런타임 알고리즘 불필요.
- ICP는 오프라인 교사 역할만으로 충분(런타임 미사용) — markerless로도 동일.

## 4. 다음 (결과 강화)
1. **held-out 평가**: 학습 안 한 에피소드(예: 31–40)로 별도 h5 → 체크포인트 로드 → dock mm·denoising 측정. (preprocessing `--max_episodes`로 슬라이스, 또는 에피소드 범위 분할)
2. **성공률 @1cm**: aux dock 포즈가 목표 ±1 cm 안인 프레임 비율.
3. **정책 롤아웃**: 샘플링한 속도 궤적으로 실제 도킹 도달(수렴 영역) 측정.
4. **개선축**: 깊이 브랜치 on, step/데이터↑, (나중 §5) 멀티 sub-goal DINO 정합.

체크포인트: `outputs/results/single_model_feasibility/2026-06-23_16-43-36/checkpoint_step_1500.pt`.
