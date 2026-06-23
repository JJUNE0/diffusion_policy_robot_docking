# Step 5 — 결과 Plot

> 선행: [04_results.md](04_results.md) · 스크립트: `scripts/plot_training.py` · 산출: `results/train_convergence.png`

---

## 1. 학습 수렴 곡선
![train convergence](../../results/train_convergence.png)

- **좌 (정책)**: denoising loss 0.95 → ~0.07 수렴. goal-조건 속도 궤적 학습.
- **우 (정밀 aux)**: dock 포즈 오차 ~70 → ~21 mm 하강(점=원시, 선=이동평균). 점선 = 1 cm 성공 허용오차. 아직 1 cm 위지만 명확히 하강 중(train).

## 2. 재생성
```bash
python scripts/plot_training.py results/train_single.log results/train_convergence.png
```
- 로그의 `Step N | Loss: .. | aux .. | dock ..mm` 라인을 파싱 → loss/dock 곡선.
- 다른 학습 로그도 인자로 교체 가능.

## 3. 다음에 추가할 plot (결과 강화 시)
- **held-out dock mm 히스토그램** (일반화).
- **성공률(≤1cm) vs step**.
- **정책 롤아웃 궤적** (예측 vs GT, 기존 `inference_*` 의 RK4 경로 재사용).
- **수렴 영역 진입률** (라벨러 onset 분포와 대조).
- (나중) 시연-개수 N ablation 곡선 (§3 novelty).
