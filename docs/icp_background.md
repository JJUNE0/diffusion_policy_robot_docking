# ICP 배경지식 + 우리 코드 위치 + 레퍼런스

> 기준일: 2026-06-23
> 목적: ICP가 뭔지(개념), 학술 레퍼런스가 어디 있는지, 그리고 **우리 코드의 어디에** 짜여 있는지를 한 문서에.
> 관련: [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md)(전체 진행), [inference_io.md](inference_io.md)(추론 I/O)

---

## 1. ICP가 뭔가 — 직관

**ICP = Iterative Closest Point(반복 최근접점).** 두 점 집합을 가장 잘 겹치게 하는 **강체 변환(회전 R + 이동 t)** 을 찾는 정합(registration) 알고리즘이다.

도킹에 적용하면:
- **템플릿(template)**: 도크가 어떻게 생겼는지 *미리 아는* 2D 점들 (우리: U자 홈+어깨).
- **스캔(scan)**: 지금 LiDAR가 찍은 점 구름.
- ICP가 푸는 것: *"이 도크가 로봇 기준으로 정확히 어디에, 몇 도 틀어져 있나?"* = SE(2) 포즈 `(x, y, θ)`.

핵심 반복:
```
초기 추측(정책/핸드오프가 대략 제공)
  │
  ▼
① 템플릿 각 점 → 가장 가까운 스캔 점 짝짓기      ← "Closest Point"
② 그 대응을 가장 잘 맞추는 (R, t) 계산
③ 적용 후 ①로 (수렴까지)                        ← "Iterative"
  │
  ▼
최종 변환 = 도크의 정밀 포즈
```

**왜 mm가 나오나**: 점 하나는 노이즈가 있어도, *아는 형상*에 수십 점을 동시에 맞추면 노이즈가 √N으로 평균화돼 포즈가 sub-mm로 수렴한다 (점 간격 7mm여도 포즈는 ~1mm). 우리 실데이터에서 확인됨.

**한계**: ICP는 **지역(local) 최적화** — 초기 추측이 어느 정도 맞아야 하고(그래서 정책의 핸드오프가 필요), 형상이 대칭이면 여러 해가 생김(aliasing).

---

## 2. 약간의 수학 (배경)

ICP가 최소화하는 목적함수는 "대응(correspondence)" 정의에 따라 둘로 갈린다.

### 2.1 Point-to-Point (원조 ICP, Besl & McKay 1992)
대응 점 쌍 `{p_i (템플릿) ↔ q_i (스캔)}` 에 대해:

$$ \min_{R,t} \sum_i \lVert R\,p_i + t - q_i \rVert^2 $$

각 반복에서 (R, t)는 **닫힌형(SVD)** 으로 풀린다 (Arun 1987 / Umeyama 1991): 중심 이동 → 공분산 행렬 SVD → 회전. 2D에서는 더 간단히 `θ = atan2(Σ(ãₓb̃_y − ã_yb̃ₓ), Σ(ãₓb̃ₓ + ã_yb̃_y))`.

문제: 희소한 LiDAR 직선(폴리라인)에서는 점들이 선 방향으로 **미끄러지며(aperture problem)** 국소최소에 갇힌다. (우리도 처음에 3–6mm로 막혔다.)

### 2.2 Point-to-Line / Point-to-Plane (우리가 쓰는 방식)
대응을 점-점 거리가 아니라 **점에서 표면(2D에선 선)까지의 수직 거리**로 잰다. 스캔 점 `q_i`의 국소 법선 `n_i`에 대해:

$$ \min_{R,t} \sum_i \big( n_i \cdot (R\,p_i + t - q_i) \big)^2 $$

- 2D **point-to-line** = **Censi PLICP (2008)**. 3D의 point-to-plane(Chen & Medioni 1992)의 2D판.
- 표면 접선 방향으로의 미끄러짐을 허용하므로 **수렴 영역이 훨씬 넓고** 직선·평면에 강하다.
- 비선형이라 **Gauss-Newton**으로 푼다: 작은 증분 `ξ=(dx,dy,dθ)`에 대해 점의 이동을 1차 근사 `p' ≈ p + [dx,dy] + dθ·[−p_y, p_x]` 하고, 법선 방향 잔차를 선형 최소제곱으로 풀어 포즈를 갱신·반복.

> 우리 구현이 정확히 이 2.2다.

---

## 3. 우리 코드는 어디에 있나

ICP 본체는 **`endgame/icp_matcher.py`** 한 파일이다 (numpy + scipy만 의존, torch 무관).

| 코드 | 위치 | 역할 (↔ §2 / 레퍼런스) |
|---|---|---|
| `ICPMatcher.match()` | [icp_matcher.py:126](../endgame/icp_matcher.py#L126) | **공개 진입점.** KD-tree 구축 → 법선 추정 → 멀티 restart → aliasing 판정 → 최선 basin 반환 |
| `ICPMatcher._icp_once()` | [icp_matcher.py:80](../endgame/icp_matcher.py#L80) | **point-to-line ICP 핵심 루프** (Gauss-Newton). ↔ §2.2, Censi 2008 |
| `_estimate_normals()` | [icp_matcher.py:52](../endgame/icp_matcher.py#L52) | k-NN PCA로 점별 법선 추정 (point-to-line의 전제) |
| `ICPResult` | [icp_matcher.py:30](../endgame/icp_matcher.py#L30) | 출력 구조체: 포즈 + 수렴진단(rms, inlier) + `ambiguous`/`is_trustworthy` |

**보조 모듈:**
- `endgame/se2.py` — SE(2) 합성/역/점변환/포즈거리 (`compose`, `transform_points`, `pose_distance`).
- `endgame/target_model.py` — 도크 템플릿(known shape) 정의·로드 (`make_template("real_dock")`).
- `endgame/config.py` — `ICPConfig`(대응거리 anneal, 수렴 임계, restart 각도 등) + `ICPConfig.for_real_dock()`.

**ICP를 쓰는 스크립트:**
- `scripts/demo_endgame.py` — 합성 스캔 단위 검증.
- `scripts/icp_real_data.py` — 실데이터 정밀도/aliasing 검증.
- `scripts/build_dock_template.py` — 여러 에피소드 정합·누적으로 공식 템플릿 생성.
- `scripts/label_subgoals.py` — 시연에 ICP를 돌려 sub-goal 자동 라벨링.

---

## 4. 우리 설계 선택 ↔ 레퍼런스 대응

| 우리 코드의 선택 | 무엇 | 근거 레퍼런스 |
|---|---|---|
| `_icp_once` Gauss-Newton 점-선 잔차 | point-to-line ICP | **Censi 2008 (PLICP)**, Chen & Medioni 1992 |
| `_estimate_normals` (kNN 공분산 최소고유벡터) | 표면 법선 추정 | 표준 normal estimation |
| `max_d = d_start·0.7^it` (대응거리 coarse→fine) + `dists ≤ max_d` | outlier 제거·점진 정밀화 | Rusinkiewicz & Levoy 2001(rejection 단계), Zhang 1994(robust ICP) |
| `cKDTree.query` 최근접 탐색 | 대응 가속 | 표준 (k-d tree) |
| `restart_yaws` 멀티 시작 + **centroid-spin** + basin 군집화 | aliasing/국소최소 방어 | ICP의 지역성 한계(서베이) — centroid-spin은 우리 고안 |
| `rms_residual`, `inlier_ratio`, `is_trustworthy` | 수렴 신뢰도 게이트 | 표준 진단 |

> **왜 글로벌 정합(Go-ICP, FGR, RANSAC)을 안 쓰나**: 정책이 핸드오프로 *대략 정렬된 초기 추측*을 주므로, 무거운 전역 탐색 없이 **지역 정밀화 + 좁은 aliasing 가드**면 충분하다. 이것이 두 영역(정책=광역, ICP=종단) 분담의 직접적 이득이다.

---

## 5. 레퍼런스 (배경지식용 핵심 논문)

**원조·기본**
- P.J. Besl, N.D. McKay (1992). *A Method for Registration of 3-D Shapes.* IEEE TPAMI 14(2). — **원조 ICP(point-to-point).**
- Y. Chen, G. Medioni (1992). *Object modelling by registration of multiple range images.* Image and Vision Computing 10(3). — **point-to-plane ICP.**
- K.S. Arun, T.S. Huang, S.D. Blostein (1987). *Least-Squares Fitting of Two 3-D Point Sets.* IEEE TPAMI 9(5). — 닫힌형 SVD 해.
- S. Umeyama (1991). *Least-squares estimation of transformation parameters between two point patterns.* IEEE TPAMI 13(4).

**2D LiDAR scan matching (우리와 가장 가까움)**
- A. Censi (2008). *An ICP variant using a point-to-line metric.* IEEE ICRA. — **PLICP, 우리 방식.**
- P. Biber, W. Straßer (2003). *The Normal Distributions Transform (NDT).* IEEE/RSJ IROS. — 대안 scan matching.

**변형·서베이**
- S. Rusinkiewicz, M. Levoy (2001). *Efficient Variants of the ICP Algorithm.* 3DIM. — ICP 변형 분류(선택/대응/가중/제거)의 고전.
- A. Segal, D. Hähnel, S. Thrun (2009). *Generalized-ICP.* RSS. — GICP.
- F. Pomerleau, F. Colas, R. Siegwart (2015). *A Review of Point Cloud Registration Algorithms for Mobile Robotics.* Foundations and Trends in Robotics. — **입문용 서베이 추천.**
- I. Vizzo et al. (2023). *KISS-ICP: In Defense of Point-to-Point ICP.* IEEE RA-L. — 최신 LiDAR 오도메트리.

**구현 참고 라이브러리(우리는 직접 구현했지만 비교용)**
- Open3D `pipelines.registration`, PCL(Point Cloud Library), libpointmatcher(ETH ASL), scipy `spatial.cKDTree`.

---

## 6. 더 읽으면 좋은 순서 (배경지식 부족할 때)
1. Pomerleau 2015 서베이 §1–3 (큰 그림).
2. Besl & McKay 1992 (원조 개념).
3. Censi 2008 PLICP (우리 방식 — 짧고 명확).
4. 그다음 우리 코드 [icp_matcher.py](../endgame/icp_matcher.py) `_icp_once`를 위 개념과 1:1로 대조.
