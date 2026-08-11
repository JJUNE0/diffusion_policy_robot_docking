# AI 모델 ↔ 서버 ↔ 로봇 통신 아키텍처 (큰 그림)

> 기준일: 2026-06-23
> 근거 코드: `plugins/`(ai-control 플러그인 + `DESIGN/`), `rc_server/`(서버 스택)
> 목적: 디테일보다 **전체 파이프라인**을 한눈에. 추론 입출력 자체는 [inference_io.md](inference_io.md), 모델 구현은 [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) 참고.

---

## 0. 한 문장 요약

로봇이 **카메라 영상(WebRTC)** 과 **센서(MQTT)** 를 서버로 올리면, **rc_server**가 이를 **AI 플러그인(ai-control)** 에 중계하고, 플러그인이 추론해 만든 **속도 궤적(trajectory)** 을 다시 **MQTT로 로봇에 내려보내** 도킹을 구동한다. 영상은 한 평면(WebRTC), 센서·명령은 다른 평면(MQTT/NATS)으로 흐른다.

---

## 1. 전체 구성도

```
        ┌──────────────────────────── ROBOT (robot-01) ────────────────────────────┐
        │  camera room1/room2 ──WebRTC publish──┐        ┌── MQTT sub: ai-command   │
        │  lidar/encoder/marker ──MQTT publish──┼──┐  ┌──┘   (= 실행할 궤적)         │
        └───────────────────────────────────────┼──┼──┼───────────────────────────┘
                       (영상 평면)               │  │  │  (명령 평면)
                          │                      │  │  │
        ┌─────────────────┼──────── rc_server (Docker 스택) ───────┼──┼─────────────┐
        │   ┌──────────┐   │   ┌──────────┐   ┌──────┐   ┌────────┐  │  ┌──────────┐ │
        │   │ LiveKit  │◄──┘   │ mqtt     │──►│ mqtt │──►│ NATS   │──┘  │ mqtt     │ │
        │   │ SFU      │       │ broker   │   │reader│   │ (내부  │     │ writer   │ │
        │   │(영상중계)│        │(mosquitto)│  └──────┘   │  버스) │◄────│(서버→로봇)│ │
        │   └────┬─────┘       └──────────┘              └───┬────┘     └────┬─────┘ │
        │        │ subscribe                    robot.data.>  │  output       │      │
        │        │                                  ┌─────────┴────┐          │      │
        │        │                                  │   RC Hub     │◄─────────┘      │
        │        │                                  │ (WS gateway) │   delivery:mqtt │
        │        │                                  └──────┬───────┘                 │
        │        │                                         │ WS JSON                 │
        └────────┼─────────────────────────────────────────┼─────────────────────────┘
                 │ WebRTC video                             │ 센서 data / 추론 output
        ┌────────┼─────────── AI 플러그인 컨테이너 (ai-control) ────────────┐
        │   ┌────▼────────────────┐         ┌─────────────────────────┐   │
        │   │ P1 Bridge Process   │  SHM/Q  │ P2 AI Process           │   │
        │   │ ·LiveKit 구독       │◄───────►│ ·우리 diffusion 모델      │   │
        │   │ ·Hub WS (센서/출력) │         │ ·inference_fn → 궤적     │   │
        │   │ ·윈도우/스냅샷/송신 │         │  (CommandStep[])         │   │
        │   └─────────────────────┘         └─────────────────────────┘   │
        └─────────────────────────────────────────────────────────────────┘
                          │ delivery: ui_overlay
                          ▼
        ┌──────────────────────────┐
        │ RC API → UI (시각화)      │  궤적/히트맵 오버레이
        └──────────────────────────┘
```

---

## 2. 구성요소 역할 (한 줄씩)

| 구성요소 | 위치 | 역할 |
|---|---|---|
| **로봇** | 현장 | 카메라(room1/room2) WebRTC 발행, 센서(lidar/encoder/marker) MQTT 발행, `ai-command` MQTT 구독·실행 |
| **LiveKit SFU** | rc_server | 영상 중계(WebRTC). 로봇이 publish, 플러그인이 subscribe |
| **MQTT broker (mosquitto)** | rc_server | 로봇↔서버 메시지 브로커 (센서 업링크 / 명령 다운링크) |
| **mqtt-reader / writer** | rc_server (mqtt-io) | MQTT↔NATS 브리지. reader=로봇 데이터 수신, writer=서버 명령 발행 |
| **NATS** | rc_server | 내부 메시지 버스 (Hub·mqtt-io·API·pipeline 연결) |
| **RC Hub** | rc_server | **WS 게이트웨이** — 플러그인 등록/센서중계/출력라우팅의 중심 |
| **RC API + UI** | rc_server | REST + 클라이언트 UI(궤적/메트릭 오버레이 시각화) |
| **Redis / RDB / MinIO** | rc_server | 설정 영속 / DB / 오브젝트 저장 |
| **ai-control 플러그인** | 별도 컨테이너 | **우리 AI 모델**. 영상+센서 받아 추론 → 궤적 출력 |

> P2(AI 프로세스)는 외부와 직접 통신하지 않는다. 모든 외부 I/O는 **Hub 경유 P1(Bridge)** 가 담당.

---

## 3. End-to-End 추론 파이프라인 (핵심)

한 번의 도킹 추론 사이클을 단계로:

```
① 업링크(로봇→서버→AI)        ② 추론(AI)            ③ 다운링크(AI→서버→로봇)     ④ 시각화
─────────────────────────    ──────────────────    ──────────────────────────   ─────────────
영상: 로봇 ─WebRTC─►          P1이 영상+센서로        P2가 궤적(CommandStep[])     output에
      LiveKit ─►P1(구독)      윈도우 구성 →           만들어 P1에 반환 →           ui_overlay도
센서: 로봇 ─MQTT─►            스냅샷 →                P1이 output{delivery:mqtt}   담아 Hub→API
      reader→NATS→Hub─►P1     P2 inference_fn         를 Hub로 WS 송신 →           →UI 오버레이
                              (diffusion 모델)        Hub→mqtt-writer→broker→
                                                      로봇 `robot/{id}/ai-command`
```

**문장으로:**
1. **영상 스트리밍**: 로봇이 room1/room2 카메라를 WebRTC로 LiveKit에 publish → 플러그인 P1이 그 room을 subscribe해 프레임을 받는다.
2. **센서 업링크**: 로봇이 lidar/encoder/marker를 MQTT로 발행 → `mqtt-reader`가 NATS(`robot.data.>`)로 변환 → Hub가 플러그인에 WS로 중계.
3. **추론**: P1이 영상·센서를 시간 정렬해 스냅샷 생성 → P2의 `inference_fn`(우리 diffusion 정책)이 **미래 속도 궤적**을 만든다(= `CommandStep[]`, 누적 dx/dtheta/dt).
4. **명령 다운링크**: P1이 결과를 `output{delivery:"mqtt"}`로 Hub에 WS 전송 → Hub가 `mqtt-writer`를 통해 MQTT 토픽 `robot/{robot_id}/ai-command`로 발행 → **로봇이 구독해 실행**.
5. **시각화(병렬)**: 같은 결과를 `delivery:"ui_overlay"`로도 보내 Hub→API→UI에서 궤적/히트맵 오버레이로 표시.

---

## 4. 두 개의 데이터 평면 (왜 나뉘나)

| 평면 | 무엇 | 경로 | 이유 |
|---|---|---|---|
| **영상 평면** | room1/room2 카메라 | 로봇 →(WebRTC)→ LiveKit →(구독)→ 플러그인 | 고대역·실시간 영상은 SFU(WebRTC)가 적합. 프레임은 플러그인 내부 **SHM ring**으로 zero-copy 전달 |
| **명령/센서 평면** | lidar·encoder·marker / ai-command | 로봇 ↔(MQTT)↔ broker ↔(NATS)↔ Hub ↔(WS)↔ 플러그인 | 작은 구조화 메시지는 MQTT/NATS pub-sub이 적합. 신뢰성·라우팅 용이 |

핵심: **영상은 WebRTC, 제어 명령·센서는 MQTT** 라는 두 평면이 Hub에서 만난다.

---

## 5. AI 플러그인 내부 (왜 프로세스 2개인가)

플러그인 1개 = 컨테이너 1개 = 내부 **mp.Process 2개**:
- **P1 (Bridge)**: LiveKit 구독, Hub WS, 센서 윈도우(`RobotDataWindows`), 스냅샷 생성, 결과 송신. 외부 I/O 전담.
- **P2 (AI)**: GPU(torch/TensorRT)로 모델 로드·추론만. **우리 코드(`inference_fn`)가 여기서 돈다.**
- 연결: 영상은 **SHM ring buffer**(zero-copy), 제어/결과는 **mp.Queue**(ControlQueue/ResultQueue).

분리 이유: 모델 crash 격리, CUDA 컨텍스트 분리, GIL 회피, 영상 zero-copy.

> 타이밍: P1의 **TickThread**(≈30Hz)가 스냅샷을 만들고, **SendThread**(0.1s 고정)가 최신 궤적을 MQTT/overlay로 내보낸다. 추론이 느리면 이전 궤적을 stride만큼 진행하며 끊김 없이 송신.

---

## 6. 우리 작업이 들어가는 자리

- **현재**: ai-control 플러그인의 `inference_fn`이 우리 **diffusion 정책(APPROACH)** 을 실행 → 속도 궤적을 CommandStep으로 반환. 이게 위 파이프라인의 ②~③.
- **goal 조건화(Loss A)**: 학습엔 배선됨. 플러그인 추론 경로(`run_postech_*` / `inference.py`)의 context에도 goal_feat 추가 필요 (현재 미배선 — [inference_io.md](inference_io.md) §A.4와 동일 공백).
- **종단 ICP(엔드게임)**: 아직 플러그인에 미연결. 연결 방법 — 등록 `required_data.sensors`에 lidar 포함(이미 가능) → P1 윈도우에 lidar 들어옴 → `inference_fn`을 `TwoRegimeController.step()`으로 감싸 **APPROACH면 정책 궤적, ENGAGED면 ICP 서보 궤적**을 같은 `CommandStep[]`으로 반환. **로봇 브리지·MQTT 경로는 그대로**, 출력 인터페이스가 동일하므로 무수정.

---

## 7. 안전/운영 메모 (큰 그림 수준)

- **검증 단계 스위치**: 플러그인은 `enable_send=false`(= `managed_resources:[]`, `no_publish_ai_command`)로 띄우면 **명령을 계산만 하고 로봇에 발행하지 않는다**(dry-run). 켤 때 reconnect+register 필요.
- **라이프사이클**: `on_demand` — UI에서 Start 누르면 RUNNING(루프 시작), Stop이면 중단. 설정은 Redis 영속.
- **실패 격리**: P2(모델) crash → 컨테이너 재시작(부분 재기동 안 함). WS/LiveKit 끊김 → 지수 백오프 재연결.
- **marker_pose**: 등록 `required_data`엔 있으나 **우리 학습/추론/엔드게임은 비의존**(검증용 옵션). markerless 시나리오에서도 동일 파이프라인 동작.

---

## 8. 더 깊은 내부 문서 (서버측 정본)

- `plugins/DESIGN/architecture.md` — 플러그인 토폴로지·SHM·Queue·타이밍·등록 (정본, 매우 상세)
- `plugins/DESIGN/ai-control.md`, `common-core.md` — ai-control 세부, 공통 코어
- `rc_server/docs/system_architecture.md`, `rc_hub_architecture.md` — 서버 전체/Hub
- `rc_server/docs/communication_robot_server_client.md`, `communication_internal_nats.md` — 통신 규약
- `rc_server/mqtt-io/README.md`, `rc_server/hub/README.md` — MQTT 브리지, Hub
