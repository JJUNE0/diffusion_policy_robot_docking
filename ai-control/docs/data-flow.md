# ai-control 데이터 흐름 — 센서/영상 → (모델) → 로봇 도킹 명령

> AI 모델 내부(DINO/TensorRT/추론 알고리즘)는 **블랙박스**로 취급한다. 이 문서는
> "데이터가 어떻게 들어와 가공되고, 모델 출력을 어떻게 로봇 명령으로 바꿔 보내는가"만 다룬다.

## 0. 한눈에

```
[로봇] ──MQTT/LiveKit──> [connection 노드] ──NATS 버스──> [ai-control 노드]
                                                              │
   ┌──────────────────────── ai-control 노드(컨테이너) ───────┴───────────────┐
   │  P1 (Bridge, asyncio)                         P2 (AI process, GPU)        │
   │  ─────────────────────                        ──────────────────         │
   │  센서 구독 → windows(버퍼/동기/보간)            ┌──> [모델] ──> horizon    │
   │  LiveKit 영상 → SHM ring                       │   (블랙박스)             │
   │  tick: 준비되면 snapshot ──ControlQueue──────────┘                        │
   │  send(0.1s): horizon ──> 궤적 보정 ──> AiCommand <──ResultQueue──────────┘ │
   └──────────────────────────────┬───────────────────────────────────────────┘
                                   │ node.send("ai_command")
                          NATS 버스 robot.{id}.ai_command
                                   │
                            [connection 노드]  ──json.dumps──> MQTT robot/{id}/ai-command
                                   │
                                [로봇] 궤적 추종(도킹)
```

핵심: **P1(I/O)과 P2(추론)가 분리된 2-프로세스**. 추론이 블로킹돼도 P1의 버스/영상 수신은 안 멈춤.

---

## 1. 입력 — 무엇을 받나

### 1.1 센서 (NATS 버스)
로봇이 MQTT로 보낸 걸 `connection` 노드가 버스 토픽으로 중계한다. ai-control은 이를 구독:

| 버스 토픽 | 타입 | 페이로드 | 용도 |
|---|---|---|---|
| `robot.{id}.lidar` | `LidarScan@1` | `{ts, points:[{x,y}]}` | (config `require_lidar:false` — 옵션) |
| `robot.{id}.encoder` | `Encoder@1` | `{ts, vx, wz}` | 필수. 측정 속도 |
| `robot.{id}.marker_pose` | `MarkerPose@1` | 마커 목록 | 옵션 |

구독·핸들러 선언: `node_sdk/node_hub.py` `NodeSpec.sensor_inputs` → `Node.declare_input`.
핸들러는 rc 봉투를 옛 포맷 subject(`robot.data.{stream}.{id}`)로 정규화해 `runner.on_data`로 넘긴다
(`node_hub.py:_make_sensor_handler`) → windows에 push(`node_sdk/runner_base.py:on_data`).

### 1.2 영상 (LiveKit, 버스 안 탐)
- `NodeHub.request_livekit_token` → `rc.media.token`(role=recorder) 호출 → 토큰 수신
  → `LIVEKIT_URL`(ws://host.docker.internal:47320)로 LiveKit room `xepler:robot:{id}` 직접 접속.
- 트랙 이름 `usb`→채널0, `orbbec`→채널1 (config `livekit_tracks_video1/2`).
- 프레임 수신 → BGR 변환 → **SHM ring buffer**에 기록, 윈도우엔 **슬롯 인덱스만** push
  (`node_sdk/livekit_video.py`, `windows.push_video`). 무거운 픽셀은 버스/큐를 안 탐.

> ⚠️ 트랙 이름이 로봇과 다르면 영상이 안 들어온다. 로그 `LiveKit 트랙 무시: name=...` 확인 → config 수정.

---

## 2. 가공 — windows (버퍼 · 동기 · 보간)

`node_sdk/windows.py` `RobotDataWindows`. 스트림별 슬라이딩 윈도우(`SlidingWindow`).

### 2.1 버퍼링
각 샘플은 `(ts, data, slow)`로 저장. `ts`는 **로봇 시각**(센서 payload의 `ts`, 영상 프레임 ts).
정렬은 **도착 시각이 아니라 이 robot-time ts** 기준 → 센서/영상 시차를 ts로 보정(버퍼링 vs 실시간 차이 무관).

### 2.2 동기 모드 (config `window_sync_mode: track`, 기준 `usb`)
기준 트랙(usb 영상)이 도착하는 **그 순간** 모든 스트림의 최신값을 함께 윈도우에 넣어 영상 케이던스에 공동등록
(`windows._push_all_from_cache`). 다른 스트림이 `window_sync_max_age_sec`(0.1s)보다 오래되면 `slow` 표기.

### 2.3 보간 (config `interpolation_enabled:true`, `interpolation_streams: encoder`)
스냅샷 준비 시 저장된 ts로 공통 시간격자에 리샘플(`node_sdk/snapshot_interpolate.py`).
`interpolation_buffer_padding:20` 만큼 여유 샘플 확보.

### 2.4 준비 판정 / slow 게이팅
- `windows.is_ready(require_video=true)`: lidar/encoder/marker는 require_* 따라, 영상은 video1·video2 둘 다 `window_size`(60)만큼 차야 함.
- 최근 tail에 `slow` 샘플이 있으면(`any_slow_in_tail`) 그 틱은 **새 추론 건너뜀**(이전 horizon 유지) — 어긋난 데이터로 추론 안 함.

---

## 3. 스냅샷 → 모델 → 결과 (P1↔P2)

`node_sdk/runner_base.py` + `ai_runtime.py`.

1. **P1 tick 루프** (`inference_fps:30` → 33ms): 윈도우 준비+신뢰 시
   `snapshot_with_timestamps()`(영상은 슬롯 인덱스 리스트) + latency → **ControlQueue**로 P2에 전달.
2. **P2**: SHM에서 프레임 attach(`is_stale`로 ring wrap 방어) → `prepare_inference_snapshot`(보간) →
   **[모델]** → `horizon`(= `List[CommandStep]`, inference_size=16) → **ResultQueue**로 P1에 반환.
   - CommandStep = `{dx, dy, dtheta, dt}` (모델이 뱉는 원자 단위). **모델 내부는 이 문서 범위 밖.**

> 영상은 ControlQueue로 픽셀이 아니라 **SHM 슬롯 인덱스**만 오감 → 큐 가벼움(제로카피).

---

## 4. 출력 — horizon → 도킹 궤적 → 로봇

P1 **send 루프** (`send_interval_sec:0.1` → 10Hz). 핵심은 horizon을 로봇이 따를 **속도 궤적**으로 바꾸는 보정
(`ai_control_node/runner.py:_build_send_tick_outputs` — 옛 코드 byte 보존).

### 4.1 궤적 보정 (요지)
- `step_dt = 1/inference_fps` (1/30s), `action_horizon:4`(메시지당 스텝 수), `stride = round(0.1*30)=3`.
- `seq_num`: 16비트 순차 증가 — 로봇이 메시지 순서/중복 판단.
- `anchor_time`: **호라이즌당 1개 고정** 기준시각 + `start_index*step_dt`. (메시지마다 재계산하면 0.1s씩 밀려서 고정.)
- `start_index`: 관측-전송 지연만큼 패딩(`observation_send_padding`/`initial_send_padding`) 후 stride만큼 전진.
- `velocity_steps = steps_to_velocity_steps(...)` → `[{vx, wz, dt}, ...]` (선속도/각속도/구간길이).
- `is_final`: 궤적 끝 or 전부 0속도면 1 (도킹 완료/정지 신호).

### 4.2 메시지 포맷 (`AiCommand@1`)
```json
{
  "version": 1,
  "num_steps": 4,
  "seq_num": 1234,
  "flags": 0,             // 0x01 = is_final
  "anchor_time": 67041.74,// 로봇 clock 기준 시각(초)
  "steps": [ {"vx":0.25,"wz":0.0,"dt":0.033}, ... ]
}
```
빌더: `ai_control_node/trajectory_payload.py:ai_command_velocity_steps_dict`.

### 4.3 전달 경로 (버스 → MQTT)
1. `runner.on_send_tick` → `hub.send_output([{delivery:"mqtt", payload: <AiCommand dict>}])`.
2. `NodeHub.send_output`(`node_hub.py`) → `node.send("ai_command", payload, keys={robot_id})`
   → 버스 토픽 **`robot.{id}.ai_command`** 발행. (+가시성 메트릭 `ai_cmd_vx`/`ai_cmd_wz`)
3. **connection 노드**(`rc_server/.../connection/__init__.py`, SINK_PORTS에 `ai_command` 추가됨)가 구독 →
   `json.dumps(payload)` 그대로 **MQTT `robot/{id}/ai-command`** 로 발행.
4. 로봇이 그 토픽을 받아 `anchor_time` 기준으로 `steps`(vx,wz,dt) 궤적을 추종 → **도킹**.

> `ENABLE_SEND=false`면 2~4를 건너뛰고(MQTT 미발행) 추론만 돈다(검증용). 기본 `true`=상시 도킹.

---

## 5. 제어권 (lease)

명령을 로봇이 받아들이게 하려면 서버 제어권을 잡는다.
- `NodeHub.start()` → `rc.control.acquire` 호출(`{robot_id, owner_key:"ai-control", session_id, source:"ai"}`)
  → `arbiter` 노드가 Redis lease 부여(`rc_server/.../arbiter`).
- 5초마다 `rc.control.keepalive`로 TTL 갱신. 종료 시 `rc.control.release`.
- (로봇이 도킹모드에서 `ai-command`를 수용하는지는 로봇 펌웨어 소관 — 실로봇 검증 필요.)

---

## 6. 타이밍 / 시계

- 정렬은 windows의 **robot-time ts**로 함(2.1) → 별도 NTP 불요. `NodeHub.robot_time_sec`는 `None` 반환 →
  `RunnerBase`가 server monotonic으로 fallback(명령 `anchor` 스케줄링용).
- `anchor_time`은 로봇 clock 기준이라, 로봇이 자기 시계로 궤적을 재생.

## 관련 파일 (전부 additional-nodes — plugins 의존 0)
- 어댑터(전송): `node_sdk/node_hub.py`, `node_sdk/node_runtime.py`
- 노드 정의: `ai-control/ai_control_node/__init__.py`
- 가공/궤적: `node_sdk/{runner_base,windows,snapshot_interpolate,livekit_video,ipc,ai_runtime}.py`,
  `ai-control/ai_control_node/{runner,trajectory_payload}.py`
- 모델: `ai-control/ai_models/`
- 코어 브리지: `rc_server/rc/rc/nodes/connection/__init__.py`(SINK `ai_command`), `rc_server/protocol/schemas/types/AiCommand@1.json`
