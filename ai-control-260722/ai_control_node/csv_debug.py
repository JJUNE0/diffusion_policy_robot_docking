"""디버그 CSV 로거 — graft 모델 급가속 진단용.

두 파일을 남긴다(프로세스 시작 시 truncate = 새로 시작할 때마다 갱신):
  - model_output.csv : 모델(diffusion)의 raw 출력 velocity 그 자체 (플러그인/P2).
  - mqtt_sent.csv    : MQTT robot/{id}/ai-command 로 실제 전송하는 값 + 로봇시간 매핑 (runner/P1).

env AI_CONTROL_CSV_LOG=1(true/on) 일 때만 활성. 경로는 AI_CONTROL_CSV_DIR (기본 /app/ai-control/csv_out).
compose에서 이 dir을 호스트로 마운트하면 밖에서 바로 열람 가능.
"""
import csv
import os
import threading

ENABLED = os.environ.get("AI_CONTROL_CSV_LOG", "").lower() in ("1", "true", "yes", "on")
_DIR = os.environ.get("AI_CONTROL_CSV_DIR", "/app/ai-control/csv_out")


class CsvLogger:
    def __init__(self, filename: str, header: list[str]) -> None:
        self._on = ENABLED
        self._lock = threading.Lock()
        self._f = None
        self._w = None
        if not self._on:
            return
        os.makedirs(_DIR, exist_ok=True)
        self._f = open(os.path.join(_DIR, filename), "w", newline="")  # 시작 시 truncate
        self._w = csv.writer(self._f)
        self._w.writerow(header)
        self._f.flush()

    def write(self, row: list) -> None:
        if not self._on:
            return
        with self._lock:
            self._w.writerow(row)
            self._f.flush()
