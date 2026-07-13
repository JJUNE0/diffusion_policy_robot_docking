"""
CSV 리플레이: encoder / marker_pose 등을 CSV에서 읽어 동일 구조로 공급.
테스트 시 로봇 없이 윈도우를 채우기 위한 데이터 소스 추상화.
실제 푸시는 run_csv_replay()를 호출하는 쪽에서 주기적으로 수행.
"""
import csv
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


def _read_csv_rows(path: str, ts_key: str = "ts") -> Iterator[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get(ts_key)
            if t is not None:
                try:
                    row["_ts"] = float(t)
                except ValueError:
                    row["_ts"] = None
            yield row


def encoder_from_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """encoder.csv 한 행 -> encoder 메시지 형태 (vx, wz 등)."""
    out = {}
    for k, v in row.items():
        if k in ("ts", "_ts"):
            continue
        try:
            out[k] = float(v)
        except (ValueError, TypeError):
            out[k] = v
    return out


def _row_to_vx_wz(row: Dict[str, Any]) -> Dict[str, float]:
    """CSV 행에서 선속도/각속도 추출. vx 또는 v, wz 또는 omega 열 지원."""
    v = row.get("vx") if "vx" in row else row.get("v")
    w = row.get("wz") if "wz" in row else row.get("omega")
    try:
        return {"vx": float(v if v is not None else 0), "wz": float(w if w is not None else 0)}
    except (ValueError, TypeError):
        return {"vx": 0.0, "wz": 0.0}


def read_vx_wz_csv(path: str, fps: float) -> Iterator[tuple]:
    """
    vx/wz 또는 v/omega 열이 있는 CSV에서 행을 읽어 (data, ts) 스트림으로 반환.
    data는 항상 {"vx": float, "wz": float}. ts는 행 인덱스와 fps로 계산: ts = index / fps.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            data = _row_to_vx_wz(row)
            ts = i / fps
            yield (data, ts)


def load_vx_wz_rows(csv_path: str) -> list:
    """vx, wz 열만 있는 CSV를 읽어 [{"vx": float, "wz": float}, ...] 리스트로 반환."""
    return [data for data, _ in read_vx_wz_csv(csv_path, 1.0)]


def run_csv_replay(
    csv_path: str,
    fps: float,
    send_interval_sec: float,
) -> list:
    """
    vx,wz CSV를 읽어 send_interval_sec 간격에 해당하는 (data, ts) 리스트를 반환.
    ts는 fps로 계산: ts = row_index / fps.
    send_interval_sec마다 한 번 보내는 시뮬레이션: frame_step = round(fps * send_interval_sec)
    (예: fps=30, send_interval_sec=0.1 -> 3프레임마다 1개, 행 0,3,6,9,...).
    반환된 리스트를 호출 측에서 push_encoder(data, ts)로 푸시하면 됨.
    """
    frame_step = max(1, round(fps * send_interval_sec))
    out: list = []
    for i, (data, ts) in enumerate(read_vx_wz_csv(csv_path, fps)):
        if i % frame_step != 0:
            continue
        out.append((data, ts))
    return out


def marker_pose_from_csv_rows(rows: list) -> Dict[str, Any]:
    """marker_pose.csv 여러 행(같은 ts) -> marker_pose 메시지 형태 (markers 리스트)."""
    if not rows:
        return {"markers": []}
    markers = []
    for row in rows:
        m = {
            "marker_id": row.get("marker_id"),
            "pos": [
                _f(row.get("pos_x")),
                _f(row.get("pos_y")),
                _f(row.get("pos_z")),
            ],
            "euler": [
                _f(row.get("euler_x")),
                _f(row.get("euler_y")),
                _f(row.get("euler_z")),
            ],
            "is_3d": row.get("is_3d"),
            "usage_type": row.get("usage_type"),
        }
        markers.append(m)
    return {"markers": markers}


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


class CSVReplaySource:
    """
    에피소드 디렉토리(encoder.csv, marker_pose.csv 등)에서 행을 읽어
    push_encoder, push_marker_pose 콜백을 호출하는 이터레이터.
    """

    def __init__(
        self,
        episode_dir: str,
        push_encoder: Callable[[Dict, float], None],
        push_marker_pose: Callable[[Dict, float], None],
        push_lidar: Optional[Callable[[Any, float], None]] = None,
    ):
        self.episode_dir = Path(episode_dir)
        self.push_encoder = push_encoder
        self.push_marker_pose = push_marker_pose
        self.push_lidar = push_lidar
        self._encoder_path = self.episode_dir / "encoder.csv"
        self._marker_path = self.episode_dir / "marker_pose.csv"
        self._lidar_path = self.episode_dir / "lidar.jsonl"

    def has_encoder(self) -> bool:
        return self._encoder_path.is_file()

    def has_marker_pose(self) -> bool:
        return self._marker_path.is_file()

    def run_once_encoder(self) -> bool:
        """encoder.csv에서 한 행씩 읽어 push_encoder 호출. 더 있으면 True."""
        if not self.has_encoder():
            return False
        try:
            with open(self._encoder_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get("ts")
                    if ts is None:
                        continue
                    try:
                        t = float(ts)
                    except ValueError:
                        continue
                    data = encoder_from_csv_row(row)
                    self.push_encoder(data, t)
                    return True
        except Exception as e:
            logger.warning("CSV replay encoder 읽기 오류: %s", e)
        return False

    def run_encoder_iterator(self) -> Iterator[tuple]:
        """encoder.csv 전체 행을 (data, ts) 이터레이터로."""
        if not self.has_encoder():
            return
        for row in _read_csv_rows(str(self._encoder_path)):
            ts = row.get("_ts")
            if ts is None:
                continue
            data = encoder_from_csv_row(row)
            yield (data, ts)

    def run_marker_pose_iterator(self) -> Iterator[tuple]:
        """marker_pose.csv를 ts별로 묶어 (data, ts) 이터레이터로."""
        if not self.has_marker_pose():
            return
        buf: Dict[float, list] = {}
        for row in _read_csv_rows(str(self._marker_path)):
            ts = row.get("_ts")
            if ts is None:
                continue
            buf.setdefault(ts, []).append(row)
        for ts, rows in sorted(buf.items()):
            data = marker_pose_from_csv_rows(rows)
            yield (data, ts)
