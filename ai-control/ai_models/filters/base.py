"""
필터 추상화: 로봇 데이터/영상을 윈도우에 넣기 전 전처리.
여러 필터를 등록해 순서대로 적용.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional

# 스트림 타입: video1, video2, lidar, marker_pose, encoder
STREAM_VIDEO1 = "video1"
STREAM_VIDEO2 = "video2"
STREAM_LIDAR = "lidar"
STREAM_MARKER_POSE = "marker_pose"
STREAM_ENCODER = "encoder"


class FilterBase(ABC):
    """단일 필터. data_type과 timestamp를 받아 변환된 데이터 또는 None(스킵) 반환."""

    @abstractmethod
    def filter(self, data: Any, data_type: str, timestamp: float) -> Optional[Any]:
        """
        data_type: video1, video2, lidar, marker_pose, encoder 중 하나.
        반환: 전처리된 데이터 (그대로 넣을 경우 data). None이면 이 샘플 스킵.
        """
        pass

    def name(self) -> str:
        return self.__class__.__name__


def apply_filters(data: Any, data_type: str, timestamp: float, filters: List[FilterBase]) -> Optional[Any]:
    """여러 필터를 순서대로 적용. 중간에 None이 나오면 스킵."""
    current = data
    for f in filters:
        if current is None:
            return None
        current = f.filter(current, data_type, timestamp)
    return current
