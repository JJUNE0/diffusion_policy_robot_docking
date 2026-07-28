"""
플러그인 인터페이스: run 모드별 진입 로직.
각 플러그인은 config 오버레이(선택)와 run(config) 진입점을 제공.
"""
from typing import Any, Dict, Optional, Protocol

from ai_control_node.config import AIControlConfig


class RunPlugin(Protocol):
    """run 모드 플러그인 규약. 플러그인 모듈은 run 함수와 선택적으로 config_overrides를 제공."""

    def run(self, config: AIControlConfig) -> None:
        """진입점. 러너 시작 후 tick 루프 등. KeyboardInterrupt 시 정리하고 종료."""
        ...


def get_config_overrides(plugin_module: Any) -> Optional[Dict[str, Any]]:
    """플러그인 모듈에 config_overrides가 있으면 반환."""
    return getattr(plugin_module, "config_overrides", None)
