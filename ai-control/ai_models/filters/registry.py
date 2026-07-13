"""
필터 레지스트리: 등록된 필터 목록을 반환.
"""
from typing import List

from .base import FilterBase

_registry: List[FilterBase] = []


def register_filter(f: FilterBase) -> None:
    _registry.append(f)


def get_registry() -> List[FilterBase]:
    return list(_registry)


def clear_registry() -> None:
    _registry.clear()
