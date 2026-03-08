"""Shared compute abstractions for FuelSense services.

Phase 0 provides minimal protocol definitions.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"


@runtime_checkable
class ComputeBackend(Protocol):
    device: DeviceType

    def warmup(self) -> None:
        ...

    def health_check(self) -> dict:
        ...
