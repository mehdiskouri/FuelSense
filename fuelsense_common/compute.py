"""Shared compute abstractions for FuelSense services.

Phase 0 provides minimal protocol definitions.
"""

from __future__ import annotations

import importlib
import logging
import os
from enum import Enum
from typing import Protocol, runtime_checkable


logger = logging.getLogger(__name__)


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"


@runtime_checkable
class ComputeBackend(Protocol):
    device: DeviceType

    def warmup(self) -> None: ...

    def health_check(self) -> dict[str, object]: ...


def resolve_device() -> DeviceType:
    """Resolve the active compute device based on env override and availability."""
    env_device = os.environ.get("FUELSENSE_DEVICE", "").lower().strip()

    if env_device == "cuda":
        try:
            torch_mod = importlib.import_module("torch")
            if bool(torch_mod.cuda.is_available()):
                return DeviceType.CUDA
        except ImportError:
            pass
        logger.warning("FUELSENSE_DEVICE=cuda but CUDA not available. Falling back to CPU.")
        return DeviceType.CPU

    if env_device == "cpu":
        return DeviceType.CPU

    try:
        torch_mod = importlib.import_module("torch")
        if bool(torch_mod.cuda.is_available()):
            logger.info("CUDA detected: %s", str(torch_mod.cuda.get_device_name(0)))
            return DeviceType.CUDA
    except ImportError:
        pass

    return DeviceType.CPU
