"""Compute backend registry placeholders for FuelSense."""

from __future__ import annotations

from typing import Dict, Type

from fuelsense_common.compute import ComputeBackend, DeviceType

_registry: Dict[str, Dict[DeviceType, Type[ComputeBackend]]] = {}
