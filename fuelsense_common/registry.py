"""Compute backend registry placeholders for FuelSense."""

from __future__ import annotations

from typing import TypeVar

from fuelsense_common.compute import ComputeBackend, DeviceType

T = TypeVar("T", bound=ComputeBackend)

_registry: dict[str, dict[DeviceType, type[ComputeBackend]]] = {}


def register_backend(name: str, device: DeviceType):
    """Register a backend class for a service and device pair."""

    def decorator(cls: type[T]) -> type[T]:
        _registry.setdefault(name, {})[device] = cls
        return cls

    return decorator


def get_backend(name: str, device: DeviceType) -> ComputeBackend:
    """Instantiate a backend by service name with CPU fallback support."""
    backends = _registry.get(name, {})
    if device in backends:
        return backends[device]()
    if DeviceType.CPU in backends:
        return backends[DeviceType.CPU]()
    raise RuntimeError(f"No backend registered for '{name}'")


def clear_registry() -> None:
    """Reset all registered backends. Useful for tests."""
    _registry.clear()
