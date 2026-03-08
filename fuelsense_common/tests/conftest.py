from __future__ import annotations

import pytest

from fuelsense_common.registry import clear_registry


@pytest.fixture(autouse=True)
def reset_backend_registry() -> None:
    clear_registry()
    yield
    clear_registry()
