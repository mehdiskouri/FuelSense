from collections.abc import Callable
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

def action(*, detail: bool, methods: list[str] | tuple[str, ...], url_path: str | None = ...) -> Callable[[_F], _F]: ...
