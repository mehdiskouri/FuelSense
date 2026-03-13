from collections.abc import Callable, Iterable
from typing import TypeVar

_F = TypeVar("_F", bound=Callable[..., object])

class _CeleryConf:
    task_queues: tuple[object, ...]

class Celery:
    conf: _CeleryConf
    def __init__(self, main: str, *args: object, **kwargs: object) -> None: ...
    def config_from_object(self, obj: str, namespace: str | None = ...) -> None: ...
    def autodiscover_tasks(self, packages: Iterable[str]) -> None: ...

class _CurrentApp:
    def send_task(
        self,
        name: str,
        args: list[object] | None = ...,
        kwargs: dict[str, object] | None = ...,
        **options: object,
    ) -> object: ...

def shared_task(*args: object, **kwargs: object) -> Callable[[_F], _F]: ...
def chord(tasks: Iterable[object], body: object) -> object: ...

current_app: _CurrentApp
