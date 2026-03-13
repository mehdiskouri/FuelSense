from typing import Any

from django.urls.resolvers import URLPattern, URLResolver

class DefaultRouter:
    urls: list[URLPattern | URLResolver]
    def register(self, prefix: str, viewset: type[Any], basename: str | None = ...) -> None: ...
