from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from django.db.models import QuerySet
from django.db.models.base import Model
from django.http import HttpRequest
from django.http.response import HttpResponseBase
from rest_framework.response import Response

_MT = TypeVar("_MT", bound=Model)

class ViewSet:
    action: str
    request: HttpRequest

    @classmethod
    def as_view(
        cls,
        actions: dict[str, str] | None = ...,
        **initkwargs: object,
    ) -> Callable[..., HttpResponseBase]: ...

class ModelViewSet(ViewSet, Generic[_MT]):  # noqa: UP046
    queryset: QuerySet[_MT] | None

    def get_object(self) -> _MT: ...
    def paginate_queryset(self, queryset: QuerySet[_MT] | Sequence[_MT]) -> Sequence[_MT] | None: ...
    def get_paginated_response(self, data: object) -> Response: ...
