from collections.abc import Sequence
from typing import TypeVar

from django.db.models import QuerySet
from django.db.models.base import Model
from django.http import HttpRequest
from rest_framework.response import Response

_MT = TypeVar("_MT", bound=Model)

class PageNumberPagination:
    page_size: int
    def paginate_queryset(
        self,
        queryset: QuerySet[_MT] | Sequence[_MT],
        request: HttpRequest,
        view: object | None = ...,
    ) -> list[_MT] | None: ...
    def get_paginated_response(self, data: object) -> Response: ...
