from django.http.response import HttpResponseBase

class Response(HttpResponseBase):
    data: object
    def __init__(self, data: object = ..., status: int | None = ..., *args: object, **kwargs: object) -> None: ...
