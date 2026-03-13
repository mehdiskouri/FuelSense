from django.http import HttpRequest

class Request(HttpRequest):
    data: object
    query_params: dict[str, str]
