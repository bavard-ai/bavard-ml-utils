import typing as t
from inspect import getmembers, ismethod

from fastapi import FastAPI


def endpoint(_func=None, **fastapi_args) -> t.Callable:
    """Decorator for identifying methods as FastAPI endpoints. All decorator
    arguments are forwarded to `fastapi.FastAPI.add_api_route`, which will
    register `method` as an API route when `WebService.to_app` is called.
    """
    def inner(method: t.Callable) -> t.Callable:
        method.is_endpoint = True
        method.fastapi_args = fastapi_args
        return method
    if _func is None:
        return inner
    else:
        # Decorator was called without arguments. Apply the decorator immediately.
        return inner(_func)


class WebService:
    """
    When inherited from and implemented, the subclass will inherit
    behavior for turning automatically into a web service.

    All sublcass methods that are decorated with `@endpoint` will be exposed as
    endpoints in a web service, and should be declared the way a FastAPI app method
    would be declared. This includes the requirement that the return value of each method be JSON
    serializeable. All parameters passed to the `@endpoint` decorator will be
    forwarded on to `fastapi.FastAPI.add_api_route`.

    As an example:

    ```python
    from pydantic import BaseModel

    class AddInput(BaseModel):
        num: int

    class MyWebService(WebService):

        @endpoint
        def hello(self):
            return {"message": "Hello world!"}

        @endpoint(methods=["POST"], path="/do-add")
        def add(self, inputs: AddInput):
            return inputs.num + 1

        def other(self):
            return "This method will not be a HTTP endpoint"

    service = MyWebService()
    app = service.to_app()
    ```

    `app` is a FastApi web service with two endpoints: `/hello` supporting GET
    requests, and `/do-add` supporting POST requests with a request body that must
    look like: `{ "num": <int> }`.
    """

    def _base_route(self) -> dict:
        """
        Landing page for the API.
        """
        cls_name = self.__class__.__name__
        return {
            "description": f"This web service was generated for the {cls_name} class",
            "class": cls_name,
            "documentation_routes": {
                "swagger": "/docs",
                "redoc": "/redoc",
                "open_api_schema": "/openapi.json",
            },
        }

    def to_app(self) -> FastAPI:
        """Converts the implementing class into an HTTP web service.
        """
        app = FastAPI()
        for name, method in self._get_endpoints().items():
            app.add_api_route(
                **{
                    # These args are overridable by `method.fastapi_args`.
                    "path": "/" + name,
                    "endpoint": method,
                    "methods": ["GET"],
                    **method.fastapi_args
                }
            )

        app.add_api_route("/", self._base_route)
        return app

    def _get_endpoints(self) -> t.Dict[str, t.Callable]:
        """
        Identifies all the methods the implementing class has that have been
        decorated with the `endpoint` decorator.
        """
        return {
            name: member
            for name, member in getmembers(self)
            if ismethod(member)
            and hasattr(member, "is_endpoint")
            and member.is_endpoint is True
        }
