from abc import ABC
import typing as t
from inspect import signature, Parameter, getmembers, ismethod

from fastapi import FastAPI
from pydantic import create_model, BaseModel


class MissingAnnotationError(Exception):
    pass


def endpoint(method: t.Callable) -> t.Callable:
    """
    Decorator for identifying methods as endpoints
    """
    method._is_endpoint = True
    return method


class WebService(ABC):
    """
    When inherited from and implemented, the subclass will inherit
    behavior for turning automatically into a web service.

    All sublcass methods that are decorated with `@endpoint` will be exposed as
    endpoints in a web service. If the method takes no arguments, its endpoint will accept
    GET requests. If the method takes arguments, its endpoint will accept POST requests with
    a request body JSON object having a key/value pair for every parameter in the method's
    signature. A couple requirements for methods decorated with `@endpoint`:

        1. All arguments in the method's signature must have type hints.
        2. The return values of the method must be JSON serializeable.

    As an example:

    ```python
    class MyWebService(WebService):

        @endpoint
        def hello(self):
            return {"message": "Hello world!"}

        @endpoint
        def add(self, num: int):
            return num + 1

        def other(self):
            return "This method will not be a HTTP endpoint"

    service = MyWebService()
    app = service.to_app()
    ```

    `app` is a FastApi web service with two endpoints: `/hello` supporting GET
    requests, and `/add` supporting POST requests with a request body that must
    look like: `{ "num": <int> }`.
    """

    def to_app(self) -> FastAPI:
        """
        Converts the implementing class into an HTTP web service.
        """
        app = FastAPI()
        for endpoint_name, endpoint in self._get_endpoints().items():
            allowed_verb = "GET"
            data_model_class = self._data_model_for_method_sig(endpoint)
            if data_model_class is not None:
                # This endpoint will accept a request body via a POST request.
                allowed_verb = "POST"
                endpoint = self._wrap_with_data_model(endpoint, data_model_class)
            app.add_api_route("/" + endpoint_name, endpoint, methods=[allowed_verb])
        return app

    def _data_model_for_method_sig(self, method: t.Callable) -> t.Optional[BaseModel]:
        """
        Creates a pydantic data model for `method`'s type signature. Returns
        `None` if `method` has no arguments (b/c then no data model is needed).
        """
        sig = signature(method)
        type_spec = {
            arg_name: param.annotation
            for arg_name, param in sig.parameters.items()
            # The `self` is not included in the data model because it is passed
            # implicitly by the method's parent object.
            if arg_name != "self"
        }
        if len(type_spec) == 0:
            return None
        for arg_name, annotation in type_spec.items():
            if annotation == Parameter.empty:
                raise MissingAnnotationError(
                    f"could not construct data model for method '{method.__name__}'; "
                    f"no typing info specified for parameter '{arg_name}'."
                )

        return create_model(
            f"ModelFor{method.__name__}",
            **{name: (typ, ...) for name, typ in type_spec.items()},
        )

    def _get_endpoints(self) -> t.Dict[str, t.Callable]:
        """
        Identifies all the methods the implementing class has that have been
        decorated with the `endpoint` decorator.
        """
        return {
            name: member
            for name, member in getmembers(self)
            if ismethod(member)
            and hasattr(member, "_is_endpoint")
            and member._is_endpoint is True
        }

    def _wrap_with_data_model(
        self, method: t.Callable, data_model_class: t.Type[BaseModel]
    ) -> t.Callable:
        """
        Returns a version of `method` that accepts `model` as its argument,
        passing the body of `model` on to `method` the way `method`'s type
        signature expects.
        """

        def wrapper(model: data_model_class):
            return method(**model.dict())

        wrapper.__name__ = f"{method.__name__}_wrapped"
        return wrapper
