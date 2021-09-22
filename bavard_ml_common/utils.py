import importlib
import typing as t
from functools import wraps


_EXTRAS = {"ml": ["numpy", "sklearn"]}


def requires_extras(**has_extras: bool):
    def decorator(callable_: t.Callable):
        @wraps(callable_)
        def check_extras(*args, **kwargs):
            for extra_name, has_extra in has_extras.items():
                if not has_extra:
                    raise ImportError(f"The {extra_name} extra is required to use {callable_.__name__}")
            return callable_(*args, **kwargs)

        return check_extras

    return decorator


def uses_extra(extra: str):
    def decorator(callable_: t.Callable):
        @wraps(callable_)
        def load_extra(*args, **kwargs):
            if extra not in _EXTRAS:
                raise ValueError(f"{extra} is not a know extra. Valid options: {_EXTRAS.keys()}")
            for dep in _EXTRAS[extra]:
                importlib.import_module(dep)
            return callable_(*args, **kwargs)

        return load_extra

    return decorator
