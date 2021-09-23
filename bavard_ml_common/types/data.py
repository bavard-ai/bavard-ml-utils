import inspect
import typing as t
from io import BytesIO

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, root_validator

from bavard_ml_common.utils import ImportExtraError


try:
    import numpy as np
except ImportError:
    raise ImportExtraError("ml", "data module")


def encode_numpy(data: np.ndarray, mode="w"):
    """
    Serializes a numpy array to `bytes` or `str`, depending on `mode`. Includes shape, datatype, and endianness
    information for perfect cross-platform reconstruction.
    """
    mf = BytesIO()
    np.save(mf, data)
    value = mf.getvalue()
    mf.close()
    if mode == "w":
        return value.decode("latin-1")
    return value


def decode_numpy(data: t.Union[str, bytes]):
    """Deserializes a numpy array from `bytes` or `str`, which was serialized using the `encode_numpy` method."""
    if isinstance(data, str):
        data = data.encode("latin-1")
    mf = BytesIO(data)
    arr = np.load(mf)
    mf.close()
    return arr


class DataModel(BaseModel):
    """
    A base class for defining pydantic models which also supports numpy fields, including serialization and
    deserialization of those fields. E.g.

    ```python
    class MyModel(DataModel):
        a: str
        b: numpy.ndarray

    model = MyModel(a="hello world", b=numpy.array([1,2,3]))
    # Serialize to a string and then back again.
    reconstructed == MyModel.parse_raw(model.json())
    # `reconstructed`'s contents are identical to `model`
    ```
    """

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            # TODO: the top level call to `NumpyModel.json()` bloats the string returned here by `encode_numpy`,
            #   because it inserts escape characters and converts '\x' to '\u00' sometimes.
            np.ndarray: lambda arr: encode_numpy(arr, mode="w")
        }

    def encode(self):
        """Returns an encoded version of this model composed of primitive Python types."""
        return jsonable_encoder(self)

    @classmethod
    def _get_fields_of_type(cls, type_: t.Type) -> t.Set[str]:
        """Get the name of the fields in this pydantic model that have an annotation of `type_`."""
        sig = inspect.signature(cls)
        return {param.name for param in sig.parameters.values() if param.annotation == type_}

    @root_validator(pre=True)
    def _validate_numpy_arrays(cls, values):
        """
        Allows numpy data to be passed in to this model's constructor in a few different forms, namely:
            - A normal `np.ndarray` object.
            - A `str` object, assumed to be produced by `encode_numpy`. It will be decoded.
            - A 'bytes` object, assumed to be produced by `encode_numpy`. It will be decoded.
        """
        numpy_fields = cls._get_fields_of_type(np.ndarray)
        for field in numpy_fields:
            value = values[field]
            if isinstance(value, np.ndarray):
                continue
            elif isinstance(value, (str, bytes)):
                values[field] = decode_numpy(value)
            else:
                raise TypeError(f"cannot process unknown type {type(value)} for {field} field")
        return values
