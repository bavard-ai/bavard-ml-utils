import typing as t
import os
from abc import ABC, abstractmethod
import pickle
import shutil
from uuid import uuid4

from sklearn.base import BaseEstimator
import joblib
import tensorflow as tf
from tensorflow.io.gfile import GFile


def make_dir_if_needed(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


class TypeSerializer(ABC):
    """
    When implemented, provides functionality for serializing instance
    of some type or group of types, for use with the `Serializer` class.
    """

    @property
    @abstractmethod
    def type_name(self) -> str:
        pass

    @property
    @abstractmethod
    def ext(self) -> str:
        """
        The filename extension, if the serializer serializes its data
        to a single file. Should be `None` otherwise.
        """
        pass

    @abstractmethod
    def serialize(self, obj: object, path: str) -> None:
        pass

    @abstractmethod
    def deserialize(self, path: str) -> object:
        pass

    @abstractmethod
    def is_serializable(self, obj: object) -> bool:
        pass

    def resolve_path(self, path: str) -> str:
        """
        Adds this serializer's extension to `path` if it has one.
        """
        return f"{path}.{self.ext}" if self.ext else path


class SklearnSerializer(TypeSerializer):
    type_name = "sklearn"
    ext = "joblib"

    def serialize(self, obj: object, path: str) -> None:
        joblib.dump(obj, path)

    def deserialize(self, path: str) -> object:
        return joblib.load(path)

    def is_serializable(self, obj: object) -> bool:
        return isinstance(obj, BaseEstimator)


class KerasSerializer(TypeSerializer):
    type_name = "keras"
    ext = None

    def serialize(self, obj: tf.keras.Model, path: str) -> None:
        obj.save(path, save_format="tf")

    def deserialize(self, path: str) -> object:
        return tf.keras.models.load_model(path)

    def is_serializable(self, obj: object) -> bool:
        return isinstance(obj, tf.keras.Model)


class _CustomPickler(pickle.Pickler):
    def __init__(
        self, pkl_file, assets_path: str, type_serializers: t.List[TypeSerializer]
    ) -> None:
        super().__init__(pkl_file)
        self._assets_path = assets_path
        self._ser_map = {ser.type_name: ser for ser in type_serializers}

    def persistent_id(self, obj: object) -> t.Optional[tuple]:
        for ser in self._ser_map.values():
            if ser.is_serializable(obj):
                # Treat `obj` as an external object and serialize it using our own
                # methods. Our serializer's type name and the path it was serialized
                # to is returned and pickled, so the pickler will know how to find
                # the object again and deserialize it.
                ser_path = ser.resolve_path(
                    os.path.join(self._assets_path, str(uuid4()))
                )
                ser.serialize(obj, ser_path)
                return (ser.type_name, ser_path)

        # No custom serializer for `obj`; pickle it using the normal way.
        return None


class _CustomUnpickler(pickle.Unpickler):
    def __init__(self, pkl_file, type_serializers: t.List[TypeSerializer]) -> None:
        super().__init__(pkl_file)
        self._ser_map = {ser.type_name: ser for ser in type_serializers}

    def persistent_load(self, pid: tuple) -> object:
        """
        This method is invoked whenever a persistent ID is encountered.
        Here, pid is the tuple returned by `ModelPickler`.
        """
        ser_type_name, ser_path = pid
        if ser_type_name not in self._ser_map:
            raise pickle.UnpicklingError(
                "cannot deserialize: an object was found which was serialized using the "
                f"{ser_type_name} serializer, and this unpickler does not have that serializer registered."
            )
        ser = self._ser_map[ser_type_name]
        return ser.deserialize(ser_path)


class Serializer:
    """
    A replacement for the `pickle.dump` and `pickle.load` functions that allows
    custom serialization behavior for different data types (e.g. `keras` models,
    `numpy` arrays, etc.). Includes support for serializing `sklearn` estimators
    and `keras` models using their native protocols. You can use your own type serializers
    and pass those in too. Just implement the `TypeSerializer` class and pass an instance
    of your class to the constructor.
    """

    def __init__(self, *custom_type_serializers: t.Tuple[TypeSerializer]) -> None:
        self._type_serializers = [
            SklearnSerializer(),
            KerasSerializer(),
        ] + list(custom_type_serializers)

        if len(self._type_serializers) != len(
            {ser.type_name for ser in self._type_serializers}
        ):
            raise ValueError(
                "The type_name of each type serializer must be unique."
                f" Currently registered names: {[ser.type_name for ser in self._type_serializers]}"
            )

    def serialize(self, obj: object, path: str, delete_existing: bool = True) -> None:
        """
        Serialize `obj` to `path`, a directory.
        """
        if os.path.exists(path) and delete_existing:
            shutil.rmtree(path)

        make_dir_if_needed(path)
        with GFile(self._get_pkl_path(path), "wb") as f:
            _CustomPickler(f, path, self._type_serializers).dump(obj)

    def deserialize(self, path: str, delete: bool = False) -> object:
        """
        Load the data that was serialized to the directory at
        `path`. It should have been serialized using this class's
        `serialize` method. If `delete==True`, `path` will be
        deleted once the deserialization is finished.
        """
        with GFile(self._get_pkl_path(path), "rb") as f:
            obj = _CustomUnpickler(f, self._type_serializers).load()

        if delete:
            shutil.rmtree(path)

        return obj

    def _get_pkl_path(self, path: str) -> str:
        return os.path.join(path, "data.pkl")
