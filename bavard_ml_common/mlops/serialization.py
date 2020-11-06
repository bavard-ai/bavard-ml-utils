import typing as t
import os
from abc import ABC, abstractmethod
import pickle
import shutil
from uuid import uuid4
import tarfile

from sklearn.base import BaseEstimator
import joblib
import tensorflow as tf


class TempDir:
    """
    Context Manager class for working with a temporary directory. Example:

    ```python
    with TempDir() as tmp_dir:
        # `tmp_dir` is a newly created directory.
        # ... do work ...

    # `tmp_dir` is automatically deleted when the `with` statement is exited.
    ```
    """

    def __init__(self, name: str = None) -> None:
        if name:
            self.name = name
        else:
            self.name = str(uuid4())

    def __enter__(self) -> str:
        os.mkdir(self.name)
        return self.name

    def __exit__(self, type, value, traceback) -> None:
        shutil.rmtree(self.name)


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
                obj_id = str(uuid4())
                obj_path = ser.resolve_path(obj_id)
                ser.serialize(obj, os.path.join(self._assets_path, obj_path))
                return (ser.type_name, obj_path)

        # No custom serializer for `obj`; pickle it using the normal way.
        return None


class _CustomUnpickler(pickle.Unpickler):
    def __init__(
        self, pkl_file, assets_path: str, type_serializers: t.List[TypeSerializer]
    ) -> None:
        super().__init__(pkl_file)
        self._assets_path = assets_path
        self._ser_map = {ser.type_name: ser for ser in type_serializers}

    def persistent_load(self, pid: tuple) -> object:
        """
        This method is invoked whenever a persistent ID is encountered.
        Here, pid is the tuple returned by `ModelPickler`.
        """
        ser_type_name, obj_path = pid
        if ser_type_name not in self._ser_map:
            raise pickle.UnpicklingError(
                "cannot deserialize: an object was found which was serialized using the "
                f"{ser_type_name} serializer, and this unpickler does not have that serializer registered."
            )
        ser = self._ser_map[ser_type_name]
        return ser.deserialize(os.path.join(self._assets_path, obj_path))


class Serializer:
    """
    A replacement for the `pickle.dump` and `pickle.load` functions that allows
    custom serialization behavior for different data types (e.g. `keras` models,
    `numpy` arrays, etc.). Includes support for serializing `sklearn` estimators
    and `keras` models using their native protocols. You can use your own type serializers
    and pass those in too. Just implement the `TypeSerializer` class and pass an instance
    of your class to the constructor.
    """

    serialize_dir_name = "serialized-data"

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

    def serialize(self, obj: object, path: str) -> None:
        """
        Serialize `obj` to `path`, which should be a file path ending in ".tar".
        """
        with TempDir() as temp_dir:
            # Serialize to a temporary directory.
            with open(self._get_pkl_path(temp_dir), "wb") as f:
                _CustomPickler(f, temp_dir, self._type_serializers).dump(obj)

            # Tar the directory to the final user-expected path.
            with tarfile.open(path, "w") as tar:
                tar.add(temp_dir, arcname=self.serialize_dir_name)

    def deserialize(self, path: str, delete: bool = False) -> object:
        """
        Load the data that was serialized to `path`. It should have
        been serialized using this class's `serialize` method. If `delete==True`,
        `path` will be deleted once the deserialization is finished.
        """
        with TempDir() as temp_dir:
            # Untar the data to a temporary directory.
            with tarfile.open(path) as tar:
                tar.extractall(temp_dir)

            # Deserialize the data
            data_dir = os.path.join(temp_dir, self.serialize_dir_name)
            with open(self._get_pkl_path(data_dir), "rb") as f:
                obj = _CustomUnpickler(f, data_dir, self._type_serializers).load()

        if delete:
            os.remove(path)

        return obj

    def _get_pkl_path(self, path: str) -> str:
        return os.path.join(path, "data.pkl")
