import typing as t
import os
from abc import ABC, abstractmethod
import pickle
import tarfile
from tempfile import TemporaryDirectory, TemporaryFile

import tensorflow as tf

from bavard_ml_common.mlops.gcs import GCSClient


class TypeSerializer(ABC):
    """
    When implemented, provides functionality for serializing instance
    of some type or group of types, for use with the `Serializer` class. Provides
    support for saving and loading from Google Cloud Storage.
    """

    @property
    @abstractmethod
    def type_name(self) -> str:
        """
        A name identifying the type this serializer serializers. Should be
        unique among all the other `TypeSerializer`s used. Should also
        contain only letters, numbers, and dashes.
        """
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
        """Adds this serializer's extension to `path` if it has one.
        """
        return f"{path}.{self.ext}" if self.ext else path


class KerasSerializer(TypeSerializer):
    type_name = "keras"
    ext = None

    def serialize(self, obj: tf.keras.Model, path: str) -> None:
        # Keras models support cloud storage out of the box.
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
        self._unique_id = 0

    def persistent_id(self, obj: object) -> t.Optional[tuple]:
        for ser in self._ser_map.values():
            if ser.is_serializable(obj):
                # Treat `obj` as an external object and serialize it using our own
                # methods. Our serializer's type name and the relative path it was serialized
                # to is returned and pickled, so the pickler will know how to find
                # the object again and deserialize it.
                obj_id = f"{ser.type_name}-{self._get_unique_id()}"
                obj_path = ser.resolve_path(obj_id)
                ser.serialize(obj, os.path.join(self._assets_path, obj_path))
                return ser.type_name, obj_path

        # No custom serializer for `obj`; pickle it using the normal way.
        return None

    def _get_unique_id(self) -> int:
        """A primary key generator.
        """
        id_ = self._unique_id
        self._unique_id += 1
        return id_


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
        Here, pid is the tuple returned by `_CustomPickler.persistent_id`.
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
    `numpy` arrays, etc.). Includes support for serializing `keras` models using their
    native protocols. You can use your own type serializers and pass those in too. Just
    implement the `TypeSerializer` class and pass an instance of it to the constructor.
    """

    _serialize_dir_name = "serialized-data"

    def __init__(self, *custom_type_serializers: TypeSerializer) -> None:
        self._type_serializers = [KerasSerializer()] + list(custom_type_serializers)

        if len(self._type_serializers) != len(
            {ser.type_name for ser in self._type_serializers}
        ):
            raise ValueError(
                "The type_name of each type serializer must be unique."
                f" Currently registered names: {[ser.type_name for ser in self._type_serializers]}"
            )

    def serialize(self, obj: object, path: str) -> None:
        """Serialize `obj` to `path`, a directory.
        """
        is_gcs_file = GCSClient.is_gcs_uri(path)
        with TemporaryDirectory() as temp_dir:
            # Serialize to a temporary directory.
            with open(self._get_pkl_path(temp_dir), "wb") as f:
                _CustomPickler(f, temp_dir, self._type_serializers).dump(obj)

            # Tar the directory to the final destination.
            if is_gcs_file:
                # Tar to GCS.
                GCSClient().to_gcs_tar(temp_dir, path)
            else:
                # Tar to local.
                with tarfile.open(path, "w") as tar:
                    tar.add(temp_dir, arcname=self._serialize_dir_name)

    def deserialize(self, path: str, delete: bool = False) -> object:
        """
        Load the data that was serialized to the directory at
        `path`. It should have been serialized using this class's
        `serialize` method. If `delete==True`, `path` will be
        deleted once the deserialization is finished.
        """
        is_gcs_file = GCSClient.is_gcs_uri(path)
        with TemporaryFile() as temp_file:
            if is_gcs_file:
                GCSClient().download_blob_to_file(path, temp_file)
                temp_file.seek(0)
                local_file = temp_file
            else:
                local_file = path

            with TemporaryDirectory() as temp_dir:
                # Untar the data to a temporary directory.
                with tarfile.open(local_file) as tar:
                    tar.extractall(temp_dir)

                # Deserialize the data
                data_dir = os.path.join(temp_dir, self._serialize_dir_name)
                with open(self._get_pkl_path(data_dir), "rb") as f:
                    obj = _CustomUnpickler(f, data_dir, self._type_serializers).load()
                # Delete the temporary directory.
            # Delete the temporary file.
        if delete:
            if is_gcs_file:
                GCSClient().delete_blob(path)
            else:
                os.remove(path)

        return obj

    @staticmethod
    def _get_pkl_path(path: str) -> str:
        return os.path.join(path, "data.pkl")


class Persistent:
    """Mixin class giving persistence behavior.
    """
    serializer = Serializer()

    def to_dir(self, path: str) -> None:
        """Serializes the full state of `self` to directory `path`.
        """
        self.serializer.serialize(self, path)

    @classmethod
    def from_dir(cls, path: str, delete: bool = False) -> "Persistent":
        """
        Deserializes a full instance of this class from directory `path`. If `delete==True`,
        the persisted instance will be deleted once loaded into memory.
        """
        obj = cls.serializer.deserialize(path, delete)
        assert isinstance(obj, cls)
        return obj
