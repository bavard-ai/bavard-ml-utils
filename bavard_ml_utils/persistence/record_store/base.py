import typing as t
from abc import ABC, abstractmethod

from bavard_ml_utils.types.data import DataModel


class Record(DataModel, ABC):
    """
    A pdyantic model that can be persisted. Override this class's :meth:`dict` method for custom serialization behavior,
    and its :meth:`parse_obj` class method for custom deserialization behavior. Numpy arrays are supported out of the
    box.
    """

    @abstractmethod
    def get_id(self) -> str:
        """Returns the primary key this record should be saved under."""
        pass


RecordT = t.TypeVar("RecordT", bound=Record)  # used to help static type checking tools


class BaseRecordStore(ABC, t.Generic[RecordT]):
    """
    Abstract base class for implementing a Data Access Object (DAO) which can save pydantic models to some back-end
    persistence solution.

    Parameters
    ----------
    record_class : Record type
        The pydantic-based class that persisted objects should be deserialized into.
    """

    def __init__(self, record_class: t.Type[Record]):
        self.record_cls = record_class

    @abstractmethod
    def save(self, record: RecordT):
        """Updates ``record`` in the database, or creates it if it's not there."""
        pass

    @abstractmethod
    def get(self, id_: str) -> t.Optional[RecordT]:
        """Retrieves a record from the database, returning ``None`` if it doesn't exist."""
        pass

    @abstractmethod
    def delete(self, id_: str) -> bool:
        """
        Deletes a record from the database, returning ``True`` if the record was deleted, and ``False`` if it didn't
        exist.
        """
        pass

    @abstractmethod
    def get_all(self, **where_equals) -> t.Iterable[RecordT]:
        """Retreives all records which satisfy the optional ``*where_equals`` equality conditions."""
        pass

    @abstractmethod
    def delete_all(self, **where_equals) -> int:
        """
        Deletes all records which satisfy the optional ``*where_equals`` equality conditions. Returns
        the number of records that were deleted.
        """
        pass
