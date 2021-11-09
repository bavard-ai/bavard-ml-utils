import typing as t

from bavard_ml_utils.persistence.record_store.base import BaseRecordStore, RecordT


class InMemoryRecordStore(BaseRecordStore[RecordT]):
    r"""
    A simple in-memory DAO for Pydantic data models. Useful for testing or other lightweight needs. Does not keep any
    indexes of non-primary key fields, so `WHERE` clause queries are :math:`\mathcal{O}(n)`.
    """

    def __init__(self, record_class: t.Type[RecordT], read_only=False):
        super().__init__(record_class, read_only)
        # Records in the db can be resolved via `self._db[id]`.
        self._db: t.Dict[str, RecordT] = {}

    def save(self, record: RecordT):
        self.assert_can_edit()
        self._db[record.get_id()] = record

    def get(self, id_: str) -> t.Optional[RecordT]:
        return self._db.get(id_)

    def delete(self, id_: str) -> bool:
        self.assert_can_edit()
        if self.get(id_) is not None:
            del self._db[id_]
            return True
        return False

    def get_all(self, **where_equals) -> t.Iterable[RecordT]:
        for record in self._db.values():
            if self._matches(record, **where_equals):
                yield record

    def delete_all(self, **where_equals) -> int:
        self.assert_can_edit()
        to_delete = [id_ for id_, record in self._db.items() if self._matches(record, **where_equals)]
        for id_ in to_delete:
            del self._db[id_]
        return len(to_delete)

    @staticmethod
    def _matches(record: RecordT, **where_equals) -> bool:
        return all(getattr(record, k) == v for k, v in where_equals.items())
