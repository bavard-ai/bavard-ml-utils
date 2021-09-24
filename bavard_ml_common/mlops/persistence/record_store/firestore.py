import os
import typing as t

from google.auth.credentials import AnonymousCredentials, Credentials
from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentSnapshot

from bavard_ml_common.mlops.persistence.record_store.base import BaseRecordStore, Record, RecordT


class FirestoreRecordStore(BaseRecordStore[RecordT]):
    """A Firestore DAO for Pydantic data models."""

    def __init__(
        self,
        collection_name: str,
        record_class: t.Type[Record],
        *,
        project: t.Optional[str] = None,
        credentials: t.Optional[Credentials] = None,
    ):
        super().__init__(record_class)
        if os.getenv("FIRESTORE_EMULATOR_HOST") is not None:
            # We are in a testing context. Make sure the client's default args
            # work in this emulator scenario.
            if credentials is None:
                credentials = AnonymousCredentials()
            if project is None:
                project = "test"
        self.collection = firestore.Client(project=project, credentials=credentials).collection(collection_name)

    def save(self, record: RecordT):
        self.collection.document(record.get_id()).set(record.dict())

    def get(self, id_: str) -> t.Optional[RecordT]:
        doc = self.collection.document(id_).get()
        if not doc.exists:
            return None
        return self.record_cls.parse_obj(doc.to_dict())

    def delete(self, id_: str) -> bool:
        doc = self.collection.document(id_).get()
        if not doc.exists:
            return False
        doc.reference.delete()
        return True

    def get_all(self, *conditions: t.Tuple[str, str, t.Any], **where_equals) -> t.Iterable[RecordT]:
        """
        Retreives all records saved under `kind` which satisfy the optional `*conditions` and `*where_equals`
        equality conditions.
        """
        for doc in self._stream(*conditions, **where_equals):
            yield self.record_cls.parse_obj(doc.to_dict())

    def delete_all(self, *conditions: t.Tuple[str, str, t.Any], **where_equals) -> int:
        """
        Deletes all records saved under `kind` which satisfy the optional `*conditions` and `*where_equals`
        conditions. Returns the number of records that were deleted.
        """
        num_deleted = 0
        for doc in self._stream(*conditions, **where_equals):
            if doc.exists:
                doc.reference.delete()
                num_deleted += 1
        return num_deleted

    def _stream(self, *conditions: t.Tuple[str, str, t.Any], **where_equals) -> t.Iterable[DocumentSnapshot]:
        query = self.collection
        for field_name, operator, value in conditions:
            query = query.where(field_name, operator, value)
        for field_name, value in where_equals.items():
            query = query.where(field_name, "==", value)
        return query.stream()
