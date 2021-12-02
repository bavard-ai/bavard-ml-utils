import operator
import os
import typing as t
from decimal import Context

from bavard_ml_utils.utils import ImportExtraError


try:
    import boto3
    from botocore.config import Config
except ImportError:
    raise ImportExtraError("aws", __name__)

from bavard_ml_utils.persistence.record_store.base import BaseRecordStore, RecordT


def remap(o: object, apply: t.Callable):
    if isinstance(o, dict):
        return {k: remap(v, apply) for k, v in o.items()}
    elif isinstance(o, (list, tuple, set)):
        return type(o)(remap(elem, apply) for elem in o)
    elif isinstance(o, (str, int, float, type(None))):
        return apply(o)
    else:
        raise AssertionError(f"remap encountered unsupported type {type(o)}")


class DynamoDBRecordStore(BaseRecordStore[RecordT]):
    """
    A DynamoDB DAO for Pydantic data models. In addition to its parent class's `WHERE equals` behavior, this class
    also supports arbitrary `WHERE` clauses (e.g. ``<=``), using Firestore's `operator string syntax <https://googleapis
    .dev/python/firestore/latest/collection.html?highlight=where#google.cloud.firestore_v1.base_collection.BaseCollectio
    nReference.where>`_.
    """

    _operators = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt, "==": operator.eq}

    def __init__(self, table_name: str, record_class: t.Type[RecordT], *, read_only=False, primary_key_field_name="id"):
        super().__init__(record_class=record_class, read_only=read_only)
        self._table = boto3.resource(
            "dynamodb", endpoint_url=os.getenv("AWS_ENDPOINT"), config=Config(region_name=os.getenv("AWS_REGION"))
        ).Table(table_name)
        self._pk = primary_key_field_name
        self._decimal_ctx = Context(prec=38)  # dynamodb has a maximum precision of 38 digits for decimal numbers

    def save(self, record: RecordT):
        self.assert_can_edit()
        item_dict = record.dict()
        item_dict = remap(item_dict, self._float_to_decimal)
        self._table.put_item(Item={**item_dict, self._pk: record.get_id()})

    def get(self, id_: str) -> t.Optional[RecordT]:
        res = self._table.get_item(Key={self._pk: id_})
        if "Item" not in res:
            return None
        return self.record_cls.parse_obj(res["Item"])

    def delete(self, id_: str) -> bool:
        self.assert_can_edit()
        if self.get(id_) is None:
            # No record exists for the given id.
            return False
        self._table.delete_item(Key={self._pk: id_})
        return True

    def get_all(self, *conditions: t.Tuple[str, str, t.Any], **where_equals) -> t.Iterable[RecordT]:
        """
        Retreives all records which satisfy the optional ``*conditions`` and ``**where_equals`` equality conditions.
        **Note**: the current implementation for this method is very slow, using the DynamoDB scan method under the
        hood.
        """
        # TODO: This is an incredibly slow way of doing this. Use secondary indexes and a DynamoDB Query instead.
        for record in self._scan():
            matches = True
            for attr, op, value in conditions:
                if not self._operators[op](getattr(record, attr), value):
                    matches = False
            for attr, value in where_equals.items():
                if getattr(record, attr) != value:
                    matches = False
            if matches:
                yield record

    def delete_all(self, *conditions: t.Tuple[str, str, t.Any], **where_equals) -> int:
        """
        Deletes all records which satisfy the optional ``*conditions`` and ``*where_equals`` conditions. Returns the
        number of records that were deleted. **Note**: the current implementation for this method is very slow, using
        the DynamoDB scan method under the hood.
        """
        # TODO: This is an incredibly slow way of doing this. Use indexes and a batch delete instead.
        self.assert_can_edit()
        num_deleted = 0
        for record in self.get_all(*conditions, **where_equals):
            self.delete(record.get_id())
            num_deleted += 1
        return num_deleted

    def _scan(self) -> t.Iterable[RecordT]:
        """Paginates over all records in the table, yielding them in an iterator."""
        done, start_key = False, None
        while not done:
            if start_key:
                res = self._table.scan(ExclusiveStartKey=start_key)
            else:
                res = self._table.scan()
            start_key = res.get("LastEvaluatedKey")
            done = start_key is None
            for item in res.get("Items", []):
                yield self.record_cls.parse_obj(item)

    def _float_to_decimal(self, a):
        """
        Convert ``a`` into a :class:`decimal.Decimal` object, if its of type :class:`float`. DynamoDB does not support
        float types; only Decimal.
        """
        if isinstance(a, float):
            # A local `decimal.Context` is needed because DynamoDB's default context will throw an error if any rounding
            # occurs.
            return self._decimal_ctx.create_decimal_from_float(a)
        return a
