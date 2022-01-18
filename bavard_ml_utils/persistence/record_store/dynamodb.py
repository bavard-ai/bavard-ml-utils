import operator
import os
import typing as t
from datetime import datetime
from decimal import Context
from functools import reduce

from bavard_ml_utils.utils import ImportExtraError


try:
    import boto3
    from boto3.dynamodb.conditions import Attr, Key
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

    def get_all(self, *conditions: t.Tuple[str, str, t.Any], **where_equals) -> t.Iterable[RecordT]:
        """Paginates over all records in the table, yielding them in an iterator.
        Retrieves all records which satisfy the optional ``*conditions`` and ``**where_equals`` equality conditions.
        """
        done, start_key, use_query = False, None, False
        conditions_expression = self._set_conditions(*conditions, **where_equals)
        if conditions_expression is None:
            conditions_expression = {}
        if "KeyConditionExpression" in conditions_expression.keys():
            use_query = True
        while not done:
            if start_key:
                if use_query:
                    res = self._table.query(ExclusiveStartKey=start_key, **conditions_expression)
                else:
                    res = self._table.scan(ExclusiveStartKey=start_key, **conditions_expression)
            else:
                if use_query:
                    res = self._table.query(**conditions_expression)
                else:
                    res = self._table.scan(**conditions_expression)
            start_key = res.get("LastEvaluatedKey")
            done = start_key is None
            for item in res.get("Items", []):
                yield self.record_cls.parse_obj(item)

    def _set_conditions(self, *conditions: t.Tuple[str, str, t.Any], **where_equals) -> dict:
        """
        FilterExpression specifies a condition that returns only items that satisfy the condition.
        All other items are discarded. However, the filter is applied only after the entire table has been scanned.

        KeyConditionExpression specifies a condition for partition key or sort key.
        it requires an equality check on a partition key value, and optional other operators check on sort key
        Using KeyConditionExpression, Query performs a direct lookup to a selected partition based on
        primary or secondary partition/hash

        scan only accepts FilterExpression argument, where as query accpets both KeyConditionExpression,
        and FilterExpression.
        Important: To use query, we must provide KeyConditionExpression --> so if KeyConditionExpression
        is not provided, we need to use scan, otherwise we use query.
        """

        filters_list, key_flag = [], False
        for attr, value in where_equals.items():
            if attr == self._pk and key_flag is False:
                key_flag = True
                key_condition = Key(attr).eq(str(value))
            else:
                filters_list.append(Attr(attr).eq(value))
        for attr, op, value in conditions:
            if attr == self._pk and op == "==" and key_flag is False:
                key_flag = True
                key_condition = Key(attr).eq(str(value))
            else:
                filters_list.append(self.choose_operator(attr, op, value))
        if len(filters_list) != 0:
            res = reduce(lambda x, y: x & y, filters_list)
        if len(filters_list) == 0 and key_flag is False:
            return {}
        if key_flag is False:
            return {"FilterExpression": res}
        if len(filters_list) == 0:
            return {"KeyConditionExpression": key_condition}
        return {"KeyConditionExpression": key_condition, "FilterExpression": res}

    @staticmethod
    def choose_operator(attr, op, value):
        if isinstance(value, datetime):
            """
            attributed with the type of datetime is recorded as isoformat,
            so the value being used for conditional search should be isoformat as well
            """
            value = value.isoformat()
        if op == "<=":
            return Attr(attr).lte(value)
        if op == "<":
            return Attr(attr).lt(value)
        if op == ">=":
            value = str(value)
            return Attr(attr).gte(value)
        if op == ">":
            return Attr(attr).gt(value)
        if op == "==":
            return Attr(attr).eq(value)

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
