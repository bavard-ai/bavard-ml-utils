import math
from datetime import datetime, timedelta, timezone
from decimal import Context
from unittest import TestCase

from bavard_ml_utils.persistence.record_store.base import BaseRecordStore, Record
from bavard_ml_utils.persistence.record_store.dynamodb import DynamoDBRecordStore
from bavard_ml_utils.persistence.record_store.firestore import FirestoreRecordStore
from bavard_ml_utils.persistence.record_store.memory import InMemoryRecordStore
from test.utils import clear_firestore, create_dynamodb_table


class Fruit(Record):
    name: str
    color: str
    is_tropical: bool
    price: float

    def get_id(self) -> str:
        return self.name

    def get_sort_key(self):
        return Context(prec=38).create_decimal_from_float(self.price)


class DatedRecord(Record):
    id: int
    createdAt: datetime
    payload: str

    def get_id(self) -> str:
        return str(self.id)

    def get_sort_key(self):
        return self.createdAt.isoformat()


class TestRecordStore(TestCase):
    def setUp(self):
        clear_firestore()
        # Create the needed DynamoDB table.
        self.fruits_table = create_dynamodb_table("fruits")
        self.data_table = create_dynamodb_table("data")
        self.fruits_table_composite_key = create_dynamodb_table(
            "fruits_composite_key", sort_key_field="price", sort_key_type="N"
        )
        self.data_table_composite_key = create_dynamodb_table(
            "data_composite_key", sort_key_field="createdAt", sort_key_type="S"
        )
        # Test multiple kinds of record stores.
        self.databases = [
            FirestoreRecordStore("fruits", Fruit),
            InMemoryRecordStore(Fruit),
            DynamoDBRecordStore("fruits", Fruit),
        ]
        self.apple = Fruit(name="apple", color="red", is_tropical=False, price=0.5)
        self.mango = Fruit(name="mango", color="yellow", is_tropical=True, price=math.pi)
        self.pear = Fruit(name="pear", color="green", is_tropical=False, price=0.75)

    def tearDown(self) -> None:
        self.fruits_table.delete()
        self.data_table.delete()
        self.fruits_table_composite_key.delete()
        self.data_table_composite_key.delete()

    def test_can_save(self):
        for db in self.databases:
            db.save(self.apple)
            apple2 = db.get("apple")
            self.assertIsNotNone(apple2)  # should exist in the db
            self.assertEqual(self.apple, apple2)  # should have serialized and deserialized correctly

    def test_can_delete(self):
        for db in self.databases:
            db.save(self.mango)
            mango2 = db.get("mango")
            self.assertIsNotNone(mango2)  # should exist in the db
            db.delete("mango")
            mango3 = db.get("mango")
            self.assertIsNone(mango3)  # should *not* exist in the db

    def test_can_get_all(self):
        for db in self.databases:
            self._create_some_records(db)
            self.assertEqual(len(list(db.get_all())), 3)
            # Test `get_all` with where clause.
            fruits = list(db.get_all(name="mango"))
            self.assertEqual(len(fruits), 1)
            mango = fruits[0]
            self.assertEqual(self.mango, mango)  # should have serialized and deserialized correctly

        # test for the table with composite key
        database = DynamoDBRecordStore("fruits_composite_key", Fruit, sort_key_field_name="price")
        self._create_some_records_compose_key(database)
        self.assertEqual(len(list(database.get_all())), 9)
        database.delete_all()

    def test_can_delete_all(self):
        for db in self.databases:
            self._create_some_records(db)
            self.assertEqual(len(list(db.get_all())), 3)
            # Delete two.
            num_deleted = db.delete_all(is_tropical=False)
            self.assertEqual(num_deleted, 2)
            fruits = list(db.get_all())
            self.assertEqual(len(fruits), 1)  # should only be one veggie left
            self.assertEqual(fruits[0], self.mango)  # it should be the right one
            # Delete everything.
            num_deleted = db.delete_all()
            self.assertEqual(num_deleted, 1)
            self.assertEqual(len(list(db.get_all())), 0)  # should be none left

        # test for the table with composite key
        database = DynamoDBRecordStore("fruits_composite_key", Fruit, sort_key_field_name="price")
        self._create_some_records_compose_key(database)
        self.assertEqual(len(list(database.get_all())), 9)
        apple2 = database.get("apple", 1.5)
        self.assertIsNotNone(apple2)  # should exist in the database
        apple3 = database.get("apple", 0.5)
        self.assertIsNotNone(apple3)  # should exist in the database
        database.delete("apple", 1.5)
        apple4 = database.get("apple", 1.5)
        self.assertIsNone(apple4)  # should *not* exist in the db
        self.assertEqual(len(list(database.get_all())), 8)
        database.delete_all()
        self.assertEqual(len(list(database.get_all())), 0)

    def test_read_only_firestore(self):
        db = FirestoreRecordStore("fruits", Fruit)
        self._create_some_records(db)
        # Connected to the same collection in Firestore.
        read_only_db = FirestoreRecordStore("fruits", Fruit, read_only=True)
        # Should be able to retrieve the data, but not edit it in any way.
        data = tuple(sorted(item.get_id() for item in read_only_db.get_all()))
        self.assertEqual(len(data), 3)
        with self.assertRaises(AssertionError):
            read_only_db.save(self.mango)
        with self.assertRaises(AssertionError):
            read_only_db.delete(self.mango.get_id())
        with self.assertRaises(AssertionError):
            read_only_db.delete_all()
        # The data should not have changed.
        data2 = tuple(sorted(item.get_id() for item in read_only_db.get_all()))
        self.assertEqual(data, data2)

    def test_read_only_in_memory(self):
        read_only_db = InMemoryRecordStore(Fruit, read_only=True)
        # Should be able to retrieve the data, but not edit it in any way.
        n_records = sum(1 for _ in read_only_db.get_all())
        self.assertEqual(n_records, 0)
        retrieved_mango = read_only_db.get(self.mango.get_id())
        self.assertIsNone(retrieved_mango)
        with self.assertRaises(AssertionError):
            read_only_db.save(self.mango)
        with self.assertRaises(AssertionError):
            read_only_db.delete(self.mango.get_id())
        with self.assertRaises(AssertionError):
            read_only_db.delete_all()

    def test_query_with_conditions(self):
        # test 1:
        databases = [FirestoreRecordStore("data", DatedRecord), DynamoDBRecordStore("data", DatedRecord)]
        for db in databases:
            # First, make some data.
            now = datetime.now(timezone.utc)
            for i in range(10):
                db.save(DatedRecord(id=i, createdAt=now - timedelta(days=i), payload="arbitrary data"))
            four_days_ago = now - timedelta(days=4)
            # Perform a conditional search.
            new_records = list(db.get_all(("createdAt", ">=", four_days_ago)))
            # Should have retrieved the five new records from the last four days.
            self.assertEqual(len(new_records), 5)
            for record in new_records:
                self.assertGreaterEqual(record.createdAt, four_days_ago)
            db.delete_all()

        database = DynamoDBRecordStore("data", DatedRecord)
        # test2: equality condition test
        now = datetime.now(timezone.utc)
        database.save(DatedRecord(id=0, createdAt=now, payload="arbitrary data"))
        # Perform a conditional search.
        new_records = list(database.get_all(("createdAt", "==", now)))
        # Should have retrieved one record.
        self.assertEqual(len(new_records), 1)
        for record in new_records:
            self.assertEqual(record.createdAt, now)
        database.delete_all()

        # test3: another equality condition test
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i, createdAt=now - timedelta(days=i), payload=f"arbitrary data{i%2}"))
        # Perform a conditional search.
        new_records = list(database.get_all(("payload", "==", "arbitrary data1")))
        # Should have retrieved the 5 new records with the payload of arbitrary data1.
        for record in new_records:
            self.assertEqual(record.payload, "arbitrary data1")
        database.delete_all()

        # test4: condition for primary key
        now = datetime.now(timezone.utc)
        database.save(DatedRecord(id=0, createdAt=now, payload="arbitrary data"))
        # Perform a conditional search on primary key.
        new_records = list(database.get_all(("id", "==", 0)))
        # Should have retrieved one record with id of 0.
        self.assertEqual(len(new_records), 1)
        for record in new_records:
            self.assertEqual(record.get_id(), "0")
        database.delete_all()

        # test5: single condition for multiple attributes
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i, createdAt=now - timedelta(days=i), payload=f"arbitrary data{i%2}"))
        four_days_ago = now - timedelta(days=4)
        # Perform a conditional search.
        new_records = list(database.get_all(("createdAt", ">=", four_days_ago), ("payload", "==", "arbitrary data1")))
        # Should have retrieved the 3 new records from the last four days arbitrary data1.
        self.assertEqual(len(new_records), 2)
        for record in new_records:
            self.assertGreaterEqual(record.createdAt, four_days_ago)
            self.assertEqual(record.payload, "arbitrary data1")
        database.delete_all()

        # test6: multiple conditions for one attribute
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i, createdAt=now - timedelta(days=i), payload="arbitrary data"))
        four_days_ago = now - timedelta(days=4)
        two_days_ago = now - timedelta(days=2)
        # Perform a conditional search.
        new_records = list(database.get_all(("createdAt", "<=", two_days_ago), ("createdAt", ">=", four_days_ago)))
        # Should have retrieved the 3 new records between last four ago and two days ago.
        self.assertEqual(len(new_records), 3)
        for record in new_records:
            self.assertGreaterEqual(record.createdAt, four_days_ago)
            self.assertLessEqual(record.createdAt, two_days_ago)
        database.delete_all()

        # test7: multiple conditions for one attribute, and one condition for another attribute
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i, createdAt=now - timedelta(days=i), payload=f"arbitrary data{i%2}"))
        four_days_ago = now - timedelta(days=4)
        two_days_ago = now - timedelta(days=2)
        # Perform a conditional search.
        new_records = list(
            database.get_all(
                ("createdAt", "<=", two_days_ago),
                ("createdAt", ">=", four_days_ago),
                ("payload", "==", "arbitrary data1"),
            )
        )
        # Should have retrieved one record between last four ago and two days ago arbitrary data1.
        self.assertEqual(len(new_records), 1)
        for record in new_records:
            self.assertGreaterEqual(record.createdAt, four_days_ago)
            self.assertLessEqual(record.createdAt, two_days_ago)
            self.assertEqual(record.payload, "arbitrary data1")
        database.delete_all()

        # test8: final test, condition only for primary key, and there is not sort key define in the table
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i % 2, createdAt=now - timedelta(days=i), payload="arbitrary data"))
        new_records = list(database.get_all(("id", "==", 0)))
        # Should have retrieved only one records with id of 0 (the very last one, items with the same primary key
        # will be overridden since there is not sort key define). if there is a sort key defined, it will be
        # tested with `test_query_compose_key_with_conditions`
        self.assertEqual(len(new_records), 1)
        for record in new_records:
            self.assertEqual(record.get_id(), "0")
        database.delete_all()

    def test_query_compose_key_with_conditions(self):
        database = DynamoDBRecordStore("data_composite_key", DatedRecord, sort_key_field_name="createdAt")
        # test1: condition only for primary key
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i % 2, createdAt=now - timedelta(days=i), payload="arbitrary data"))
        new_records = list(database.get_all(("id", "==", 0)))
        # Should have retrieved five records with id of 0.
        self.assertEqual(len(new_records), 5)
        for record in new_records:
            self.assertEqual(record.get_id(), "0")
        database.delete_all()

        # test2: the previous test, it makes sure delete all worked properly
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i % 2, createdAt=now - timedelta(days=i), payload="arbitrary data"))
        new_records = list(database.get_all(("id", "==", 0)))
        # Should have retrieved five records with id of 0.
        self.assertEqual(len(new_records), 5)
        for record in new_records:
            self.assertEqual(record.get_id(), "0")
        database.delete_all()

        # test3: again, condition only for primary key
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i % 3, createdAt=now - timedelta(days=i), payload="arbitrary data"))
        new_records = list(database.get_all(("id", "==", 2)))
        # Should have retrieved three records with id of 2.
        self.assertEqual(len(new_records), 3)
        for record in new_records:
            self.assertEqual(record.get_id(), "2")
        database.delete_all()

        # test4: condition for both primary key and sort key
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i % 2, createdAt=now - timedelta(days=i), payload="arbitrary data"))
        four_days_ago = now - timedelta(days=4)
        new_records = list(database.get_all(("id", "==", 1), ("createdAt", ">=", four_days_ago)))
        # Should have retrieved two records with id of 1 from last four days.
        self.assertEqual(len(new_records), 2)
        for record in new_records:
            self.assertEqual(record.get_id(), "1")
            self.assertGreaterEqual(record.createdAt, four_days_ago)
        database.delete_all()

        # test5: multiple condition for sort key
        database = DynamoDBRecordStore("data_composite_key", DatedRecord, sort_key_field_name="createdAt")
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i % 2, createdAt=now - timedelta(days=i), payload="arbitrary data"))
        four_days_ago = now - timedelta(days=4)
        two_days_ago = now - timedelta(days=2)
        new_records = list(database.get_all(("createdAt", ">=", four_days_ago), ("createdAt", "<=", two_days_ago)))
        # Should have retrieved three records between four days ago and two days ago.
        self.assertEqual(len(new_records), 3)
        for record in new_records:
            self.assertGreaterEqual(record.createdAt, four_days_ago)
            self.assertLessEqual(record.createdAt, two_days_ago)
        database.delete_all()

        # test6: multiple condition for sort key and one condition on another attribute
        database = DynamoDBRecordStore("data_composite_key", DatedRecord, sort_key_field_name="createdAt")
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i % 2, createdAt=now - timedelta(days=i), payload=f"arbitrary data{i%2}"))
        four_days_ago = now - timedelta(days=4)
        two_days_ago = now - timedelta(days=2)
        new_records = list(
            database.get_all(
                ("createdAt", ">=", four_days_ago),
                ("createdAt", "<=", two_days_ago),
                ("payload", "==", "arbitrary data1"),
            )
        )
        # Should have retrieved one record between four days ago and two days ago with the payload of arbitrary data1.
        self.assertEqual(len(new_records), 1)
        for record in new_records:
            self.assertGreaterEqual(record.createdAt, four_days_ago)
            self.assertLessEqual(record.createdAt, two_days_ago)
            self.assertEqual(record.payload, "arbitrary data1")
        database.delete_all()

        # test7: condition for primary key and  multiple condition for sort key
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i % 2, createdAt=now - timedelta(days=i), payload=f"arbitrary data{i%2}"))
        four_days_ago = now - timedelta(days=4)
        two_days_ago = now - timedelta(days=2)
        new_records = list(
            database.get_all(("createdAt", ">=", four_days_ago), ("id", "==", 1), ("createdAt", "<=", two_days_ago))
        )
        # Should have retrieved one records with id of 1 between four days ago and two days ago.
        self.assertEqual(len(new_records), 1)
        for record in new_records:
            self.assertEqual(record.get_id(), "1")
            self.assertGreaterEqual(record.createdAt, four_days_ago)
            self.assertLessEqual(record.createdAt, two_days_ago)
        database.delete_all()

        # test8: condition for primary key, sort key, and another attribute
        now = datetime.now(timezone.utc)
        for i in range(10):
            database.save(DatedRecord(id=i % 2, createdAt=now - timedelta(days=i), payload=f"arbitrary data{i % 2}"))
        four_days_ago = now - timedelta(days=4)
        new_records = list(
            database.get_all(("createdAt", ">=", four_days_ago), ("id", "==", 0), ("payload", "==", "arbitrary data0"))
        )
        # Should have retrieved three records with id from four days ago with the payload of arbitrary data1.
        self.assertEqual(len(new_records), 3)
        for record in new_records:
            self.assertEqual(record.get_id(), "0")
            self.assertGreaterEqual(record.createdAt, four_days_ago)
            self.assertEqual(record.payload, "arbitrary data0")
        database.delete_all()

    def _create_some_records(self, db: BaseRecordStore):
        db.save(self.apple)
        db.save(self.mango)
        db.save(self.pear)

    @staticmethod
    def _create_some_records_compose_key(db: BaseRecordStore):
        db.save(Fruit(name="apple", color="red", is_tropical=False, price=0.5))
        db.save(Fruit(name="apple", color="red", is_tropical=False, price=1))
        db.save(Fruit(name="apple", color="red", is_tropical=False, price=1.5))
        db.save(Fruit(name="apple", color="red", is_tropical=False, price=2))
        db.save(Fruit(name="mango", color="yellow", is_tropical=True, price=math.pi))
        db.save(Fruit(name="mango", color="yellow", is_tropical=True, price=2 * math.pi))
        db.save(Fruit(name="mango", color="yellow", is_tropical=True, price=3 * math.pi))
        db.save(Fruit(name="pear", color="green", is_tropical=False, price=0.75))
        db.save(Fruit(name="pear", color="green", is_tropical=False, price=1.5))
