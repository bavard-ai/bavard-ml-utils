from unittest import TestCase

from bavard_ml_utils.persistence.record_store.base import BaseRecordStore, Record
from bavard_ml_utils.persistence.record_store.firestore import FirestoreRecordStore
from bavard_ml_utils.persistence.record_store.memory import InMemoryRecordStore
from test.utils import clear_database


class Fruit(Record):
    name: str
    color: str
    is_tropical: bool

    def get_id(self) -> str:
        return self.name


class TestRecordStore(TestCase):
    def setUp(self):
        clear_database()
        # Test multiple kinds of record stores.
        self.databases = [FirestoreRecordStore("fruits", Fruit), InMemoryRecordStore(Fruit)]
        self.apple = Fruit(name="apple", color="red", is_tropical=False)
        self.mango = Fruit(name="mango", color="yellow", is_tropical=True)
        self.pear = Fruit(name="pear", color="green", is_tropical=False)

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

    def _create_some_records(self, db: BaseRecordStore):
        db.save(self.apple)
        db.save(self.mango)
        db.save(self.pear)
