import typing as t
from unittest import TestCase

import numpy as np
from pydantic import BaseModel

from bavard_ml_utils.types.data import DataModel
from bavard_ml_utils.types.utils import hash_model


class MyModel(BaseModel):
    a: str
    b: int
    c: float
    d: bool
    e: t.List[str]
    f: dict
    g: bytes


class MyNumpyModel(DataModel):
    a: int
    b: np.ndarray


class TestUtils(TestCase):
    def setUp(self):
        self.model = MyModel(a="a", b=1, c=2.0, d=True, e=["b", "c"], f={"a": 1}, g=b"123")
        self.np_model = MyNumpyModel(a=123, b=np.random.normal(size=(5, 5)))

    def test_hash(self):
        # Should be able to hash a model, and a deserialized version of the same model should give the same hash.
        # The same goes for a numpy-compatible model.
        for model in [self.model, self.np_model]:
            hash1 = hash_model(model)
            hash2 = hash_model(model.__class__.parse_raw(model.json()))
            self.assertEqual(hash1, hash2)
