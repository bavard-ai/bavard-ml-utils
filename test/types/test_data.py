from unittest import TestCase

import numpy as np

from bavard_ml_utils.types.data import DataModel, decode_numpy, encode_numpy


class InnerModel(DataModel):
    data: np.ndarray


class TestModel(DataModel):
    string: str
    array: np.ndarray
    inner: InnerModel

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.string != other.string:
            return False
        if not (self.array.shape == other.array.shape):
            return False
        if not (self.array == other.array).all():
            return False
        if not (self.inner.data.shape == other.inner.data.shape):
            return False
        if not (self.inner.data == other.inner.data).all():
            return False
        return True


class TestData(TestCase):
    def setUp(self):
        self.array = np.random.normal(size=(3, 4))
        self.array2 = np.random.normal(size=(5, 4))
        self.model = TestModel(string="hello", array=self.array, inner=InnerModel(data=self.array2))

    def test_can_serialize(self):
        raw = self.model.json(ensure_ascii=False)
        rebuilt = TestModel.parse_raw(raw)
        # The model should have reconstructed perfectly.
        self.assertEqual(self.model, rebuilt)

    def test_can_convert_to_object(self):
        d = self.model.dict()
        rebuilt = TestModel.parse_obj(d)
        # The model should have reconstructed perfectly.
        self.assertEqual(self.model, rebuilt)

    def test_numpy_serialization(self):
        for dtype in ["uint8", "int16", "float32"]:
            for mode in ["w", "wb"]:
                data = np.random.normal(size=(9, 11)).astype(dtype)
                deserialized = decode_numpy(encode_numpy(data, mode))
                self.assertTrue(np.array_equal(data, deserialized))
