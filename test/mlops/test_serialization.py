from unittest import TestCase
import inspect
import shutil

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from transformers import ReformerModelWithLMHead, ReformerTokenizer

from bavard_ml_common.mlops.serialization import Serializer, TypeSerializer


class ReformerModelSerializer(TypeSerializer):
    type_name = "hf"
    ext = None

    def serialize(self, obj: ReformerModelWithLMHead, path: str) -> None:
        obj.save_pretrained(path)

    def deserialize(self, path: str) -> object:
        return ReformerModelWithLMHead.from_pretrained(path)

    def is_serializable(self, obj: object) -> bool:
        return isinstance(obj, ReformerModelWithLMHead)


class TestClass:
    class_attr = "class_attr"

    def __init__(self, *, public_attr) -> None:
        self.public_attr = public_attr
        self._private_attr = "_private_attr"

    def __eq__(self, obj: object) -> bool:
        return (
            type(self) == type(obj)
            and self.class_attr == obj.class_attr
            and self.public_attr == obj.public_attr
            and self._private_attr == obj._private_attr
        )


class TestSklearnModel:
    def __init__(self, *, C: float = 1.0) -> None:
        self._fitted = False
        self._clf = LogisticRegression(C=C)
        self.C = C

    def fit(self, X, y) -> None:
        self._clf.fit(X, y)
        self._fitted = True

    def predict(self, X):
        assert self._fitted
        return self._clf.predict(X)

    def __eq__(self, obj: object) -> bool:
        unfitted_check = (
            type(self) == type(obj) and self.C == obj.C and self._fitted == obj._fitted
        )
        if not self._fitted:
            return unfitted_check
        else:
            return (
                unfitted_check
                and (self._clf.coef_ == obj._clf.coef_).all()
                and (self._clf.intercept_ == obj._clf.intercept_).all()
            )


class TestKerasModel:
    def __init__(self, *, n_units: int) -> None:
        self.n_units = n_units
        self._fitted = False

    def fit(self, X, y) -> None:
        inputs = tf.keras.Input(shape=(X.shape[1],))
        outputs = tf.keras.layers.Dense(self.n_units)(inputs)
        outputs = tf.keras.layers.Dense(y.shape[1], activation="softmax")(outputs)
        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self._model.compile(loss="categorical_crossentropy")
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X):
        assert self._fitted
        return self._model.predict(X)

    def __eq__(self, obj: object) -> bool:
        unfitted_check = (
            type(self) == type(obj)
            and self.n_units == obj.n_units
            and self._fitted == obj._fitted
        )

        if not unfitted_check or not self._fitted:
            return unfitted_check

        if len(self._model.layers) != len(obj._model.layers):
            return False

        # Do a layer-wise check of all the weights.
        for layer_a, layer_b in zip(self._model.layers, obj._model.layers):
            wa, wb = layer_a.get_weights(), layer_b.get_weights()
            for mat_a, mat_b in zip(wa, wb):
                if not tf.math.reduce_all(tf.math.equal(mat_a, mat_b)):
                    return False

        return True


class TestHfModel:
    def __init__(self, max_length: int = 20):
        # A very small model we can use for tests.
        self.max_length = max_length
        self._model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")
        self._tokenizer = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")

    def predict(self, x: str) -> str:
        encoded_x = self._tokenizer.encode(x, return_tensors="pt")
        generated = self._model.generate(encoded_x, max_length=self.max_length)[0]
        return self._tokenizer.decode(generated)


class TestSerialization(TestCase):
    def setUp(self) -> None:
        self.temp_dir = "temp-data"
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.n_features = len(iris.feature_names)

    def test_can_serialize_basic(self) -> None:
        # Can serialize when no custom serializers are used.
        data = {"foo": 1, "bar": None, "baz": [1, 2, 3], 0: "Hello"}
        serializer = Serializer()

        serializer.serialize(data, self.temp_dir)
        loaded_data = serializer.deserialize(self.temp_dir, True)

        self.assertDictEqual(data, loaded_data)

    def test_can_serialize_class(self) -> None:
        obj = TestClass(public_attr=1)
        serializer = Serializer()

        serializer.serialize(obj, self.temp_dir)
        loaded_obj = serializer.deserialize(self.temp_dir, True)

        self.assertEqual(obj, loaded_obj)

    def test_sklearn_serializer(self) -> None:
        model = TestSklearnModel(C=0.5)
        serializer = Serializer()

        # Unfit models should be equal.
        serializer.serialize(model, self.temp_dir)
        loaded_model = serializer.deserialize(self.temp_dir, True)
        self.assertEqual(model, loaded_model)

        # Fit models should be equal.
        model.fit(self.X, self.y)
        serializer.serialize(model, self.temp_dir)
        loaded_fit_model = serializer.deserialize(self.temp_dir, True)
        self.assertEqual(model, loaded_fit_model)

        # Predictions should be identical.
        first_two = self.X[:2, :]
        self.assertTrue(
            (model.predict(first_two) == loaded_fit_model.predict(first_two)).all()
        )

    def test_keras_serializer(self) -> None:
        model = TestKerasModel(n_units=10)
        serializer = Serializer()

        # Unfit models should be equal.
        serializer.serialize(model, self.temp_dir)
        loaded_model = serializer.deserialize(self.temp_dir, True)
        self.assertEqual(model, loaded_model)

        # Fit models should be equal.
        model.fit(self.X, tf.keras.utils.to_categorical(self.y))
        serializer.serialize(model, self.temp_dir)
        loaded_fit_model = serializer.deserialize(self.temp_dir, True)
        self.assertEqual(model, loaded_fit_model)

        # Predictions should be identical.
        first_two = self.X[:2, :]
        self.assertTrue(
            (model.predict(first_two) == loaded_fit_model.predict(first_two)).all()
        )

    def test_hf_serializer(self) -> None:
        model = TestHfModel(max_length=25)
        serializer = Serializer(ReformerModelSerializer())

        # Should serialize and deserialize successfully.
        serializer.serialize(model, self.temp_dir)
        loaded_model = serializer.deserialize(self.temp_dir, True)

        # Predictions should be identical.
        self.assertEqual(model.predict("A few months later"), loaded_model.predict("A few months later"))

    def test_can_load_moved_object(self) -> None:
        move_location = "moved-dir"
        model = TestKerasModel(n_units=10)
        serializer = Serializer()
        serializer.serialize(model, self.temp_dir)
        shutil.move(self.temp_dir, move_location)

        # Unfit models should be equal.
        loaded_model = serializer.deserialize(move_location, True)
        self.assertEqual(model, loaded_model)

        # Fit models should be equal.
        model.fit(self.X, tf.keras.utils.to_categorical(self.y))
        serializer.serialize(model, self.temp_dir)
        shutil.move(self.temp_dir, move_location)
        loaded_fit_model = serializer.deserialize(move_location, True)
        self.assertEqual(model, loaded_fit_model)

        # Predictions should be identical.
        first_two = self.X[:2, :]
        self.assertTrue(
            (model.predict(first_two) == loaded_fit_model.predict(first_two)).all()
        )

    @staticmethod
    def _get_member_types(obj: object) -> dict:
        return {key: type(member) for key, member in inspect.getmembers(obj)}
