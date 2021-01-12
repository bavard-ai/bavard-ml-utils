from unittest import TestCase
import typing as t
import statistics

from fastapi.testclient import TestClient
from pydantic import BaseModel

from bavard_ml_common.mlops.web_service import WebService, endpoint


class FitInput(BaseModel):
    X: t.List[t.List[float]]
    y: t.List[float]


class Clf(WebService):
    """
    Clf stands for classifier.
    """

    def fit(self, X, y) -> None:
        self._mode = statistics.mode(y)

    @endpoint(methods=["POST"])
    def predict(self, X: t.List[t.List[float]]) -> dict:
        return {"predictions": [self._mode] * len(X)}


class ClfNoTypes(WebService):
    def fit(self, X, y) -> None:
        self._mode = statistics.mode(y)

    @endpoint(methods=["POST"])
    def predict(self, X) -> dict:
        return {"predictions": [self._mode] * len(X)}


class ClfMultipleEndpoints(WebService):
    def fit(self, X, y) -> None:
        self._mode = statistics.mode(y)

    @endpoint(path="/my-custom-predict-path", methods=["POST"])
    def predict(self, X: t.List[t.List[float]]) -> dict:
        return {"predictions": [self._mode] * len(X)}

    @endpoint
    def mode(self) -> dict:
        return {"mode": self._mode}


class ClfWithFitEndpoint(WebService):
    @endpoint(methods=["POST"])
    def fit(self, dataset: FitInput) -> None:
        self._mode = statistics.mode(dataset.y)

    @endpoint(methods=["POST"])
    def predict(self, X: t.List[t.List[float]]) -> dict:
        return {"predictions": [self._mode] * len(X)}


class TestWebService(TestCase):
    def setUp(self) -> None:
        self.X = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
        self.y = [1, 2, 2, 3, 3, 3, 4, 4, 5]
        self.y2 = [2, 2, 2, 2, 2, 2, 2, 2, 2]

    def test_web_service(self) -> None:
        client = self._get_client(Clf)

        # Has a home page
        res = client.get("/")
        self.assertEqual(res.status_code, 200)

        # Can handle basic predict request.
        res = client.post("/predict", json=[[1], [1], [1]])
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["predictions"], [3, 3, 3])

        # Performs request body data validation
        res = client.post("/predict", json={"foo": [[1], [1], [1]]})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json()["detail"][0]["msg"], "value is not a valid list")

        res = client.post("/predict", json=[["a"], ["b"], ["c"]])
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json()["detail"][0]["msg"], "value is not a valid float")

        # Knows how to handle unknown routes properly.
        self.assertEqual(client.get("/foo").status_code, 404)

    def test_multiple_endpoints(self) -> None:
        # `WebService` should be able to create multiple endpoints
        # for a single class.

        client = self._get_client(ClfMultipleEndpoints)
        # Can handle custom predict path.
        res = client.post("/my-custom-predict-path", json=[[1], [1], [1]])
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["predictions"], [3, 3, 3])

        res = client.get("/mode")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["mode"], 3)

    def test_can_alter_state(self) -> None:
        # The state of the model should be able to be altered even while
        # its a web service, if methods to do so are exposed in an endpoint.
        model = ClfWithFitEndpoint()
        model.fit(FitInput(X=self.X, y=self.y))
        app = model.to_app()
        client = TestClient(app)

        # Can predict as normal
        res = client.post("/predict", json=[[1], [1], [1]])
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["predictions"], [3, 3, 3])

        # Can fit on new dataset via post request. This is likely not a good idea
        # in practice because I think the model would be blocked until fitting
        # is finished.
        res = client.post("/fit", json={"X": self.X, "y": self.y2})
        self.assertEqual(res.status_code, 200)

        # `/predict` now matches the new state of the model.
        res = client.post("/predict", json=[[1], [1], [1]])
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["predictions"], [2, 2, 2])

    def _get_client(self, cls: t.Type) -> TestClient:
        model = cls()
        model.fit(self.X, self.y)
        app = model.to_app()
        return TestClient(app)
