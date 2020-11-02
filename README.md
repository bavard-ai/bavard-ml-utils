# bavard-ml-common

## Web Service Example

```python
import typing as t

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import uvicorn

from bavard_ml_common.mlops.web_service import endpoint, WebService 


class DTModel(WebService):
    """
    A machine learning model that can turn into a web service.
    """

    def __init__(self) -> None:
        self._dt = DecisionTreeClassifier()
        self._fitted = False

    def fit(self, X, y) -> None:
        self._dt.fit(X, y)
        self._fitted = True

    # Type annotations are required for all `@endpoint` method arguments.
    @endpoint
    def predict(self, X: t.List[t.List[float]]):
        assert self._fitted
        return self._dt.predict(X).tolist()

    @endpoint
    def feature_importances(self) -> list:
        assert self._fitted
        return self._dt.feature_importances_.tolist()


# Fit a ML model on a dataset.
iris = load_iris()
X = iris.data
y = iris.target
model = DTModel()
model.fit(X, y)

# Convert the model into a `fastapi` web service.
api = model.to_app()

# Run the web service.
uvicorn.run(api, host="0.0.0.0", use_colors=True, log_level="debug")

# The fitted model is now available at 0.0.0.0:8000/,
# with a `/predict` endpoint and a `/feature_importances` endpoint.
# API documentation for the fitted model is at a `/docs` endpoint.
```

## Developing Locally

Install dependencies:

```
pip3 install -e .
pip3 install -r requirements-test.txt
```

Then, run the tests using pytest:

```
python3 -m unittest
```

## Releasing The Package

Releasing the package is automatically handled by CI, but three steps must be taken to trigger a successful release:

1. Increment the `VERSION` variable in `setup.py` to the new desired version (e.g. `VERSION="1.1.1"`)
2. Commit and tag the repo with the **exact same** value you populated the `VERSION` variable with (e.g. `git tag 1.1.1`)
3. Push the commit and tag to remote. These can be done together using: `git push --atomic origin <branch name> <tag>`

CI will then release the package to pypi with that version once the commit and tag are pushed.
