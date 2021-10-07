# bavard-ml-utils

[![CircleCI Build Status](https://circleci.com/gh/bavard-ai/bavard-ml-utils/tree/main.svg?style=shield)](https://circleci.com/gh/bavard-ai/bavard-ml-utils/tree/main)
[![PyPI Version](https://badge.fury.io/py/bavard-ml-utils.svg)](https://badge.fury.io/py/bavard-ml-utils)
[![PyPI Downloads](https://pepy.tech/badge/bavard-ml-utils)](https://pepy.tech/project/bavard-ml-utils)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bavard-ml-utils)](https://pypi.org/project/bavard-ml-utils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A package of common code and utilities for machine learning and MLOps. Includes classes and methods for:

1. ML model serialization/deserialization
2. Google Cloud Storage IO operations
3. Converting a ML model into a runnable web service
4. Common ML model evaluation utilities
5. Common data structures/models used across the Bavard AI organization
6. ML model artifact persistence and version management
7. And more

This package maintains common data structures used across our organization. They can all be found in the `bavard_ml_utils.types` sub-package, and are all [Pydantic](https://pydantic-docs.helpmanual.io/) data models. For example the `bavard_ml_utils.types.agent.AgentConfig` class represents a chatbot's configuration and training data, and is used heavily across Bavard.

API docs for this package can be found [here](https://docs-bavard-ml-utils.web.app/).

## Getting Started

To begin using the package, use your favorite package manager to install it from PyPi. For example, using pip:

```
pip install bavard-ml-utils
```

Some of the features in this repo require more heavy weight dependencies, like Google Cloud Platform related utilities, or utilities specific to machine-learning. If you try to import those features, they will tell you if you do not have the correct package extra installed. For example, many of the features in the `bavard_ml_utils.gcp` sub-package require the `gcp` extra. To install `bavard-ml-utils` with that extra:

```
pip install bavard-ml-utils[gcp]
```

You can then begin using any package features that require GCP dependencies.

## Developing Locally

Before making any new commits or pull requests, please complete these steps.

1. Install the Poetry package manager for Python if you do not already have it. Installation instructions can be found [here](https://python-poetry.org/docs/#installation).
2. Clone the project:
   ```
   git clone https://github.com/bavard-ai/bavard-ml-utils.git
   cd bavard-ml-utils
   ```
3. Install the dependencies, including all dev dependencies and extras:
   ```
   poetry install --extras "gcp ml"
   ```
4. Install the pre-commit hooks, so they will run before each local commit. This includes linting, auto-formatting, and import sorting:
   ```
   pre-commit install
   ```

## Testing Locally

With Docker and docker-compose installed, run this script from the project root:

```
./scripts/lint-and-test-package.sh
```

## Releasing The Package

Releasing the package is automatically handled by CI, but three steps must be taken to trigger a successful release:

1. Use Poetry's [`version` command](https://python-poetry.org/docs/cli/#version) to bump the package's version.
2. Commit and tag the repo with the exact same version the package was bumped to, e.g. `1.0.0` (note there is no preceding `v`.)
3. Push the commit and tag to remote. These can be done together using: `git push --atomic origin <branch name> <tag>`

CI will then build release the package to pypi with that version once the commit and tag are pushed.
