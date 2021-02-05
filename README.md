# bavard-ml-common

A package of common code and utilities for machine learning and MLOps. Includes classes and methods for:

1. ML model serialization/deserialization
2. Google Cloud Storage IO operations
3. Converting an ML model into a runnable web service
4. Common ML model evaluation utilities
5. And more

## Testing Locally

With Docker and docker-compose installed, run:

```
./scripts/lint-and-test-package.sh
```

## Releasing The Package

Releasing the package is automatically handled by CI, but three steps must be taken to trigger a successful release:

1. Increment the `VERSION` variable in `setup.py` to the new desired version (e.g. `VERSION="1.1.1"`)
2. Commit and tag the repo with the **exact same** value you populated the `VERSION` variable with (e.g. `git tag 1.1.1`)
3. Push the commit and tag to remote. These can be done together using: `git push --atomic origin <branch name> <tag>`

CI will then release the package to pypi with that version once the commit and tag are pushed.
