#!/bin/bash

set -eo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..  # go to project root

pkg_version="$(poetry version --short)"
if [pkg_version -ne $CIRCLE_TAG]; then
  echo "Git tag: $CIRCLE_TAG does not match the version of this package: $pkg_version"
fi

poetry publish --build --username __token__ --password $PYPI_PASSWORD
