#!/bin/bash

# Builds the docs site for this package. Assumes all the package's dependencies are already installed.

set -eo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/../docs"

# Scan all source code and generate API documentation from the doc strings.
sphinx-apidoc --force --no-toc --module-first --separate --ext-viewcode --output-dir ./src ../bavard_ml_utils
# Create a static web page version of the documentation.
make html
