#!/bin/bash

set -eo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/.."  # go to project root

# Perform linting

pip3 install pre-commit
pre-commit run check-yaml --all-files
pre-commit run check-json --all-files
pre-commit run black --all-files
pre-commit run flake8 --all-files

# Run the unit tests

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
# Needed for emulators to work, see:
# https://github.com/openssl/openssl/issues/5845#issuecomment-378601109
export LD_LIBRARY_PATH=/usr/local/lib

cd dockerfiles

docker-compose -f docker-compose.yml build --progress=plain

docker-compose -f docker-compose.yml up \
  --abort-on-container-exit \
  --exit-code-from test

docker-compose -f docker-compose.yml rm --force
