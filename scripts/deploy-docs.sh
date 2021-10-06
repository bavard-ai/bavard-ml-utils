#!/bin/bash

# Builds and deploys the docs site for this package.

firebase_token=$1

set -eo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR/../dockerfiles"

image_tag="docs-bavard-ml-utils"

docker build --file Dockerfile --target docs --tag "$image_tag" ..
docker run --env FIREBASE_TOKEN=$firebase_token "$image_tag"
