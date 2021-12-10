#!/bin/bash

set -eo pipefail

# Make sure the dependent services are ready before executing the tests.
while ! nc -z gcs_emulator 4443; do sleep 1; done
while ! nc -z pubsub_emulator 4442; do sleep 1; done
while ! nc -z firestore_emulator 8081; do sleep 1; done
while ! nc -z localstack 4566; do sleep 1; done

echo "running the unit tests"
python -m unittest --verbose
