FROM google/cloud-sdk:alpine

# Install Java 8 JRE (required for Firestore emulator).
RUN apk add --update --no-cache openjdk8-jre

# Install Firestore Emulator.
RUN gcloud components install cloud-firestore-emulator beta --quiet

ENV PORT 8080
EXPOSE "$PORT"

ENV FIRESTORE_PROJECT_ID "test"

RUN gcloud config set project "${FIRESTORE_PROJECT_ID}"

ENTRYPOINT gcloud beta emulators firestore start --project=${FIRESTORE_PROJECT_ID} --host-port="0.0.0.0:${PORT}"