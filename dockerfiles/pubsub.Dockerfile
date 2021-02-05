FROM google/cloud-sdk:alpine

# Install Java 8 JRE (required for Pubsub emulator).
RUN apk add --update --no-cache openjdk8-jre

# Install Pubsub Emulator.
RUN gcloud components install pubsub-emulator beta --quiet

ENV PORT 4442
EXPOSE "$PORT"

ENV PUBSUB_PROJECT_ID "test"

ENTRYPOINT gcloud config set project "${PUBSUB_PROJECT_ID}" \
    && gcloud beta emulators pubsub start --host-port="0.0.0.0:${PORT}"
