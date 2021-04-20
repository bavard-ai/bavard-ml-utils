import os
import typing as t
import json

from google.auth.credentials import AnonymousCredentials
from google.cloud.pubsub_v1 import PublisherClient


class PubSub:
    """A GCP pub-sub helper client that works out of the box with the GCP pus-sub emulator.
    """

    def __init__(self, project_id: str, credentials=None) -> None:
        if os.getenv("PUBSUB_EMULATOR_HOST") is not None:
            # We are in a testing context. Make sure the client's default args
            # work in this emulator scenario.
            if credentials is None:
                credentials = AnonymousCredentials()
        self._project_id = project_id
        self.client = PublisherClient(credentials=credentials)

    def publish(self, topic_id: str, msg: t.Any) -> None:
        topic_path = self.topic_path(topic_id)
        data_bytes = json.dumps(msg).encode("utf-8")
        self.client.publish(topic_path, data_bytes)

    def topic_path(self, topic_id: str) -> str:
        return self.client.topic_path(self._project_id, topic_id)
