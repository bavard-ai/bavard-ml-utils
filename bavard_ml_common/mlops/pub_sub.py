import typing as t
import json

from google.cloud.pubsub_v1 import PublisherClient


class PubSub:
    def __init__(self, project_id: str) -> None:
        self._project_id = project_id
        self._pub_client = PublisherClient()

    def publish(self, topic_id: str, msg: t.Any) -> None:
        topic_path = self._pub_client.topic_path(self._project_id, topic_id)
        data_bytes = json.dumps(msg).encode("utf-8")
        self._pub_client.publish(topic_path, data_bytes)

    def notify_model_ready(
        self, agent_uname: str, publish_id: str, model_name: str, model_version: str
    ) -> None:
        self.publish(
            "chatbot-service-training-jobs",
            {
                "EVENT_TYPE": "MODEL_READY",
                "AGENT_UNAME": agent_uname,
                "PUBLISH_ID": publish_id,
                "MODEL_NAME": model_name,
                "MODEL_VERSION": model_version,
            },
        )
