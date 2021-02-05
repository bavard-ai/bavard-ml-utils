import json
from unittest import TestCase

from google.cloud.pubsub_v1 import SubscriberClient
from google.auth.credentials import AnonymousCredentials

from bavard_ml_common.mlops.pub_sub import PubSub
from test.config import PUBSUB_PROJECT_ID


class TestPubsubClient(TestCase):

    def setUp(self):
        self.recieved_message = False

    def _sub_callback(self, message):
        self.recieved_message = True
        return message.data

    def test_pubsub_client(self):
        publisher = PubSub(PUBSUB_PROJECT_ID)
        subscriber = SubscriberClient(credentials=AnonymousCredentials())

        # Create a topic
        topic = publisher.topic_path("test-topic")
        publisher.client.create_topic(request={"name": topic})

        # Create a subscription
        subscription = subscriber.subscription_path(PUBSUB_PROJECT_ID, "test-subscription")
        subscriber.create_subscription(request={"name": subscription, "topic": topic})

        # Publish to the topic
        publisher.publish("test-topic", "Hello world!")

        # Pull the subscription
        response = subscriber.pull(request={"subscription": subscription, "max_messages": 1})

        # Make sure the publish happened and the subscriber recieved it.
        self.assertEqual(len(response.received_messages), 1)
        message = response.received_messages[0].message.data.decode("utf-8")
        message = json.loads(message)
        self.assertEqual(message, "Hello world!")
