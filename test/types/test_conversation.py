from unittest import TestCase

from fastapi.encoders import jsonable_encoder

from bavard_ml_common.types.conversations.actions import Actor
from bavard_ml_common.types.conversations.conversation import Conversation
from test.utils import load_json_file


class TestConversation(TestCase):
    def setUp(self):
        self.raw_conv = load_json_file("test/data/conversations/last-turn-user.json")

    def test_serialization(self):
        conv = Conversation.parse_obj(self.raw_conv)
        re_raw = jsonable_encoder(conv)
        # The object should be fully reconstructable.
        self.assertEqual(conv, Conversation.parse_obj(re_raw))

    def test_make_validation_pairs(self):
        convs, next_actions = Conversation.parse_obj(self.raw_conv).make_validation_pairs()
        self.assertEqual(len(convs), len(next_actions))

        # Each validation conversation should end with a user action
        for conv in convs:
            self.assertEqual(conv.turns[-1].actor, Actor.USER)
