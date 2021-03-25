from unittest import TestCase

from bavard_ml_common.types.conversations.actions import Actor
from bavard_ml_common.types.conversations.conversation import Conversation


class TestConversation(TestCase):
    def setUp(self):
        self.conv = Conversation.parse_file("test/data/conversations/last-turn-user.json")

    def test_make_validation_pairs(self):
        convs, next_actions = Conversation.parse_obj(self.conv).make_validation_pairs()
        self.assertEqual(len(convs), len(next_actions))

        # Each validation conversation should end with a user action
        for conv in convs:
            self.assertEqual(conv.turns[-1].actor, Actor.USER)
