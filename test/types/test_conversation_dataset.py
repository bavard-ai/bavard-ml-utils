from unittest import TestCase

from bavard_ml_common.types.conversations.actions import Actor
from bavard_ml_common.types.conversations.conversation import Conversation, ConversationDataset


class TestConversationDataset(TestCase):
    def setUp(self):
        self.dataset = ConversationDataset.from_conversations([
            Conversation.parse_file("test/data/conversations/last-turn-user.json")
        ])

    def test_make_validation_pairs(self):
        convs, next_actions = self.dataset.make_validation_pairs()
        self.assertEqual(len(convs), len(next_actions))

        # Each validation conversation should end with a user action
        for conv in convs:
            self.assertEqual(conv.turns[-1].actor, Actor.USER)
