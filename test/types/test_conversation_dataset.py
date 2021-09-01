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

    def test_properties(self):
        self.assertSetEqual(
            self.dataset.unique_slots(),
            {
                "hotel-bookday",
                "hotel-bookpeople",
                "hotel-bookstay",
                "hotel-internet",
                "hotel-name",
                "hotel-parking",
                "hotel-pricerange",
                "hotel-type",
                "nomatches"
            }
        )
        self.assertSetEqual(self.dataset.unique_intents(), {"Hotel-Inform", "Hotel-Request"})
        self.assertSetEqual(self.dataset.unique_tag_types(), {"pricerange", "bookstay", "bookday", "bookpeople"})
        self.assertSetEqual(
            self.dataset.unique_labels(), {"Hotel-Inform", "Booking-Request", "Booking-NoBook", "general-reqmore"}
        )
