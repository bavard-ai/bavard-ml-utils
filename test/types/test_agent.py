from unittest import TestCase

from fastapi.encoders import jsonable_encoder

from bavard_ml_common.types.agent import Agent
from bavard_ml_common.types.conversations.actions import UserAction
from bavard_ml_common.types.conversations.conversation import TrainingConversation, Conversation
from bavard_ml_common.types.conversations.dialogue_turns import UserDialogueTurn
from test.utils import load_json_file


class TestAgent(TestCase):
    def test_filters_bad_training_conversations(self):
        agent = Agent.parse_file("test/data/agents/test-agent.json")

        # Empty conversations should be filtered out.
        num_convs = len(agent.trainingConversations)
        agent.trainingConversations.append(TrainingConversation(conversation=Conversation(turns=[])))
        self.assertEqual(len(agent.trainingConversations), num_convs + 1)
        agent = Agent.parse_obj(jsonable_encoder(agent))
        # The added conversation should be filtered out.
        self.assertEqual(len(agent.trainingConversations), num_convs)

        # Conversations with no agent actions should be filtered out.
        agent.trainingConversations.append(
            TrainingConversation(
                conversation=Conversation(turns=[UserDialogueTurn(userAction=UserAction(type="UTTERANCE_ACTION"))])
            )
        )
        self.assertEqual(len(agent.trainingConversations), num_convs + 1)
        agent = Agent.parse_obj(jsonable_encoder(agent))
        # The added conversation should be filtered out.
        self.assertEqual(len(agent.trainingConversations), num_convs)

    def test_adds_nlu_examples_from_training_conversations(self):
        raw_agent = load_json_file("test/data/agents/bavard.json")
        n_nlu_examples = len(raw_agent["nluData"]["examples"])
        # When parsed, the `Agent` object should have more nlu examples than `n_nlu_examples`.
        self.assertGreater(len(Agent.parse_obj(raw_agent).nluData.examples), n_nlu_examples)
