from unittest import TestCase

from fastapi.encoders import jsonable_encoder

from bavard_ml_common.types.agent import AgentConfig, AgentExport
from bavard_ml_common.types.conversations.actions import UserAction
from bavard_ml_common.types.conversations.conversation import Conversation
from bavard_ml_common.types.conversations.dialogue_turns import UserDialogueTurn
from test.utils import load_json_file


class TestAgent(TestCase):
    def test_filters_bad_training_conversations(self):
        agent_config = AgentExport.parse_file("test/data/agents/test-agent.json").config

        # Empty conversations should be filtered out.
        num_convs = len(agent_config.trainingConversations)
        agent_config.trainingConversations.append(Conversation(turns=[]))
        self.assertEqual(len(agent_config.trainingConversations), num_convs + 1)
        agent_config = AgentConfig.parse_obj(jsonable_encoder(agent_config))
        agent_config.filter_no_agent_convs()

        # The added conversation should be filtered out.
        self.assertEqual(len(agent_config.trainingConversations), num_convs)

        # Conversations with no agent actions should be filtered out.
        agent_config.trainingConversations.append(
            Conversation(turns=[UserDialogueTurn(userAction=UserAction(type="UTTERANCE_ACTION"))])
        )
        self.assertEqual(len(agent_config.trainingConversations), num_convs + 1)
        agent_config.filter_no_agent_convs()
        # The added conversation should be filtered out.
        self.assertEqual(len(agent_config.trainingConversations), num_convs)

    def test_adds_nlu_examples_from_training_conversations(self):
        raw_agent = load_json_file("test/data/agents/bavard.json")["config"]
        n_nlu_examples = sum(len(examples) for examples in raw_agent["intentExamples"].values())
        agent = AgentConfig.parse_obj(raw_agent)
        agent.incorporate_training_conversations()
        # The `agent` object should now have more nlu examples than `n_nlu_examples`.
        self.assertGreater(len(list(agent.all_nlu_examples())), n_nlu_examples)

    def test_excludes_examples_with_unknown_labels(self):
        messy_agent = AgentExport.parse_file("test/data/agents/invalid-nlu-examples.json").config
        messy_agent.remove_unknown_intent_examples()

        # Examples with unregistered intents or tag types should be filtered out.
        self.assertEqual(len(list(messy_agent.all_nlu_examples())), 2)
        for ex in messy_agent.all_nlu_examples():
            self.assertIn(ex.intent, messy_agent.intent_names())
            for tag in ex.tags or []:
                self.assertIn(tag.tagType, messy_agent.tag_names())
