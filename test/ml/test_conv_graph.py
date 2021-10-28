from unittest import TestCase

from bavard_ml_utils.ml.conv_graph import ConvGraph
from bavard_ml_utils.types.agent import AgentExport
from bavard_ml_utils.types.conversations.actions import UserAction
from bavard_ml_utils.types.conversations.dialogue_turns import UserDialogueTurn


class TestConvGraph(TestCase):
    def setUp(self):
        self.bavard_data = AgentExport.parse_file("test/data/agents/bavard.json").config.to_conversation_dataset(
            expand=False
        )
        self.bavard_graph = ConvGraph(self.bavard_data)
        self.toy_data = AgentExport.parse_file("test/data/agents/toy.json").config.to_conversation_dataset(expand=False)
        self.toy_graph = ConvGraph(self.toy_data)
        # Based on the training data, there are two acceptable agent responses to the `ask_how_doing` user intent.
        self.last_turns = [
            UserDialogueTurn(
                userAction=UserAction(type="UTTERANCE_ACTION", utterance="how's it going?", intent="ask_how_doing")
            ),
            UserDialogueTurn(
                userAction=UserAction(type="UTTERANCE_ACTION", utterance="how's it going?", intent="ask_how_doing")
            ),
            UserDialogueTurn(
                userAction=UserAction(type="UTTERANCE_ACTION", utterance="what's the weather?", intent="ask_weather")
            ),
        ]

    def test_has_correct_actions(self):
        # The conv graph should only have the actions and intents seen in the training data, and should have *all*
        # of them, because they all occur in one or more dialogue states in the training data.
        valid_actions = set(self.bavard_data.unique_actions()).union(self.bavard_data.unique_intents()).union({None})
        graph_actions = {action for n, action in self.bavard_graph.graph.nodes.data("action")}
        self.assertEqual(graph_actions, valid_actions)

    def test_has_correct_weight(self):
        # The weights of all the edges in the graph should add up to the number of turns in the training data.
        total_weight = sum(weight for u, v, weight in self.bavard_graph.graph.edges.data("weight"))
        num_turns = sum(len(c.turns) for c in self.bavard_data)
        self.assertEqual(total_weight, num_turns)

    def test_soft_accuracy(self):
        acc = self.toy_graph.soft_accuracy(["state_mood_good", "state_mood_bad", "state_weather"], self.last_turns)
        self.assertEqual(acc, 1)
        acc = self.toy_graph.soft_accuracy(["state_mood_good", "state_mood_good", "state_weather"], self.last_turns)
        self.assertEqual(acc, 1)
        acc = self.toy_graph.soft_accuracy(["foo", "state_mood_good", "state_weather"], self.last_turns)
        self.assertAlmostEqual(acc, 2 / 3)
        acc = self.toy_graph.soft_accuracy(["foo", "state_mood_good", "bar"], self.last_turns)
        self.assertAlmostEqual(acc, 1 / 3)

    def test_balanced_soft_accuracy(self):
        acc = self.toy_graph.balanced_soft_accuracy(
            ["state_mood_good", "state_mood_bad", "state_weather"], self.last_turns
        )
        self.assertEqual(acc, 1)
        acc = self.toy_graph.balanced_soft_accuracy(
            ["state_mood_good", "state_mood_good", "state_weather"], self.last_turns
        )
        self.assertEqual(acc, 1)
        acc = self.toy_graph.balanced_soft_accuracy(["foo", "state_mood_good", "state_weather"], self.last_turns)
        self.assertAlmostEqual(acc, (0 / 0.5 + 1 / 1.5 + 1) / 3)
        acc = self.toy_graph.balanced_soft_accuracy(["foo", "state_mood_good", "bar"], self.last_turns)
        self.assertAlmostEqual(acc, (0 / 0.5 + 1 / 1.5 + 0 / 1) / 3)
