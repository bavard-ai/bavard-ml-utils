from unittest import TestCase

from fastapi.encoders import jsonable_encoder

from bavard_ml_common.types.conversations.actions import UserAction, AgentAction, TagValue
from bavard_ml_common.types.conversations.dialogue_turns import (
    UserDialogueTurn, DialogueState, SlotValue, AgentDialogueTurn
)


class TestDialogueTurns(TestCase):
    def test_user_dialogue_turn_serialization(self):
        turn = UserDialogueTurn(
            state=DialogueState(
                slotValues=[SlotValue(name="slot3", value="foo"), SlotValue(name="slot1", value="bar")]
            ),
            userAction=UserAction(
                intent="intent2",
                utterance="I utter.",
                tags=[TagValue(tagType="tagtype1", value="value1")],
                type="UTTERANCE_ACTION"
            )
        )
        self.assertEqual(turn, UserDialogueTurn.parse_obj(jsonable_encoder(turn)))

    def test_agent_dialogue_turn_serialization(self):
        turn = AgentDialogueTurn(
            state=DialogueState(slotValues=[SlotValue(name="slot2", value="foo")]),
            agentAction=AgentAction(name="action2", utterance="I utter also.", type="UTTERANCE_ACTION")
        )
        self.assertEqual(turn, AgentDialogueTurn.parse_obj(jsonable_encoder(turn)))

    def test_can_handle_no_state(self):
        user_turn = UserDialogueTurn.parse_obj(
            {
                "userAction": {
                    "type": "UTTERANCE_ACTION",
                    "utterance": "Ok thank you, that's great to know. What are your prices like? Are they competitive?",
                    "intent": "ask_pricing",
                    "tags": []
                },
                "actor": "USER"
            }
        )
        self.assertEqual(user_turn.state, None)

    def test_can_handle_no_intent(self):
        user_turn = UserDialogueTurn.parse_obj({
            "actor": "USER",
            "userAction": {
                "type": "UTTERANCE_ACTION",
                "utterance": "What are your prices like?",
                "tags": []
            },
            "timestamp": 1607471164994
        })
        self.assertEqual(user_turn.userAction.intent, None)
