import typing as t

from pydantic import BaseModel

from bavard_ml_common.types.conversations.actions import Actor
from bavard_ml_common.types.conversations.dialogue_turns import DialogueTurn


class Conversation(BaseModel):
    turns: t.List[DialogueTurn]

    @property
    def num_agent_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.actor == Actor.AGENT)

    @property
    def is_last_turn_user(self) -> bool:
        return False if len(self.turns) == 0 else self.turns[-1].actor == Actor.USER

    def make_validation_pairs(self) -> t.Tuple[t.List["Conversation"], t.List[str]]:
        """
        Expands this conversation into as many conversations as possible,
        under the constraint that each conversation end with a user action
        and have an agent action following it.
        """
        cls = self.__class__
        val_convs = []
        next_actions = []
        for conv in self.expand():
            val_convs.append(cls(turns=conv.turns[:-1]))
            next_actions.append(conv.turns[-1].agentAction.name)
        return val_convs, next_actions

    def expand(self) -> t.List["Conversation"]:
        cls = self.__class__
        convs = []
        for i in range(len(self)):
            if self.turns[i].actor == Actor.AGENT:
                convs.append(cls(turns=self.turns[:i + 1]))
        return convs

    def __len__(self):
        return len(self.turns)

    def get_final_user_utterance(self) -> str:
        """
        Retrieves the user utterance from the most recent
        turn of the conversation. If the most recent turn is not
        a user turn or doesn't have an utterance, returns the empty string.
        """
        if self.is_last_turn_user:
            last_turn = self.turns[-1]
            if last_turn.userAction.type == "UTTERANCE_ACTION":
                if last_turn.userAction.translatedUtterance is not None:
                    return last_turn.userAction.translatedUtterance
                return last_turn.userAction.utterance
        return ""


class TrainingConversation(BaseModel):
    conversation: Conversation
