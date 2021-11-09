import typing as t

from pydantic import BaseModel

from bavard_ml_utils.ml.dataset import LabeledDataset
from bavard_ml_utils.types.conversations.actions import Actor, AgentAction
from bavard_ml_utils.types.conversations.dialogue_turns import AgentDialogueTurn, DialogueTurn, UserDialogueTurn
from bavard_ml_utils.types.nlu import NLUExample, NLUExampleDataset


class Conversation(BaseModel):
    id: t.Optional[str]
    agentId: t.Optional[str]
    turns: t.List[DialogueTurn]

    @property
    def num_agent_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.actor == Actor.AGENT)

    @property
    def last_turn(self) -> t.Optional[DialogueTurn]:
        return None if len(self.turns) == 0 else self.turns[-1]

    @property
    def intents_used(self):
        intents = set(
            turn.userAction.intent
            for turn in self.turns
            if turn.actor == Actor.USER and turn.userAction.intent is not None
        )
        intents.discard("")
        intents.discard(None)
        return intents

    @property
    def actions_used(self):
        actions = set(turn.agentAction.name for turn in self.turns if turn.actor == Actor.AGENT)
        actions.discard("")
        return actions

    def expand(self) -> t.List["Conversation"]:
        """Turns this conversation into a list of its partial conversations that each end with an agent action."""
        cls = self.__class__
        convs = []
        for i in range(len(self)):
            if self.turns[i].actor == Actor.AGENT:
                convs.append(cls(turns=self.turns[: i + 1]))
        return convs

    def __len__(self):
        return len(self.turns)

    def get_final_user_utterance(self) -> str:
        """
        Retrieves the user utterance from the most recent
        turn of the conversation. If the most recent turn is not
        a user turn or doesn't have an utterance, returns the empty string.
        """
        last_turn = self.last_turn
        if isinstance(last_turn, UserDialogueTurn) and last_turn.userAction.type == "UTTERANCE_ACTION":
            return last_turn.userAction.translatedUtterance or last_turn.userAction.utterance or ""
        return ""


class ConversationDataset(LabeledDataset[Conversation]):
    def get_label(self, item: Conversation) -> str:
        """
        Returns the name of the final agent action in ``item`` (a :class:`Conversation`), and ensures the last turn
        in the conversation is in fact an agent turn.
        """
        if not isinstance(item.last_turn, AgentDialogueTurn):
            raise AssertionError("conversations in a ConversationDataset must have an agent action as final last turn.")
        return item.last_turn.agentAction.name

    @classmethod
    def from_conversations(cls, convs: t.List[Conversation], expand=True) -> "ConversationDataset":
        """
        Safely builds a dataset from Conversations which may or may not have agent actions as the final turn.
        Expands each conversation in ``convs`` into as many possible conversations as possible, under the constraint
        that each conversation end with an agent action. If ``expand==False``, expansion is bypassed, and the regular
        conversations are used.
        """
        result = []
        for conv in convs:
            if expand:
                result += conv.expand()
            else:
                result.append(conv)
        return cls(result)

    def turns(self) -> t.Iterable[DialogueTurn]:
        """Iterates over all turns of all conversations in the dataset."""
        for conv in self:
            for turn in conv.turns:
                yield turn

    def user_turns(self) -> t.Iterable[UserDialogueTurn]:
        for turn in self.turns():
            if isinstance(turn, UserDialogueTurn):
                yield turn

    def agent_turns(self) -> t.Iterable[AgentDialogueTurn]:
        for turn in self.turns():
            if isinstance(turn, AgentDialogueTurn):
                yield turn

    def unique_intents(self) -> t.Set[str]:
        intents = set()
        for turn in self.user_turns():
            if turn.userAction.intent is not None:
                intents.add(turn.userAction.intent)
        return intents

    def unique_actions(self) -> t.Dict[str, AgentAction]:
        """Returns a mapping of unique action names, each to an example of that action found in the dataset."""
        actions = {}
        for turn in self.agent_turns():
            actions[turn.agentAction.name] = turn.agentAction
        return actions

    def unique_tag_types(self) -> t.Set[str]:
        tag_types = set()
        for turn in self.user_turns():
            if turn.userAction.tags is not None:
                for tag in turn.userAction.tags:
                    tag_types.add(tag.tagType)
        return tag_types

    def unique_slots(self) -> t.Set[str]:
        slots = set()
        for turn in self.turns():
            if turn.state is not None:
                for slot_name in turn.state.slotValues:
                    slots.add(slot_name)
        return slots

    def to_nlu_dataset(self):
        examples = []
        for turn in self.user_turns():
            action = turn.userAction
            if action.utterance is not None:
                if action.ood:
                    examples.append(NLUExample(text=action.utterance, isOOD=True))
                elif action.intent is not None:
                    examples.append(NLUExample(text=action.utterance, intent=action.intent))
        return NLUExampleDataset(examples)

    def make_validation_pairs(self) -> t.Tuple["ConversationDataset", t.List[str]]:
        """
        Takes all the conversations and returns each one having all its turns but the
        last agent action. Also returns a tuple of those last agent actions.

        Returns
        -------
        tuple of (ConversationDataset, str)
            The first list is the list of conversations. The second
            is the list of the names of the next actions that should
            be taken, given the conversations; one action per conversation.
        """
        cls = self.__class__
        val_convs = []
        next_actions = []
        for conv in self:
            val_convs.append(Conversation(turns=conv.turns[:-1]))
            next_actions.append(self.get_label(conv))
        return cls(val_convs), next_actions
