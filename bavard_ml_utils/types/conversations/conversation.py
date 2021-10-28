import typing as t

from pydantic import BaseModel

from bavard_ml_utils.ml.dataset import LabeledDataset
from bavard_ml_utils.types.conversations.actions import Actor, AgentAction
from bavard_ml_utils.types.conversations.dialogue_turns import DialogueTurn
from bavard_ml_utils.types.nlu import NLUExample, NLUExampleDataset


class Conversation(BaseModel):
    id: t.Optional[str]
    agentId: t.Optional[str]
    turns: t.List[DialogueTurn]

    @property
    def num_agent_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.actor == Actor.AGENT)

    @property
    def is_last_turn_user(self) -> bool:
        return False if len(self.turns) == 0 else self.turns[-1].actor == Actor.USER

    @property
    def is_last_turn_agent(self) -> bool:
        return False if len(self.turns) == 0 else self.turns[-1].actor == Actor.AGENT

    @property
    def intents_used(self):
        return set(
            turn.userAction.intent
            for turn in self.turns
            if turn.actor == Actor.USER and turn.userAction.intent is not None
        )

    @property
    def actions_used(self):
        return set(turn.agentAction.name for turn in self.turns if turn.actor == Actor.AGENT)

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
        if self.is_last_turn_user:
            last_turn = self.turns[-1]
            if last_turn.userAction.type == "UTTERANCE_ACTION":
                if last_turn.userAction.translatedUtterance is not None:
                    return last_turn.userAction.translatedUtterance
                return last_turn.userAction.utterance
        return ""


class ConversationDataset(LabeledDataset[Conversation]):
    def get_label(self, item: Conversation) -> str:
        """
        Returns the name of the final agent action in ``item`` (a :class:`Conversation`), and ensures the last turn
        in the conversation is in fact an agent turn.
        """
        if not item.is_last_turn_agent:
            raise AssertionError("conversations in a ConversationDataset must have an agent action as final last turn.")
        return item.turns[-1].agentAction.name

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

    def turns(self, actor: t.Optional[Actor] = None) -> t.Iterable[DialogueTurn]:
        """Iterates over all turns of all conversations in the dataset, having ``actor``, if supplied."""
        check_actor = actor is not None
        for conv in self:
            for turn in conv.turns:
                if check_actor:
                    if turn.actor == actor:
                        yield turn
                else:
                    yield turn

    def unique_intents(self) -> t.Set[str]:
        intents = set()
        for turn in self.turns(Actor.USER):
            if turn.userAction.intent is not None:
                intents.add(turn.userAction.intent)
        return intents

    def unique_actions(self) -> t.Dict[str, AgentAction]:
        """Returns a mapping of unique action names, each to an example of that action found in the dataset."""
        actions = {}
        for turn in self.turns(Actor.AGENT):
            actions[turn.agentAction.name] = turn.agentAction
        return actions

    def unique_tag_types(self) -> t.Set[str]:
        tag_types = set()
        for turn in self.turns(Actor.USER):
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
        for turn in self.turns(Actor.USER):
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
