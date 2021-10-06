import typing as t
from collections import defaultdict
from itertools import chain

from pydantic import BaseModel

from bavard_ml_utils.types.conversations.actions import Actor
from bavard_ml_utils.types.conversations.conversation import Conversation, ConversationDataset
from bavard_ml_utils.types.nlu import NLUExample, NLUExampleDataset


class Intent(BaseModel):
    name: str


class Slot(BaseModel):
    name: str


class AgentActionDefinition(BaseModel):
    name: str


class AgentConfig(BaseModel):
    """A configuration for a chatbot, including its NLU and dialogue policy training data."""

    name: str
    """The chatbot's name."""

    agentId: t.Optional[str]
    """The chatbot's unique id."""

    actions: t.List[AgentActionDefinition]
    """The unique actions this chatbot can take when interacting with users."""

    language: str = "en"
    """The language the text in this chatbot's training data is written in. Defaults to English."""

    intents: t.List[Intent]
    """The unique intents this chatbot can recognize in the things users say to it."""

    tagTypes: t.List[str]
    """The custom named entities this chatbot can recognize in a user's speech."""

    slots: t.List[Slot]
    """The potential values this chatbot can track and store over the course of a conversation with a user."""

    intentExamples: t.Dict[str, t.List[NLUExample]]
    """
    The training data that can be used to teach this chatbot how to recognize the :attr:`intents` that it has defined.
    """

    intentOODExamples: t.List[str] = []
    """
    Optional examples of utterances that a user might tell this chatbot which do not belong to any of its known
    :attr:`intents`.
    """

    trainingConversations: t.List[Conversation]
    """
    Conversations which can be used to train the chatbot what next action to take, out of its possible :attr:`actions`,
    given a conversation's state so far.
    """

    def to_nlu_dataset(self, include_ood=False) -> NLUExampleDataset:
        """
        Converts the NLU examples in this config's :attr:`intentExamples`, :attr:`intentOODExamples`, and
        :attr:`trainingConversations` into an NLU dataset, for an NLU machine learning model to train on.
        """
        copy = self.copy(deep=True)
        copy.clean()
        copy.incorporate_training_conversations()
        return NLUExampleDataset(copy.all_nlu_examples(include_ood))

    def to_conversation_dataset(self, expand=True) -> ConversationDataset:
        """
        Converts this config's :attr:`trainingConversations` into a conversation dataset, for a dialogue poliy
        machine learning model to train on.
        """
        copy = self.copy(deep=True)
        copy.clean()
        return ConversationDataset.from_conversations(copy.trainingConversations, expand)

    def all_nlu_examples(self, include_ood=False) -> t.List[NLUExample]:
        examples = list(chain.from_iterable(self.intentExamples.values()))
        if include_ood:
            examples += [NLUExample(text=text, isOOD=True) for text in self.intentOODExamples]
        return examples

    def intent_names(self) -> t.Set[str]:
        return set(intent.name for intent in self.intents)

    def tag_names(self) -> t.Set[str]:
        return set(self.tagTypes)

    def clean(self):
        """Filters out invalid and unusable training data from the config."""
        self.remove_unknown_intent_examples()
        self.filter_no_agent_convs()

    def remove_unknown_intent_examples(self):
        """
        Filters out all of the chatbot's NLU examples whose intents are not explicitly defined in its :attr:`intents`,
        or whose tags are not explicitly defined in its :attr:`tagTypes`.
        """
        filtered = defaultdict(list)
        valid_intents = self.intent_names()
        valid_tag_types = self.tag_names()

        for example in self.all_nlu_examples():
            if example.intent not in valid_intents:
                continue
            if any(tag.tagType not in valid_tag_types for tag in example.tags or []):
                continue
            filtered[example.intent].append(example)

        self.intentExamples = filtered

    def filter_no_agent_convs(self):
        """Removes all training conversations from this chatbot config which have no agent turns."""
        # We only include training conversations that have at least one agent action.
        self.trainingConversations = [c for c in self.trainingConversations if c.num_agent_turns > 0]

    def incorporate_training_conversations(self):
        """Adds to this agent's NLU examples any valid examples present in its training conversations."""
        valid_intents = self.intent_names()
        for conv in self.trainingConversations:
            for turn in conv.turns:
                if turn.actor == Actor.USER:
                    if turn.userAction.intent in valid_intents and turn.userAction.utterance:
                        self.intentExamples[turn.userAction.intent].append(
                            NLUExample(intent=turn.userAction.intent, text=turn.userAction.utterance, tags=[])
                        )

    @classmethod
    def from_conversation_dataset(cls, convs: ConversationDataset, name: str, **kwargs):
        """Builds an agent config from a conversation dataset, including all its conversations and NLU examples."""
        nlu_examples = convs.to_nlu_dataset()
        examples_by_intent = defaultdict(list)
        for ex in nlu_examples:
            if not ex.isOOD and ex.intent is not None:
                examples_by_intent[ex.intent].append(ex)
        return cls(
            name=name,
            actions=[AgentActionDefinition(name=action) for action in convs.unique_actions()],
            intents=[Intent(name=intent) for intent in convs.unique_intents()],
            tagTypes=list(convs.unique_tag_types()),
            slots=[Slot(name=slot) for slot in convs.unique_slots()],
            intentOODExamples=list({ex.text for ex in nlu_examples if ex.isOOD}),
            intentExamples=examples_by_intent,
            trainingConversations=list(convs),
            **kwargs
        )


class AgentExport(BaseModel):
    """
    Wrapper data structure for an :class:`AgentConfig` object. Often, Bavard agent configs are saved as JSON files in
    this format.
    """

    config: AgentConfig
