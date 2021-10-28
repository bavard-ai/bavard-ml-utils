import typing as t
from collections import defaultdict
from itertools import chain

from loguru import logger
from pydantic import BaseModel

from bavard_ml_utils.types.conversations.actions import Actor, AgentAction
from bavard_ml_utils.types.conversations.conversation import Conversation, ConversationDataset
from bavard_ml_utils.types.nlu import NLUExample, NLUExampleDataset


class Intent(BaseModel):
    name: str


class Slot(BaseModel):
    name: str


class AgentConfig(BaseModel):
    """A configuration for a chatbot, including its NLU and dialogue policy training data."""

    name: str
    """The chatbot's name."""

    agentId: t.Optional[str]
    """The chatbot's unique id."""

    actions: t.List[AgentAction]
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

    def action_names(self) -> t.Set[str]:
        return set(a.name for a in self.actions)

    def clean(self):
        """Filters out invalid and unusable training data from the config."""
        self.filter_invalid_intent_examples()
        self.filter_no_agent_convs()
        self.filter_invalid_intent_convs()
        self.filter_invalid_action_convs()

    def filter_invalid_intent_examples(self):
        """
        Filters out all of the chatbot's NLU examples whose intents are not explicitly defined in its :attr:`intents`,
        or whose tags are not explicitly defined in its :attr:`tagTypes`.
        """
        filtered = defaultdict(list)
        valid_intents, valid_tag_types = self.intent_names(), self.tag_names()
        invalid_intents, invalid_tag_types = set(), set()
        lost_to_intents, lost_to_tags = 0, 0

        for example in self.all_nlu_examples():
            example_tag_types = set(tag.tagType for tag in example.tags or [])
            if example.intent not in valid_intents:
                invalid_intents.add(example.intent)
                lost_to_intents += 1
                continue
            if not example_tag_types.issubset(valid_tag_types):
                invalid_tag_types.update(example_tag_types - valid_tag_types)
                lost_to_tags += 1
                continue
            filtered[example.intent].append(example)

        self._warn_lost_data("NLU examples", lost_to_intents, "intents", invalid_intents)
        self._warn_lost_data("NLU examples", lost_to_tags, "tag types", invalid_tag_types)
        self.intentExamples = filtered

    def filter_no_agent_convs(self):
        """Removes all training conversations from this chatbot config which have no agent turns."""
        self.trainingConversations = [c for c in self.trainingConversations if c.num_agent_turns > 0]

    def filter_invalid_intent_convs(self):
        """Removes all training conversations that use any intents not defined in the chatbot's :attr:`intents`."""
        valid_intents = set(intent.name for intent in self.intents)
        invalid_intents = set()
        new_convs = []
        num_lost = 0
        for c in self.trainingConversations:
            if c.intents_used.issubset(valid_intents):
                new_convs.append(c)
            else:
                num_lost += 1
                invalid_intents.update(c.intents_used - valid_intents)
        self._warn_lost_data("training conversations", num_lost, "intents", invalid_intents)
        self.trainingConversations = new_convs

    def filter_invalid_action_convs(self):
        """
        Removes all training conversations that use any agent actions not defined in the chatbot's :attr:`actions`.
        """
        valid_actions = set(a.name for a in self.actions)
        invalid_actions = set()
        new_convs = []
        num_lost = 0
        for c in self.trainingConversations:
            if c.actions_used.issubset(valid_actions):
                new_convs.append(c)
            else:
                num_lost += 1
                invalid_actions.update(c.actions_used - valid_actions)
        self._warn_lost_data("training conversations", num_lost, "actions", invalid_actions)
        self.trainingConversations = new_convs

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
            actions=[action.copy(deep=True) for action in convs.unique_actions().values()],
            intents=[Intent(name=intent) for intent in convs.unique_intents()],
            tagTypes=list(convs.unique_tag_types()),
            slots=[Slot(name=slot) for slot in convs.unique_slots()],
            intentOODExamples=list({ex.text for ex in nlu_examples if ex.isOOD}),
            intentExamples=examples_by_intent,
            trainingConversations=list(convs),
            **kwargs,
        )

    def _warn_lost_data(self, lost_type: str, num_lost: int, invalid_type: str, invalid: set):
        if num_lost > 0:
            logger.warning(
                f"There are {len(invalid)} {invalid_type} found in {lost_type} that are not defined "
                f"in the {self.name} config. Because of this, {num_lost} {lost_type} are being filtered out. "
                f"Consider either adding these {invalid_type} to your config or removing use of them from your "
                f"{lost_type}: {invalid}"
            )


class AgentExport(BaseModel):
    """
    Wrapper data structure for an :class:`AgentConfig` object. Often, Bavard agent configs are saved as JSON files in
    this format.
    """

    config: AgentConfig
