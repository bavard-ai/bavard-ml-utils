import typing as t
from collections import defaultdict
from itertools import chain

from pydantic import BaseModel

from bavard_ml_common.types.conversations.actions import Actor
from bavard_ml_common.types.conversations.conversation import Conversation, ConversationDataset
from bavard_ml_common.types.nlu import NLUExample, NLUExampleDataset


class Intent(BaseModel):
    name: str


class Slot(BaseModel):
    name: str


class AgentActionDefinition(BaseModel):
    name: str


class AgentConfig(BaseModel):
    """A subset of the `IAgentConfig` interface in our `agent-config` repo.
    """
    name: str
    agentId: t.Optional[str]
    actions: t.List[AgentActionDefinition]
    language: str = "en"
    intents: t.List[Intent]
    tagTypes: t.List[str]
    slots: t.List[Slot]
    intentExamples: t.Dict[str, t.List[NLUExample]]
    intentOODExamples: t.List[str] = []
    trainingConversations: t.List[Conversation]

    def to_nlu_dataset(self, include_ood=False) -> NLUExampleDataset:
        copy = self.copy(deep=True)
        copy.clean()
        copy.incorporate_training_conversations()
        return NLUExampleDataset(copy.all_nlu_examples(include_ood))

    def to_conversation_dataset(self) -> ConversationDataset:
        copy = self.copy(deep=True)
        copy.clean()
        return ConversationDataset.from_conversations(copy.trainingConversations)

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
        """Filters out invalid and unusable training data.
        """
        self.remove_unknown_intent_examples()
        self.filter_no_agent_convs()

    def remove_unknown_intent_examples(self):
        """Filters out any example in `examples` whose intent is not in `intents`, or whose tags are not in `tag_types`.
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
        # We only include training conversations that have at least one agent action.
        self.trainingConversations = [c for c in self.trainingConversations if c.num_agent_turns > 0]

    def incorporate_training_conversations(self):
        """Adds to this agent's NLU examples any valid examples present in its training conversations.
        """
        valid_intents = self.intent_names()
        for conv in self.trainingConversations:
            for turn in conv.turns:
                if turn.actor == Actor.USER:
                    if turn.userAction.intent in valid_intents and turn.userAction.utterance:
                        self.intentExamples[turn.userAction.intent].append(
                            NLUExample(intent=turn.userAction.intent, text=turn.userAction.utterance, tags=[])
                        )


class AgentExport(BaseModel):
    """A subset of the `IAgentExport` interface in our `chatbot-service` repo.
    """
    config: AgentConfig
