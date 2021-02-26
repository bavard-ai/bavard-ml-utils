import typing as t
from collections import defaultdict
from itertools import chain

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from pydantic import BaseModel, validator

from bavard_ml_common.ml.utils import make_stratified_folds
from bavard_ml_common.types.conversations.conversation import TrainingConversation, Conversation
from bavard_ml_common.types.nlu import NLUTrainingData
from bavard_ml_common.types.conversations.actions import Actor


class Intent(BaseModel):
    name: str


class Slot(BaseModel):
    name: str


class AgentActionDefinition(BaseModel):
    name: str


class AgentConfig(BaseModel):
    uname: str
    actions: t.List[AgentActionDefinition]
    intents: t.List[Intent]
    tagTypes: t.List[str]
    slots: t.List[Slot]
    language: str = "en"


class Agent(BaseModel):
    config: AgentConfig
    nluData: NLUTrainingData
    trainingConversations: t.List[TrainingConversation]

    def __init__(self, *, config: dict, nluData: dict, trainingConversations: t.List[dict], **data):
        """
        Use a custom constructor to automatically add all NLU examples present in this agent's
        training conversations to it's official NLU examples, so those can be used for NLU training
        as well.
        """
        nluData = NLUTrainingData.parse_obj(nluData)
        nluData.incorporate_training_conversations([TrainingConversation.parse_obj(c) for c in trainingConversations])
        super().__init__(config=config, nluData=nluData.dict(), trainingConversations=trainingConversations, **data)

    @validator("trainingConversations")
    def only_keep_valid_training_convs(cls, v):
        # We only include training conversations that have at least one agent action.
        return [c for c in v if c.conversation.num_agent_turns > 0]

    def split(self, split_ratio: float, shuffle: bool = True, seed: int = 0) -> tuple:
        """Splits `agent` into two different training/test conversation sets."""
        convs = [x.conversation for x in self.trainingConversations]
        convs_a, convs_b = train_test_split(
            convs,
            test_size=split_ratio,
            random_state=seed,
            shuffle=shuffle,
        )
        return (
            self.build_from_convs(convs_a, f"{self.config.uname}-a"),
            self.build_from_convs(convs_b, f"{self.config.uname}-b"),
        )

    def make_validation_pairs(self) -> t.Tuple[t.List[Conversation], t.List[str]]:
        """
        Takes all the conversations in `agent` and expands them into
        many conversations, with all conversations ending with a user
        action.

        Returns
        -------
        tuple of lists
            The first list is the list of raw conversations. The second
            is the list of the names of the next actions that should
            be taken, given the conversations; one action per conversation.
        """
        all_convs = []
        all_next_actions = []
        for conv in self.trainingConversations:
            convs, next_actions = conv.conversation.make_validation_pairs()
            all_convs += convs
            all_next_actions += next_actions
        return all_convs, all_next_actions

    def expand(self, *, balance: bool = False, seed: int = 0) -> "Agent":
        """
        Takes `agent` and makes a new agent with all the old agent's training conversations
        expanded into more conversations. Makes as many conversations as possible under the
        constraints that each conversation have the full dialogue from the beginning of the
        conversation forward, and that each conversation end with an agent action, making the
        resulting conversation useful for training dialogue models to decide which agent action
        to take, given a conversation history.
        """
        all_convs: t.List[Conversation] = []
        for conv in self.trainingConversations:
            all_convs += conv.conversation.expand()

        if balance:
            # Partition the conversations by their final agent action,
            # then upsample till each action has equal representation.
            convs_by_action = defaultdict(list)
            for conv in all_convs:
                convs_by_action[conv.turns[-1].agentAction.name].append(conv)
            n_majority = max(len(examples) for examples in convs_by_action.values())
            all_convs = list(
                chain.from_iterable(
                    resample(convs, replace=True, n_samples=n_majority, random_state=seed)
                    for convs in convs_by_action.values()
                )
            )

        return self.build_from_convs(all_convs, self.config.uname)

    @classmethod
    def build_from_convs(cls, conversations: t.List[Conversation], uname: str) -> "Agent":
        """Builds an agent from conversations only.
        """

        actions = set()
        intents = set()
        tag_types = set()
        slot_names = set()

        for conv in conversations:
            for turn in conv.turns:
                if turn.actor == Actor.AGENT:
                    actions.add(turn.agentAction.name)
                elif turn.actor == Actor.USER:
                    action_body = turn.userAction
                    intents.add(action_body.intent)
                    if action_body.tags is not None:
                        tag_types.update(tag.tagType for tag in action_body.tags)
                    if turn.state is not None and turn.state.slotValues is not None:
                        slot_names.update(sv.name for sv in turn.state.slotValues)

        return cls(
            config=AgentConfig(
                uname=uname,
                actions=[AgentActionDefinition(name=action) for action in actions],
                intents=[Intent(name=intent) for intent in intents],
                tagTypes=list(tag_types),
                slots=[Slot(name=slot) for slot in slot_names]
            ),
            trainingConversations=[TrainingConversation(conversation=conv) for conv in conversations],
            nluData=NLUTrainingData(intents=[], examples=[], tagTypes=[])
        )

    def get_action_distribution(self) -> dict:
        """
        Counts the number of each type of action present in `agent`'s training
        conversations.
        """
        counts = defaultdict(int)
        for tc in self.trainingConversations:
            for turn in tc.conversation.turns:
                if turn.actor == Actor.AGENT:
                    counts[turn.agentAction.name] += 1
        return counts

    @classmethod
    def concat(cls, *agents: "Agent") -> "Agent":
        """
        Takes the concatenation of multiple agent objects, returning a single agent
        object with all the training conversations included.
        """
        convs = list(chain.from_iterable(agent.trainingConversations for agent in agents))
        convs = [c.conversation for c in convs]
        unames = [a.config.uname for a in agents]
        return cls.build_from_convs(convs, "+".join(unames))

    def to_folds(self, nfolds: int, *, shuffle: bool = True, seed: int = 0) -> t.Tuple["Agent"]:
        """
        Splits an expanded version of `agent` into `nfolds` random subsets, stratified by action label.
        The stratification will only be accurate when the agents returned by this method are consumed in
        `predict_single` mode.
        """
        convs = self.expand(balance=False, seed=seed).trainingConversations
        convs = [c.conversation for c in convs]
        action_labels = [c.turns[-1].agentAction.name for c in convs]
        folds = make_stratified_folds(convs, action_labels, nfolds, shuffle, seed)
        return tuple(self.build_from_convs(convs, f"{self.config.uname}-{i}") for i, convs in enumerate(folds))
