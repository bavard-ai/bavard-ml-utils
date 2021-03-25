import typing as t
from collections import defaultdict
from itertools import chain

from sklearn.utils import resample

from bavard_ml_common.ml.utils import make_stratified_folds
from bavard_ml_common.types.conversations.actions import Actor
from bavard_ml_common.types.conversations.conversation import Conversation


def make_validation_pairs(convs: t.List[Conversation]) -> t.Tuple[t.List[Conversation], t.List[str]]:
    """
    Takes all the conversations in `convs` and expands them into
    many conversations, with all conversations ending with a user
    action.

    Returns
    -------
    tuple of lists
        The first list is the list of conversations. The second
        is the list of the names of the next actions that should
        be taken, given the conversations; one action per conversation.
    """
    all_convs = []
    all_next_actions = []
    for conv in convs:
        convs, next_actions = conv.make_validation_pairs()
        all_convs += convs
        all_next_actions += next_actions
    return all_convs, all_next_actions


def expand(convs: t.List[Conversation]) -> t.List[Conversation]:
    """
    Takes `convs` and expands them into more conversations. Makes as many conversations
    as possible under the constraints that each conversation have the full dialogue from
    the beginning of the conversation forward, and that each conversation end with an agent
    action, making the resulting conversation useful for training dialogue models to decide
    which agent action to take, given a conversation history.
    """
    all_convs: t.List[Conversation] = []
    for conv in convs:
        all_convs += conv.expand()
    return all_convs


def balance(convs: t.List[Conversation], seed: int = 0) -> t.List[Conversation]:
    """
    Partitions the conversations by their final agent action,
    then upsamples till each action has equal representation.
    """
    convs_by_action = defaultdict(list)
    for conv in convs:
        convs_by_action[conv.turns[-1].agentAction.name].append(conv)
    n_majority = max(len(examples) for examples in convs_by_action.values())
    return list(
        chain.from_iterable(
            resample(convs, replace=True, n_samples=n_majority, random_state=seed)
            for convs in convs_by_action.values()
        )
    )


def get_action_distribution(convs: t.List[Conversation]) -> dict:
    """
    Counts the number of each type of action present in `agent`'s training
    conversations.
    """
    counts = defaultdict(int)
    for conv in convs:
        for turn in conv.turns:
            if turn.actor == Actor.AGENT:
                counts[turn.agentAction.name] += 1
    return counts


def to_folds(
    convs: t.List[Conversation], nfolds: int, *, shuffle: bool = True, seed: int = 0
) -> t.Tuple[t.List[Conversation]]:
    """Splits `convs` into `nfolds` random subsets, stratified by the final agent turn's action label.
    """
    action_labels = [c.turns[-1].agentAction.name for c in convs]
    return make_stratified_folds(convs, action_labels, nfolds, shuffle, seed)
