import typing as t
from itertools import chain
from collections import defaultdict, Counter

from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from sklearn.utils import resample
from bavard_ml_common.ml.utils import make_stratified_folds

from bavard_ml_common.types.conversations.actions import Actor
from bavard_ml_common.types.conversations.conversation import TrainingConversation


class TrainingTag(BaseModel):
    """A tag as it appears in NLU training data.
    """
    tagType: str
    start: int
    end: int


class NLUExample(BaseModel):
    intent: t.Optional[str]
    text: str
    tags: t.List[TrainingTag]


class NLUTrainingData(BaseModel):
    """NLU data for an agent.
    """
    intents: t.List[str]
    tagTypes: t.List[str]
    examples: t.List[NLUExample]

    def split(self, split_ratio: float, shuffle: bool = True, seed: int = 0) -> tuple:
        """
        Splits `self` into two different training sets, stratified by their intent labels.
        """
        examples = self.examples
        intent_labels = [ex.intent for ex in examples]
        examples_a, examples_b = train_test_split(
            examples,
            test_size=split_ratio,
            random_state=seed,
            shuffle=shuffle,
            stratify=intent_labels,
        )
        return (
            self.build_from_examples(examples_a),
            self.build_from_examples(examples_b),
        )

    def to_folds(self, nfolds: int, shuffle: bool = True, seed: int = 0) -> tuple:
        """
        Splits `self` into `nfolds` random subsets, stratified by intent label.
        """
        examples = self.examples
        intent_labels = [ex.intent for ex in examples]
        folds = make_stratified_folds(examples, intent_labels, nfolds, shuffle, seed)
        return tuple(self.build_from_examples(examples) for examples in folds)

    @classmethod
    def build_from_examples(cls, examples: t.List[NLUExample]) -> "NLUTrainingData":
        """
        Builds an `NLUTrainingData` object from examples.
        """
        return cls(examples=examples, intents=cls.get_intents(examples), tagTypes=cls.get_tag_types(examples))

    @staticmethod
    def get_intents(examples: t.List[NLUExample]) -> t.List[str]:
        return list(set(ex.intent for ex in examples))

    @staticmethod
    def get_tag_types(examples: t.List[NLUExample]) -> t.List[str]:
        return list(set(tag.tagType for ex in examples for tag in ex.tags))

    @classmethod
    def concat(cls, *nlu_datas: "NLUTrainingData") -> "NLUTrainingData":
        """
        Takes the concatenation of multiple nluData objects, returning a single nluData
        object with all the examples from `nlu_datas`.
        """
        examples = list(
            chain.from_iterable(nlu_data.examples for nlu_data in nlu_datas)
        )
        return cls.build_from_examples(examples)

    def incorporate_training_conversations(self, convs: t.List[TrainingConversation]):
        """Adds any valid NLU examples present in `convs` to self's NLU examples.
        """
        for conv in convs:
            for turn in conv.conversation.turns:
                if turn.actor == Actor.USER:
                    if turn.userAction.intent in self.intents and turn.userAction.utterance:
                        self.examples.append(
                            NLUExample(intent=turn.userAction.intent, text=turn.userAction.utterance, tags=[])
                        )

    def balance_by_intent(self, seed: int = 0) -> "NLUTrainingData":
        """
        Makes a new version of `self`, where the minority classes
        are upsampled to have the same number examples as the majority class.
        """
        examples_by_intent = defaultdict(list)
        for ex in self.examples:
            examples_by_intent[ex.intent].append(ex)
        n_majority_intent = max(len(examples) for examples in examples_by_intent.values())
        upsampled = list(
            chain.from_iterable(
                resample(examples, replace=True, n_samples=n_majority_intent, random_state=seed)
                for examples in examples_by_intent.values()
            )
        )
        return self.build_from_examples(upsampled)

    def get_intent_distribution(self) -> Counter:
        """
        Counts the number of each type of intent present in `self`.
        The returned `Counter` object can be treated as a dictionary e.g.
        `my_intent_count = counter["my_intent"]`.
        """
        return Counter([ex.intent for ex in self.examples])

    def filter_out_unknown(self) -> "NLUTrainingData":
        """Returns a new copy of the NLU data, where examples containing unknonw intents or tags are filtered out.
        """
        new_data = NLUTrainingData(intents=self.intents, tagTypes=self.tagTypes, examples=[])
        for ex in self.examples:
            if ex.intent not in self.intents:
                # We only allow examples for the agent's registered intents. This is probably invalid/old data.
                continue

            if any(tag.tagType not in self.tagTypes for tag in ex.tags):
                # The same goes for NER tag types.
                continue

            new_data.examples.append(ex)

        return new_data
