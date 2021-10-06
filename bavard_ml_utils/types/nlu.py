import typing as t

from pydantic import BaseModel

from bavard_ml_utils.ml.dataset import LabeledDataset


class TrainingTag(BaseModel):
    """A tag as it appears in NLU training data."""

    tagType: str
    """The type of named entity that is referenced e.g. ``"person"``, ``"location"``, ``"size"``, etc."""

    start: int
    """The starting index where this tag can be found in its parent utterance (inclusive)."""

    end: int
    """The ending index where this tag can be found in its parent utterance (exclusive)."""


class NLUExample(BaseModel):
    intent: t.Optional[str]
    """The intent of :attr:`text`. When ``text=="Hello, how are you?"``, the intent might be ``"greet"``."""

    text: str
    """The text of the example, e.g. "Hello, how are you?\""""

    tags: t.Optional[t.List[TrainingTag]]
    """Any tags that are present in :attr:`text`."""

    isOOD = False
    """``True`` if :attr:`text` does not belong to any intent in this example's dataset."""


class NLUExampleDataset(LabeledDataset[NLUExample]):
    """A dataset of :class:`NLUExample` objects."""

    def get_label(self, item: NLUExample) -> t.Optional[str]:
        """Returns the intent of ``item``, an :class:`NLUExample`."""
        return item.intent

    def unique_tag_types(self) -> t.Set[str]:
        return set(tag.tagType for ex in self for tag in ex.tags or [])
