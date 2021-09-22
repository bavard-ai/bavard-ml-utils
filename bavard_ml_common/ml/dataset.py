import typing as t
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from itertools import chain

from bavard_ml_common.utils import requires_extras


try:
    from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
    from sklearn.utils import resample
except ImportError:
    _has_ml_deps = False
else:
    _has_ml_deps = True


_T = t.TypeVar("_T")


class LabeledDataset(t.List[_T], ABC):
    """
    Essentially an abstract typed array with attached helper functions related to categorically-labeled
    datasets. Subclassing instances can be used as regular lists with indexeing, etc. The only method
    that needs to be implemented is `get_label`, which should return the label associated with
    an instance of this dataset.
    """

    def __init__(self, items: t.Optional[t.Iterable[_T]] = None):
        super().__init__(items)

    @abstractmethod
    def get_label(self, item: _T) -> t.Any:
        pass

    def labels(self) -> list:
        return [self.get_label(item) for item in self]

    def unique_labels(self) -> set:
        return set(self.get_label(item) for item in self)

    def get_label_distribution(self) -> Counter:
        """
        Counts the number of each type of label present in `self`.
        The returned `Counter` object can be treated as a dictionary e.g.
        `my_label_count = counter["my_label"]`.
        """
        return Counter(self.labels())

    @requires_extras(ml=_has_ml_deps)
    def cv(
        self, nfolds: int = 5, nrepeats: int = 1, seed: int = 0
    ) -> t.Iterator[t.Tuple["LabeledDataset", "LabeledDataset"]]:
        """
        Yields train/test splits for `nfolds` cross-validation, stratified by label.
        Repeats the random splitting `nrepeats` times.
        """
        cls = self.__class__
        rskf = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=seed)
        for train_index, test_index in rskf.split(self, self.labels()):
            yield cls([self[i] for i in train_index]), cls(self[i] for i in test_index)

    @requires_extras(ml=_has_ml_deps)
    def balance(self, seed: int = 0) -> "LabeledDataset":
        """
        Makes a new version of `self`, where the minority labels
        are upsampled to have the same number items as the majority label.
        """
        cls = self.__class__
        items_by_label = defaultdict(list)
        for item in self:
            items_by_label[self.get_label(item)].append(item)
        n_majority_label = max(len(items) for items in items_by_label.values())
        upsampled = chain.from_iterable(
            resample(items, replace=True, n_samples=n_majority_label, random_state=seed)
            for items in items_by_label.values()
        )
        return cls(upsampled)

    @requires_extras(ml=_has_ml_deps)
    def split(
        self, test_size: t.Union[float, int, None] = None, seed: int = 0, shuffle: bool = True
    ) -> t.Tuple["LabeledDataset", "LabeledDataset"]:
        """Returns a train/test split of `self`, stratified by label."""
        cls = self.__class__
        train, test = train_test_split(
            self, test_size=test_size, random_state=seed, shuffle=shuffle, stratify=self.labels()
        )
        return cls(train), cls(test)
