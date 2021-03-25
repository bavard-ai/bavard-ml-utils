from unittest import TestCase
import typing as t

from bavard_ml_common.ml.dataset import LabeledDataset


Pair = t.Tuple[str, int]


class PairDataset(LabeledDataset[Pair]):
    def get_label(self, item: Pair) -> int:
        return item[1]


class TestLabeledDataset(TestCase):
    def setUp(self) -> None:
        self.dataset = PairDataset([("a", 0), ("b", 0), ("c", 0), ("d", 1), ("e", 1), ("f", 2), ("g", 2)])
        self.labels = set(self.dataset.labels())

    def test_balance_by_intent(self) -> None:
        label_distribution = self.dataset.get_label_distribution()
        majority_class_n = label_distribution.most_common(1)[0][1]

        balanced = self.dataset.balance()
        balanced_labels = set(balanced.labels())

        # The labels should still be the same
        self.assertSetEqual(set(self.labels), set(balanced_labels))
        # Each intent's examples should have been upsampled.
        balanced_label_distribution = balanced.get_label_distribution()
        for label in self.labels:
            self.assertEqual(balanced_label_distribution[label], majority_class_n)

    def test_cv(self):
        # Each cross-validation fold should have all the correct labels, and have the full dataset as well.
        for train, test in self.dataset.cv(2, nrepeats=2):
            # The folds should still be PairDataset instances.
            self.assertIsInstance(train, PairDataset)
            self.assertIsInstance(test, PairDataset)
            # Train fold should have all labels represented in it.
            self.assertSetEqual(set(train.labels()), self.labels)
            # Test fold should have all labels represented in it.
            self.assertSetEqual(set(test.labels()), self.labels)
            # The folds should contain all the samples.
            self.assertEqual(len(self.dataset), len(train) + len(test))
