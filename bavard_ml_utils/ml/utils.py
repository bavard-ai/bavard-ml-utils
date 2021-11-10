import typing as t


try:
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils import shuffle as do_shuffle
except ImportError:
    _has_ml_deps = False
else:
    _has_ml_deps = True

from bavard_ml_utils.utils import requires_extras


_T = t.TypeVar("_T")


def leave_one_out(items: t.List[_T]) -> t.Iterable[t.Tuple[_T, t.List[_T]]]:
    """
    Performs leave-one-out cross validation. Cycles through ``items``. On each ith item ``item_i``, it yields
    ``item_i``, as well as all items in ``items`` except ``item_i`` as a list. So given ``items==[1,2,3,4]``, the first
    iteration would yield ``(1, [2,3,4])``, the second will yield ``(2, [1,3,4])``, and so on.
    """
    for i, item in enumerate(items):
        yield item, items[:i] + items[i + 1 :]


@requires_extras(ml=_has_ml_deps)
def make_stratified_folds(
    data: t.Sequence[_T], labels: t.Sequence, nfolds: int, shuffle: bool = True, seed: int = 0
) -> t.Tuple[t.List[_T], ...]:
    """
    Takes ``data``, a list, and breaks it into ``nfolds`` chunks. Each chunk is stratified
    by `labels`.
    """
    skf = StratifiedKFold(n_splits=nfolds, shuffle=shuffle, random_state=seed)
    fold_indices = [indices for _, indices in skf.split(data, labels)]
    folds = tuple([data[i] for i in indices] for indices in fold_indices)
    if shuffle:
        folds = tuple(do_shuffle(fold, random_state=seed) for fold in folds)
    return folds


@requires_extras(ml=_has_ml_deps)
def aggregate_dicts(dicts: t.Sequence[dict], agg: str = "mean") -> dict:
    """
    Aggregates a list of dictionaries into a single dictionary. All dictionaries in ``dicts`` should have the same keys.
    All values for a given key are aggregated into a single value using ``agg``. Returns a single dictionary with the
    aggregated values.

    Parameters
    ----------
    dicts : sequence of dicts
        The dictionaries to aggregate.
    agg : {'mean', 'stdev', 'sum', 'median', 'min', 'max'}
        Name of the method to use to aggregate the values of `dicts` with.
    """
    aggs: t.Dict[str, t.Callable] = {
        "mean": np.mean,
        "stdev": np.std,
        "sum": np.sum,
        "median": np.median,
        "min": np.min,
        "max": np.max,
    }
    assert len(dicts) > 0
    keys = dicts[0].keys()
    result = {}
    for key in keys:
        values = [d[key] for d in dicts]
        if isinstance(values[0], dict):
            # Recurse
            result[key] = aggregate_dicts(values, agg)
        else:
            result[key] = aggs[agg](values, axis=0)
    return result


@requires_extras(ml=_has_ml_deps)
def onehot(data, axis=-1, dtype=None):
    """A pure numpy implementation of the one-hot encoding function for arrays of arbitrary dimensionality.
    (`Source <https://stackoverflow.com/a/63840293>`_).
    """
    if dtype is None:
        dtype = np.float32
    pos = axis if axis >= 0 else data.ndim + axis + 1
    shape = list(data.shape)
    shape.insert(pos, data.max() + 1)
    out = np.zeros(shape, dtype)
    ind = list(np.indices(data.shape, sparse=True))
    ind.insert(pos, data)
    out[tuple(ind)] = True
    return out
