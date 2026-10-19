"""Differential learning-curve checks against the original pinned sklearn body."""
import subprocess
import warnings

import numpy as np
from sklearn import config_context
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold


def check_learning(ns, dummy, root):
    X = np.zeros((8, 2))
    y = np.arange(8, dtype=float) ** 2
    weights = np.arange(1, 9, dtype=float)
    folds = list(KFold(4).split(X))
    sizes = [6, 2, 4, 2]
    actual = {}
    rng_next = None
    output = subprocess.check_output([str(root / 'builds/learning')], text=True)
    for line in output.splitlines():
        fields = line.split()
        if fields[0] == 'learning':
            actual.setdefault(fields[1], {})[int(fields[2]), int(fields[3])] = (
                int(fields[4]), *map(float, fields[5:]))
        elif fields[0] == 'rng_next':
            rng_next = int(fields[1])

    def fail_score(estimator, X, y):
        raise ValueError('intentional scoring failure')

    uneven = [(np.array(a), np.array(b)) for a, b in (
        ([7, 5, 3, 1, 0, 2], [4, 6]),
        ([6, 4, 0], [7, 5, 3, 2, 1]),
        ([1, 2, 3, 4, 5, 6, 7], [0]),
    )]
    options = {
        'default': {},
        'absolute': dict(train_sizes=sizes, return_times=True),
        'fractional': dict(train_sizes=[1., .01, .5, .1, .99]),
        'shuffled': dict(train_sizes=sizes, shuffle=True, random_state=42),
        'weighted_fit': dict(train_sizes=sizes, shuffle=True, random_state=42,
                             params={'sample_weight': weights}),
        'scoring_failed': dict(train_sizes=sizes, scoring=fail_score, error_score=-123),
        'uneven': dict(train_sizes=sizes, cv=uneven),
        'strided': dict(train_sizes=np.array([1., .5, .01])[::-1]),
    }
    expected = {}
    model = dummy.DummyRegressor().fit(X, y)
    original = model.predict(X).copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, kwargs in options.items():
            expected[name] = ns['learning_curve'](model, X, y, **(dict(cv=iter(folds)) | kwargs))
        rng = np.random.RandomState(42)
        for name in ('mutable_first', 'mutable_second'):
            expected[name] = ns['learning_curve'](model, X, y, cv=iter(folds),
                train_sizes=sizes, shuffle=True, random_state=rng)
        assert rng_next == rng.randint(2**32, dtype=np.uint32)
        with config_context(enable_metadata_routing=True):
            estimator = dummy.DummyRegressor().set_fit_request(sample_weight=True)
            scorer = make_scorer(r2_score).set_score_request(sample_weight=True)
            expected['weighted_both'] = ns['learning_curve'](
                estimator, X, y, cv=iter(folds), train_sizes=sizes,
                shuffle=True, random_state=42, scoring=scorer,
                params={'sample_weight': weights})
    np.testing.assert_array_equal(model.predict(X), original)
    assert set(actual) == set(expected)
    cells = 0
    for name, result in expected.items():
        absolute, train, test = result[:3]
        assert len(actual[name]) == train.size
        for size, fold in np.ndindex(train.shape):
            size_got, *scores = actual[name][size, fold]
            assert size_got == absolute[size]
            np.testing.assert_allclose(scores, [train[size, fold], test[size, fold]],
                rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=f'{name}, {size}, {fold}')
            cells += 2
        if len(result) == 5:
            for timing in result[3:]:
                assert timing.shape == train.shape and timing.dtype == np.float64
                assert np.all(timing >= 0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for kwargs in (
            dict(train_sizes=[0]), dict(train_sizes=[-1]), dict(train_sizes=[7]),
            dict(train_sizes=[0.]), dict(train_sizes=[-.1]), dict(train_sizes=[1.1]),
            dict(train_sizes=[]), dict(exploit_incremental_learning=True),
            dict(scoring=fail_score, error_score='raise'),
        ):
            try:
                ns['learning_curve'](dummy.DummyRegressor(), X, y, cv=folds, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f'expected invalid-input failure: {kwargs}')
        for policy in (-99, np.nan, 'raise'):
            try:
                ns['learning_curve'](dummy.DummyRegressor(strategy='invalid'), X, y,
                                     cv=folds, error_score=policy)
            except ValueError:
                pass
            else:
                raise AssertionError('all fits must fail even with numeric error_score')

    log = []

    class Probe(RegressorMixin, BaseEstimator):
        def __init__(self, parameter=7):
            self.parameter = parameter

        def fit(self, X, y, mode=0):
            assert not hasattr(self, 'fitted_')
            log.append(int(y.sum()) * 10 + self.parameter)
            if mode == 1 and y[0] == 0:
                raise ValueError('intentional partial fit failure')
            self.fitted_ = True
            return self

        def score(self, X, y):
            assert self.fitted_
            return self.parameter + 10 + y[0]

    def partial_score(estimator, X, y):
        if y[0] == 0:
            raise ValueError('intentional per-side score failure')
        return estimator.score(X, y)

    probe_actual = {}
    order = []
    for line in subprocess.check_output([str(root / 'builds/learning_lifecycle')], text=True).splitlines():
        fields = line.split()
        if fields[0] == 'order':
            order.append(int(fields[1]))
        elif fields[0] == 'probe':
            probe_actual.setdefault(fields[1], {})[int(fields[2]), int(fields[3])] = tuple(map(float, fields[4:]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, kwargs in (
            ('order', {}),
            ('partial', dict(params={'mode': 1}, error_score=-999)),
            ('scoring', dict(scoring=partial_score, error_score=-456)),
        ):
            log.clear()
            _, train, test = ns['learning_curve'](Probe(), X, np.arange(8.),
                cv=folds, train_sizes=[6, 1, 3], **kwargs)
            if name == 'order':
                assert order == log
            assert len(probe_actual[name]) == train.size
            for size, fold in np.ndindex(train.shape):
                np.testing.assert_array_equal(probe_actual[name][size, fold],
                                              [train[size, fold], test[size, fold]])
                cells += 2
    print(f'PASS original learning_curve: {cells} score cells, exact sizes/order/RNG, weights and failure policies')
    return dict(score_cells=cells, cases=list(expected), execution_order=order,
                next_rng_word=rng_next, rtol=1e-12, atol=1e-12)
