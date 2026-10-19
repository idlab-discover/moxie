"""Compare native curves with the original pinned validation_curve body."""
import subprocess
import warnings

import numpy as np
from sklearn import config_context
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold


def check_curves(ns, dummy, root):
    X = np.zeros((8, 2))
    y = np.arange(8, dtype=float) ** 2
    weights = np.arange(1, 9, dtype=float)
    folds = list(KFold(4).split(X))
    values = ['median', 'mean', 'median', 'invalid']
    actual = {}
    for line in subprocess.check_output([str(root / 'builds/curves')], text=True).splitlines():
        fields = line.split()
        if fields[0] == 'curve':
            key = fields[1]
            actual.setdefault(key, {})[int(fields[2]), int(fields[3])] = tuple(map(float, fields[4:]))

    def fail_score(estimator, X, y):
        raise ValueError('intentional scoring failure')

    cases = {
        'mixed': dict(param_range=values, error_score=-123),
        'weighted_fit': dict(param_range=values, error_score=-123, params={'sample_weight': weights}),
        'all_failed': dict(param_range=['invalid', 'also_invalid'], error_score=-321),
        'nan_failed': dict(param_range=['invalid', 'also_invalid']),
        'scoring_failed': dict(param_range=['mean'], error_score=-456, scoring=fail_score),
    }
    expected = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, options in cases.items():
            # A generator also checks the reference's one-pass split consumption.
            expected[name] = ns['validation_curve'](
                dummy.DummyRegressor(), X, y, param_name='strategy',
                cv=iter(folds), **options,
            )
        with config_context(enable_metadata_routing=True):
            estimator = dummy.DummyRegressor().set_fit_request(sample_weight=True)
            scorer = make_scorer(r2_score).set_score_request(sample_weight=True)
            expected['weighted_both'] = ns['validation_curve'](
                estimator, X, y, param_name='strategy', param_range=values,
                cv=iter(folds), scoring=scorer, params={'sample_weight': weights},
                error_score=-123,
            )

    assert set(actual) == set(expected)
    cells = 0
    for name, (train, test) in expected.items():
        assert len(actual[name]) == train.size
        for parameter, fold in np.ndindex(train.shape):
            np.testing.assert_allclose(
                actual[name][parameter, fold],
                [train[parameter, fold], test[parameter, fold]],
                rtol=1e-12, atol=1e-12, equal_nan=True,
                err_msg=f'{name}: parameter {parameter}, fold {fold}',
            )
            cells += 2

    # Match failures outside numeric error_score, and explicit raising policies.
    for options in (
        dict(param_name='unknown', param_range=['mean']),
        dict(param_name='strategy', param_range=['invalid'], error_score='raise'),
        dict(param_name='strategy', param_range=['mean'], scoring=fail_score, error_score='raise'),
        dict(param_name='strategy', param_range=['mean'], params={'sample_weight': np.ones(1)}, error_score='raise'),
        dict(param_name='strategy', param_range=[]),
    ):
        try:
            ns['validation_curve'](dummy.DummyRegressor(), X, y, cv=folds, **options)
        except (ValueError, IndexError):
            pass
        else:
            raise AssertionError(f'expected failure: {options}')

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
            raise ValueError('intentional score failure for one side of a fold')
        return estimator.score(X, y)

    model = Probe()
    model.fitted_ = True
    output = subprocess.check_output([str(root / 'builds/curve_lifecycle')], text=True)
    order = []
    probe_actual = {'partial': {}, 'scoring': {}}
    for line in output.splitlines():
        fields = line.split()
        if fields[0] == 'order':
            order.append(int(fields[1]))
        elif fields[0] in probe_actual:
            probe_actual[fields[0]][int(fields[1]), int(fields[2])] = tuple(map(float, fields[3:]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ns['validation_curve'](model, X, np.arange(8.), cv=iter(folds),
                               param_name='parameter', param_range=[3, 1, 3])
        assert order == log
        for name, options in (
            ('partial', dict(params={'mode': 1}, error_score=-999)),
            ('scoring', dict(scoring=partial_score, error_score=-456)),
        ):
            train, test = ns['validation_curve'](
                model, X, np.arange(8.), cv=iter(folds),
                param_name='parameter', param_range=[3, 1, 3], **options,
            )
            assert len(probe_actual[name]) == train.size
            for parameter, fold in np.ndindex(train.shape):
                np.testing.assert_array_equal(
                    probe_actual[name][parameter, fold],
                    [train[parameter, fold], test[parameter, fold]],
                )
                cells += 2
    assert model.parameter == 7 and model.fitted_
    print(f'PASS original validation_curve: {cells} score cells, exact execution order, weights and failure policies')
    return dict(score_cells=cells, cases=list(expected), probe_cases=list(probe_actual),
                execution_order=order, rtol=1e-12, atol=1e-12)
