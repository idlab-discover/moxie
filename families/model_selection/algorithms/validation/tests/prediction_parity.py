"""Compare all native prediction containers with the pinned sklearn CV bodies."""
import subprocess
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold


def check_predictions(ns, dummy, root):
    X = np.zeros((8, 2))
    y = np.array(['é', 'z', 'a', 'a', 'z', 'z', 'z', 'z'])
    multi_y = np.column_stack([y, ['yes'] * 2 + ['no'] * 6])
    folds = [(train, test[::-1]) for train, test in KFold(4).split(X)]
    weights = np.arange(1., 9.)
    arrays, strings, containers = {}, {}, {}
    for line in subprocess.check_output([str(root / 'builds/prediction')], text=True).splitlines():
        f = line.split()
        if f[0] == 'response':
            arrays.setdefault(f[1], {})[tuple(map(int, f[2:5]))] = float(f[5])
        elif f[0] == 'string':
            strings.setdefault(f[1], {})[int(f[2]), int(f[3])] = f[4]
        elif f[0] == 'container':
            containers[f[1]] = (bool(int(f[2])), int(f[3]))
    expected = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected_strings = {
            name: ns['cross_val_predict'](dummy.DummyClassifier(), X, target, cv=folds)
            for name, target in [('single', y), ('multi', multi_y)]
        }
        for method in ['predict_proba', 'predict_log_proba']:
            expected[method] = ns['cross_val_predict'](dummy.DummyClassifier(), X, y,
                                                     cv=folds, method=method)
        expected['weighted'] = ns['cross_val_predict'](dummy.DummyClassifier(), X, y,
            cv=folds, method='predict_proba', params={'sample_weight': weights})
        expected['multi'] = ns['cross_val_predict'](dummy.DummyClassifier(), X, multi_y,
            cv=folds, method='predict_proba')
        expected['multi_log'] = ns['cross_val_predict'](dummy.DummyClassifier(), X, multi_y,
            cv=folds, method='predict_log_proba')
        binary_y = np.array(['z'] * 4 + ['a'] * 4)
        for name, method in [('binary_missing', 'predict_proba'), ('binary_missing_log', 'predict_log_proba')]:
            expected[name] = ns['cross_val_predict'](dummy.DummyClassifier(), X, binary_y,
                cv=list(KFold(2).split(X)), method=method)
    assert set(arrays) == set(expected) and set(strings) == set(expected_strings)
    cells = 0
    for name, prediction in expected_strings.items():
        matrix = prediction[:, None] if prediction.ndim == 1 else prediction
        assert len(strings[name]) == matrix.size
        for row, col in np.ndindex(matrix.shape):
            assert strings[name][row, col] == matrix[row, col]
            cells += 1
    for name, prediction in expected.items():
        is_list = isinstance(prediction, list)
        values = prediction if is_list else [prediction]
        assert containers[name] == (is_list, len(values))
        assert len(arrays[name]) == sum(value.size for value in values)
        for output, value in enumerate(values):
            actual = np.array([[arrays[name][output, row, col] for col in range(value.shape[1])]
                               for row in range(value.shape[0])])
            np.testing.assert_allclose(actual, value, rtol=1e-12, atol=1e-12)
            missing = value == np.finfo(value.dtype).min
            np.testing.assert_array_equal(actual[missing], value[missing])
            cells += value.size

    class DecisionProbe(ClassifierMixin, BaseEstimator):
        def __init__(self, mode=0, dtype='float64'):
            self.mode = mode
            self.dtype = dtype

        def fit(self, X, y):
            assert not hasattr(self, 'classes_')
            if self.mode == 2:
                raise ValueError('intentional fit failure')
            self.classes_ = np.unique(y)
            return self

        def decision_function(self, X):
            k = len(self.classes_)
            if k <= 2:
                return (X[:, 0] + self.classes_[-1]).astype(self.dtype)
            columns = np.arange(k * (k - 1) // 2) if self.mode == 1 else self.classes_
            return (10 * X[:, :1] + columns[None, :]).astype(self.dtype)

    class IndicatorProbe(DecisionProbe):
        def fit(self, X, y):
            self.classes_ = np.arange(y.shape[1])
            return self

    actual_decisions = {}
    for line in subprocess.check_output([str(root / 'builds/prediction_lifecycle')], text=True).splitlines():
        f = line.split()
        if f[0] == 'decision':
            actual_decisions.setdefault(f[1], {})[int(f[2]), int(f[3])] = float(f[4])
    X = np.arange(12.)[:, None]
    missing_y = (np.arange(12) // 2 + 1) * 10
    full_y = np.arange(12) % 4 + 10
    binary_y = np.arange(12) % 2 + 10
    folds = list(KFold(3).split(X))
    expected_decisions = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, model, target in (
            ('missing64', DecisionProbe(), missing_y),
            ('missing32', DecisionProbe(dtype='float32'), missing_y),
            ('binary', DecisionProbe(), binary_y),
            ('ovo', DecisionProbe(mode=1), full_y),
            ('indicator', IndicatorProbe(), (np.arange(12)[:, None] + np.arange(3)) % 2),
        ):
            expected_decisions[name] = ns['cross_val_predict'](model, X, target,
                cv=folds, method='decision_function')
        for model, target, cv in (
            (DecisionProbe(mode=1), missing_y, folds),
            (DecisionProbe(mode=2), missing_y, folds),
            (DecisionProbe(), np.arange(12) // 6, list(KFold(2).split(X))),
        ):
            try:
                ns['cross_val_predict'](model, X, target, cv=cv, method='decision_function')
            except ValueError:
                pass
            else:
                raise AssertionError('expected decision/fit failure')
    assert set(actual_decisions) == set(expected_decisions)
    for name, expected_array in expected_decisions.items():
        assert len(actual_decisions[name]) == expected_array.size
        for index in np.ndindex(expected_array.shape):
            key = (index[0], -1) if expected_array.ndim == 1 else index
            # Float32's decimal output round-trips through its own dtype.
            actual_value = expected_array.dtype.type(actual_decisions[name][key])
            assert actual_value == expected_array[index], (name, index)
            cells += 1
    print(f'PASS original cross_val_predict: {cells} prediction cells, String/multioutput, probabilities/logs, Float32/64 decision shapes')
    return dict(cells=cells, cases=list(expected), decision_cases=list(expected_decisions),
                string_cases=list(expected_strings), rtol=1e-12, atol=1e-12)
