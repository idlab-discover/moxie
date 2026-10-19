"""Pinned permutation-test body, including resplitting and refit lifecycle."""
import subprocess
import numpy as np
from sklearn import config_context
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, StratifiedKFold


def check_permutations(ns, dummy, root):
    X = np.zeros((8, 2))
    y = np.arange(8.) ** 2
    weights = np.arange(1., 9.)
    groups = np.array(['z' if i % 3 == 0 else 'a' for i in range(8)])
    folds = list(KFold(4).split(X))
    actual = {}
    next_word = None
    for line in subprocess.check_output([str(root / 'builds/permutation')], text=True).splitlines():
        f = line.split()
        if f[0] == 'permutation':
            actual[f[1]] = [float(f[2]), [], float(f[3])]
        elif f[0] == 'null':
            assert int(f[2]) == len(actual[f[1]][1])
            actual[f[1]][1].append(float(f[3]))
        elif f[0] == 'permutation_rng':
            next_word = int(f[1])
    expected = {}
    for name, options in (
        ('plain', {}),
        ('grouped', dict(groups=groups, random_state=42)),
        ('weighted', dict(params={'sample_weight': weights}, random_state=42)),
        ('singletons', dict(groups=np.arange(8.) / 2, n_permutations=1)),
    ):
        expected[name] = ns['permutation_test_score'](dummy.DummyRegressor(), X, y,
            cv=folds, **(dict(n_permutations=7) | options))
    with config_context(enable_metadata_routing=True):
        model = dummy.DummyRegressor().set_fit_request(sample_weight=True)
        scorer = make_scorer(r2_score).set_score_request(sample_weight=True)
        expected['both'] = ns['permutation_test_score'](model, X, y, cv=folds,
            n_permutations=7, random_state=42, scoring=scorer, params={'sample_weight': weights})
    rng = np.random.RandomState(42)
    for name in ['mutable_first', 'mutable_second']:
        expected[name] = ns['permutation_test_score'](dummy.DummyRegressor(), X, y,
            cv=folds, groups=groups, random_state=rng, n_permutations=7)
    assert next_word == rng.randint(2**32, dtype=np.uint32)
    expected['stratified'] = ns['permutation_test_score'](dummy.DummyRegressor(), X,
        np.arange(8.) % 2, cv=StratifiedKFold(2), n_permutations=7)
    assert set(actual) == set(expected)
    for name, (score, scores, pvalue) in expected.items():
        np.testing.assert_allclose(actual[name][0], score, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[name][1], scores, rtol=1e-12, atol=1e-12)
        assert actual[name][2] == pvalue, name

    refits = []

    class Probe(RegressorMixin, BaseEstimator):
        def fit(self, X, y, mode=99):
            assert mode == 99
            self.fit_count_ = getattr(self, 'fit_count_', 0) + 1
            refits.append(self.fit_count_)
            return self

        def score(self, X, y):
            return float(self.fit_count_)

    model = Probe()
    model.fit_count_ = 77
    score, scores, pvalue = ns['permutation_test_score'](model, X, y,
        cv=folds, n_permutations=3)
    assert score == 2.5 and pvalue == 1 and model.fit_count_ == 77
    np.testing.assert_array_equal(scores, [2.5] * 3)
    output = subprocess.check_output([str(root / 'builds/permutation_lifecycle')], text=True)
    native_refits = [int(line.split()[1]) for line in output.splitlines() if line.startswith('refit ')]
    assert native_refits == refits
    score, scores, pvalue = ns['permutation_test_score'](model, X, y,
        cv=folds, n_permutations=3, scoring=lambda *args: np.nan)
    assert np.isnan(score) and np.isnan(scores).all() and pvalue == .25
    count = sum(1 + len(value[1]) for value in actual.values())
    print(f'PASS original permutation_test_score: {count} scores, exact p-values/RNG and clone/refit lifecycle')
    return dict(cases=list(actual), scores=count, refits=native_refits,
                next_rng_word=next_word, rtol=1e-12, atol=1e-12)
