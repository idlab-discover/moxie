"""Pinned permutation-test body, including resplitting and refit lifecycle."""
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn import config_context
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.svm import SVC


def check_svc_permutations(ns, stratified_kfold, oracle, binary):
    """Compare the Iris and noise scenarios from the pinned gallery."""
    iris_path = "sklearn/datasets/data/iris.csv"
    assert not subprocess.check_output(
        ["git", "-C", str(oracle), "status", "--porcelain", "--", iris_path],
        text=True,
    )
    iris = np.loadtxt(oracle / iris_path, skiprows=1, delimiter=",")
    X, y = iris[:, :-1], iris[:, -1].astype(np.int64)
    noise = np.random.RandomState(0).normal(size=(len(y), 20))
    datasets = {"iris": X, "noise": noise}
    n_permutations = 1000
    with TemporaryDirectory() as directory:
        paths = []
        for name, features in datasets.items():
            case = Path(directory) / name
            paths.append(case)
            with case.open("w") as fixture:
                print(*features.shape, file=fixture)
                # Pass Float64 feature bits as signed integers for exact input.
                for label, row in zip(y, features.view(np.int64)):
                    print(label, *row, file=fixture)
        values = subprocess.check_output([str(binary), *paths], text=True)

    actual = np.array([float(line) for line in values.splitlines()])
    actual = actual.reshape(len(datasets), n_permutations + 2)
    for native, (name, features) in zip(actual, datasets.items()):
        model = SVC(kernel="linear", random_state=7)
        cv = stratified_kfold(2, shuffle=True, random_state=0)
        score, null_scores, pvalue = ns["permutation_test_score"](
            model, features, y, scoring="accuracy", cv=cv,
            n_permutations=n_permutations, random_state=0, n_jobs=1,
        )
        # Accuracies are ratios of counts. Allow Float64 rounding of fold means.
        np.testing.assert_allclose(
            native[:-1], np.r_[score, null_scores], rtol=0, atol=1e-15, err_msg=name,
        )
        assert native[-1] == pvalue, name
        print(f"PASS pinned permutation_test_score with linear SVC ({name}): "
              f"{n_permutations + 1} scores and exact p-value {pvalue}")


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
