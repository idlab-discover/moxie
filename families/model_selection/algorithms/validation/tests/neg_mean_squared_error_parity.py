"""Compare native regression scores with the pinned sklearn sources."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import sklearn

REFERENCE = Path(__file__).resolve().parents[6] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
DUMMY = "sklearn/dummy.py"
REGRESSION = "sklearn/metrics/_regression.py"
SCORER = "sklearn/metrics/_scorer.py"
SPLIT = "sklearn/model_selection/_split.py"
SEARCH = "sklearn/model_selection/_search.py"
SCORER_NAMES = ("neg_mean_squared_error", "neg_mean_absolute_error", "neg_root_mean_squared_error")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(binary):
    return subprocess.check_output([Path(binary).resolve()], text=True).splitlines()


def compare_probe(binary, dummy, scorers):
    actual = {
        (scorer, dtype, case): float(score)
        for scorer, dtype, case, score in map(str.split, run(binary))
    }
    expected = {}
    for dtype in (np.float32, np.float64):
        name = np.dtype(dtype).name
        X = np.zeros((3, 1), dtype=dtype)
        y = np.arange(6, dtype=dtype).reshape(3, 2) * np.array([1, 2], dtype=dtype)
        y /= dtype(3)
        weights = np.array([0.25, -0.125, 1.0], dtype=dtype)
        model = dummy.DummyRegressor().fit(X, y)
        for scorer_name, scorer in scorers.items():
            expected[scorer_name, name, "multioutput"] = scorer(model, X, y[::-1])
            expected[scorer_name, name, "weighted"] = scorer(model, X, y[::-1], sample_weight=weights[::-1])
        vector = np.arange(3, dtype=dtype)
        model.fit(X, vector)
        for scorer_name, scorer in scorers.items():
            expected[scorer_name, name, "singleton_column"] = scorer(model, X, vector[:, None])
            with np.errstate(invalid="ignore"):
                expected[scorer_name, name, "signed_weights"] = scorer(
                    model, X, vector, sample_weight=np.array([-1, 3, 0], dtype=dtype),
                )

    assert actual.keys() == expected.keys(), actual
    for key, value in expected.items():
        tolerance = 2e-7 if key[1] == "float32" else 1e-13
        np.testing.assert_allclose(actual[key], value, rtol=tolerance, atol=tolerance, err_msg=str(key))
    print(f"PASS regression scorer parity ({len(expected)} cases)")


def compare_dense_adapter(binary, dummy, scorers):
    split = load("sklearn.model_selection._moxie_scorer_split", SPLIT)
    search = load("sklearn.model_selection._moxie_scorer_search", SEARCH)
    mae, rmse = scorers["neg_mean_absolute_error"], scorers["neg_root_mean_squared_error"]
    rows = np.arange(12)
    X = np.zeros((len(rows), 1))
    y = np.column_stack((rows % 4, rows**2 % 17)).astype(float)
    weights = rows + 1.0
    cv = split.check_cv(None, y)
    expected = {}
    for fold, (train, test) in enumerate(cv.split(X, y)):
        fitted = dummy.DummyRegressor().fit(X[train], y[train])
        expected["mae_fold", fold] = mae(fitted, X[test], y[test], sample_weight=weights[test])
        expected["rmse_fold", fold] = rmse(fitted, X[test], y[test])
    grid = search.GridSearchCV(
        dummy.DummyRegressor(), {"strategy": ["mean", "median"]}, cv=cv, scoring=mae,
    ).fit(X, y)
    expected["best_index", 0] = grid.best_index_
    expected["best_score", 0] = grid.best_score_
    expected["refit_score", 0] = grid.score(X, y)
    for candidate in range(2):
        for fold in range(5):
            expected["search_fold", candidate * 5 + fold] = grid.cv_results_[f"split{fold}_test_score"][candidate]

    actual = {(case, int(index)): float(score) for case, index, score in map(str.split, run(binary))}
    assert actual.keys() == expected.keys(), actual
    assert actual["best_index", 0] == expected["best_index", 0]
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=1e-13, atol=1e-13, err_msg=str(key))
    print(f"PASS dense named scoring parity ({len(expected)} CV and search values)")


def main(probe_binary, dense_adapter_binary=None):
    assert sklearn.__version__ == "1.9.1"
    assert subprocess.check_output(["git", "-C", REFERENCE, "rev-parse", "HEAD"], text=True).strip() == REVISION
    assert not subprocess.check_output(
        ["git", "-C", REFERENCE, "status", "--porcelain", "--", DUMMY, REGRESSION, SCORER, SPLIT, SEARCH],
        text=True,
    )
    dummy = load("sklearn._moxie_regression_dummy", DUMMY)
    regression = load("sklearn.metrics._moxie_regression_reference", REGRESSION)
    scoring = load("sklearn.metrics._moxie_scorer_reference", SCORER)
    scorers = {name: scoring.get_scorer(name) for name in SCORER_NAMES}
    for name, scorer in scorers.items():
        # The pinned scorer module imports public metrics from the installed sklearn.
        scorer._score_func = getattr(regression, name.removeprefix("neg_"))
    compare_probe(probe_binary, dummy, scorers)
    if dense_adapter_binary:
        compare_dense_adapter(dense_adapter_binary, dummy, scorers)


if __name__ == "__main__":
    main(*sys.argv[1:])
