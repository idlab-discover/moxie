"""Compare native named scorers and Pipeline search with pinned sklearn."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT.parent / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
SOURCES = (
    "sklearn/metrics/_classification.py",
    "sklearn/metrics/_scorer.py",
    "sklearn/model_selection/_split.py",
    "sklearn/pipeline.py",
)
NAMES = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ORACLE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output(binary):
    return subprocess.check_output([binary], text=True).splitlines()


def main():
    assert sklearn.__version__ == "1.9.1"
    assert subprocess.check_output(
        ["git", "-C", ORACLE, "rev-parse", "HEAD"], text=True,
    ).strip() == REVISION
    assert not subprocess.check_output(
        ["git", "-C", ORACLE, "status", "--porcelain", "--", *SOURCES], text=True,
    )
    metrics = load("sklearn.metrics._moxie_classification_reference", SOURCES[0])
    scorer_module = load("sklearn.metrics._moxie_scorer_reference", SOURCES[1])
    split = load("sklearn.model_selection._moxie_split_reference", SOURCES[2])
    pipeline = load("sklearn._moxie_pipeline_reference", SOURCES[3])
    scorers = {name: scorer_module.get_scorer(name) for name in NAMES}
    for name, scorer in scorers.items():
        scorer._score_func = getattr(metrics, f"{name.removesuffix('_macro')}_score")

    truth = np.array([0, 1, 2, 0, 2, 1])
    predicted = np.array([1, 1, 0, 0, 0, 1])
    weights = np.array([1, 8, 3, 1, 2, 4], dtype=np.float32)
    seen = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for line in output(sys.argv[1]):
            target, metric, value = line.split()
            y, estimates = truth, predicted
            if target in ("strings", "string_target"):
                labels = np.array(["a", "b", "c"])
                y, estimates = labels[truth], labels[predicted]
            elif target.endswith("_array"):
                y = y.reshape(-1, 1)
            scorer = scorers[metric]
            expected = scorer._score_func(y, estimates, sample_weight=weights, **scorer._kwargs)
            np.testing.assert_allclose(float(value), expected, rtol=0, atol=1e-13, err_msg=line)
            seen.add((target, metric))
    assert len(seen) == 36
    print("PASS 36 weighted classification scorer values against the pinned metrics")

    rows = np.arange(30)
    X = np.column_stack([rows % 3 + rows / 10, (rows * 7) % 5])
    y = rows % 3
    sample_weight = rows % 7 + 1
    model = pipeline.Pipeline([("scale", StandardScaler()), ("svc", SVC())])
    cv = list(split.check_cv(None, y, classifier=True).split(X, y))
    precision = scorers["precision_macro"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        weighted = [
            precision(model.fit(X[train], y[train]), X[test], y[test], sample_weight=sample_weight[test])
            for train, test in cv
        ]
        search = GridSearchCV(
            model, {"svc__C": [0.1, 1.0, 10.0]}, scoring=precision, cv=cv,
        ).fit(X, y)
        expected = {
            "weighted_cv": weighted,
            "search_fold": np.column_stack(
                [search.cv_results_[f"split{fold}_test_score"] for fold in range(len(cv))]
            ).ravel(),
            "best_index": [search.best_index_],
            "best_score": [search.best_score_],
            "refit_score": [search.score(X, y)],
        }
    actual = {}
    for line in output(sys.argv[2]):
        key, value = line.split()
        actual.setdefault(key, []).append(float(value))
    assert actual.keys() == expected.keys()
    for key, values in expected.items():
        if key == "best_index":
            np.testing.assert_array_equal(actual[key], values, err_msg=key)
        else:
            np.testing.assert_allclose(actual[key], values, rtol=0, atol=1e-13, err_msg=key)
    print(f"PASS 23 Pipeline CV and search values against pinned scorer and splitter semantics at {REVISION}")


if __name__ == "__main__":
    main()
