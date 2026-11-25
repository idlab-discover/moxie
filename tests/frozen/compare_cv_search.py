"""Check dense CV and search over fitted frozen estimators against pinned sklearn."""

import ast
from pathlib import Path
import subprocess

import numpy as np
from numpy.testing import assert_allclose
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import NotFittedError
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"


def data(n, shift):
    i = np.arange(n)
    return np.column_stack((i % 2 * 3 + i % 7 / 4 + shift, i % 11 / 3 - 2 * shift))


def expected():
    X_train = data(20, 0)
    y_train = np.arange(20) % 2
    X = data(24, 1.5)
    y = np.arange(24) // 3 % 2
    with np.testing.assert_raises(NotFittedError):
        cross_val_score(FrozenEstimator(LinearDiscriminantAnalysis()), X, y, error_score="raise")
    scores = {}
    predictions = {}
    for name, model in (
        ("lda", LinearDiscriminantAnalysis()),
        ("pipeline", make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())),
    ):
        frozen = FrozenEstimator(model.fit(X_train, y_train))
        for cv_name, cv in (("default", None), ("kfold", KFold(n_splits=3))):
            fold_scores = cross_val_score(frozen, X, y, cv=cv)
            for fold, score in enumerate(fold_scores):
                scores["frozen_cv", name, cv_name, str(fold)] = score
            if cv is None:
                mean_score = np.mean(fold_scores)
        for search_name, search in (
            ("grid", GridSearchCV(frozen, {})),
            ("random", RandomizedSearchCV(frozen, {}, n_iter=1, random_state=42)),
        ):
            # Pinned search assigns the result of FrozenEstimator.set_params,
            # which is None. An empty grid should evaluate the same frozen fit.
            with np.testing.assert_raises_regex(AttributeError, "NoneType"):
                search.fit(X, y)
            scores["frozen_search", name, search_name] = mean_score
        predictions[name] = frozen.predict(X).tolist()
    return scores, predictions


def main():
    checkout = ROOT.parent / "scikit-learn"
    assert subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "HEAD"], text=True,
    ).strip() == REVISION
    for module in (
        "discriminant_analysis.py", "frozen/_frozen.py", "model_selection/_search.py",
        "model_selection/_split.py", "model_selection/_validation.py", "pipeline.py",
        "preprocessing/_data.py",
    ):
        pinned = subprocess.check_output(
            ["git", "-C", checkout, "show", f"{REVISION}:sklearn/{module}"],
        )
        assert (Path(sklearn.__file__).parent / module).read_bytes() == pinned, module

    output = subprocess.check_output(
        [ROOT / "builds/tests/frozen/test_frozen_composition"], text=True,
    )
    actual = {}
    actual_predictions = {}
    for line in output.splitlines():
        if line.startswith(("frozen_cv ", "frozen_search ")):
            *key, value = line.split()
            assert tuple(key) not in actual, key
            actual[tuple(key)] = float(value)
        elif line.startswith("frozen_refit "):
            _, name, predictions = line.split(maxsplit=2)
            assert name not in actual_predictions, name
            actual_predictions[name] = ast.literal_eval(predictions)
    reference, reference_predictions = expected()
    assert actual.keys() == reference.keys(), (actual.keys(), reference.keys())
    for key, score in reference.items():
        if key[0] == "frozen_cv":
            assert actual[key] == score, (key, actual[key], score)
        else:
            # A search averages five independently rounded Float64 fold scores.
            assert_allclose(actual[key], score, atol=1e-15, rtol=0, err_msg=str(key))
    assert actual_predictions == reference_predictions
    print(
        f"PASS {len(reference)} frozen CV/search score comparisons and "
        f"{len(reference_predictions)} refit prediction arrays against pinned sklearn"
    )


if __name__ == "__main__":
    main()
