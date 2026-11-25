"""Run the four dense heterogeneous pipeline checks against pinned sklearn."""

from inspect import getfile
from pathlib import Path
import sys

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import compose_parity_io as io


def check_tree_and_gaussian():
    sources = {
        Pipeline: "sklearn/pipeline.py",
        StandardScaler: "sklearn/preprocessing/_data.py",
        DecisionTreeClassifier: "sklearn/tree/_classes.py",
        GaussianNB: "sklearn/naive_bayes.py",
    }
    io.verify_reference(sources.values())
    for estimator, relative in sources.items():
        assert Path(getfile(estimator)).read_bytes() == (io.REFERENCE / relative).read_bytes(), (
            f"Install scikit-learn built from {io.REVISION} to compare {relative}"
        )

    i = np.arange(12)
    group = i % 2
    X = np.column_stack((i // 2 + 2 * group, 10 + (7 * i) % 19 + 3 * group))
    y = np.where(group == 0, 7, -2)
    i = np.arange(4)
    query = np.column_stack((2 * i + 0.5, 12 + (5 * i) % 11))
    expected = np.where(i % 2 == 0, 7, -2)
    binary = ROOT / "builds/tests/pipeline/test_heterogeneous_compositions"
    actual = io.parse_results(io.run(ROOT, binary, capture=True).stdout)
    assert set(actual) == {"tree", "gaussian"}

    for name, estimator, dtype, params, tolerance in (
        ("tree", DecisionTreeClassifier(random_state=0, max_depth=2), np.float32,
         {"decisiontreeclassifier__max_depth": 1}, 0),
        ("gaussian", GaussianNB(), np.float64, {"gaussiannb__var_smoothing": 0.05}, 1e-12),
    ):
        train, held_out = X.astype(dtype), query.astype(dtype)
        model = make_pipeline(StandardScaler(), estimator).fit(train, y)
        native = actual[name]
        np.testing.assert_array_equal(native["classes"], model.classes_)
        np.testing.assert_array_equal(native["prediction"], model.predict(held_out))
        assert native["score"] == model.score(held_out, expected)
        error = io.close(native["probability"], model.predict_proba(held_out), 0, tolerance,
                         f"{name} held-out probabilities")
        model.set_params(**params).fit(train, y + 10)
        np.testing.assert_array_equal(native["refit_classes"], model.classes_)
        np.testing.assert_array_equal(native["refit_prediction"], model.predict(held_out))
        refit_error = io.close(native["refit_probability"], model.predict_proba(held_out),
                               0, tolerance, f"{name} refit probabilities")
        print(f"PASS: {name} classes, predictions and score exact; "
              f"max probability error {max(error, refit_error):.3g}", flush=True)


if __name__ == "__main__":
    check_tree_and_gaussian()
    for script in ("select_k_best_cv_parity.py", "calibrated_pipeline_parity.py"):
        io.run(ROOT, sys.executable, Path(__file__).with_name(script))
