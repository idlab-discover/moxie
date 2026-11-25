"""Compare the native FixedThresholdClassifier contract with pinned sklearn."""
import ast
from pathlib import Path
import subprocess

import numpy as np
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier

ROOT = Path(__file__).resolve().parents[2]


def expected():
    for dtype in ("float32", "float64"):
        X = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=dtype)[:, None]
        y = np.array([-3] * 4 + [7] * 4, dtype=np.int64)
        test = np.array([-3, -1, 0, 1, 3], dtype=dtype)[:, None]
        truth = np.array([-3, -3, -3, 7, 7], dtype=np.int64)
        weights = np.array([1, 1, 5, 2, 1], dtype=dtype)
        model = FixedThresholdClassifier(FrozenEstimator(
            LogisticRegression(fit_intercept=False).fit(X, y)
        ))
        yield "default", dtype, model.predict(test)
        yield "weighted-score", dtype, model.score(test, truth, sample_weight=weights)
        strided = np.empty(10, dtype=dtype)[::-2]
        strided[:] = weights
        yield "strided-score", dtype, model.score(test, truth, sample_weight=strided)
        model.set_params(threshold=0.9)
        yield "probability", dtype, model.predict(test)
        model.set_params(pos_label=-3, response_method="predict_proba")
        yield "first-probability", dtype, model.predict(test)
        model.set_params(threshold=1.2, response_method="decision_function")
        yield "first-decision", dtype, model.predict(test)
        model.set_params(threshold="auto")
        yield "first-decision-auto", dtype, model.predict(test)
        fitted = model.fit(X, y[::-1])
        yield "frozen-refit", dtype, fitted.predict(test)
        yield "frozen-clone", dtype, clone(fitted).predict(test)

    X = np.array([-3, -2, -1, 1, 2, 3], dtype=np.float64)[:, None]
    y = np.array([-3, -3, -3, 7, 7, 7], dtype=np.int64)
    weights = np.array([2, 1, 3, 2, 1, 4], dtype=np.float64)
    estimator = LogisticRegression(fit_intercept=False)
    model = FixedThresholdClassifier(estimator, threshold=0.78)
    with config_context(enable_metadata_routing=True):
        estimator.set_fit_request(sample_weight=True)
        model.fit(X, y, sample_weight=weights)
        yield "weighted-fit", "float64", model.predict(X)
        model.fit(X, y[::-1], sample_weight=weights)
        yield "repeat-fit", "float64", model.predict(X)
    lda = FixedThresholdClassifier(
        FrozenEstimator(LinearDiscriminantAnalysis().fit(X, y)), threshold=0.9,
    )
    yield "lda-threshold", "float64", lda.predict(X)
    yield "lda-score", "float64", lda.score(X, y)


def main():
    binary = ROOT / "builds/tests/frozen/test_fixed_threshold"
    output = subprocess.check_output([binary], text=True)
    actual = {}
    for line in output.splitlines():
        case, dtype, value = line.split(maxsplit=2)
        actual[case, dtype] = ast.literal_eval(value)

    oracle = {(case, dtype): value for case, dtype, value in expected()}
    assert actual.keys() == oracle.keys(), (actual.keys(), oracle.keys())
    for key, value in oracle.items():
        assert_array_equal(actual[key], value, err_msg=str(key))
    print(f"PASS {len(oracle)} exact fixed-threshold sklearn comparisons")


if __name__ == "__main__":
    main()
