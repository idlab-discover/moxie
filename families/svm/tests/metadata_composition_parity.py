"""Compare alias-routed SVM multioutput fits against pinned scikit-learn."""

from collections import defaultdict
from pathlib import Path
import subprocess

import numpy as np
from sklearn import config_context, multioutput, svm
from sklearn.base import clone
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.utils import _metadata_requests


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[2] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"


def main():
    assert subprocess.check_output(
        ["git", "-C", REFERENCE, "rev-parse", "HEAD"], text=True,
    ).strip() == REVISION
    for module in (multioutput, svm._base, svm._classes, _metadata_requests):
        source = Path(*module.__name__.split(".")).with_suffix(".py")
        expected = subprocess.check_output(["git", "-C", REFERENCE, "show", f"{REVISION}:{source}"])
        assert Path(module.__file__).read_bytes() == expected, module.__file__

    rows = np.arange(8)
    X = np.column_stack((rows - 3.5, rows % 3))
    classification = np.column_stack((rows >= 4, rows % 2)).astype(np.int64)
    regression = np.column_stack((rows % 3 - 0.5 * rows, rows % 2 + 0.25 * rows))
    weights = np.ones(len(rows))
    weights[2] = 8
    weights[5] = 2

    actual = defaultdict(list)
    output = subprocess.check_output([ROOT / "builds/test_metadata_composition"], text=True)
    for line in output.splitlines():
        name, value = line.split()
        actual[name].append(float(value))

    cases = [
        (svm.SVC(kernel="linear", tol=1e-9), classification),
        (svm.NuSVC(kernel="linear", nu=0.3, tol=1e-9), classification),
        (svm.LinearSVC(dual=False, tol=1e-9, random_state=0), classification),
        (svm.SVR(kernel="linear", tol=1e-9), regression),
        (svm.NuSVR(kernel="rbf", tol=1e-9), regression),
        (svm.LinearSVR(
            loss="squared_epsilon_insensitive", dual=False, tol=1e-9, random_state=0,
        ), regression),
    ]
    with config_context(enable_metadata_routing=True):
        with np.testing.assert_raises(UnsetMetadataPassedError):
            multioutput.MultiOutputRegressor(svm.SVR(kernel="linear")).fit(
                X, regression, sample_weight=weights,
            )
        for child, y in cases:
            parent = (
                multioutput.MultiOutputClassifier if y is classification
                else multioutput.MultiOutputRegressor
            )
            name = type(child).__name__
            child.set_fit_request(sample_weight="weights").set_score_request(sample_weight=True)
            routed = parent(child).fit(X, y, weights=weights).predict(X)
            plain = parent(clone(child)).fit(X, y).predict(X)
            assert np.any(np.abs(routed - plain) > 1e-5), name
            if y is classification:
                np.testing.assert_array_equal(actual[name], routed.ravel(), err_msg=name)
            else:
                # Solver termination and reduction order may differ in Float64.
                np.testing.assert_allclose(actual[name], routed.ravel(), rtol=2e-6, atol=2e-7, err_msg=name)

    assert actual.keys() == {type(child).__name__ for child, _ in cases}
    print("PASS six SVM multioutput metadata fits against pinned sklearn (96 predictions)")


if __name__ == "__main__":
    main()
