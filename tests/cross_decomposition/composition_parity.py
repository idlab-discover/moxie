"""Compare PLS dense CV/search with the pinned scikit-learn sources.

The locked scikit-learn wheel supplies scoring and validation infrastructure.
"""

import importlib.util
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn

sys.dont_write_bytecode = True
from pls.parity import REFERENCE, REVISION, load_reference, verify_reference


def load_selection_module(name):
    path = f"sklearn/model_selection/_{name}.py"
    assert not subprocess.check_output(
        ["git", "-C", REFERENCE, "status", "--porcelain", "--", path], text=True
    )
    spec = importlib.util.spec_from_file_location(
        f"sklearn.model_selection._moxie_{name}_reference", REFERENCE / path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(dtype, outputs):
    """Reproduce the deterministic data() and targets() composition fixture."""
    row = np.arange(53)[:, None]
    col = np.arange(4)
    latent = np.sin(row * 0.37) * 2.0
    noise = np.sin((row * 31 + col * 17) * 12.9898)
    noise += 0.5 * np.cos((row * 7 + col) * 3.1)
    x = (latent * (col % 3 + 1) + noise * (col + 1) + 5.0 * col).astype(dtype)
    y = np.empty((len(row), outputs or 1), dtype=dtype)
    for target in range(y.shape[1]):
        values = np.full(len(row), 3.0 + target)
        for j in col:
            values += x[:, j].astype(np.float64) * ((j + target) % 4 - 1) * 0.5
        y[:, target] = values + np.cos((row[:, 0] * 13 + target) * 1.7)
    return x, y if outputs else y.ravel()


def main(runner):
    assert sklearn.__version__ == "1.9.1"
    verify_reference()
    pls = load_reference()
    search = load_selection_module("search")
    search.check_cv = load_selection_module("split").check_cv

    records = defaultdict(list)
    for line in subprocess.check_output([runner.resolve()], text=True).splitlines():
        if line.startswith("PASS:"):
            continue
        name, outputs, *values = line.split()
        records[name, int(outputs)].append([float(value) for value in values])

    candidates = [3, 1, 2]
    for dtype, outputs in ((np.float64, 0), (np.float32, 2)):
        x, y = fixture(dtype, outputs)
        reference = search.GridSearchCV(
            pls.PLSRegression(), {"n_components": candidates}, error_score="raise"
        ).fit(x, y)
        folds = np.asarray(records["fold", outputs])
        expected_order = [(count, fold) for count in candidates for fold in range(5)]
        np.testing.assert_array_equal(folds[:, :2], expected_order)
        expected_scores = [
            reference.cv_results_[f"split{fold}_test_score"][candidate]
            for candidate in range(len(candidates))
            for fold in range(5)
        ]
        # Both estimators promote to Float64. Allow for reduction order and
        # platform math library rounding in the deterministic fixture.
        score_tol = 1e-10
        np.testing.assert_allclose(folds[:, 2], expected_scores, rtol=0, atol=score_tol)
        [(best_index, best_score)] = records["winner", outputs]
        assert best_index == reference.best_index_
        np.testing.assert_allclose(best_score, reference.best_score_, rtol=0, atol=score_tol)

        prediction = reference.predict(x)
        actual = np.asarray(records["predict", outputs])
        np.testing.assert_array_equal(actual[:, 0], np.arange(prediction.size))
        np.testing.assert_allclose(actual[:, 1], prediction.ravel(), rtol=1e-10, atol=1e-10)

    print(f"PASS pinned PLS search parity with {REVISION} (two fixtures, 30 fold scores)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
