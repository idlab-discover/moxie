"""Compare the native scorer probe with the pinned sklearn scorer."""
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.metrics import get_scorer


def main(binary):
    oracle = Path(os.environ["MOXIE_SKLEARN_ORACLE"]).resolve()
    assert Path(sklearn.__file__).resolve().is_relative_to(oracle)
    reference = Path(__file__).resolve().parents[6] / "scikit-learn"
    revision = subprocess.check_output(
        ["git", "-C", reference, "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
    scorer = get_scorer("neg_mean_squared_error")
    actual = {
        (dtype, case): float(score)
        for dtype, case, score in (
            line.split()
            for line in subprocess.check_output([Path(binary).resolve()], text=True).splitlines()
        )
    }
    expected = {}
    for dtype in (np.float32, np.float64):
        name = np.dtype(dtype).name
        X = np.zeros((3, 1), dtype=dtype)
        y = np.arange(6, dtype=dtype).reshape(3, 2) * np.array([1, 2], dtype=dtype)
        y /= dtype(3)
        weights = np.array([0.25, -0.125, 1.0], dtype=dtype)
        model = DummyRegressor().fit(X, y)
        expected[name, "multioutput"] = scorer(model, X, y[::-1])
        expected[name, "weighted"] = scorer(model, X, y[::-1], sample_weight=weights[::-1])
        vector = np.arange(3, dtype=dtype)
        model.fit(X, vector)
        expected[name, "singleton_column"] = scorer(model, X, vector[:, None])

    assert actual.keys() == expected.keys(), actual
    for key, value in expected.items():
        tolerance = 2e-7 if key[0] == "float32" else 1e-13
        np.testing.assert_allclose(actual[key], value, rtol=tolerance, atol=tolerance, err_msg=str(key))
    print(f"PASS negative MSE scorer parity ({len(expected)} cases)")


if __name__ == "__main__":
    main(sys.argv[1])
