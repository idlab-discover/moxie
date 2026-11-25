"""Compare iterative k-NN training fills and replay with pinned sklearn."""

import inspect
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import _base as installed_neighbors_base

from parity import REVISION, SOURCE, load, oracle


def fixture(start, rows, dtype):
    i = np.arange(start, start + rows)[:, None]
    j = np.arange(4)
    values = ((((i + 3) * (i + 17 * j + 8) * (j + 9)) % 251) / 8).astype(dtype)
    values[(i % 4 == (j + 1) % 4) & ((start != 0) | (j < 3))] = np.nan
    return values


def matrix(lines, shape):
    assert tuple(map(int, next(lines).split())) == shape
    return np.array([float(next(lines)) for _ in range(np.prod(shape))]).reshape(shape)


def main(binary):
    iterative, _ = oracle()
    regression = "sklearn/neighbors/_regression.py"
    neighbors_base = "sklearn/neighbors/_base.py"
    for path in (regression, neighbors_base):
        pinned = subprocess.check_output(["git", "-C", SOURCE, "show", f"{REVISION}:{path}"])
        assert (SOURCE / path).read_bytes() == pinned, path
    assert Path(installed_neighbors_base.__file__).read_bytes() == (SOURCE / neighbors_base).read_bytes()
    regressor = load(regression).KNeighborsRegressor
    for method in (regressor.fit, regressor.predict):
        assert Path(inspect.unwrap(method).__code__.co_filename) == SOURCE / regression

    lines = iter(subprocess.check_output([binary], text=True).splitlines())
    for dtype, tolerance in ((np.float32, 3e-6), (np.float64, 2e-12)):
        train, query = fixture(0, 32, dtype), fixture(32, 7, dtype)
        model = iterative(
            estimator=regressor(n_neighbors=10), random_state=0, max_iter=40, tol=1e-2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fitted = model.fit_transform(train)
        replay = model.transform(query)
        assert fitted.dtype == replay.dtype == dtype
        assert tuple(map(int, next(lines).split())) == (
            model.n_iter_, model.n_features_with_missing_, len(model.imputation_sequence_),
        )
        for step in model.imputation_sequence_:
            assert tuple(map(int, next(lines).split())) == (
                step.feat_idx, step.estimator.n_features_in_, step.estimator.n_samples_fit_,
            )
        for source, expected in ((train, fitted), (query, replay)):
            actual = matrix(lines, expected.shape)
            observed = ~np.isnan(source)
            np.testing.assert_array_equal(actual[observed], source[observed])
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        print(f"PASS {dtype.__name__}: training and held-out fills, {model.n_iter_} rounds")
    assert next(lines, None) is None


if __name__ == "__main__":
    main(sys.argv[1])
