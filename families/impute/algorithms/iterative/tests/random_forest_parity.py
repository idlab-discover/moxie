"""Compare the iterative random forest gallery configuration with pinned sklearn."""

import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

from parity import REVISION, SOURCE, oracle
from knn_regressor_parity import matrix


def fixture(start, rows):
    i = np.arange(start, start + rows)
    values = np.column_stack((i, i**2 / 32, i**3 / 1024)).astype(np.float64)
    j = np.arange(3)
    values[(i[:, None] % 7 == j) & ((start != 0) | (j < 2))] = np.nan
    return values


def main(binary):
    iterative, _ = oracle()
    package_dir = Path(sklearn.__file__).parent.parent
    forest_path = "sklearn/ensemble/_forest.py"
    for relative in (
        forest_path,
        "sklearn/tree/_classes.py",
        "sklearn/tree/_criterion.pyx",
        "sklearn/tree/_partitioner.pyx",
        "sklearn/tree/_splitter.pyx",
        "sklearn/tree/_tree.pyx",
    ):
        pinned = subprocess.check_output(["git", "-C", SOURCE, "show", f"{REVISION}:{relative}"])
        assert (package_dir / relative).read_bytes() == pinned, relative
    for method in (RandomForestRegressor.fit, RandomForestRegressor.predict):
        assert Path(inspect.unwrap(method).__code__.co_filename) == package_dir / forest_path
    model = iterative(
        estimator=RandomForestRegressor(
            n_estimators=5, max_depth=10, bootstrap=True, max_samples=0.5,
            n_jobs=2, random_state=0,
        ),
        random_state=0, max_iter=40, tol=1e-1,
    )
    train, query = fixture(0, 32), fixture(32, 7)
    fitted = model.fit_transform(train)
    replay = model.transform(query)
    assert fitted.dtype == replay.dtype == np.float64

    lines = iter(subprocess.check_output([binary], text=True).splitlines())
    assert tuple(map(int, next(lines).split())) == (
        model.n_iter_, model.n_features_with_missing_, len(model.imputation_sequence_),
    )
    for step in model.imputation_sequence_:
        trees = step.estimator.estimators_
        assert tuple(map(int, next(lines).split())) == (
            step.feat_idx, step.estimator.n_features_in_, len(trees),
        )
        assert [int(next(lines)) for _ in step.neighbor_feat_idx] == list(step.neighbor_feat_idx)
        assert [int(next(lines)) for _ in trees] == [tree.random_state for tree in trees]
    max_error = 0.0
    for source, expected in ((train, fitted), (query, replay)):
        actual = matrix(lines, expected.shape)
        observed = ~np.isnan(source)
        np.testing.assert_array_equal(actual[observed], source[observed])
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        max_error = max(max_error, np.max(np.abs(actual - expected)))
    assert next(lines, None) is None
    print(f"PASS Float64 against {REVISION}: {model.n_iter_} rounds, "
          f"exact sequence and forest seeds, max fill error {max_error:.3g}")


if __name__ == "__main__":
    main(sys.argv[1])
