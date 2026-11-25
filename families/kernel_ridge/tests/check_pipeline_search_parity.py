"""Compare the composition contract with a build of the local sklearn oracle."""
from collections import defaultdict
from pathlib import Path
import os
import subprocess

import numpy as np
import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from check import compile, run

ROOT = Path(__file__).resolve().parents[1]
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"


def main():
    # The oracle import must come from the private install of the pinned source.
    oracle = Path(os.environ["MOXIE_SKLEARN_ORACLE"]).resolve()
    assert Path(sklearn.__file__).resolve().is_relative_to(oracle)
    reference = ROOT.parents[2] / "scikit-learn"
    revision = subprocess.check_output(
        ["git", "-C", reference, "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == REVISION

    binary = ROOT / "builds/test_composition"
    compile(ROOT / "tests/test_composition.mojo", binary)
    result = run(binary, "parity", capture=True)
    actual = defaultdict(list)
    for line in result.stdout.splitlines():
        key, *values = line.split()
        if key != "PASS:":
            actual[key].append([float(value) for value in values])

    index = np.arange(40)
    x = (index * 37 % 40) / 8
    y = np.sin(x) + np.where(index % 5 == 0, (index * 13 % 7) / 3 - 1, 0)
    X = x[:, None]
    search = GridSearchCV(
        make_pipeline(StandardScaler(), KernelRidge(kernel="rbf")),
        {"kernelridge__alpha": [0.01, 0.1, 1.0], "kernelridge__gamma": [0.1, 1.0]},
        cv=5,
        error_score="raise",
    ).fit(X, y)
    results = search.cv_results_
    np.testing.assert_array_equal(actual["params"], [
        [p["kernelridge__alpha"], p["kernelridge__gamma"]] for p in results["params"]
    ])
    np.testing.assert_array_equal(np.ravel(actual["rank"]), results["rank_test_score"])
    np.testing.assert_array_equal(np.ravel(actual["best"]), [search.best_index_])
    # Scaling, RBF reduction and Cholesky use different floating-point
    # operation orders. This fixture is Float64 with alpha >= 0.01.
    for observed, expected in (
        (np.reshape(actual["fold"], (6, 5)),
         np.column_stack([results[f"split{i}_test_score"] for i in range(5)])),
        (np.ravel(actual["mean"]), results["mean_test_score"]),
        (np.ravel(actual["prediction"]), search.predict(X)),
    ):
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-12)
    print("PASS: pipeline grid candidate order, 30 fold scores, means, ranks, winner and 40 refit predictions")


if __name__ == "__main__":
    main()
