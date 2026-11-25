# Gallery scenario adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
"""Compare the native Iris runner with the pinned search, splitter and CV bodies."""
from collections import defaultdict
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def check_svc_search(reference, cross_val_score, kfold, svc, iris_path, binary):
    iris = np.loadtxt(iris_path, delimiter=",", skiprows=1)
    X = np.ascontiguousarray(iris[:, :4])
    y = iris[:, 4].astype(int)
    with tempfile.TemporaryDirectory(prefix="moxie-iris-search-") as temporary:
        fixture = Path(temporary) / "iris"
        with fixture.open("w") as stream:
            for label, row in zip(y, X.view(np.int64)):
                print(label, *row, file=stream)
        output = subprocess.check_output([str(binary), str(fixture)], text=True)

    actual = defaultdict(list)
    for line in output.splitlines():
        kind, trial, field, *values = line.split()
        actual[kind, int(trial), field].append([float(value) for value in values])

    def compare(kind, trial, field, expected, *, rtol=0, atol=0):
        key = kind, trial, field
        native = np.asarray(actual.pop(key))
        expected = np.asarray(expected)
        assert native.size == expected.size, (key, native.shape, expected.shape)
        np.testing.assert_allclose(
            native.reshape(expected.shape), expected, rtol=rtol, atol=atol,
            err_msg=str(key),
        )

    def compare_search(kind, trial, search):
        results = search.cv_results_
        candidates = [(p["C"], p["gamma"], rank) for p, rank in zip(
            results["params"], results["rank_test_score"], strict=True,
        )]
        compare(kind, trial, "candidate", candidates)
        summaries = np.column_stack([
            results[f"{stat}_{field}_score"]
            for field in ("test", "train") for stat in ("mean", "std")
        ])
        compare(kind, trial, "summary", summaries, atol=1e-15)
        compare(kind, trial, "best", [
            search.best_index_, search.best_score_, search.best_params_["C"],
            search.best_params_["gamma"],
        ], atol=1e-15)
        for field in ("train", "test"):
            expected = np.column_stack([
                results[f"split{fold}_{field}_score"] for fold in range(4)
            ])
            compare(kind, trial, field, expected, atol=1e-15)
        if trial == 0:
            compare(kind, trial, "predict", search.predict(X))
            compare(kind, trial, "support", search.best_estimator_.support_)
            # Kernel arithmetic can round differently in the native SVC.
            compare(kind, trial, "decision", search.decision_function(X), rtol=1e-9, atol=1e-10)

    def shuffled_folds(seed):
        return kfold(4, shuffle=True, random_state=seed)

    grid = {"C": [1.0, 10.0, 100.0], "gamma": [0.01, 0.1]}
    model = svc(kernel="rbf")
    for trial in range(30):
        search = reference.GridSearchCV(
            model, grid, cv=shuffled_folds(trial), return_train_score=True,
        ).fit(X, y)
        compare_search("grid", trial, search)
        nested = reference.GridSearchCV(model, grid, cv=shuffled_folds(trial))
        compare(
            "grid", trial, "nested",
            cross_val_score(nested, X, y, cv=shuffled_folds(trial)), atol=1e-15,
        )

    random = reference.RandomizedSearchCV(
        model, grid, n_iter=4, random_state=0, cv=shuffled_folds(0), return_train_score=True,
    ).fit(X, y)
    compare_search("random", 0, random)
    assert not actual, actual.keys()
    print("PASS Iris SVC: 30 grid searches, 120 outer-fold scores, and seeded randomized search")
    print("PASS Iris SVC: refit predictions, support indices, and decision values for both searches")
