"""Compare search fold scores for fit-time group splitters."""
from collections import defaultdict
import subprocess

import numpy as np
import sklearn


def check_fit_time_cv(reference, split, dummy, binary):
    rows = np.arange(12)
    X = rows.astype(float).reshape(-1, 1)
    y = ((rows * 5) % 13).astype(float)
    groups = rows % 4
    weights = rows.astype(float) + 1
    constants = {"constant": [2.0, 5.0, 8.0, 11.0]}
    model = dummy.DummyRegressor(strategy="constant", constant=2.0)

    actual = defaultdict(list)
    for line in subprocess.check_output([str(binary)], text=True).splitlines():
        case, field, *values = line.split()
        actual[case, field].append([float(value) for value in values])

    def compare(case, search):
        results = search.cv_results_
        candidates = [[p["constant"], rank] for p, rank in zip(
            results["params"], results["rank_test_score"], strict=True,
        )]
        np.testing.assert_array_equal(actual.pop((case, "candidate")), candidates)
        np.testing.assert_array_equal(
            actual.pop((case, "best")), [[search.best_index_, search.predict(X)[0]]],
        )
        for field in ("train", "test"):
            scores = np.column_stack([
                results[f"split{fold}_{field}_score"] for fold in range(search.n_splits_)
            ])
            native = np.asarray(actual.pop((case, field))).reshape(scores.shape)
            np.testing.assert_allclose(native, scores, rtol=0, atol=1e-14)

    grid = reference.GridSearchCV(model, constants, cv=split.GroupKFold(3), return_train_score=True)
    compare("grid", grid.fit(X, y, groups=groups))

    train, _ = next(split.KFold(3).split(X))
    with sklearn.config_context(enable_metadata_routing=True):
        weighted_model = model.set_fit_request(sample_weight=True).set_score_request(sample_weight=True)
        random = reference.RandomizedSearchCV(
            weighted_model, constants, n_iter=3, random_state=42,
            cv=split.LeaveOneGroupOut(), return_train_score=True,
        )
        compare("random", random.fit(X[train], y[train], groups=groups[train], sample_weight=weights[train]))

    assert not actual, actual.keys()
    print("PASS pinned sklearn fit-time CV: GroupKFold grid and weighted LeaveOneGroupOut randomized search")
