"""Compare splitter-based scoring, returned folds and fit weights to pinned sklearn."""
from collections import defaultdict
import subprocess

import numpy as np


def check_splitter_cv(reference, split, dummy, binary):
    actual = defaultdict(list)
    for line in subprocess.check_output([str(binary)], text=True).splitlines():
        case, fold, kind, value = line.split()
        actual[case, int(fold), kind].append(float(value))

    rows = np.arange(12)
    X = rows.astype(np.float64).reshape(-1, 1)
    y = ((rows * 5) % 13).astype(np.float64)
    model = dummy.DummyRegressor()
    groups = rows % 4
    labels = rows % 2
    grouped = split.GroupKFold(3)
    stratified = split.StratifiedKFold(3, shuffle=True, random_state=42)
    # An explicit split target selects folds independently of the regression
    # target. sklearn accepts the resulting fold iterable for that case.
    cases = [
        ("grouped", grouped, groups, {"sample_weight": rows.astype(float) + 1}),
        ("stratified", list(stratified.split(X, labels)), None, None),
    ]
    for case, cv, split_groups, params in cases:
        result = reference["cross_validate"](
            model, X, y, cv=cv, groups=split_groups, params=params,
            return_train_score=True, return_indices=True, return_estimator=True,
        )
        scores = reference["cross_val_score"](
            model, X, y, cv=cv, groups=split_groups, params=params,
        )
        for fold in range(len(scores)):
            for side in ("train", "test"):
                np.testing.assert_array_equal(
                    actual.pop((case, fold, f"{side}_index")), result["indices"][side][fold],
                )
                np.testing.assert_allclose(
                    actual.pop((case, fold, side)), [result[f"{side}_score"][fold]],
                    rtol=0, atol=1e-13,
                )
            np.testing.assert_allclose(
                actual.pop((case, fold, "score")), [scores[fold]], rtol=0, atol=1e-13,
            )
    assert not actual, actual
    print("PASS splitter CV: six exact folds and 18 scores within 1e-13")
