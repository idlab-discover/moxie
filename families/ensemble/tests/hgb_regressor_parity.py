"""Compare the public dense histogram regressor with the pinned sklearn build."""

from pathlib import Path
import subprocess

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from hgb_losses_parity import cases as loss_cases
from hgb_splitting_parity import bits64

BUILDS = Path(__file__).resolve().parents[1] / "builds"


def cases():
    for index, (loss, q, X, query, y, _, weights, stages) in enumerate(loss_cases()):
        yield loss, q, X, query, y, weights, stages, 0, 0, False, index
    for seed in (0, 41, 421):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(167, 4))
        X[::17, 1] = np.nan
        y = X[:, 0] + np.nan_to_num(X[:, 1]) - (X[:, 2] > 0) + rng.normal(size=len(X))
        query = rng.normal(size=(31, 4))
        query[::4, 1] = np.nan
        weights = rng.integers(1, 8, len(X)) / 4
        for mode in (0, 1, 2, 3, 5):
            for constraints, sampled in ((0, True), (1, False), (1, True)):
                yield "squared_error", 0.5, X, query, y, weights, 18, mode, constraints, sampled, seed
        for constraints in (2, 3):
            yield "squared_error", 0.5, X, query, y, weights, 6, 2, constraints, False, seed
    # The original Poisson target is valid even when its sole positive row
    # lands in the internal validation set.
    X = np.arange(10, dtype=np.float64).reshape(-1, 1)
    y = np.zeros(len(X))
    split_seed = np.random.RandomState(2).randint(np.iinfo(np.uint32).max, dtype="uint64")
    y[np.random.RandomState(split_seed).permutation(len(y))[0]] = 1.0
    yield "poisson", 0.5, X, X, y, None, 4, 2, 0, False, 2
    rng = np.random.default_rng(123)
    X = rng.normal(size=(10_001, 2))
    y = np.ones(len(X))
    yield "squared_error", 0.5, X, X[:5], y, None, 10, 4, 0, False, 123


def explicit_validation(query):
    return np.arange(len(query)) / 12, np.linspace(0.5, 2, len(query))


def fixture(case):
    loss, q, X, query, y, weights, stages, mode, constraints, sampled, seed = case
    fields = [
        loss, *bits64([q]), *X.shape, len(query), stages, int(weights is not None),
        mode, constraints, int(sampled), seed,
        *bits64(X.ravel()), *bits64(query.ravel()), *bits64(y),
    ]
    if weights is not None:
        fields.extend(bits64(weights))
    if mode == 3:
        for values in explicit_validation(query):
            fields.extend(bits64(values))
    return " ".join(map(str, fields))


def reference(case):
    loss, q, X, query, y, weights, stages, mode, constraints, sampled, seed = case
    validation_size = {1: None, 5: 9}.get(mode, 0.1)
    validation = {}
    if mode == 3:
        y_val, w_val = explicit_validation(query)
        validation = dict(X_val=query, y_val=y_val, sample_weight_val=w_val)
    model = HistGradientBoostingRegressor(
        loss=loss, quantile=q, max_iter=stages, early_stopping="auto" if mode == 4 else mode != 0,
        max_bins=16, max_leaf_nodes=5, max_depth=3, learning_rate=0.125,
        min_samples_leaf=3, l2_regularization=0.25, max_features=0.6 if sampled else 1.0,
        validation_fraction=validation_size, n_iter_no_change=3, tol=0.125, random_state=seed,
        monotonic_cst=[1, 1, 0, 0] if constraints == 1 else None,
        interaction_cst={1: [[0, 1], [2, 3]], 2: "pairwise", 3: "no_interactions"}.get(constraints),
    ).fit(X, y, sample_weight=weights, **validation)
    attrs = [model.n_iter_, model.n_features_in_, model.n_trees_per_iteration_,
             model.do_early_stopping_, model.is_categorical_ is not None,
             len(model.train_score_), len(model.validation_score_)]
    return attrs, model.score(X, y, sample_weight=weights), model.train_score_, model.validation_score_, list(model.staged_predict(query)), model.predict(query)


def main():
    assert Path(sklearn.__file__).resolve().is_relative_to(BUILDS / "oracle-env")
    all_cases = list(cases())
    fixtures = BUILDS / "hgb_regressor_fixtures.txt"
    fixtures.write_text("".join(f"{fixture(case)}\n" for case in all_cases))
    completed = subprocess.run(
        [BUILDS / "hgb_regressor_driver", fixtures], check=True, capture_output=True, text=True
    )
    records = iter(completed.stdout.splitlines())

    def read_values(n):
        return np.array([next(records) for _ in range(n)], dtype=np.int64).view(np.float64)

    scores = predictions = 0
    with threadpool_limits(limits=1):
        for index, case in enumerate(all_cases):
            attrs, r2, train, val, stages, prediction = reference(case)
            context = f"case {index} {case[0]}, mode={case[7]}, seed={case[10]}"
            assert_array_equal(list(map(int, next(records).split())), attrs, err_msg=context)
            for expected in ([r2], train, val):
                assert_allclose(read_values(len(expected)), expected, atol=1e-12, rtol=1e-12,
                                equal_nan=True, err_msg=context)
                scores += len(expected)
            for expected in [*stages, prediction]:
                assert_allclose(read_values(len(expected)), expected, atol=1e-12, rtol=1e-12,
                                err_msg=context)
                predictions += len(expected)
    assert next(records, None) is None
    print(f"PASS: {len(all_cases)} regression fits, {scores} scores, {predictions} predictions")


if __name__ == "__main__":
    main()
