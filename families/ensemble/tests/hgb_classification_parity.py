"""Validate native log losses and their composition with histogram trees."""

from pathlib import Path
import subprocess

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sklearn
from sklearn._loss.loss import HalfBinomialLoss, HalfMultinomialLoss
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from hgb_splitting_parity import bits64

BUILDS = Path(__file__).resolve().parents[1] / "builds"


def cases():
    for n_classes in (1, 2, 3, 5, 11):
        for n in (21, 69, 143):
            for mode in ("unweighted", "weighted", "single_weighted_class"):
                rng = np.random.default_rng(n + n_classes)
                X = rng.normal(size=(n, 4))
                X[2::9, 1] = np.nan
                order = np.argsort(X[:, 0] + 0.5 * np.nan_to_num(X[:, 1]) + X[:, 2] ** 2)
                y = np.empty(n)
                y[order] = np.arange(n) * n_classes // n
                query = np.concatenate((X[:8], rng.normal(size=(7, 4))))
                query[::3, 2] = np.nan
                raw = rng.normal(size=(n, 1 if n_classes <= 2 else n_classes)) * 3
                raw[0, :] = -1000
                raw[1, :] = 1000
                raw[2:7, 0] = [-37, -2, 18, 33.3, 37]
                weights = None
                if mode != "unweighted":
                    weights = rng.integers(1, 18, size=n).astype(np.float64) / 8
                    weights[::7] = 0
                    if mode == "single_weighted_class":
                        weights[y != n_classes - 1] = 0
                        weights[order[-1]] = 1.0
                yield n_classes, X, query, y, raw, weights


def fixture(case):
    n_classes, X, query, y, raw, weights = case
    return " ".join(map(str, [
        n_classes, *X.shape, len(query), 4, int(weights is not None),
        *bits64(X.ravel()), *bits64(query.ravel()), *bits64(y), *bits64(raw.ravel()),
        *([] if weights is None else bits64(weights)),
    ]))


def main():
    assert Path(sklearn.__file__).resolve().is_relative_to(BUILDS / "oracle-env")
    fixtures = BUILDS / "hgb_classification_fixtures.txt"
    all_cases = list(cases())
    fixtures.write_text("".join(f"{fixture(case)}\n" for case in all_cases))
    completed = subprocess.run(
        [BUILDS / "hgb_classification_driver", fixtures], check=True,
        capture_output=True, text=True,
    )
    records = iter(completed.stdout.splitlines())
    derivatives = predictions = scores = 0

    def compare(expected, context):
        expected = np.asarray(expected, dtype=np.float64)
        values = np.array([next(records) for _ in range(expected.size)], dtype=np.int64)
        assert_allclose(values.view(np.float64), expected.ravel(),
                        atol=1e-12, rtol=1e-12, err_msg=context)

    with threadpool_limits(limits=1):
        for index, (n_classes, X, query, y, raw, weights) in enumerate(all_cases):
            binary = n_classes <= 2
            context = f"case {index}, n_classes={n_classes}, n={len(y)}"
            loss = HalfBinomialLoss() if binary else HalfMultinomialLoss(n_classes=n_classes)
            raw_loss = raw[:, 0] if binary else raw
            compare(loss.fit_intercept_only(y, weights), context)
            compare(loss(y, raw_loss, weights), context)
            compare(loss.predict_proba(raw_loss), context)
            gradient = np.empty_like(raw_loss, dtype=np.float32)
            hessian = np.empty_like(gradient)
            loss.gradient_hessian(y, raw_loss, weights, gradient_out=gradient, hessian_out=hessian)
            expected = np.column_stack([
                gradient.ravel(order="F").view(np.int32),
                hessian.ravel(order="F").view(np.int32),
            ])
            actual = np.array([next(records).split() for _ in expected], dtype=np.int32)
            assert_array_equal(actual, expected, err_msg=context)
            derivatives += expected.size

            model = HistGradientBoostingClassifier(
                max_iter=4, early_stopping=False, random_state=0, max_bins=16,
                max_leaf_nodes=5, max_depth=3, learning_rate=0.125,
                min_samples_leaf=3, l2_regularization=0.25,
            ).fit(X, y, sample_weight=weights)
            for train_raw, query_raw, probability in zip(
                model._staged_raw_predict(X), model._staged_raw_predict(query),
                model.staged_predict_proba(query), strict=True,
            ):
                compare(loss(y, train_raw[:, 0] if binary else train_raw, weights), context)
                compare(query_raw, context)
                compare(probability, context)
                scores += 1
                predictions += query_raw.size + probability.size

    assert next(records, None) is None
    print(f"hgb classification loss parity: {len(all_cases)} cases, {derivatives} exact "
          f"derivative values, {scores} stage losses and {predictions} raw/probability "
          "stage predictions within atol=rtol=1e-12")


if __name__ == "__main__":
    main()
