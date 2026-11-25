"""Compare regression objectives and staged trees with source-built sklearn."""

from pathlib import Path
import subprocess

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sklearn
from sklearn._loss.loss import (
    AbsoluteError,
    HalfGammaLoss,
    HalfPoissonLoss,
    HalfSquaredError,
    PinballLoss,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from hgb_splitting_parity import bits64

BUILDS = Path(__file__).resolve().parents[1] / "builds"
LOSSES = {
    "squared_error": HalfSquaredError,
    "absolute_error": AbsoluteError,
    "poisson": HalfPoissonLoss,
    "gamma": HalfGammaLoss,
    "quantile": PinballLoss,
}


def cases():
    for name in LOSSES:
        quantiles = (0.1, 0.5, 0.83) if name == "quantile" else (0.5,)
        for quantile in quantiles:
            for weighted in (False, True):
                for seed, n in enumerate((1, 6, 67, 147, 221)):
                    rng = np.random.default_rng(seed)
                    X = rng.normal(size=(n, 4))
                    X[10::11, 1] = np.nan
                    signal = X[:, 0] + 1.5 * np.nan_to_num(X[:, 1])
                    signal += (X[:, 2] < 0) * 2.0 + 0.3 * rng.normal(size=n)
                    y = np.exp(signal / 3) if name in ("poisson", "gamma") else signal
                    if name == "poisson" and n > 1:
                        y[::5] = 0.0
                    if weighted:
                        weights = rng.integers(1, 12, n).astype(np.float64) / 4
                        weights[1::7] = 0.0
                    else:
                        weights = None
                    query = np.vstack((X[:8], rng.normal(size=(13, 4))))
                    query[::3, 0] = np.nan
                    raw = rng.normal(size=n) * 3
                    if name in ("absolute_error", "quantile"):
                        raw[::3] = y[::3]  # subgradient at a kink
                    yield (name, quantile, X, query, y, raw, weights, 3 if n < 7 else 7)
    X = np.arange(8, dtype=np.float64).reshape(-1, 1)
    raw = np.array([-75.0, -50.0, -35.0, -20.0, 0.0, 20.0, 35.0, 50.0])
    for name in ("poisson", "gamma"):
        y = np.array([1.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0, 8.0]) * 1e-20
        if name == "poisson":
            y[::2] = 0.0
        yield (name, 0.5, X, X, y, raw, None, 3)


def reference(case):
    name, quantile, X, query, y, raw, weights, n_stages = case
    params = {"quantile": quantile} if name == "quantile" else {}
    loss = LOSSES[name](sample_weight=weights, **params)
    baseline = loss.fit_intercept_only(y, weights)
    score = loss(y, raw, weights)
    gradients = np.empty_like(y, dtype=np.float32)
    hessians = np.empty_like(gradients)
    gradient_args = {
        "sample_weight": weights,
        "gradient_out": gradients,
    }
    if loss.constant_hessian:
        loss.gradient(y, raw, **gradient_args)
        hessians.fill(1)
    else:
        loss.gradient_hessian(y, raw, hessian_out=hessians, **gradient_args)
    derivative_bits = np.column_stack([gradients.view(np.int32), hessians.view(np.int32)])
    stage_scores = []
    predictions = []
    if n_stages:
        model = HistGradientBoostingRegressor(
            loss=name, max_iter=n_stages, early_stopping=False, random_state=0,
            max_bins=16, max_leaf_nodes=5, max_depth=3, learning_rate=0.125,
            min_samples_leaf=3, l2_regularization=0.25, **params,
        ).fit(X, y, sample_weight=weights)
        for raw_train in model._staged_raw_predict(X):
            stage_scores.append(loss(y, raw_train[:, 0], weights))
        for raw_query, prediction in zip(
            model._staged_raw_predict(query), model.staged_predict(query), strict=True
        ):
            predictions.append(np.column_stack([raw_query[:, 0], prediction]))
    return [baseline, score], derivative_bits, stage_scores, predictions


def fixture(case):
    name, quantile, X, query, y, raw, weights, n_stages = case
    fields = [
        name, *bits64([quantile]), *X.shape, len(query), n_stages, int(weights is not None),
        *bits64(X.ravel()), *bits64(query.ravel()), *bits64(y), *bits64(raw),
    ]
    if weights is not None:
        fields.extend(bits64(weights))
    return " ".join(map(str, fields))


def main():
    assert Path(sklearn.__file__).resolve().is_relative_to(BUILDS / "oracle-env")
    all_cases = list(cases())
    fixtures = BUILDS / "hgb_losses_fixtures.txt"
    fixtures.write_text("".join(f"{fixture(case)}\n" for case in all_cases))
    completed = subprocess.run(
        [BUILDS / "hgb_losses_driver", fixtures], check=True, capture_output=True, text=True
    )
    records = iter(completed.stdout.splitlines())

    def read_ints(n, dtype=np.int64):
        return np.array([next(records).split() for _ in range(n)], dtype=dtype)

    derivatives = predictions = scores = 0
    with threadpool_limits(limits=1):
        for index, case in enumerate(all_cases):
            name, quantile, X, query, y, _, weights, n_stages = case
            context = f"case {index}, {name}, q={quantile}, n={len(X)}, weighted={weights is not None}"
            expected, derivative_bits, stage_scores, stage_predictions = reference(case)
            assert_allclose(read_ints(1).view(np.float64)[0], expected,
                            atol=1e-12, rtol=1e-12, err_msg=context)
            assert_array_equal(read_ints(len(y), np.int32), derivative_bits, err_msg=context)
            derivatives += 2 * len(y)
            for stage in range(n_stages):
                assert_allclose(read_ints(1).view(np.float64).item(), stage_scores[stage],
                                atol=1e-12, rtol=1e-12, err_msg=context)
                assert_allclose(read_ints(len(query)).view(np.float64), stage_predictions[stage],
                                atol=1e-12, rtol=1e-12, err_msg=context)
                predictions += 2 * len(query)
                scores += 1
    assert next(records, None) is None
    print(f"hgb regression loss parity: {len(all_cases)} cases, {derivatives} exact "
          f"derivative values, {scores} stage losses and {predictions} raw/linked "
          "stage predictions within atol=rtol=1e-12")


if __name__ == "__main__":
    main()
