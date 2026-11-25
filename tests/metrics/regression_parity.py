"""Pinned sklearn cases for the dense Mojo regression metrics."""
import warnings

import numpy as np
from sklearn import config_context
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    explained_variance_score,
    make_scorer,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_score


METRICS = {
    "mae": mean_absolute_error,
    "median": median_absolute_error,
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "r2": r2_score,
    "ev": explained_variance_score,
}


def regression_cases():
    truth = np.array([[1.0, 8.0], [2.0, 8.0], [4.0, 8.0], [6.0, 8.0]])
    pred = np.array([[1.0, 7.0], [3.0, 5.0], [2.0, 7.0], [8.0, 4.0]])
    weights = np.array([1.0, 0.0, 1.0, 2.0])
    outputs = np.array([0.25, 0.75])
    cases = [
        ("vector", [3.0, -0.5, 2.0, 7.0], [[2.5], [0.0], [2.0], [8.0]], None, None, True),
        ("multi", truth, pred, weights, outputs, True),
        ("strided", truth[::-1, ::-1], pred, weights[::-1], outputs[::-1], True),
        ("signed", truth, pred, [-1.0, 2.0, 0.5, 3.0], None, True),
        ("constant_perfect", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], None, None, False),
        ("constant_shifted", [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], None, None, False),
        ("constant_multi", truth, pred, None, None, False),
        ("all_constant", [[1.0, 4.0]] * 3, [[1.0, 3.0], [1.0, 4.0], [1.0, 5.0]], None, None, True),
        ("singleton", [[1.0, 2.0]], [[2.0, 2.0]], None, None, True),
        ("float32", np.array([0.2, 1.7, 2.1], np.float32), np.array([0.1, 1.2, 2.6], np.float32), None, None, True),
        ("mixed", np.array([0, 1, 2], np.int64), np.array([0.1, 1.2, 2.6], np.float32), None, None, True),
        ("integral", np.array([0, 1, 2], np.int64), np.array([2, -1, 4], np.int64), None, None, True),
    ]
    expected = {}
    with warnings.catch_warnings(record=True) as log, np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("always", UndefinedMetricWarning)
        for name, y_true, y_pred, sample_weight, output_weight, force_finite in cases:
            columns = np.shape(y_true)[1] if np.ndim(y_true) == 2 else 1
            aggregate = []
            variance_weighted = []
            for key, metric in METRICS.items():
                kwargs = {"sample_weight": sample_weight}
                if key in ("r2", "ev"):
                    kwargs["force_finite"] = force_finite
                mode = "uniform_average" if output_weight is None else output_weight
                aggregate.append(metric(y_true, y_pred, multioutput=mode, **kwargs))
                raw = metric(y_true, y_pred, multioutput="raw_values", **kwargs)
                # The native raw API always returns a rank-1 array, including undefined R².
                expected[f"{name}_{key}"] = np.broadcast_to(raw, (columns,))
                if columns > 1 and key in ("r2", "ev"):
                    variance_weighted.append(
                        metric(y_true, y_pred, multioutput="variance_weighted", **kwargs)
                    )
            expected[f"{name}_aggregate"] = aggregate
            if columns > 1:
                expected[f"{name}_variance"] = variance_weighted
            else:
                expected[f"{name}_max"] = [max_error(y_true, y_pred)]
    for name, sw in [
        ("gap", [1, 0, 1, 0]),
        ("leading", [0, 1, 0, 1]),
        ("zero_total", [0, 2, 1, -3]),
    ]:
        expected[f"median_{name}"] = [
            median_absolute_error([0.0] * 4, [1.0, 2.0, 3.0, 4.0], sample_weight=sw)
        ]
    emitted = [f"{entry.category.__name__}: {entry.message}" for entry in log]
    return expected, emitted


def regression_errors():
    vector = [3.0, -0.5, 2.0, 7.0]
    matrix = [[1.0, 8.0], [2.0, 8.0], [4.0, 8.0], [6.0, 8.0]]
    cases = [
        (mean_absolute_error, vector, [0.0, 1.0], {}, ValueError),
        (mean_squared_error, vector, np.zeros((4, 1, 1)), {}, ValueError),
        (r2_score, vector, matrix, {}, ValueError),
        (explained_variance_score, vector, [0, 1, 2, np.nan], {}, ValueError),
        (max_error, matrix, matrix, {}, ValueError),
        (median_absolute_error, [], [], {}, ValueError),
        (mean_absolute_error, np.empty((4, 0)), np.empty((4, 0)), {}, ValueError),
        (mean_absolute_error, vector, vector, {"sample_weight": [0, 1]}, ValueError),
        (mean_absolute_error, vector, vector, {"sample_weight": [0, 1, 2, np.inf]}, ValueError),
        (mean_absolute_error, vector, vector, {"sample_weight": np.arange(4)[:, None]}, ValueError),
        (median_absolute_error, vector, vector, {"sample_weight": [0, 0, 0, 0]}, ValueError),
        (mean_absolute_error, vector, vector, {"sample_weight": [1, -1, 0, 0]}, ZeroDivisionError),
        (mean_absolute_error, vector, vector, {"multioutput": [0, 1]}, ValueError),
        (mean_absolute_error, matrix, matrix, {"multioutput": vector}, ValueError),
        (mean_squared_error, matrix, matrix, {"multioutput": [1, -1]}, ZeroDivisionError),
        (r2_score, matrix, matrix, {"multioutput": "invalid"}, ValueError),
    ]
    for metric, truth, pred, kwargs, error in cases:
        with np.testing.assert_raises(error):
            metric(truth, pred, **kwargs)
    return len(cases)


def regression_cv_scores():
    sample = np.arange(12)
    X = sample[:, None].astype(np.float64)
    y = ((sample * sample + sample) % 13).astype(np.float64)
    weights = (sample % 5 + 1).astype(np.float64)
    with config_context(enable_metadata_routing=True):
        estimator = DummyRegressor().set_fit_request(sample_weight=True)
        return {
            name: cross_val_score(
                estimator,
                X,
                y,
                scoring=make_scorer(metric, greater_is_better=name == "r2").set_score_request(
                    sample_weight=True
                ),
                cv=KFold(3),
                params={"sample_weight": weights},
                error_score="raise",
            )
            for name, metric in [("r2", r2_score), ("neg_rmse", root_mean_squared_error)]
        }
