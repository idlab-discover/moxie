"""Compare native dense estimators with the pinned local sklearn build.

Setup (from the repository root):
    uv venv builds/reference-env
    uv pip install --python builds/reference-env/bin/python ../scikit-learn -Ccompile-args=-j2
    builds/reference-env/bin/python tests/check_sklearn.py
    # Optional reproduction of a contiguous case range:
    builds/reference-env/bin/python tests/check_sklearn.py --start 503 --stop 507

Python creates reference fixtures and launches Mojo; the library is pure Mojo.
"""
from pathlib import Path
import argparse
import inspect
import re
import json
import os
import subprocess

import numpy as np
import sklearn
from sklearn import naive_bayes as nb

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
assert subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip() == REVISION
assert Path(inspect.getfile(nb)).read_bytes() == (REFERENCE / "sklearn/naive_bayes.py").read_bytes(), "Install the pinned local checkout"
OUT = ROOT / "builds" / "naive-bayes-reference"
OUT.mkdir(parents=True, exist_ok=True)


def literal(value):
    if isinstance(value, (str, np.str_)):
        return json.dumps(str(value), ensure_ascii=False)
    if value is None:
        return "None"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return 'Float64("nan")'
        if np.isposinf(value):
            return 'Float64("inf")'
        if np.isneginf(value):
            return '-Float64("inf")'
        return repr(float(value))
    return "[" + ", ".join(literal(v) for v in value) + "]"


lines = ['''import numojo as nm
from std.math import isfinite, isnan
from std.testing import assert_true, assert_equal, assert_raises
from moxie._labels import StringTarget
from moxie._naive_bayes_dense import dense_from_lists, NumericDomain, StringDomain
from moxie.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB

@no_inline
def close(actual: Float64, expected: Float64) raises:
    if isnan(expected):
        assert_true(isnan(actual))
    elif not isfinite(expected):
        assert_equal(actual, expected)
    else:
        assert_true(isfinite(actual))
        if abs(actual - expected) > 1e-7 + 1e-7 * abs(expected):
            print("Mismatch: actual", actual, "expected", expected)
        assert_true(abs(actual - expected) <= 1e-7 + 1e-7 * abs(expected))
''']
lines.append("""
@no_inline
def close_vector(actual: List[Float64], expected: List[Float64]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        close(actual[i], expected[i])

@no_inline
def close_matrix(actual: List[List[Float64]], expected: List[List[Float64]]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        close_vector(actual[i], expected[i])

@no_inline
def close_vector(actual: nm.NDArray[DType.float64], expected: List[Float64]) raises:
    assert_equal(actual.ndim, 1)
    assert_equal(actual.shape[0], len(expected))
    for i in range(len(expected)):
        close(actual.item(i), expected[i])

@no_inline
def close_matrix(actual: nm.NDArray[DType.float64], expected: List[List[Float64]]) raises:
    assert_equal(actual.ndim, 2)
    assert_equal(actual.shape[0], len(expected))
    for i in range(len(expected)):
        assert_equal(actual.shape[1], len(expected[i]))
        for j in range(len(expected[i])):
            close(actual.item(i, j), expected[i][j])

@no_inline
def close_cube(actual: List[List[List[Float64]]], expected: List[List[List[Float64]]]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        close_matrix(actual[i], expected[i])

@no_inline
def equal_labels(actual: List[Int], expected: List[Int]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        assert_equal(actual[i], expected[i])
""")
lines.append("""
@no_inline
def close_cube(actual: List[nm.NDArray[DType.float64]], expected: List[List[List[Float64]]]) raises:
    assert_equal(len(actual), len(expected))
    for i in range(len(actual)):
        close_matrix(actual[i], expected[i])

@no_inline
def equal_labels[dtype: DType](actual: nm.NDArray[dtype], expected: List[Int]) raises:
    assert_equal(actual.ndim, 1)
    assert_equal(actual.size, len(expected))
    for i in range(len(expected)):
        assert_equal(actual.item(i), SIMD[dtype, 1](expected[i]))

@no_inline
def equal_labels[dtype: DType](actual: nm.NDArray[dtype], expected: List[Float64]) raises:
    assert_equal(actual.ndim, 1)
    assert_equal(actual.size, len(expected))
    for i in range(len(expected)):
        assert_equal(actual.item(i), SIMD[dtype, 1](expected[i]))

@no_inline
def equal_labels(actual: StringTarget, expected: List[String]) raises:
    assert_equal(actual.rank(), 1)
    assert_equal(actual.rows(), len(expected))
    for i in range(len(expected)):
        assert_equal(actual.get(i), expected[i])
""")
case_names = []


expected_counter = 0


def compare(expr, value, *, labels=False):
    global expected_counter
    if isinstance(value, list) or isinstance(value, np.ndarray) and value.ndim != 0:
        leaf = value
        rank = 0
        while isinstance(leaf, (list, np.ndarray)):
            rank += 1
            leaf = leaf[0]
        integer = isinstance(leaf, (int, np.integer))
        dtype = "String" if isinstance(leaf, (str, np.str_)) else ("Int" if integer else "Float64")
        for _ in range(rank):
            dtype = f"List[{dtype}]"
        method = "equal_labels" if integer or labels else {1: "close_vector", 2: "close_matrix", 3: "close_cube"}[rank]
        expected_counter += 1
        varname = f"expected{expected_counter}"
        lines.append(f"    var {varname}: {dtype} = {literal(value)}")
        lines.append(f"    {method}({expr}, {varname})")
    elif isinstance(value, (int, np.integer)):
        lines.append(f"    assert_equal({expr}, {int(value)})")
    else:
        lines.append(f"    close({expr}, {literal(value)})")


def case(name, params, X, y, Q, *, weights=None, chunks=None, classes=None, dtype=None, label=None):
    scalar_weight = weights is not None and np.isscalar(weights)
    # Retained native deviation: Gaussian's default state/output dtype is
    # Float64 even for Float32 storage. Preserve the rounded storage values,
    # then use sklearn's Float64 arithmetic as the oracle for that contract.
    # Explicit Float32 parity is checked by algorithms/gaussian/tests/parity.py.
    widened_gaussian = name == "GaussianNB" and dtype == "float32"
    if widened_gaussian:
        X = X.astype(np.float32).astype(np.float64)
        Q = Q.astype(np.float32).astype(np.float64)

    def target(variable, value):
        if label == "string":
            lines.append(f"    var {variable} = StringTarget({literal(value)})")
        elif label:
            lines.append(f"    var {variable} = nm.zeros[DType.{label}](nm.Shape({len(value)}))")
            for i, v in enumerate(value):
                lines.append(f"    {variable}[nm.Item({i})] = {literal(v)}")
        else:
            lines.append(f"    var {variable}: List[Int] = {literal(value)}")

    def matrix(variable, value):
        data_name = variable + "_data" if dtype else variable
        lines.append(f"    var {data_name}: List[List[Float64]] = {literal(value.astype(float))}")
        if dtype:
            lines.append(f"    var {variable} = dense_from_lists({data_name}).astype[DType.{dtype}]()")

    ident = f"case_{len(case_names)}"
    case_names.append(ident)
    lines.append(f"\n@no_inline\ndef {ident}() raises:")
    lines.append(f'    print("{ident}: {name}, {params}, weighted={weights is not None}, incremental={chunks is not None}, dtype={dtype}")')
    if widened_gaussian:
        lines.append('    print("Retained deviation: default Gaussian uses a Float64 oracle for Float32 storage")')
    args = []
    for k, v in params.items():
        if k == "alpha" and isinstance(v, list):
            lines.append(f"    var alpha_values = nm.zeros[DType.float64](nm.Shape({len(v)}))")
            for f, value in enumerate(v):
                lines.append(f"    alpha_values[nm.Item({f})] = {literal(value)}")
            args.append("alpha_values=alpha_values^")
        elif k == "min_categories" and isinstance(v, list):
            lines.append(f"    var minima: List[Int] = {literal(v)}")
            args.append("min_categories_values=minima^")
        elif isinstance(v, list):
            lines.append(f"    var {k}: List[Float64] = {literal(v)}")
            args.append(f"{k}={k}^")
        else:
            args.append(f"{k}={literal(v)}")
    domain = "[StringDomain]" if label == "string" else (f"[NumericDomain[DType.{label}]]" if label else "")
    lines.append(f"    var model = {name}{domain}(" + ", ".join(args) + ")")
    # The pinned Bernoulli vector denominator broadcasts alpha along classes.
    # For the documented corrected semantics, compose unmodified scalar sklearn
    # fits, one feature at a time; use sklearn itself for joint/probability output.
    vector_bernoulli = name == "BernoulliNB" and isinstance(params.get("alpha"), list)
    oracle_params = dict(params, alpha=1.0) if vector_bernoulli else params
    model = getattr(nb, name)(**oracle_params)
    feature_models = ([nb.BernoulliNB(**dict(params, alpha=float(alpha)))
                       for alpha in params["alpha"]] if vector_bernoulli else [])
    def snapshot():
        if feature_models:
            model.feature_log_prob_ = np.column_stack([
                feature_model.feature_log_prob_[:, 0] for feature_model in feature_models])
        attrs = ["classes_", "class_count_", "n_features_in_"]
        if name == "GaussianNB":
            attrs += ["theta_", "var_", "epsilon_", "class_prior_"]
        else:
            attrs += ["class_log_prior_", "feature_log_prob_"]
            attrs += ["category_count_", "n_categories_"] if name == "CategoricalNB" else ["feature_count_"]
            if name == "ComplementNB":
                attrs += ["feature_all_"]
        for attr in attrs:
            compare("model." + attr, getattr(model, attr), labels=attr == "classes_")
        query = Q.copy()
        if name == "CategoricalNB":
            for f, counts in enumerate(model.category_count_):
                invalid = np.trunc(query[:, f]) >= counts.shape[1]
                query[invalid, f] = np.trunc(query[invalid, f]).astype(int) % counts.shape[1]
        matrix("Q", query)
        for method in ["predict", "predict_proba", "predict_log_proba", "predict_joint_log_proba"]:
            with np.errstate(all="ignore"):
                expected = getattr(model, method)(query)
            lines.append(f"    var {method}_result = model.{method}(Q)")
            compare(method + "_result", expected, labels=method == "predict")
        # Accuracy includes labels outside the fitted classes; they count as wrong.
        score_target = np.resize(y, len(Q))
        score_weights = np.arange(1, len(Q) + 1, dtype=float)
        target("target", score_target)
        lines.append(f"    var score_weights: List[Float64] = {literal(score_weights)}")
        compare("model.score(Q, target)", model.score(query, score_target))
        compare("model.score(Q, target, sample_weight=score_weights.copy())", model.score(query, score_target, sample_weight=score_weights))

    if chunks is None:
        matrix("X", X)
        target("y", y)
    if weights is not None and chunks is None:
        weight_type = "Float64" if scalar_weight else "List[Float64]"
        lines.append(f"    var weights: {weight_type} = {literal(weights)}")
    if chunks is None:
        model.fit(X, y, sample_weight=weights)
        for f, feature_model in enumerate(feature_models):
            feature_model.fit(X[:, f:f + 1], y, sample_weight=weights)
        suffix = ", sample_weight=weights.copy()" if weights is not None else ""
        lines.append(f"    model.fit(X, y{suffix})")
        snapshot()
    else:
        if classes is None:
            classes = np.unique(y)
        for j, ix in enumerate(chunks):
            model.partial_fit(X[ix], y[ix], classes=classes if j == 0 else None, sample_weight=None if weights is None else (weights if scalar_weight else weights[ix]))
            for f, feature_model in enumerate(feature_models):
                feature_model.partial_fit(X[ix, f:f + 1], y[ix],
                    classes=classes if j == 0 else None,
                    sample_weight=None if weights is None else (weights if scalar_weight else weights[ix]))
            matrix(f"X{j}", X[ix])
            target(f"y{j}", y[ix])
            if j == 0:
                target("classes", classes)
            suffix = ", classes=classes^" if j == 0 else ""
            if weights is not None:
                weight_type = "Float64" if scalar_weight else "List[Float64]"
                batch_weights = weights if scalar_weight else weights[ix]
                lines.append(f"    var w{j}: {weight_type} = {literal(batch_weights)}")
                suffix += f", sample_weight=w{j}.copy()"
            lines.append(f"    model.partial_fit(X{j}, y{j}{suffix})")
            # Scope per-update temporaries without changing the fitted model.
            lines.append('    if True:')
            start = len(lines)
            snapshot()
            lines[start:] = ['    ' + line for line in lines[start:]]


rng = np.random.default_rng(427)
X = rng.integers(0, 5, size=(12, 4))
y = np.array([10, -4, 2, -4, 10, 2, 10, 2, -4, 2, 10, -4])
Q = rng.integers(0, 5, size=(5, 4))
weights = np.array([0, 1, 2, 0.5, 3, 1, 0.25, 2, 0.5, 1, 2, 1.0])
chunks = [slice(0, 4), slice(4, 8), slice(8, 12)]
configs = {
    "GaussianNB": [{}, {"var_smoothing": 0.2}, {"priors": [0.2, 0.3, 0.5]}],
    "MultinomialNB": [{}, {"alpha": 0.3, "fit_prior": False}, {"class_prior": [0.2, 0.3, 0.5]}, {"alpha": 0.0, "force_alpha": False}],
    "ComplementNB": [{}, {"norm": True}, {"alpha": 0.2, "class_prior": [0.2, 0.3, 0.5]}],
    "BernoulliNB": [{}, {"binarize": 2.0}, {"alpha": 0.4, "fit_prior": False}],
    "CategoricalNB": [{}, {"min_categories": 6}, {"alpha": 0.4, "fit_prior": False}],
}
for name, options in configs.items():
    for params in options:
        case(name, params, X, y, Q)
        case(name, params, X, y, Q, weights=weights, chunks=chunks)
    # Absent classes remain in probability columns.
    case(name, {}, X[:4], y[:4], Q, chunks=[slice(None)], classes=np.array([20, 10, 2, -4]))
    case(name, {}, X, np.full(len(y), -4), Q, weights=weights)

case("GaussianNB", {}, X.astype(float) - 10, y, Q.astype(float) - 10, weights=weights)
case("GaussianNB", {"var_smoothing": 0.0}, X, y, Q)
case("BernoulliNB", {"binarize": None}, (X > 2).astype(float), y, (Q > 2).astype(float), weights=weights)
# Force exact zero probabilities and their sklearn NaN/Inf consequences.
for name in ["MultinomialNB", "ComplementNB", "BernoulliNB", "CategoricalNB"]:
    case(name, {"alpha": 0.0}, np.array([[0, 0], [1, 0], [0, 1], [0, 2]]), np.array([1, 1, 2, 2]), np.array([[0, 0], [0, 1]]))
# A later categorical batch has fewer categories: retained buffers must not shrink.
case("CategoricalNB", {}, np.array([[3, 2], [1, 0], [0, 0], [0, 0]]), np.array([1, 2, 1, 2]), np.array([[3, 2], [0, 0]]), chunks=[slice(0, 2), slice(2, 4)])

# Per-feature minima, unseen allocated categories, weighted updates and growth.
for minima in ([3, 4], [0, 1], [2, 6]):
    for chunks in (None, [slice(0, 2), slice(2, 4)]):
        case("CategoricalNB", {"min_categories": minima, "alpha": 0.4},
             np.array([[3, 2], [1, 0], [0, 0], [0, 4]]), np.array([1, 2, 1, 2]),
             np.array([[3, 2], [0, 3], [1, 4]]), weights=np.array([1., 0.5, 2., 3.]), chunks=chunks)

# Native array Bernoulli branches, including degenerate zero smoothing.
for dtype in ("float32", "float64"):
    case("BernoulliNB", {"binarize": 0.1},
         np.array([[0.1, 0.2], [0.2, 0.1]], dtype=dtype), np.array([1, 2]),
         np.array([[0.1, 0.1]], dtype=dtype), dtype=dtype)
    for params in ({}, {"binarize": 0.5, "alpha": 0.3},
                   {"binarize": None}, {"alpha": 0.0},
                   {"alpha": 0.0, "force_alpha": False},
                   {"fit_prior": False}, {"class_prior": [0.2, 0.8]}):
        data = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0]], dtype=dtype)
        if params.get("binarize") == 0.5:
            data = data * 2 - 0.5  # includes values equal to the threshold
        for parts in (None, [slice(0, 2), slice(2, 4)]):
            case("BernoulliNB", params, data, np.array([9, 9, -2, -2]), data,
                 weights=np.array([0.5, 2.0, 1.0, 3.0]), chunks=parts, dtype=dtype)
    case("BernoulliNB", {}, data[:2], np.array([9, 9]), data,
         chunks=[slice(None)], classes=np.array([9, -2, 17]), dtype=dtype)
    case("BernoulliNB", {"binarize": None},
         np.array([[0.25, 0.5], [-0.125, 0.75], [0.5, 0.25], [0.75, 0.125]], dtype=dtype),
         np.array([9, 9, -2, -2]), np.array([[0.125, 0.5]], dtype=dtype), dtype=dtype)

# Corrected vector smoothing: rectangular class/feature dimensions, single
# class/feature, absent classes, priors, flooring and zero-induced nonfinites.
for dtype in (None, "float32", "float64"):
    for features, labels in ((3, [9, 9, -2, -2]), (2, [9, 9, -2, -2]),
                              (1, [9, 9, -2, -2]), (3, [9, 9, 9, 9])):
        data = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0]],
                        dtype=dtype or "float64")[:, :features]
        for alpha, force in (([0.25, 0.5, 2.0][:features], True),
                             ([0., 1e-12, 2.0][:features], False),
                             ([0., 0.5, 2.0][:features], True)):
            for parts in (None, [slice(0, 2), slice(2, 4)]):
                case("BernoulliNB", {"alpha": alpha, "force_alpha": force},
                     data, np.array(labels), data, weights=np.array([0.5, 2., 1., 3.]),
                     chunks=parts, dtype=dtype)
    for params in ({"binarize": None, "fit_prior": False},
                   {"binarize": 0.5, "class_prior": [0.2, 0.8]}):
        data = np.array([[0.25, 0.5, 0.75], [0.75, 0.25, 0.5]], dtype=dtype or "float64")
        case("BernoulliNB", dict(params, alpha=[0.25, 0.5, 2.]), data,
             np.array([9, 9]), data, chunks=[slice(None)], classes=np.array([-2, 9]), dtype=dtype)

# Categorical NDArray fitting, growth, retained columns, priors and zero smoothing.
for dtype in ("float32", "float64"):
    data = np.array([[3, 2], [1, 0], [0, 0], [0, 4]], dtype=dtype)
    for params in ({}, {"min_categories": [6, 7]}, {"min_categories": [0, 1]},
                   {"alpha": 0.0}, {"alpha": 0.0, "force_alpha": False},
                   {"fit_prior": False}, {"class_prior": [0.2, 0.8]}):
        for parts in (None, [slice(0, 2), slice(2, 4)]):
            case("CategoricalNB", params, data, np.array([9, -2, 9, -2]), data,
                 weights=np.array([0.5, 2.0, 1.0, 3.0]), chunks=parts, dtype=dtype)
    case("CategoricalNB", {}, data[:2], np.array([9, 9]), data[:2],
         chunks=[slice(None)], classes=np.array([9, -2, 17]), dtype=dtype)
    case("CategoricalNB", {}, data, np.full(4, 9), data, dtype=dtype)

# All label domains, both array dtypes and the List feature compatibility API.
for name in ("MultinomialNB", "ComplementNB", "BernoulliNB", "CategoricalNB"):
    for label in ("int64", "float32", "float64", "string"):
        values = ([2**60 + 3, -(2**60) + 7] if label == "int64" else
                  (["éclair", "apple"] if label == "string" else [9, -2]))
        labels = np.array([values[0], values[0], values[1], values[1]],
                          dtype=str if label == "string" else label)
        for dtype in (None, "float32", "float64"):
            data = np.array([[0, 1], [0, 2], [1, 0], [2, 0]], dtype=dtype or "float64")
            for parts in (None, [slice(0, 2), slice(2, 4)]):
                case(name, {}, data, labels, data, weights=np.array([0.5, 2., 1., 3.]),
                     chunks=parts, dtype=dtype, label=label)

# Input compatibility: every supported storage dtype and all five estimators.
for dtype in ("bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
              "uint32", "uint64", "float32", "float64"):
    data = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=dtype)
    for name in configs:
        for weight in (None, 2.5):
            for parts in (None, [slice(0, 2), slice(2, 4)]):
                case(name, {}, data, np.array([9, -2, 9, -2]), data,
                     weights=weight, chunks=parts, dtype=dtype)
    # These would fail if thresholds were cast to integer/bool input dtype.
    for threshold in (0.5, 1.5, 256.5):
        case("BernoulliNB", {"binarize": threshold}, data,
             np.array([9, -2, 9, -2]), data, dtype=dtype, weights=2.5)

# Scalar weights compose with List features and every native target domain.
for name in configs:
    for dtype in (None, "int16", "bool"):
        for label in (None, "int64", "float32", "float64", "string"):
            data = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=dtype or "float64")
            labels = np.array(["a", "b", "a", "b"]) if label == "string" else np.array([9, -2, 9, -2])
            for parts in (None, [slice(0, 2), slice(2, 4)]):
                case(name, {}, data, labels, data, weights=0.5,
                     chunks=parts, dtype=dtype, label=label)

# Fixed-seed stress: randomized shapes, rare/late classes, uneven partitions,
# extreme positive weights and count scales. Every update invokes snapshot().
stress_rng = np.random.default_rng(20261005)
for trial in range(12):
    rows = int(stress_rng.integers(7, 26))
    features = int(stress_rng.choice([1, 2, 3, 7, 17]))
    count = int(stress_rng.integers(2, 5))
    labels = np.full(rows, -7)
    labels[-(count - 1):] = np.arange(1, count) * 11
    cuts = sorted(set([0, 1, rows - 1, rows] + stress_rng.integers(1, rows, 3).tolist()))
    parts = [slice(a, b) for a, b in zip(cuts, cuts[1:])]
    weights = 10.0 ** stress_rng.uniform(-12, 12, rows)
    for name in configs:
        data = stress_rng.integers(0, 5, (rows, features)).astype(float)
        params = {}
        if name in ("MultinomialNB", "ComplementNB"):
            data *= [1e-100, 1., 1e100][trial % 3]
        elif name == "GaussianNB":
            data = stress_rng.normal(size=(rows, features)) * [1e-100, 1., 1e100][trial % 3]
        elif name == "CategoricalNB":
            params = {"min_categories": [6]}
        case(name, params, data, labels, data[:3], weights=weights,
             chunks=parts, dtype="float64")

# Exact ties and nearby decisions; labels are always exact, never tolerance-based.
for name in configs:
    data = np.array([[0., 1.], [1., 0.], [0., 1.], [1., 0.]])
    for delta in (0., 1e-10, -1e-10):
        case(name, {}, data, np.array([-3, -3, 8, 8]), data,
             weights=np.array([1., 1., 1., 1. + delta]), dtype="float64",
             chunks=[slice(0, 2), slice(2, 4)])

# Reference-compatible categorical truncation and singleton minimum broadcasting.
for dtype in (None, "float32", "float64"):
    data = np.array([[-0.9, 1.9], [1.1, 0.9], [2.9, 3.1], [0.1, 2.8]], dtype=dtype or float)
    case("CategoricalNB", {"min_categories": [5]}, data, np.array([9, -2, 9, -2]),
         data, chunks=[slice(0, 1), slice(1, 3), slice(3, 4)], dtype=dtype)

# Retained deviations are checked against the unmodified reference separately.
# Negative weights with positive class totals succeed in sklearn; native rejects
# them transactionally. All-zero batches are rejected by the pinned reference.
for name in configs:
    X = np.array([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
    y = np.array([0, 0, 1, 1])
    for weights in (np.array([-1., 2., 1., 1.]), np.zeros(4)):
        reference = getattr(nb, name)()
        with np.errstate(all="ignore"):
            try:
                reference.fit(X, y, sample_weight=weights)
            except ValueError as error:
                assert not weights.any() and "non-zero" in str(error)
            else:
                assert weights.any(), "reference must reject an all-zero batch"
                np.testing.assert_array_equal(reference.class_count_,
                                              [weights[:2].sum(), weights[2:].sum()])
        # Exercise the reference incremental path independently as well.
        reference = getattr(nb, name)().fit(X, y)
        with np.errstate(all="ignore"):
            try:
                reference.partial_fit(X, y, sample_weight=weights)
            except ValueError as error:
                assert not weights.any() and "non-zero" in str(error)
            else:
                assert weights.any(), "reference must reject an all-zero update"
                np.testing.assert_array_equal(reference.class_count_,
                                              [2 + weights[:2].sum(), 2 + weights[2:].sum()])
        ident = f"case_{len(case_names)}"
        case_names.append(ident)
        lines.extend([f"\n@no_inline\ndef {ident}() raises:",
                      f"    var model = {name}()",
                      f"    var X = dense_from_lists({literal(X)})",
                      "    var y: List[Int] = [0, 0, 1, 1]",
                      "    model.fit(X, y)"])
        baseline = getattr(nb, name)().fit(X, y)
        for method in ("fit", "partial_fit"):
            lines.extend(["    with assert_raises():",
                          f"        model.{method}(X, y, sample_weight=List[Float64]({literal(weights)}))"])
            compare("model.class_count_", baseline.class_count_)
            compare("model.predict(X)", baseline.predict(X), labels=True)

# Pin the Bernoulli bug itself: square dimensions silently mis-broadcast,
# rectangular dimensions raise. Native featurewise expectations above come
# from independent scalar sklearn fits, and are deliberately NOT direct parity.
for features in (2, 3):
    data = np.array([[0., 1., 1.], [1., 0., 0.], [1., 1., 0.]])[:, :features]
    alpha = np.array([0.25, 2., 3.])[:features]
    try:
        buggy = nb.BernoulliNB(alpha=alpha).fit(data, [0, 0, 1])
    except ValueError:
        assert features == 3
    else:
        corrected = np.column_stack([nb.BernoulliNB(alpha=a).fit(data[:, f:f+1], [0, 0, 1]).feature_log_prob_[:, 0]
                                     for f, a in enumerate(alpha)])
        assert features == 2 and not np.allclose(buggy.feature_log_prob_, corrected)

# The default-dtype deviation can be visible in probabilities, not merely
# low bits: Float32 loses log(2) while normalizing joint scores near -2e9.
for arithmetic, expected in ((np.float32, 1.0), (np.float64, 0.5)):
    data = np.array([[0., 1.], [1., 0.]], dtype=arithmetic)
    oracle = nb.GaussianNB().partial_fit(data, [9, -2], classes=[-2, 9])
    np.testing.assert_allclose(oracle.predict_proba(np.ones((1, 2), dtype=arithmetic)),
                               expected, rtol=1e-7, atol=1e-7)

# Bound compilation memory while retaining every case and its original index.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--start", type=int, default=0, help="first case index (inclusive)")
parser.add_argument("--stop", type=int, default=len(case_names), help="last case index (exclusive)")
parser.add_argument("--batch-size", type=int, default=32, help="maximum cases per native compilation")
options = parser.parse_args()
if not 0 <= options.start < options.stop <= len(case_names):
    parser.error(f"require 0 <= start < stop <= {len(case_names)}")
if options.batch_size < 1:
    parser.error("batch-size must be positive")
text = "\n".join(lines) + "\n"
# Fluent mutation returns are intentionally discarded by these oracle fixtures.
text = re.sub(r"(?m)^(\s*)([A-Za-z_]\w*\.(?:fit|partial_fit|set_params)\()", r"\1_ = \2", text)
starts = list(re.finditer(r"@no_inline\ndef case_\d+\(\) raises:", text))
assert len(starts) == len(case_names)
prefix = text[:starts[0].start()]
print(f"Reference: sklearn {sklearn.__version__}, {REVISION}", flush=True)
env = os.environ.copy()
for key, directory in (("MODULAR_CACHE_DIR", ROOT / ".pixi/modular-cache"),
                       ("TMPDIR", ROOT / "builds/tmp")):
    directory.mkdir(parents=True, exist_ok=True)
    env[key] = str(directory)
executed = []
for first in range(options.start, options.stop, options.batch_size):
    last = min(first + options.batch_size, options.stop)
    end = starts[last].start() if last < len(starts) else len(text)
    batch = prefix + text[starts[first].start():end] + "\ndef main() raises:\n"
    for ident in case_names[first:last]:
        batch += f"    {ident}()\n"
    batch += f'    print("Passed {last - first} reference/regression cases [{first}, {last})")\n'
    source = OUT / f"reference_cases_{first}_{last}.mojo"
    source.write_text(batch)
    subprocess.run(["pixi", "run", "mojo", "run", "--fp-mode=contract=off", "-I",
                    str(ROOT / "builds/deps"), str(source)], cwd=ROOT, env=env, check=True)
    executed.extend(case_names[first:last])
assert executed == case_names[options.start:options.stop]
print(f"Passed all {len(executed)} reference/regression cases; Bernoulli vectors use scalar sklearn fits; "
      "default Gaussian Float32 storage uses Float64 oracle arithmetic", flush=True)
