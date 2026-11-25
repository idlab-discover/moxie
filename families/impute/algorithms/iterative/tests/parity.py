"""Dense IterativeImputer comparisons with pinned imputer and BayesianRidge bodies."""
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import sklearn.impute._base
import sklearn.linear_model._base

FAMILY = Path(__file__).resolve().parents[3]
SOURCE = FAMILY.parents[2] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
FILES = [
    "sklearn/impute/_iterative.py",
    "sklearn/impute/_base.py",
    "sklearn/impute/tests/test_impute.py",
    "sklearn/linear_model/_bayes.py",
    "sklearn/linear_model/_base.py",
]
sys.path.insert(0, str(FAMILY / "tools"))
from check import assemble, compile_source, run


def load(relative):
    path = SOURCE / relative
    name = relative.removesuffix(".py").replace("/", ".") + "_pinned"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def installed(name, module):
    previous = sys.modules[name]
    sys.modules[name] = module
    try:
        yield
    finally:
        sys.modules[name] = previous


def oracle():
    assert subprocess.check_output(
        ["git", "-C", SOURCE, "rev-parse", "HEAD"], text=True,
    ).strip() == REVISION
    assert not subprocess.check_output(
        ["git", "-C", SOURCE, "status", "--porcelain", "--", *FILES], text=True,
    )
    with installed("sklearn.impute._base", load(FILES[1])):
        iterative = load(FILES[0]).IterativeImputer
    with installed("sklearn.linear_model._base", load(FILES[4])):
        bayesian = load(FILES[3]).BayesianRidge
    for method, path in [(iterative.fit_transform, FILES[0]), (iterative.transform, FILES[0]),
                         (bayesian.fit, FILES[3]), (bayesian.predict, FILES[3])]:
        assert Path(inspect.unwrap(method).__code__.co_filename) == SOURCE / path
    return iterative, bayesian


DRIVER = """import numojo as nm
from moxie.impute import IterativeBound, IterativeImputer
from moxie.impute._plumbing import MissingValue
from moxie._random import seed_global_random_state
from moxie.linear_model import BayesianRidge

@no_inline
def emit[dtype: DType](X: nm.NDArray[dtype]) raises:
    print(X.shape[0], X.shape[1])
    for i in range(X.size):
        print(Float64(X.unsafe_get(i)))

@no_inline
def report[dtype: DType](
    name: String, model: IterativeImputer[dtype],
    fitted: nm.NDArray[dtype], transformed: nm.NDArray[dtype],
) raises:
    print(name)
    emit(fitted)
    emit(transformed)
    print(model.n_iter_, model.n_features_in_, model.n_features_with_missing_)
    for feature_name in model.get_feature_names_out():
        print(feature_name)
    print(len(model.imputation_sequence_))
    for step in model.imputation_sequence_:
        print(step.feat_idx, len(step.neighbor_feat_idx))
        for j in step.neighbor_feat_idx:
            print(j)
    for i in range(model.initial_imputer_.statistics_.size):
        print(model.initial_imputer_.statistics_.unsafe_get(i))
"""


def cases():
    rng = np.random.default_rng(415)
    X = rng.normal(size=(16, 5))
    X[:, 3] = 2 * X[:, 0] - X[:, 1] + 0.3 * X[:, 3]
    X[:, 4] = 5
    X[[0, 2, 9], 0] = np.nan
    X[[1, 5, 7, 10], 1] = np.nan
    X[[6, 8, 11], 2] = np.nan
    X[[0, 3], 3] = np.nan
    Z = rng.normal(size=(4, 5))
    Z[[0, 2], 0] = np.nan
    Z[[2, 3], 2] = np.nan
    Z[[1], 4] = np.nan
    for order in ("ascending", "descending", "roman", "arabic", "random"):
        for skip in (False, True):
            yield f"{order}-{skip}", X, Z, dict(imputation_order=order, skip_complete=skip)
    for nearest in (1, 3, 8):
        yield f"nearest-{nearest}", X, Z, dict(n_nearest_features=nearest, imputation_order="random")
    yield "repeat-random", X, Z, dict(n_nearest_features=3, imputation_order="random")
    yield "default-stream", X, Z, dict(random_state=None, n_nearest_features=3, imputation_order="random")
    yield "scaled-tol", X, Z, dict(max_iter=20, tol=0.02)
    yield "zero-tol", X, Z, dict(max_iter=4, tol=0.0)
    yield "scalar-bounds", X, Z, dict(min_value=-0.1, max_value=0.2)
    yield "all-missing-transform", X, np.full_like(Z, np.nan), {}
    yield "one-feature", X[:, :1], Z[:, :1], {}
    yield "all-missing-fit", np.full_like(X, np.nan), Z, {}
    yield "all-missing-kept", np.full_like(X, np.nan), Z, dict(keep_empty_features=True)
    yield "all-missing-no-indicator", np.full_like(X, np.nan), Z, dict(add_indicator=False)
    complete = np.nan_to_num(X)
    for skip in (False, True):
        yield f"new-mask-{skip}", complete, Z, dict(skip_complete=skip)
    yield "zero-scale", np.zeros_like(X), Z, {}
    finite_X = np.nan_to_num(X, nan=-10.0)
    finite_X[:, 2] = -10
    finite_Z = np.nan_to_num(Z, nan=-10.0)
    for keep in (False, True):
        for strategy in ("mean", "median", "most_frequent", "constant"):
            yield f"empty-{keep}-{strategy}", finite_X, finite_Z, dict(
                missing_values=-10.0, keep_empty_features=keep,
                initial_strategy=strategy, fill_value=0.75,
                min_value=[-1, -2, -3, -4, -5], max_value=[1, 2, 3, 4, 5],
            )
    yield "initial-only", X, Z, dict(max_iter=0, initial_strategy="median")
    yield "configured-bayesian", X, Z, dict(estimator=dict(max_iter=1, fit_intercept=False))


def scalar(x):
    if np.isnan(x):
        return 'Float64("nan")'
    return f"Float64({float(x)!r})"


def integer_cases():
    X = [[1, 3, 2], [4, 7, 7], [6, -1, 2], [8, 10, 8], [-1, 18, 9]]
    Z = [[-1, 5, 3], [8, -1, 6], [1, 9, -1]]
    options = dict(missing_values=-1)
    yield "integer-contiguous", np.array(X, "int64"), np.array(Z, "int32"), options
    yield "integer-reversed", np.array(X, "int16")[::-1, ::-1], np.array(Z, "int16")[::-1, ::-1], options
    padded = np.zeros((len(X), 6), dtype="int32")
    padded[:, 1::2] = X
    yield "integer-strided", padded[:, 1::2], np.array(Z, "float64"), options
    yield "integer-query", np.array(X, "float64"), np.array(Z, "int64"), options
    yield "integer-nan-marker", np.arange(15, dtype="uint8").reshape(5, 3), np.array(Z, "int64"), {}
    marker = 2 ** 53
    # Values that differ in integer storage can share the same Float64 marker.
    yield "integer-rounded-marker", np.array([
        [1, 1, marker], [2, 3, marker + 1], [3, 5, marker], [4, 8, marker + 1],
    ], "uint64"), np.array([[3, 5, marker + 1], [1, 2, marker]], "uint64"), dict(
        missing_values=float(marker),
    )


def array(name, values):
    dtype = values.dtype.name
    rows, cols = values.shape
    if values.dtype.kind in "iu":
        # Recreate each integer layout with its original element strides and
        # offset so the native overload sees the same logical matrix.
        owner = values
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        lines = [f"    var {name}_owner = nm.zeros[DType.{dtype}](nm.Shape({owner.size}))"]
        mojo_type = dtype.capitalize().replace("Uint", "UInt")
        lines += [
            f"    {name}_owner.unsafe_set({i}, {mojo_type}({int(x)}))"
            for i, x in enumerate(owner.flat)
        ]
        strides = ", ".join(str(s // values.itemsize) for s in values.strides)
        offset = (values.ctypes.data - owner.ctypes.data) // values.itemsize
        return lines + [
            f"    var {name} = {name}_owner.view_with_layout("
            f"nm.Shape({rows}, {cols}), nm.Strides({strides}), {offset})",
        ]
    return [
        f"    var {name} = nm.zeros[DType.{dtype}](nm.Shape({rows}, {cols}))",
        *(f"    {name}.unsafe_set({i}, {scalar(x)}.cast[DType.{dtype}]())"
          for i, x in enumerate(values.flat)),
    ]


def value(key, x, dtype):
    if key in ("min_value", "max_value"):
        if isinstance(x, list):
            return f"IterativeBound({key}^)"
        return f"IterativeBound({scalar(x)})"
    if key == "estimator":
        args = ", ".join(f"{k}={v}" for k, v in x.items())
        return f"BayesianRidge[DType.{dtype}]({args})"
    if key == "missing_values":
        return f"MissingValue[DType.{dtype}]({scalar(x)}.cast[DType.{dtype}]())"
    if key == "fill_value":
        return f"{scalar(x)}.cast[DType.{dtype}]()"
    if isinstance(x, str):
        return f'"{x}"'
    return str(x)


def main():
    iterative, bayesian = oracle()
    expected = {}
    definitions = []
    calls = []
    inputs = [
        (dtype, tag, X.astype(dtype), Z.astype(dtype), overrides)
        for dtype in ("float64", "float32")
        for tag, X, Z, overrides in cases()
    ]
    inputs.extend(("float64", *case) for case in integer_cases())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dtype, tag, X, Z, overrides in inputs:
            name = f"{tag}-{dtype}"
            options = dict(random_state=17, max_iter=4, add_indicator=True) | overrides
            reference = iterative(**(options | {
                "estimator": bayesian(**options.get("estimator", {})),
            }))
            if options["random_state"] is None:
                np.random.seed(17)
            fitted = reference.fit_transform(X)
            repeat = tag == "repeat-random"
            if repeat:
                fitted = reference.fit_transform(X)
            transformed = reference.transform(Z)
            assert fitted.dtype.name == transformed.dtype.name == dtype, name
            expected[name] = (reference, fitted, transformed)
            function = f"case_{len(calls)}"
            definitions += ["@no_inline", f"def {function}() raises:"]
            definitions += array("X", X) + array("Z", Z)
            for key in ("min_value", "max_value"):
                if isinstance(options.get(key), list):
                    values = ", ".join(map(scalar, options[key]))
                    definitions.append(f"    var {key}: List[Float64] = [{values}]")
            args = ", ".join(f"{k}={value(k, v, dtype)}" for k, v in options.items())
            definitions.append(f"    var model = IterativeImputer[DType.{dtype}]({args})")
            if options["random_state"] is None:
                definitions.append("    seed_global_random_state(17)")
            definitions.append("    var fitted = model.fit_transform(X)")
            if repeat:
                definitions.append("    fitted = model.fit_transform(X)")
            definitions += [f'    report[DType.{dtype}]("{name}", model, fitted, model.transform(Z))', ""]
            calls.append(f"    {function}()")

    build = FAMILY / "builds" / "iterative"
    build.mkdir(parents=True, exist_ok=True)
    source = build / "parity_driver.mojo"
    source.write_text(DRIVER + "\n".join(["", *definitions, "def main() raises:", *calls, ""]))
    assemble(FAMILY)
    binary = build / "parity_driver"
    compile_source(source, binary)
    output = run(binary, capture=True).stdout
    (build / "native.txt").write_text(output)
    (build / "reference.txt").write_text(
        REVISION + "\n" + "\n".join(
            f"{hashlib.sha256((SOURCE / path).read_bytes()).hexdigest()} {path}" for path in FILES
        ) + "\n",
    )
    records = iter(output.splitlines())
    worst = {"float64": 0.0, "float32": 0.0}
    for name, (model, fitted, transformed) in expected.items():
        assert next(records) == name
        dtype = fitted.dtype.name
        tolerance = 3e-4 if dtype == "float32" else 5e-8
        for want in (fitted, transformed):
            shape = tuple(map(int, next(records).split()))
            assert shape == want.shape, (name, shape, want.shape)
            observed = np.array([float(next(records)) for _ in range(want.size)]).reshape(shape)
            np.testing.assert_allclose(observed, want, rtol=tolerance, atol=tolerance, err_msg=name)
            worst[dtype] = max(worst[dtype], np.max(np.abs(observed - want), initial=0))
        counts = tuple(map(int, next(records).split()))
        assert counts == (model.n_iter_, model.n_features_in_,
                          getattr(model, "n_features_with_missing_", 0)), (name, counts)
        assert [next(records) for _ in range(fitted.shape[1])] == list(model.get_feature_names_out()), name
        assert int(next(records)) == len(model.imputation_sequence_), name
        for step in model.imputation_sequence_:
            feature, n = map(int, next(records).split())
            assert feature == step.feat_idx, name
            assert [int(next(records)) for _ in range(n)] == list(step.neighbor_feat_idx), name
        statistics = [float(next(records)) for _ in range(model.n_features_in_)]
        np.testing.assert_allclose(statistics, model.initial_imputer_.statistics_.astype(float),
                                   rtol=tolerance, atol=tolerance, err_msg=name)
    assert next(records, None) is None
    print(f"PASS: {len(expected)} iterative cases against {REVISION}. Max absolute error {worst}.")


if __name__ == "__main__":
    main()
