"""Compare three real LinearRegression compositions to pinned scikit-learn."""
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "families/linear_model/tools"))
from compiler_limits import compiler_command
import parity_io as oracle

OUTPUT = ROOT / "builds/linear-regression-importance-parity"
HEADER = """\
import numojo as nm
from std.testing import assert_equal, assert_false
from moxie.feature_selection import (
    DenseSupervisedCV, FeatureCount, LinearRegressionImportance,
    RFE, RFECV, SelectFromModel, SequentialCV,
)
from moxie.linear_model import LinearRegression

def dump[dtype: DType](array: nm.NDArray[dtype]) raises:
    print(array.ndim)
    for axis in range(array.ndim):
        print(array.shape[axis])
    for i in range(array.size):
        print(array.unsafe_get(i))
"""


def pinned_estimators():
    base, from_model, rfe = [
        f"sklearn/feature_selection/{name}.py" for name in ("_base", "_from_model", "_rfe")
    ]
    linear = "sklearn/linear_model/_base.py"
    oracle.verify_reference([base, from_model, rfe, linear])
    common = oracle.load_pinned(base, register=True)
    regressor = oracle.load_pinned(linear).LinearRegression
    model = oracle.load_pinned(from_model)
    elimination = oracle.load_pinned(rfe)
    oracle.pinned_code(regressor.fit, linear)
    oracle.pinned_code(regressor.predict, linear)
    oracle.pinned_code(model.SelectFromModel.fit, from_model)
    oracle.pinned_code(elimination.RFE.fit, rfe)
    oracle.pinned_code(elimination.RFECV.fit, rfe)
    assert model._get_feature_importances is common._get_feature_importances
    assert elimination._get_feature_importances is common._get_feature_importances
    return regressor, (model.SelectFromModel, elimination.RFE, elimination.RFECV)


def fixture(dtype, multioutput, regressor, selectors):
    rng = np.random.default_rng(146)
    X = rng.normal(size=(24, 4)).astype(dtype)
    y = X @ (np.array([[5, 0.4], [1.7, -3], [-0.5, 2.1], [0.2, 0.1]], dtype=dtype))
    y += rng.normal(0, 0.14, y.shape).astype(dtype)
    if not multioutput:
        y = y[:, 0].copy()
    weights = np.linspace(0.25, 2.0, len(y), dtype=dtype)
    lines = [f'def check_{int(multioutput)}() raises:']
    expected = {}
    exact_count = 0

    def add(line):
        lines.append("    " + line)

    def array(name, values):
        flat = ", ".join(repr(float(value)) for value in values.flat)
        add(f"var {name} = nm.array[dtype]([{flat}], {list(values.shape)})")

    def exact(expression, reference):
        nonlocal exact_count
        if isinstance(reference, np.ndarray):
            reference = reference.tolist()
        literal = repr(reference)
        add(f"assert_equal({expression}, {literal})")
        exact_count += np.size(reference)

    def dump(key, expression, values, array_dtype="dtype"):
        add(f'print("{key}")')
        if np.ndim(values) == 0:
            add("print(0)")
            add(f"print({expression})")
        else:
            reference_dtype = dtype if array_dtype == "dtype" else np.float64
            assert np.asarray(values).dtype == reference_dtype
            add(f"dump[{array_dtype}]({expression})")
        expected[key] = np.asarray(values)

    array("X", X)
    array("y", y)
    array("weights", weights)
    cases = (
        ("sfm", 'SelectFromModel(E(LinearRegression[dtype]()))', {}, True),
        ("rfe", 'RFE(E(LinearRegression[dtype]()), n_features_to_select=FeatureCount(2), importance_getter="coef_")',
         {"n_features_to_select": 2, "importance_getter": "coef_"}, True),
        ("rfecv", 'RFECV[DenseSupervisedCV[E, dtype]](E(LinearRegression[dtype]()), cv=SequentialCV(3))',
         {"cv": 3}, False),
    )
    for (name, constructor, options, weighted), selector in zip(cases, selectors, strict=True):
        reference = selector(regressor(), **options)
        fit_options = {"sample_weight": weights} if weighted else {}
        reference.fit(X, y, **fit_options)
        add(f"var {name} = {constructor}")
        add(f"_ = {name}.fit(X, y.copy(){', sample_weight=weights' if weighted else ''})")
        add(f"assert_false({name}.estimator.__sklearn_is_fitted__())")
        exact(f"{name}.get_support()[List[Bool]]", reference.get_support())
        dump(f"{name}:transform", f"{name}.transform(X)", reference.transform(X))
        add(f"var {name}_fitted = {name}.estimator_()")
        dump(f"{name}:coef", f'{name}_fitted.importances("coef_")', reference.estimator_.coef_)
        if name == "sfm":
            dump("sfm:threshold", "sfm.threshold_()", reference.threshold_)
            continue
        exact(f"{name}.ranking_()", reference.ranking_)
        dump(f"{name}:predict", f"{name}.predict(X)", reference.predict(X))
        score_weight = ", weights.copy()" if weighted else ""
        dump(f"{name}:score", f"{name}.score(X, y{score_weight})", reference.score(X, y, **fit_options))
        if name == "rfecv":
            add("var results = rfecv.cv_results_()")
            for key, values in reference.cv_results_.items():
                if key == "n_features":
                    exact(f'results["{key}"][List[Int]]', values)
                elif key.endswith(("_support", "_ranking")):
                    item = "Bool" if key.endswith("_support") else "Int"
                    exact(f'results["{key}"][List[List[{item}]]]', values)
                else:
                    dump(f"rfecv:{key}", f'results["{key}"][nm.NDArray[DType.float64]]',
                         values, array_dtype="DType.float64")
    return "\n".join(lines), expected, exact_count


def main():
    regressor, selectors = pinned_estimators()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dtype in (np.float32, np.float64):
        name = np.dtype(dtype).name
        fixtures = [fixture(dtype, multi, regressor, selectors) for multi in (False, True)]
        source = OUTPUT / f"{name}.mojo"
        binary = source.with_suffix("")
        header = HEADER + f"\ncomptime dtype = DType.{name}\ncomptime E = LinearRegressionImportance[dtype]\n"
        source.write_text("\n\n".join([header, *(code for code, _, _ in fixtures),
                                      "def main() raises:\n    check_0()\n    check_1()\n"]))
        subprocess.run(compiler_command([
            "mojo", "build", "-O3", "-I", ROOT / "builds/deps", source, "-o", binary,
        ]), check=True)
        actual = iter(subprocess.check_output([binary], text=True).splitlines())
        for multi, (_, expected, exact_count) in enumerate(fixtures):
            # Match the well-conditioned least-squares owner tolerances.
            tolerance = 3e-4 if dtype == np.float32 else 1e-8
            for key, value in expected.items():
                assert next(actual) == key
                ndim = int(next(actual))
                assert ndim == value.ndim, key
                shape = tuple(int(next(actual)) for _ in range(ndim))
                assert shape == value.shape, key
                result = np.array([float(next(actual)) for _ in range(value.size)]).reshape(shape)
                np.testing.assert_allclose(result, value, rtol=tolerance, atol=tolerance, err_msg=key)
            count = exact_count + sum(value.size for value in expected.values())
            print(f"PASS {name}/{['vector', 'multioutput'][multi]} ({count} comparisons)", flush=True)
        assert next(actual, None) is None
    print(f"PASS LinearRegression selector parity against scikit-learn {oracle.REVISION}")


if __name__ == "__main__":
    main()
