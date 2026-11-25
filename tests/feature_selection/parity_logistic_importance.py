"""Compare dense logistic model selectors to pinned scikit-learn."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from compiler_limits import compiler_command

OUTPUT = ROOT / "builds/logistic-importance-parity"
HEADER = """\
import numojo as nm
from moxie.feature_selection import DenseSupervisedCV, FeatureCount, LogisticRegressionImportance, RFE, RFECV, SelectFromModel, SequentialCV
from moxie.linear_model import LogisticRegression
from std.testing import assert_equal, assert_false

def dump[dtype: DType](array: nm.NDArray[dtype]) raises:
    print(array.ndim)
    for axis in range(array.ndim):
        print(array.shape[axis])
    for i in range(array.size):
        print(array.unsafe_get(i))
"""


def pinned_estimators():
    # Reuse the owner oracle, including the pinned loss and Newton solver.
    path = ROOT / "families/linear_model/algorithms/logistic/tests/parity.py"
    spec = importlib.util.spec_from_file_location("logistic_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logistic = module.oracle()
    from parity_io import load_pinned, pinned_code, verify_reference

    base, model, rfe = [
        f"sklearn/feature_selection/{name}.py" for name in ("_base", "_from_model", "_rfe")
    ]
    verify_reference([base, model, rfe])
    common = load_pinned(base, register=True)
    from_model = load_pinned(model)
    selectors = load_pinned(rfe)
    pinned_code(from_model.SelectFromModel.fit, model)
    pinned_code(selectors.RFE.fit, rfe)
    pinned_code(selectors.RFECV.fit, rfe)
    assert from_model._get_feature_importances is selectors._get_feature_importances
    assert selectors._get_feature_importances is common._get_feature_importances
    return logistic, (from_model.SelectFromModel, selectors.RFE, selectors.RFECV)


def fixture(name, dtype, n_classes, solver, logistic, selector_type):
    rng = np.random.default_rng(842)
    groups = rng.permutation(np.tile(np.arange(n_classes), 10))
    X = rng.normal(0, 0.55, (len(groups), 5)).astype(dtype)
    X[:, 0] += 2 * groups
    X[:, 2] += 1.5 * (groups == 1)
    X[:, 4] -= 0.75 * (groups >= 1)
    y = 2**53 + groups.astype(np.int64)
    weights = np.linspace(0.5, 2, len(y), dtype=dtype)
    weighted = name != "RFECV"
    weight_kw = {"sample_weight": weights} if weighted else {}
    options = {
        "SelectFromModel": {},
        "RFE": {"n_features_to_select": 2},
        "RFECV": {"cv": 3},
    }
    getter = "auto" if n_classes == 2 else "coef_"
    model = logistic(C=0.5, solver=solver, tol=1e-9, max_iter=500)
    ref = selector_type(model, importance_getter=getter, **options[name])
    ref.fit(X, y, **weight_kw)

    lines = [
        HEADER,
        f"comptime dtype = DType.{dtype}",
        "comptime E = LogisticRegressionImportance[dtype]",
        "def main() raises:",
    ]
    numeric = []
    comparisons = 0

    def add(line):
        lines.append("    " + line)

    def exact(expression, value):
        nonlocal comparisons
        add(f"assert_equal({expression}, {value})")
        comparisons += 1

    def exact_list(expression, values):
        exact(f"len({expression})", len(values))
        for i, value in enumerate(values):
            exact(f"{expression}[{i}]", value)

    def label_array(expression, values):
        exact(f"{expression}.ndim", 1)
        exact(f"{expression}.size", len(values))
        for i, value in enumerate(values):
            exact(f"{expression}[nm.Item({i})]", f"Int64({value})")

    def dump(expression, values):
        values = np.asarray(values)
        add(f"dump({expression})")
        numeric.extend([values.ndim, *values.shape, *values.flat])

    add(f"var X = nm.zeros[dtype](nm.Shape({len(y)}, {X.shape[1]}))")
    for i, j in np.ndindex(X.shape):
        add(f"X[nm.Item({i}, {j})] = {float(X[i, j])}")
    add(f"var y = nm.zeros[DType.int64](nm.Shape({len(y)}))")
    for i, value in enumerate(y):
        add(f"y[nm.Item({i})] = {value}")
    if weighted:
        add(f"var weights = nm.zeros[dtype](nm.Shape({len(y)}))")
        for i, value in enumerate(weights):
            add(f"weights[nm.Item({i})] = {float(value)}")
    add(f'var est = E(LogisticRegression[dtype](C=0.5, solver="{solver}", tol=1e-9, max_iter=500))')
    selector = {
        "SelectFromModel": "SelectFromModel(est^",
        "RFE": "RFE(est^, n_features_to_select=FeatureCount(2)",
        "RFECV": "RFECV[DenseSupervisedCV[E, dtype]](est^, cv=SequentialCV(3)",
    }
    add(f'var selector = {selector[name]}, importance_getter="{getter}")')
    target = "y.copy()" if name == "SelectFromModel" else "y"
    weight_arg = ", sample_weight=weights" if weighted else ""
    add(f"_ = selector.fit(X, {target}{weight_arg})")
    add("assert_false(selector.estimator.__sklearn_is_fitted__())")
    add("var support = selector.get_support()[List[Bool]].copy()")
    exact_list("support", ref.get_support())
    dump("selector.transform(X)", ref.transform(X))
    add("var fitted = selector.estimator_()")
    classes_source = "fitted" if name == "SelectFromModel" else "selector"
    add(f"var classes = {classes_source}.classes_()")
    label_array("classes", ref.estimator_.classes_)
    dump('fitted.importances("coef_")', ref.estimator_.coef_)
    if name == "SelectFromModel":
        add("print(selector.threshold_())")
        numeric.append(ref.threshold_)
    else:
        add("var ranking = selector.ranking_()")
        exact_list("ranking", ref.ranking_)
        add("var prediction = selector.predict(X)")
        label_array("prediction", ref.predict(X))
    receiver = "fitted" if name == "SelectFromModel" else "selector"
    reference = ref.estimator_ if name == "SelectFromModel" else ref
    for method in ("decision_function", "predict_proba", "predict_log_proba"):
        dump(f"{receiver}.{method}(X)", getattr(reference, method)(X))
    score_weight = ", weights.copy()" if weighted else ""
    add(f"print({receiver}.score(X, y{score_weight}))")
    numeric.append(reference.score(X, y, **weight_kw))
    if name == "RFECV":
        add("var results = selector.cv_results_()")
        for key, column in ref.cv_results_.items():
            if key == "n_features":
                add('var counts = results["n_features"][List[Int]].copy()')
                exact_list("counts", column)
            elif key.endswith(("_support", "_ranking")):
                typ = "Bool" if key.endswith("_support") else "Int"
                name_var = f"result_{key}"
                add(f'var {name_var} = results["{key}"][List[List[{typ}]]].copy()')
                exact(f"len({name_var})", len(column))
                for i, values in enumerate(column):
                    exact_list(f"{name_var}[{i}]", values)
            else:
                dump(f'results["{key}"][nm.NDArray[DType.float64]]', column)
    return "\n".join(lines) + "\n", numeric, comparisons


def main():
    logistic, selectors = pinned_estimators()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for dtype, classes, solver in (
        ("float32", 2, "newton-cholesky"), ("float64", 3, "lbfgs")
    ):
        for selector_type in selectors:
            name = selector_type.__name__
            code, expected, exact = fixture(name, dtype, classes, solver, logistic, selector_type)
            source = OUTPUT / f"{name}-{dtype}.mojo"
            binary = source.with_suffix("")
            source.write_text(code)
            command = compiler_command(
                ["mojo", "build", "-O3", "-I", ROOT / "builds/deps", source, "-o", binary]
            )
            with source.with_suffix(".log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=log)
            if result.returncode:
                raise RuntimeError(f"See compiler log {source.with_suffix('.log')}")
            actual = np.array(subprocess.check_output([binary], text=True).split(), dtype=float)
            # Independent solver termination and Float32 rounding can affect
            # probabilities and coefficients. Selection and label identities are exact.
            tolerance = 1e-5 if dtype == "float32" else 1e-6
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
            total = len(expected) + exact
            count += total
            print(f"PASS {name}/{dtype}/{classes} classes/{solver} ({total} comparisons)", flush=True)
    print(f"PASS LogisticRegression selector parity ({count} comparisons)")


if __name__ == "__main__":
    main()
