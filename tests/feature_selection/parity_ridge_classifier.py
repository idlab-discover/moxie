"""Compare dense RidgeClassifier selectors to pinned scikit-learn."""
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "families/linear_model/tools"))
from compiler_limits import compiler_command
import parity_io as oracle

OUTPUT = ROOT / "builds/ridge-classifier-parity"
HEADER = """\
import numojo as nm
from moxie.feature_selection import (
    DenseSupervisedCV, FeatureCount, RFE, RFECV, RidgeClassifierImportance,
    SelectFromModel, SequentialCV,
)
from moxie.linear_model import RidgeClassifier
from std.testing import assert_equal, assert_false

def dump[dtype: DType](array: nm.NDArray[dtype]) raises:
    print(array.ndim)
    for axis in range(array.ndim):
        print(array.shape[axis])
    for i in range(array.size):
        print(array.unsafe_get(i))
"""


def pinned_estimators():
    linear_base = "sklearn/linear_model/_base.py"
    ridge = "sklearn/linear_model/_ridge.py"

    base, model, rfe = [
        f"sklearn/feature_selection/{name}.py" for name in ("_base", "_from_model", "_rfe")
    ]
    oracle.verify_reference([linear_base, ridge, base, model, rfe])
    oracle.load_pinned(linear_base, register=True)
    classifier = oracle.load_pinned(ridge).RidgeClassifier
    oracle.pinned_code(classifier.fit, ridge)
    oracle.pinned_code(classifier.decision_function, linear_base)
    common = oracle.load_pinned(base, register=True)
    from_model = oracle.load_pinned(model)
    selectors = oracle.load_pinned(rfe)
    oracle.pinned_code(from_model.SelectFromModel.fit, model)
    oracle.pinned_code(selectors.RFE.fit, rfe)
    oracle.pinned_code(selectors.RFECV.fit, rfe)
    assert from_model._get_feature_importances is selectors._get_feature_importances
    assert selectors._get_feature_importances is common._get_feature_importances
    return classifier, (from_model.SelectFromModel, selectors.RFE, selectors.RFECV)


def fixture(name, dtype, n_classes, multilabel, classifier, selector_type):
    rng = np.random.default_rng(842)
    groups = rng.permutation(np.tile(np.arange(n_classes), 10))
    X = rng.normal(0, 0.55, (len(groups), 5)).astype(dtype)
    X[:, 0] += 2 * groups
    X[:, 2] += 1.5 * (groups == 1)
    X[:, 4] -= 0.75 * (groups >= 1)
    y = 2**53 + groups.astype(np.int64)
    if multilabel:
        y = np.column_stack((groups == 0, groups >= 1, groups == 2)).astype(np.int64)
    weights = np.linspace(0.5, 2, len(y), dtype=dtype)
    weighted = name != "RFECV"
    weight_kw = {"sample_weight": weights} if weighted else {}
    options = {
        "SelectFromModel": {},
        "RFE": {"n_features_to_select": 2},
        "RFECV": {"cv": 3},
    }
    getter = "auto" if n_classes == 2 else "coef_"
    model = classifier(alpha=0.5, solver="cholesky")
    ref = selector_type(model, importance_getter=getter, **options[name])
    ref.fit(X, y, **weight_kw)

    lines = [
        HEADER,
        f"comptime dtype = DType.{dtype}",
        "comptime E = RidgeClassifierImportance[dtype]",
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
        exact(f"{expression}.ndim", values.ndim)
        for axis, size in enumerate(values.shape):
            exact(f"{expression}.shape[{axis}]", size)
        for index in np.ndindex(values.shape):
            item = ", ".join(str(i) for i in index)
            exact(f"{expression}[nm.Item({item})]", f"Int64({values[index]})")

    def dump(expression, values, array_dtype="dtype"):
        values = np.asarray(values)
        add(f"dump[{array_dtype}]({expression})")
        numeric.extend([values.ndim, *values.shape, *values.flat])

    def array(variable, values, dtype_arg):
        flat = ", ".join(repr(value) for value in values.ravel().tolist())
        add(f"var {variable} = nm.array[{dtype_arg}]([{flat}], {list(values.shape)})")

    array("X", X, "dtype")
    array("y", y, "DType.int64")
    if weighted:
        array("weights", weights, "dtype")
    add('var est = E(RidgeClassifier[dtype](alpha=0.5, solver="cholesky"))')
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
    dump(f"{receiver}.decision_function(X)", reference.decision_function(X))
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
                dump(f'results["{key}"][nm.NDArray[DType.float64]]', column, "DType.float64")
    return "\n".join(lines) + "\n", numeric, comparisons


def main():
    classifier, selectors = pinned_estimators()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for dtype, classes, multilabel in (("float32", 2, False), ("float64", 3, False), ("float64", 3, True)):
        target_kind = "multilabel" if multilabel else f"{classes}-class"
        for selector_type in selectors:
            name = selector_type.__name__
            code, expected, exact = fixture(name, dtype, classes, multilabel, classifier, selector_type)
            source = OUTPUT / f"{name}-{dtype}-{target_kind}.mojo"
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
            # The reference factorizes in the feature dtype. The native owner
            # widens certain operations and uses a distinct Cholesky factorization.
            tolerance = 3e-4 if dtype == "float32" else 1e-8
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
            total = len(expected) + exact
            count += total
            print(f"PASS {name}/{dtype}/{target_kind} ({total} comparisons)", flush=True)
    print(f"PASS RidgeClassifier selector parity against scikit-learn {oracle.REVISION} ({count} comparisons)")


if __name__ == "__main__":
    main()
