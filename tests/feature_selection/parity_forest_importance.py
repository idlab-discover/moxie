"""Check dense forest selectors with the source-built, pinned sklearn oracle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "scikit-learn"
ORACLE = ROOT / "families/ensemble/builds/oracle-env"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
OUTPUT = ROOT / "builds/forest-importance-parity"
sys.path.insert(0, str(ROOT / "tools"))

from compiler_limits import compiler_command

CASES = (
    ("SelectFromModel", "RandomForestClassifier", "float32", "string"),
    ("RFE", "RandomForestClassifier", "float32", "string"),
    ("RFECV", "RandomForestClassifier", "float32", "string"),
    ("RFE", "ExtraTreesClassifier", "float32", "int64"),
    ("RFECV", "RandomForestRegressor", "float64", "regression"),
    ("RFE", "RandomForestRegressor", "float64", "regression"),
    ("SelectFromModel", "ExtraTreesRegressor", "float32", "regression"),
)

HEADER = """\
import numojo as nm
from moxie._labels import StringTarget
from moxie.base import clone
from moxie.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from moxie.feature_selection import DenseSupervisedCV, FeatureCount, ForestImportance, RFE, RFECV, SelectFromModel, SequentialCV
from std.math import isnan, nan
from std.testing import assert_equal, assert_false, assert_raises, assert_true

def dump[dtype: DType](values: nm.NDArray[dtype]) raises:
    print(values.ndim)
    for axis in range(values.ndim):
        print(values.shape[axis])
    for i in range(values.size):
        print(values.unsafe_get(i))

def dump_list[T: Writable & Copyable](values: List[T]):
    print(len(values))
    for value in values:
        print(value)

def dump_labels(values: StringTarget) raises:
    print(1)
    print(values.rows())
    for i in range(values.rows()):
        print(values.get(i))
"""


def fixture(dtype, domain):
    rng = np.random.RandomState(52)
    X = rng.normal(size=(24, 4)).astype(dtype)
    group = np.tile([0, 1], 12)
    X[:, 2] += 8 * group
    if domain == "string":
        y = np.array(["é", "alpha"])[group]
    elif domain == "int64":
        y = np.array([2**53 + 1, 2**53], dtype=np.int64)[group]
    else:
        y = 10 * X[:, 2].astype(np.float64) + 2 * X[:, 0]
    weights = np.linspace(1, 2, len(X)).astype(dtype)
    return X, y, weights


def source(case, X, y, weights):
    selector, model, dtype, domain = case
    if domain == "string":
        model_type = f"{model}[string_labels=True]"
        target = f"StringTarget({json.dumps(y.tolist(), ensure_ascii=False)})"
    elif domain == "int64":
        model_type = f"{model}[label_dtype=DType.int64]"
        target = f"nm.array[DType.int64]({y.tolist()}, [{len(y)}])"
    else:
        model_type = f"{model}[DType.float64]" if dtype == "float64" else model
        target = f"nm.array[DType.float64]({y.tolist()}, [{len(y)}])"
    constructor = f"{model_type}(n_estimators=5, max_features=1.0, max_depth=2, bootstrap=False, random_state=11)"
    code = [HEADER]

    def add(line):
        code.append("    " + line)

    code.append("def main() raises:")
    add(f"var X = nm.array[DType.{dtype}]({X.ravel().tolist()}, [{len(X)}, {X.shape[1]}])")
    add(f"var y = {target}")
    if selector != "RFECV":
        # The values occupy a backwards strided view with a nonzero offset.
        backing = [0.0, *weights[::-1].tolist()]
        add(f"var backing = nm.array[DType.{dtype}]({backing}, [{len(backing)}])")
        add(f"var weights = backing.view_with_layout(nm.Shape({len(X)}), nm.NDArrayStrides(-1), {len(X)})")
    add(f"var adapter = ForestImportance({constructor})")
    add('with assert_raises(contains="NotFittedError"):')
    add('    _ = adapter.importances("auto")')
    add('with assert_raises(contains="requires y"):')
    add("    adapter.fit_selection(X, None, None)")
    add('with assert_raises(contains="unsupported importance_getter"):')
    add('    _ = adapter.importances("coef_")')
    if selector == "RFECV":
        add(f"var model = RFECV[DenseSupervisedCV[type_of(adapter), DType.{dtype}]](adapter^, cv=SequentialCV(3))")
        add("_ = model.fit(X, y)")
    elif selector == "RFE":
        add("var model = RFE(adapter^, n_features_to_select=FeatureCount(2))")
        add("_ = model.fit(X, y, sample_weight=weights)")
    else:
        add('var model = SelectFromModel(adapter^, importance_getter="feature_importances_")')
        add("_ = model.fit(X, y.copy(), sample_weight=weights)")
    add("assert_false(model.estimator.__sklearn_is_fitted__())")
    add("assert_false(clone(model).__sklearn_is_fitted__())")
    add("dump_list(model.get_support()[List[Bool]])")
    add("dump(model.transform(X))")
    add("var fitted = model.estimator_()")
    add('dump(fitted.importances("auto"))')
    add('assert_equal(fitted.importances("auto").size, fitted.n_features())')
    if selector == "SelectFromModel":
        add("print(model.threshold_())")
        add('var invalid_weights = nm.zeros[DType.' + dtype + '](nm.Shape(1))')
        add('with assert_raises(contains="sample_weight.shape"):')
        add("    _ = model.fit(X, y.copy(), sample_weight=invalid_weights)")
        add("assert_equal(model.estimator_().n_features(), 4)")
        add("var missing = X.copy()")
        add(f'for j in range({X.shape[1]}):')
        add(f'    missing[nm.Item(0, j)] = nan[DType.{dtype}]()')
        add("assert_true(isnan(model.transform(missing)[nm.Item(0, 0)]))")
        add("var prefit = SelectFromModel(fitted.copy(), prefit=True)")
        add("assert_equal(prefit.get_support()[List[Bool]], model.get_support()[List[Bool]])")
        add("_ = prefit.fit(X)")
        add("var snapshot = prefit.estimator_()")
        add("assert_equal(len(snapshot.model), 5)")
        add("snapshot.model.feature_importances_[nm.Item(0)] = 42")
        add("_ = snapshot.model.set_params(n_estimators=7, warm_start=True)")
        add("_ = snapshot.fit(X, y)")
        add("assert_equal(len(snapshot.model), 7)")
        add("assert_equal(len(prefit.estimator_().model), 5)")
        add("assert_equal(prefit.get_support()[List[Bool]], model.get_support()[List[Bool]])")
        add(f'var different = X.view_with_layout(nm.Shape({len(X)}, 2), nm.NDArrayStrides(4, 1), 0)')
        add("_ = model.fit(different, y.copy())")
        add("assert_equal(model.estimator_().n_features(), 2)")
        add("assert_equal(fitted.n_features(), 4)")
    else:
        add("dump_list(model.ranking_())")
        add(f'{"dump_labels" if domain == "string" else "dump"}(model.predict(X))')
        add("print(model.score(X, y" + (", weights.copy()" if selector == "RFE" else "") + "))")
    if domain != "regression":
        predictor = "fitted" if selector == "SelectFromModel" else "model"
        add(f"dump({predictor}.predict_proba(X))")
        add(f"dump({predictor}.predict_log_proba(X))")
        add(f'{"dump_list" if domain == "string" else "dump"}({predictor}.classes_())')
    if selector == "RFECV":
        add("var results = model.cv_results_()")
        add('dump_list(results["n_features"][List[Int]])')
        for fold in range(3):
            add(f'dump(results["split{fold}_test_score"][nm.NDArray[DType.float64]])')
            for key, kind in (("support", "Bool"), ("ranking", "Int")):
                add(f'for values in results["split{fold}_{key}"][List[List[{kind}]]]:')
                add("    dump_list(values)")
    return "\n".join(code) + "\n"


def compare(case, X, y, weights, stdout):
    import sklearn
    from sklearn import ensemble, feature_selection

    assert ORACLE in Path(sklearn.__file__).parents, sklearn.__file__
    receipt = ROOT / "families/ensemble/docs/private/forest_engine/oracle-build.json"
    assert json.loads(receipt.read_text())["revision"] == REVISION
    selector, forest, dtype, domain = case
    model = getattr(ensemble, forest)(
        n_estimators=5, max_features=1.0, max_depth=2, bootstrap=False, random_state=11
    )
    if selector == "SelectFromModel":
        reference = feature_selection.SelectFromModel(model, importance_getter="feature_importances_")
    elif selector == "RFE":
        reference = feature_selection.RFE(model, n_features_to_select=2)
    else:
        reference = feature_selection.RFECV(model, cv=3)
    reference.fit(X, y, **({} if selector == "RFECV" else {"sample_weight": weights}))
    lines = iter(stdout.splitlines())
    count = 0

    def check(values, read=str):
        nonlocal count
        values = list(values)
        actual = [read(next(lines)) for _ in values]
        if read is float:
            # Native forest reductions and the Float32 selector's importance
            # storage can round. Feature choices, shapes, and labels are exact.
            np.testing.assert_allclose(actual, values, rtol=2e-6, atol=2e-7)
        else:
            assert actual == values, (actual, values)
        count += len(values)

    def check_list(values, read=int):
        check([len(values)], int)
        check(values, read)

    def check_array(values, read=float):
        check([values.ndim, *values.shape], int)
        check(values.flat, read)

    def boolean(line):
        assert line in ("True", "False")
        return line == "True"

    check_list(reference.get_support(), boolean)
    check_array(reference.transform(X))
    fitted = reference.estimator_
    check_array(fitted.feature_importances_)
    if selector == "SelectFromModel":
        check([reference.threshold_], float)
    else:
        check_list(reference.ranking_)
        check_array(reference.predict(X), {"string": str, "int64": int, "regression": float}[domain])
        check([reference.score(X, y, **({"sample_weight": weights} if selector == "RFE" else {}))], float)
    if domain != "regression":
        data = X if selector == "SelectFromModel" else reference.transform(X)
        check_array(fitted.predict_proba(data))
        with np.errstate(divide="ignore"):
            check_array(fitted.predict_log_proba(data))
        if domain == "string":
            check_list(fitted.classes_, str)
        else:
            check_array(fitted.classes_, int)
    if selector == "RFECV":
        results = reference.cv_results_
        check_list(results["n_features"])
        for fold in range(3):
            check_array(results[f"split{fold}_test_score"])
            for key, read in (("support", boolean), ("ranking", int)):
                for values in results[f"split{fold}_{key}"]:
                    check_list(values, read)
    assert next(lines, None) is None
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--build-only", action="store_true")
    action.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    assert subprocess.check_output(["git", "-C", REFERENCE, "rev-parse", "HEAD"], text=True).strip() == REVISION
    OUTPUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for case in CASES:
        selector, forest, dtype, domain = case
        X, y, weights = fixture(dtype, domain)
        binary = OUTPUT / f"{selector}-{forest}"
        if not args.compare_only:
            driver = binary.with_suffix(".mojo")
            driver.write_text(source(case, X, y, weights))
            with binary.with_suffix(".log").open("w") as log:
                try:
                    subprocess.run(compiler_command([
                        "mojo", "build", "-O3", "-I", ROOT / "builds/deps", driver, "-o", binary
                    ]), check=True, stdout=log, stderr=log)
                except subprocess.CalledProcessError:
                    print(f"Compiler log in {log.name}", file=sys.stderr)
                    raise
        if not args.build_only:
            count = compare(case, X, y, weights, subprocess.check_output([binary], text=True))
            total += count
            print(f"PASS {selector}/{forest}/{dtype}/{domain} ({count} comparisons)", flush=True)
    if not args.build_only:
        print(f"PASS forest selector parity ({total} comparisons)")


if __name__ == "__main__":
    main()
