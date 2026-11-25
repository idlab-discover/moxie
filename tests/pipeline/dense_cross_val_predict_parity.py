"""Check native dense predictions against pinned sklearn CV and Pipeline."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import sklearn
import sklearn.model_selection._validation as validation
import sklearn.utils._metadata_requests as metadata
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ORACLE = Path(__file__).resolve().parents[3] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
VALIDATION = "sklearn/model_selection/_validation.py"
METADATA = "sklearn/utils/_metadata_requests.py"
SPLIT = "sklearn/model_selection/_split.py"
DUMMY = "sklearn/dummy.py"
PIPELINE = "sklearn/pipeline.py"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ORACLE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_functions(file, namespace, names):
    path = ORACLE / file
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {function.name for function in functions} == set(names)
    for function in functions:
        function.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)


def main():
    assert sklearn.__version__ == "1.9.1"
    assert subprocess.check_output(
        ["git", "-C", ORACLE, "rev-parse", "HEAD"], text=True,
    ).strip() == REVISION
    assert not subprocess.check_output(
        ["git", "-C", ORACLE, "status", "--porcelain", "--",
         VALIDATION, METADATA, SPLIT, DUMMY, PIPELINE], text=True,
    )
    split = load("sklearn.model_selection._moxie_split_reference", SPLIT)
    dummy = load("sklearn._moxie_dummy_reference", DUMMY)
    pipeline = load("sklearn._moxie_pipeline_reference", PIPELINE)
    routing = vars(metadata).copy()
    load_functions(METADATA, routing, {"_manual_routing"})
    namespace = dict(
        vars(validation), check_cv=split.check_cv, _manual_routing=routing["_manual_routing"],
        array_device=lambda x: None, move_to=lambda x, **kwargs: np.asarray(x),
    )
    load_functions(VALIDATION, namespace, {"cross_val_predict", "_fit_and_predict"})
    predict = namespace["cross_val_predict"]

    rows = np.arange(12)
    X = rows.astype(np.float32).reshape(-1, 1)
    y = (rows % 5).astype(np.float32)
    multi = np.column_stack([y, (rows * 3 % 7).astype(np.float32)])
    weights = (rows + 1).astype(np.float32)
    folds = list(split.KFold().split(X))[::-1]
    expected_regression = predict(dummy.DummyRegressor(), X, y)
    expected_weighted = predict(
        dummy.DummyRegressor(), X, multi, cv=folds, params={"sample_weight": weights},
    )

    rows = np.arange(30)
    classes = (rows * 11 % 31) // 11
    X = np.column_stack([1.5 * classes + rows / 20, (rows * 7) % 5])
    labels = np.asarray(["red", "green", "blue"])[classes]
    model = pipeline.Pipeline([("scale", StandardScaler()), ("svc", SVC())])
    expected_labels = predict(model, X, labels)
    assert np.any(expected_labels != predict(model, X, labels, cv=split.KFold()))

    actual = {}
    for line in subprocess.check_output([sys.argv[1]], text=True).splitlines():
        name, value = line.split()
        actual.setdefault(name, []).append(value)
    assert actual.keys() == {"regression", "weighted", "classification"}
    # Independent Float32 mean implementations can round at different steps.
    np.testing.assert_allclose(
        np.asarray(actual["regression"], dtype=np.float32), expected_regression,
        rtol=0, atol=5e-7,
    )
    np.testing.assert_allclose(
        np.asarray(actual["weighted"], dtype=np.float32).reshape(expected_weighted.shape),
        expected_weighted, rtol=0, atol=5e-7,
    )
    np.testing.assert_array_equal(actual["classification"], expected_labels)
    print("PASS pinned cross_val_predict: 36 Float32 cells and 30 exact String SVC labels")


if __name__ == "__main__":
    main()
