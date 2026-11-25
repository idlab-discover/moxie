"""Compare dense GPC with the pinned local sklearn implementation."""
from itertools import combinations
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch
import warnings

import numpy as np
import sklearn.gaussian_process
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path(os.environ.get("MOXIE_WORKSPACE", ROOT.parent)) / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
BUILD = ROOT / "builds/tests/gaussian_process/classification"


def load_pinned(path, name):
    identity = subprocess.check_output(
        ["git", "-C", REFERENCE, "rev-parse", "HEAD"], text=True
    ).strip()
    assert identity == REVISION, identity
    changes = subprocess.check_output(
        ["git", "-C", REFERENCE, "status", "--porcelain", "--", path],
        text=True,
    ).strip()
    assert not changes, changes
    path = REFERENCE / path
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def configure(gpc, kernels, case, strategy):
    if case == "default":
        return gpc(multi_class=strategy)
    length_scale = 0.01 if case.startswith("restart") else 0.8
    kernel = kernels.ConstantKernel(1.0, (0.01, 100.0)) * kernels.RBF(length_scale, (0.001, 10.0))
    kwargs = {}
    if case in ("fixed", "warm", "cold"):
        kwargs["optimizer"] = None
    if case in ("warm", "cold"):
        kwargs.update(max_iter_predict=1, warm_start=case == "warm")
    if case == "restarts":
        kwargs.update(n_restarts_optimizer=3, random_state=21)
    return gpc(kernel, multi_class=strategy, **kwargs)


def reference(model, X, y, query, *, check_vote_tie=False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    binary = model.n_classes_ == 2
    result = {
        "classes": model.classes_, "prediction": model.predict(query),
        "score": model.score(X, y), "lml": model.log_marginal_likelihood(),
        "theta": model.kernel_.theta,
        "warnings": sum(w.category.__name__ == "ConvergenceWarning" for w in caught),
    }
    if check_vote_tie:
        votes = np.zeros((len(query), model.n_classes_), dtype=int)
        pairs = combinations(range(model.n_classes_), 2)
        for (i, j), estimator in zip(pairs, model.base_estimator_.estimators_, strict=True):
            votes[np.arange(len(query)), np.where(estimator.predict(query), j, i)] += 1
        assert np.any(result["prediction"] != model.classes_[votes.argmax(axis=1)])
    if binary or model.multi_class == "one_vs_rest":
        result["probability"] = model.predict_proba(query)
    if binary:
        result["mean"], result["variance"] = model.latent_mean_and_variance(query)
        theta = model.kernel_.theta + 0.1
        result["shifted"], result["gradient"] = model.log_marginal_likelihood(theta, True)
    else:
        theta = model.kernel_.theta + 0.02 * np.arange(1, model.kernel_.theta.size + 1)
        result["shifted"] = model.log_marginal_likelihood(theta)
        estimators = model.base_estimator_.estimators_
        result["components"] = len(estimators)
        result["shared_lml"] = model.log_marginal_likelihood(np.zeros(estimators[0].kernel_.n_dims))
    result["cloned_theta"] = model.kernel_.theta
    result["inplace_lml"] = model.log_marginal_likelihood(theta, clone_kernel=False)
    result["inplace_theta"] = model.kernel_.theta
    result["cached_lml"] = model.log_marginal_likelihood_value_
    return result


def native(case, kind, X, y, query, strategy):
    fixture = BUILD / "cases.txt"
    header = f"{X.shape[0]} {X.shape[1]} {query.shape[0]}"
    data = " ".join(str(float(v)) for a in (X, y, query) for v in a.ravel())
    fixture.write_text(header + "\n" + data + "\n")
    run = subprocess.run(
        [BUILD / "gpc_driver", fixture, kind, case, strategy],
        check=True, text=True, capture_output=True,
    )
    result = {}
    for line in run.stdout.splitlines():
        key, shape, values = line.split("|")
        dims = tuple(map(int, shape.split(","))) if shape else ()
        result[key] = np.fromstring(values, sep=" ").reshape(dims)
    return result


def pinned_classifier():
    kernels = load_pinned("sklearn/gaussian_process/kernels.py", "pinned_gpc_kernels")
    multiclass_utils = load_pinned("sklearn/utils/multiclass.py", "pinned_multiclass_utils")
    with patch.dict(sys.modules, {"sklearn.utils.multiclass": multiclass_utils}):
        multiclass = load_pinned("sklearn/multiclass.py", "pinned_multiclass")
    with patch.dict(sys.modules, {
        "sklearn.gaussian_process.kernels": kernels,
        "sklearn.multiclass": multiclass,
    }):
        classifier = load_pinned("sklearn/gaussian_process/_gpc.py", "pinned_gpc").GaussianProcessClassifier
    return classifier, kernels


def main():
    gpc, kernels = pinned_classifier()

    rng = np.random.RandomState(12)
    X = rng.normal(size=(28, 2))
    y = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 > 1.2, 12, -5)
    query = rng.normal(size=(9, 2))
    one_vs_rest = "one_vs_rest"
    one_vs_one = "one_vs_one"
    cases = [
        (case, one_vs_rest, X, y, query, f"binary/{case}")
        for case in ("default", "fixed", "optimized", "restart_baseline", "restarts", "warm", "cold")
    ]
    multi_X = np.linspace(0, 10, 30)[:, None]
    multi_y = np.choose(np.digitize(np.sin(multi_X[:, 0]), [-0.35, 0.35]), [19, -7, 3])
    multi_query = np.linspace(0.17, 9.83, 73)[:, None]
    iris_X, iris_y = load_iris(return_X_y=True)
    iris_X = iris_X[:, :2]
    for strategy in (one_vs_rest, one_vs_one):
        cases.extend(
            (case, strategy, multi_X, multi_y, multi_query, f"{strategy}/{case}")
            for case in ("default", "fixed", "warm", "cold", "restarts")
        )
        cases.append(("optimized", strategy, iris_X, iris_y, iris_X[::3], f"{strategy}/iris"))
    tie_rng = np.random.RandomState(1)
    tie_X = tie_rng.normal(size=(30, 2))
    tie_y = tie_rng.permutation(np.repeat([-7, 3, 19], 10))
    tie_query = tie_rng.normal(size=(240, 2))[103:104]
    cases.append(("fixed", one_vs_one, tie_X, tie_y, tie_query, "one_vs_one/vote_tie"))
    exact = {"classes", "prediction", "score", "warnings", "components"}
    maxima = {}
    baseline = {}
    count = 0
    for case, strategy, X, y, query, label in cases:
        for kind, dtype in (("f64", np.float64), ("f32", np.float32)):
            actual = native(case, kind, X, y, query, strategy)
            expected = reference(
                configure(gpc, kernels, case, strategy), X.astype(dtype), y, query.astype(dtype),
                check_vote_tie=label == "one_vs_one/vote_tie",
            )
            if case == "restart_baseline":
                baseline[kind] = actual["lml"]
            if case == "restarts":
                if label.startswith("binary/"):
                    assert actual["lml"] > baseline[kind] + 1
                repeated = native(case, kind, X, y, query, strategy)
                np.testing.assert_array_equal(actual["theta"], repeated["theta"])
                count += 2
            assert actual.keys() == expected.keys()
            for key, want in expected.items():
                got = actual[key]
                assert got.shape == np.shape(want), (label, kind, key, got.shape, np.shape(want))
                if key in exact:
                    np.testing.assert_array_equal(got, want, err_msg=f"{label}/{kind}/{key}")
                else:
                    # Float32 kernels may reduce distances at input precision in
                    # sklearn. Native features are widened before all kernel math.
                    # Optimizer termination further amplifies small slope changes.
                    if case in ("default", "fixed", "warm", "cold"):
                        tolerance = 5e-7 if kind == "f32" else 2e-10
                    else:
                        tolerance = 2e-5 if kind == "f32" else 2e-7
                    np.testing.assert_allclose(
                        got, want, atol=tolerance, rtol=tolerance,
                        err_msg=f"{label}/{kind}/{key}",
                    )
                    if got.size:
                        maxima[kind] = max(maxima.get(kind, 0), float(np.max(np.abs(got - want))))
                count += 1
    print(f"PASS: {count} GPC comparisons and restart checks, max absolute errors {maxima}")


if __name__ == "__main__":
    main()
