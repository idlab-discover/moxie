"""Compare dense binary GPC with the pinned local sklearn implementation."""
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch
import warnings

import numpy as np
import sklearn.gaussian_process

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path(os.environ.get("MOXIE_WORKSPACE", ROOT.parent)) / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
BUILD = ROOT / "builds/tests/gaussian_process/classification"
GP = Path("sklearn/gaussian_process")


def load_pinned(filename, name):
    identity = subprocess.check_output(
        ["git", "-C", REFERENCE, "rev-parse", "HEAD"], text=True
    ).strip()
    assert identity == REVISION, identity
    changes = subprocess.check_output(
        ["git", "-C", REFERENCE, "status", "--porcelain", "--", GP / filename],
        text=True,
    ).strip()
    assert not changes, changes
    path = REFERENCE / GP / filename
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def configure(gpc, kernels, case):
    if case == "default":
        return gpc()
    length_scale = 0.01 if case.startswith("restart") else 0.8
    kernel = kernels.ConstantKernel(1.0, (0.01, 100.0)) * kernels.RBF(length_scale, (0.001, 10.0))
    kwargs = {}
    if case in ("fixed", "warm", "cold"):
        kwargs["optimizer"] = None
    if case in ("warm", "cold"):
        kwargs.update(max_iter_predict=1, warm_start=case == "warm")
    if case == "restarts":
        kwargs.update(n_restarts_optimizer=3, random_state=21)
    return gpc(kernel, **kwargs)


def reference(model, X, y, query):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    mean, variance = model.latent_mean_and_variance(query)
    result = {
        "classes": model.classes_, "prediction": model.predict(query),
        "probability": model.predict_proba(query), "mean": mean, "variance": variance,
        "score": model.score(X, y), "lml": model.log_marginal_likelihood(),
        "theta": model.kernel_.theta,
        "warnings": sum(w.category.__name__ == "ConvergenceWarning" for w in caught),
    }
    theta = model.kernel_.theta + 0.1
    result["shifted"], result["gradient"] = model.log_marginal_likelihood(theta, True)
    result["cloned_theta"] = model.kernel_.theta
    result["inplace_lml"] = model.log_marginal_likelihood(theta, clone_kernel=False)
    result["inplace_theta"] = model.kernel_.theta
    result["cached_lml"] = model.log_marginal_likelihood_value_
    return result


def native(case, kind, X, y, query):
    fixture = BUILD / "cases.txt"
    header = f"{X.shape[0]} {X.shape[1]} {query.shape[0]}"
    data = " ".join(str(float(v)) for a in (X, y, query) for v in a.ravel())
    fixture.write_text(header + "\n" + data + "\n")
    run = subprocess.run(
        [BUILD / "gpc_driver", fixture, kind, case],
        check=True, text=True, capture_output=True,
    )
    result = {}
    for line in run.stdout.splitlines():
        key, shape, values = line.split("|")
        dims = tuple(map(int, shape.split(","))) if shape else ()
        result[key] = np.fromstring(values, sep=" ").reshape(dims)
    return result


def main():
    kernels = load_pinned("kernels.py", "pinned_gpc_kernels")
    with patch.dict(sys.modules, {"sklearn.gaussian_process.kernels": kernels}):
        gpc = load_pinned("_gpc.py", "pinned_gpc").GaussianProcessClassifier

    rng = np.random.RandomState(12)
    X = rng.normal(size=(28, 2))
    y = np.where(X[:, 0] ** 2 + X[:, 1] ** 2 > 1.2, 12, -5)
    query = rng.normal(size=(9, 2))
    exact = {"classes", "prediction", "score", "warnings"}
    maxima = {}
    baseline = {}
    count = 0
    cases = ("default", "fixed", "optimized", "restart_baseline", "restarts", "warm", "cold")
    for case in cases:
        for kind, dtype in (("f64", np.float64), ("f32", np.float32)):
            actual = native(case, kind, X, y, query)
            expected = reference(configure(gpc, kernels, case), X.astype(dtype), y, query.astype(dtype))
            if case == "restart_baseline":
                baseline[kind] = actual["lml"]
            if case == "restarts":
                assert actual["lml"] > baseline[kind] + 1
                repeated = native(case, kind, X, y, query)
                np.testing.assert_array_equal(actual["theta"], repeated["theta"])
                count += 2
            assert actual.keys() == expected.keys()
            for key, want in expected.items():
                got = actual[key]
                assert got.shape == np.shape(want), (case, kind, key, got.shape, np.shape(want))
                if key in exact:
                    np.testing.assert_array_equal(got, want, err_msg=f"{case}/{kind}/{key}")
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
                        err_msg=f"{case}/{kind}/{key}",
                    )
                    if got.size:
                        maxima[kind] = max(maxima.get(kind, 0), float(np.max(np.abs(got - want))))
                count += 1
    print(f"PASS: {count} binary GPC comparisons, max absolute errors {maxima}")


if __name__ == "__main__":
    main()
