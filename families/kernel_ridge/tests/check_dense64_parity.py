#!/usr/bin/env python3
"""Generate sklearn/SciPy parity fixtures and run a native Mojo test binary.

The exact local estimator AND shared solver are loaded against a pinned sklearn
support wheel. Python and external linear algebra are validation tools only.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import warnings

import numpy as np
import scipy
from scipy.linalg import lstsq
import sklearn
from sklearn.metrics.pairwise import pairwise_kernels

ROOT = Path(__file__).resolve().parents[1]
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
SEED = 681094


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(checkout):
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected {REVISION}, found {revision}")
    estimator = load_module("sklearn._moxie_kernel_ridge", checkout / "sklearn/kernel_ridge.py")
    # The full dev _ridge module imports newer support APIs than the wheel.
    # Compile its exact solver function, without rewriting its algorithm.
    solver_path = checkout / "sklearn/linear_model/_ridge.py"
    tree = ast.parse(solver_path.read_text(), filename=str(solver_path))
    solver = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_solve_cholesky_kernel")
    namespace = {"np": np, "linalg": scipy.linalg, "warnings": warnings}
    exec(compile(ast.Module(body=[solver], type_ignores=[]), str(solver_path), "exec"), namespace)
    estimator._solve_cholesky_kernel = namespace["_solve_cholesky_kernel"]
    return estimator.KernelRidge


def encode_array(value):
    value = np.asarray(value, dtype=np.float64)
    rows = value.shape[0]
    cols = value.shape[1] if value.ndim == 2 else 1
    data = ",".join(repr(float(x)) for x in value.ravel())
    return f"{value.ndim}:{rows}:{cols}:{data}"


def encode_parameter(value):
    if value is None:
        return "-"
    if np.ndim(value) == 0:
        return f"s:{float(value)!r}"
    return ",".join(repr(float(x)) for x in value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sklearn-checkout", type=Path, default=ROOT.parents[2] / "scikit-learn")
    args = parser.parse_args()
    KernelRidge = reference(args.sklearn_checkout.resolve())
    rng = np.random.default_rng(SEED)
    records = []
    counts = {}

    def add(kind, *fields):
        records.append("|".join([kind] + list(map(str, fields))))
        counts[kind] = counts.get(kind, 0) + 1

    def fit_case(X, y, query, *, kernel="precomputed", alpha=1.0, gamma=None,
                 degree=3.0, coef0=1.0, weights=None, tolerance=2e-9, expect_error=False):
        params = (encode_array(X), encode_array(y), encode_parameter(alpha), kernel,
                  "-" if gamma is None else repr(gamma), degree, coef0, encode_parameter(weights))
        model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        if expect_error:
            try:
                model.fit(X, y, sample_weight=weights)
            except (ValueError, np.linalg.LinAlgError, TypeError):
                add("fit_error", *params)
            else:
                raise AssertionError("Reference unexpectedly accepted invalid case")
            return
        # Native all-zero weight lists are an explicit retained extension.
        # New support validators reject these arrays; scalar zero gives the
        # identical reference system and bypasses that array-only restriction.
        reference_weights = (0.0 if weights is not None and np.ndim(weights) > 0
                             and np.all(np.asarray(weights) == 0) else weights)
        model.fit(X, y, sample_weight=reference_weights)
        predicted = model.predict(query)
        score_y = rng.normal(size=predicted.shape)
        score_weights = rng.uniform(0.1, 2, size=len(query))
        score = model.score(query, score_y, sample_weight=score_weights)
        add("fit", *params, tolerance, encode_array(model.dual_coef_), encode_array(query),
            encode_array(predicted), encode_array(score_y), encode_parameter(score_weights), repr(float(score)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Positive-definite precomputed systems; all output ranks and alpha modes.
        for i in range(100):
            n = int(rng.integers(1, 22))
            m = int(rng.integers(1, 5))
            features = rng.normal(size=(n, n + 2))
            K = features @ features.T + 0.2 * np.eye(n)
            y = rng.normal(size=(n, m)) if i % 3 else rng.normal(size=n)
            alpha = [0.0, 0.1, 2.0][i % 3]
            if i % 4 == 0:
                alpha = rng.uniform(0.1, 2, size=m if y.ndim == 2 else 1)
            if i % 7 == 0:
                alpha = np.full(m if y.ndim == 2 else 1, 0.3)
            weights = None
            if i % 3 == 1:
                weights = rng.uniform(0, 3, size=n)
                weights[0] = 0
            elif i % 3 == 2:
                weights = 2.5
            fit_case(K, y, rng.normal(size=(7, n)), alpha=alpha, weights=weights)

        # Named kernels: direct kernel values plus fitted behavior.
        names = ["linear", "rbf", "laplacian", "poly", "polynomial", "sigmoid", "cosine", "chi2", "additive_chi2"]
        for name in names:
            for i in range(16):
                n, d = 9, 4
                X = rng.normal(size=(n, d))
                Q = rng.normal(size=(5, d))
                if name in ("chi2", "additive_chi2"):
                    X, Q = np.abs(X), np.abs(Q)
                if i % 4 == 0:
                    X[0] = 0
                    Q[0] = 0
                gamma = None if i % 2 == 0 and name != "chi2" else 0.3
                degree = 2.0 if i % 2 else 3.0
                coef0 = 0.7
                gram = pairwise_kernels(X, Q, metric=name, filter_params=True,
                                        gamma=gamma, degree=degree, coef0=coef0)
                add("pairwise", encode_array(X), encode_array(Q), name,
                    "-" if gamma is None else gamma, degree, coef0, encode_array(gram))
                y = rng.normal(size=(n, 2)) if i % 2 else rng.normal(size=n)
                # Indefinite kernels exercise shared-alpha fallback; unequal
                # alpha gets a large enough diagonal to remain positive definite.
                alpha = [30.0, 31.0] if i % 5 == 1 and y.ndim == 2 else 0.8
                weights = rng.uniform(0, 2, size=n) if i % 3 == 0 else None
                fit_case(X, y, Q, kernel=name, alpha=alpha, gamma=gamma, degree=degree,
                         coef0=coef0, weights=weights)
        # Large nearby values expose cancellation and arithmetic-order changes
        # in sklearn.metrics.pairwise._euclidean_distances.
        X = np.array([[95306579.74822405], [95306575.48487124]])
        Q = X[::-1].copy()
        gram = pairwise_kernels(X, Q, metric="rbf", gamma=0.1)
        add("pairwise", encode_array(X), encode_array(Q), "rbf",
            0.1, 3.0, 1.0, encode_array(gram))
        fit_case(X, [1., 2.], Q, kernel="rbf", gamma=0.1)
        fit_case(X, [[1., 2.], [2., 1.]], Q, kernel="rbf", gamma=0.1,
                 weights=[0.5, 2.0])
        # Singular shared-alpha systems (vector, one-column and multioutput).
        for n in (2, 4, 10):
            K = np.ones((n, n))
            for m in (1, 3):
                for rank in (1, 2):
                    if rank == 1 and m != 1:
                        continue
                    y = rng.normal(size=(n, m)) if rank == 2 else rng.normal(size=n)
                    fit_case(K, y, rng.normal(size=(5, n)), alpha=0.0)
                    fit_case(np.zeros((n, n)), y, np.zeros((5, n)), alpha=0.0, weights=0.0)
        # Weighted rank-deficient Gram matrices force fallback via a zero row.
        for i in range(20):
            X = rng.integers(-3, 4, size=(12, 3)).astype(float)
            w = rng.integers(1, 4, size=12).astype(float) ** 2
            w[0] = 0
            fit_case(X, rng.normal(size=(12, 2)), rng.normal(size=(5, 3)),
                     kernel="linear", alpha=0.0, weights=w, tolerance=2e-8)
        # Zero gamma and fractional polynomial degree are legal configurations.
        for name in ("rbf", "laplacian", "poly", "sigmoid", "chi2"):
            X = rng.uniform(0, 1, size=(6, 3))
            fit_case(X, rng.normal(size=6), rng.uniform(0, 1, size=(5, 3)),
                     kernel=name, gamma=0.0, degree=2.5)
        # Nonsymmetric precomputed inputs: successful Cholesky uses upper
        # triangle; fallback least squares uses the entire matrix.
        fit_case(np.array([[2., 1.], [0., 2.]]), [1., 3.], np.eye(2), alpha=0.0)
        fit_case(np.array([[0., 2.], [1., 0.]]), [1., 3.], np.eye(2), alpha=0.0)
        fit_case(np.array([[1., 2.], [2., 1.]]), [1., 3.], np.eye(2), alpha=0.0)
        fit_case(np.ones((3, 3)), np.ones((3, 2)), np.ones((2, 3)), alpha=[0.0, 1.0], expect_error=True)
        fit_case(np.eye(2), [1., 3.], np.eye(2), weights=[1., -1.], expect_error=True)
        fit_case(np.ones((2, 3)), [1., 3.], np.ones((1, 3)), expect_error=True)
        fit_case(np.eye(2), [1., 3.], np.eye(2), alpha=-1.0, expect_error=True)
        fit_case(np.eye(2), [1., 3.], np.eye(2), kernel="chi2", expect_error=True)
        # Fallback preserves scale and truncates numerically null singular values.
        for scale in (1e-150, 1.0, 1e150):
            A = np.diag([1.0, 1e-12, 1e-17]) * scale
            y = np.array([[1.0], [1e-12], [1e-17]]) * scale
            add("lstsq", encode_array(A), encode_array(y), encode_array(lstsq(A, y)[0]), 2e-12)
        # Check Jacobi against SciPy independently of which path Cholesky takes.
        for i in range(75):
            n = int(rng.integers(2, 18))
            rank = max(1, n // 2) if i % 3 == 0 else n
            A = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, n))
            y = rng.normal(size=(n, 3))
            expected = lstsq(A, y)[0]
            add("lstsq", encode_array(A), encode_array(y), encode_array(expected), 2e-8)

    output = ROOT / "builds/dense64-parity"
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "reference.txt"
    fixture.write_text("\n".join(records) + "\n")
    metadata = {"seed": SEED, "cases": counts, "reference_revision": REVISION,
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "sklearn_support_wheel": sklearn.__version__, "platform": platform.platform(),
                "pairwise_tolerance": 2e-12, "fit_tolerance": 2e-9, "lstsq_tolerance": 2e-8}
    metadata["mojo"] = subprocess.check_output(["mojo", "--version"], cwd=ROOT, text=True).strip()
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    binary = output / "runner"
    subprocess.run(["mojo", "build", "-O3", "--fp-mode=contract=off", "-I", "builds/deps", "tests/dense64_parity_runner.mojo", "-o", str(binary)], cwd=ROOT, check=True)
    result = subprocess.run([str(binary), str(fixture)], cwd=ROOT, text=True, capture_output=True)
    (output / "result.txt").write_text(result.stdout + result.stderr)
    print(result.stdout, end="")
    print(result.stderr, end="")
    result.check_returncode()


if __name__ == "__main__":
    main()
