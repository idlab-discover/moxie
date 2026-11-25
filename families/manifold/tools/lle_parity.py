"""Compare dense LLE and its classifier pipeline against pinned sklearn."""
import argparse
import json
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import eigh
from sklearn.datasets import load_digits, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from check import ROOT, assemble, compile_file, environment
from isomap_parity import pinned, read_matrix

WORK = ROOT / "builds/lle-parity"
BINARY = WORK / "driver"
METHODS = ("standard", "modified", "hessian", "ltsa")


def matrix(name, values):
    path = WORK / f"{name}.txt"
    with path.open("w") as stream:
        stream.write(f"{values.shape[0]} {values.shape[1]}\n")
        np.savetxt(stream, values, fmt="%.17g")
    return path


def native(X, query, settings, operation, labels=None):
    args = [
        BINARY, X.dtype, matrix("train", X), matrix("query", query),
        settings["n_neighbors"], settings["n_components"], settings["method"],
        settings.get("reg", 0.001), settings.get("neighbors_algorithm", "auto"),
        settings.get("eigen_solver", "dense"), operation,
    ]
    if labels is not None:
        args.append(matrix("labels", labels[:, None]))
    result = subprocess.run(
        list(map(str, args)), check=True, capture_output=True, text=True, env=environment(ROOT),
    )
    assert not result.stderr, result.stderr
    return iter(result.stdout.splitlines())


def signs(first, expected):
    return np.where(np.sum(first * expected, axis=0) < 0, -1, 1)


def compare(reference, X, query, label, *, n_neighbors, n_components, method, **options):
    settings = dict(n_neighbors=n_neighbors, n_components=n_components, method=method, **options)
    expected = reference.LocallyLinearEmbedding(eigen_solver="dense", **settings).fit(X)
    native_settings = dict(settings, eigen_solver="auto" if method == "standard" else "dense")
    lines = native(X, query, native_settings, "estimator")
    error, n_features, n_samples = next(lines).split()
    assert int(n_features) == expected.n_features_in_
    assert int(n_samples) == expected.nbrs_.n_samples_fit_
    assert [next(lines) for _ in range(n_components)] == list(expected.get_feature_names_out())
    assert expected.embedding_.dtype == expected.transform(query).dtype == np.float64
    actual = read_matrix(lines, np.dtype("float64"))
    actual_projected = read_matrix(lines, np.dtype("float64"))
    assert not list(lines)

    direction = signs(actual, expected.embedding_)
    assert_allclose(actual * direction, expected.embedding_, atol=4e-8, rtol=4e-8, err_msg=label)
    assert_allclose(float(error), expected.reconstruction_error_, atol=5e-10, rtol=4e-8, err_msg=label)
    # Pinned sklearn casts fit data to Float64. Its transform stores weights
    # in the query dtype before the Float64 embedding product.
    query_tol = 3e-7 if X.dtype == np.float32 else 4e-8
    assert_allclose(actual_projected * direction, expected.transform(query),
                    atol=query_tol, rtol=query_tol, err_msg=label)

    function_options = {key: value for key, value in settings.items() if key != "neighbors_algorithm"}
    expected_function, expected_error = reference.locally_linear_embedding(
        X, eigen_solver="dense", **function_options,
    )
    lines = native(X, query, settings, "function")
    actual_error = float(next(lines))
    actual_function = read_matrix(lines, np.dtype("float64"))
    assert not list(lines)
    expected_dtype = np.float32 if X.dtype == np.float32 and method == "standard" else np.float64
    assert expected_function.dtype == expected_dtype
    # The Float32 sklearn function fits in the input dtype before the
    # eigensolve, which affects small eigenvalues and their eigenvectors.
    function_tol = 3e-3 if X.dtype == np.float32 else 4e-8
    assert_allclose(actual_function * signs(actual_function, expected_function), expected_function,
                    atol=function_tol, rtol=function_tol, err_msg=label)
    assert_allclose(actual_error, expected_error,
                    atol=2e-6 if X.dtype == np.float32 else 5e-10, rtol=function_tol, err_msg=label)
    print(f"PASS {label}: estimator, held-out transform, function")


def degenerate_hessian(reference, X, query):
    settings = dict(n_neighbors=7, n_components=2, method="hessian", eigen_solver="dense")
    operator = []
    solve = reference.null_space

    def capture(matrix, *args, **kwargs):
        # Capture the pinned implementation's independently assembled operator.
        operator.append(matrix.copy())
        return solve(matrix, *args, **kwargs)

    with patch.object(reference, "null_space", capture):
        expected = reference.LocallyLinearEmbedding(**settings).fit(X)
    assert len(operator) == 1
    operator = operator[0]
    values = eigh(operator, eigvals_only=True)
    # The skipped eigenpair and first returned eigenpair share a null space.
    assert np.max(np.abs(values[:2])) < 1e-10
    assert values[2] > 1e-8

    lines = native(X, query, settings, "estimator")
    error = float(next(lines).split()[0])
    for _ in range(settings["n_components"]):
        next(lines)
    embedding = read_matrix(lines, np.dtype("float64"))
    projected = read_matrix(lines, np.dtype("float64"))
    assert not list(lines)
    assert_allclose(embedding.T @ embedding, np.eye(2), atol=1e-10)
    assert_allclose(operator @ embedding, embedding * values[1:3], atol=1e-10)
    assert_allclose(error, expected.reconstruction_error_, atol=1e-10)
    nearest = expected.nbrs_.kneighbors(query, return_distance=False)
    weights = reference.barycenter_weights(query, X, nearest, reg=expected.reg)
    assert_allclose(projected, np.einsum("qk,qkd->qd", weights, embedding[nearest]), atol=1e-10)
    print("PASS Hessian LLE with a degenerate null space")


def composition(reference):
    X, y = make_blobs(n_samples=96, n_features=4, centers=2, cluster_std=3.5, random_state=21)
    train, query = X[:72], X[72:]
    settings = dict(n_neighbors=11, n_components=2, method="standard", eigen_solver="dense")
    expected = make_pipeline(
        reference.LocallyLinearEmbedding(**settings), KNeighborsClassifier(n_neighbors=3),
    ).fit(train, y[:72]).predict(query)
    lines = native(train, query, settings, "pipeline", labels=y[:72])
    size = int(next(lines))
    assert size == len(expected)
    assert_array_equal(np.fromiter(lines, dtype=float), expected)
    assert len(np.unique(expected)) > 1
    print("PASS LocallyLinearEmbedding → KNeighborsClassifier pipeline")


def estimator_parity(reference):
    WORK.mkdir(parents=True, exist_ok=True)
    assemble(ROOT)
    compile_file(ROOT / "tests/lle_driver.mojo", BINARY)
    rng = np.random.default_rng(917)
    for dtype in (np.float64, np.float32):
        for features, k in ((4, 10), (10, 9)):
            raw = rng.normal(size=(44, features)).astype(dtype)
            query = rng.normal(size=(9, features)).astype(dtype)
            # Equalize sample radii in the narrow fixture so Hessian/LTSA do
            # not leave a sample outside every other sample's neighborhood.
            X = raw / np.linalg.norm(raw, axis=1, keepdims=True) if features == 10 else raw
            for method in METHODS:
                compare(reference, X, query, f"{dtype.__name__} {features} features/{k} neighbors {method}",
                        n_neighbors=k, n_components=2, method=method, reg=0.007,
                        neighbors_algorithm="ball_tree" if features == 4 else "kd_tree")
            if dtype == np.float64 and features == 10:
                degenerate_hessian(reference, raw, query)
    X = rng.normal(size=(30, 4))
    compare(reference, X, X[:4] + 0.01, "one component", n_neighbors=8, n_components=1,
            method="standard", neighbors_algorithm="brute")
    composition(reference)


def notebook_parity(reference):
    notebooks = ROOT.parents[1] / "notebooks"
    sys.path.insert(0, str(notebooks / "_support"))
    from moxie_nb import Session

    path = notebooks / "manifold/plot_lle_digits.ipynb"
    cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]}
    X = load_digits(n_class=6).data
    nb = Session("manifold/plot_lle_digits_lle_validation")
    namespace = dict(X=X, S=X[:200], nb=nb, np=np, LocallyLinearEmbedding=reference.LocallyLinearEmbedding)
    exec(cells["lle-digits-reference"], namespace)
    signature, body = cells["lle-digits-native"].split("\n", 1)
    assert signature == "%%mojo lle_digits"
    nb.run_mojo("lle_digits", body)
    exec(cells["lle-digits-parity"], namespace)
    for check in nb.checks:
        print(check.status, check.name, check.max_abs)
    nb.assert_parity()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="store_true", help="Run the focused digits LLE cells")
    args = parser.parse_args()
    reference = pinned("sklearn/manifold/_locally_linear.py", "sklearn.manifold._pinned_lle")
    if args.notebook:
        notebook_parity(reference)
    else:
        estimator_parity(reference)
