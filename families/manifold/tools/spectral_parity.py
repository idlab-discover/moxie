"""Compare native dense Laplacian eigenmaps against pinned sklearn."""
import argparse
import json
import subprocess
import sys
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from sklearn.datasets import load_digits

from check import ROOT, assemble, compile_file, environment
from isomap_parity import pinned, read_matrix

WORK = ROOT / "builds/spectral-parity"
BINARY = WORK / "driver"


def native(operation, x, n_components, *options):
    path = WORK / "input.txt"
    with path.open("w") as stream:
        stream.write(f"{x.shape[0]} {x.shape[1]}\n")
        np.savetxt(stream, x, fmt="%.17g")
    result = subprocess.run(
        list(map(str, (BINARY, operation, x.dtype, path, n_components, *options))),
        check=True, capture_output=True, text=True, env=environment(ROOT),
    )
    assert not result.stderr, result.stderr
    return iter(result.stdout.splitlines())


def eigenspace(actual, expected, adjacency, *, normed, skip, label, degenerate=False):
    assert actual.shape == expected.shape, label
    dtype = expected.dtype
    assert actual.dtype == dtype, label
    tol = 3e-5 if dtype == np.float32 else 2e-10
    if not degenerate:
        assert_allclose(actual, expected, atol=tol, rtol=tol, err_msg=label)
    symmetric = (adjacency.astype(np.float64) + adjacency.T) * 0.5
    operator, scales = laplacian(symmetric, normed=normed, return_diag=True)
    vectors = actual.astype(np.float64)
    expected_vectors = expected.astype(np.float64)
    if normed:
        vectors = vectors * scales[:, None]
        expected_vectors = expected_vectors * scales[:, None]
    if degenerate:
        assert_allclose(vectors @ vectors.T, expected_vectors @ expected_vectors.T,
                        atol=tol, rtol=tol, err_msg=label)
    eigenvalues = eigh(operator, eigvals_only=True)[skip : skip + actual.shape[1]]
    assert_allclose(operator @ vectors, vectors * eigenvalues, atol=tol, rtol=tol, err_msg=label)
    assert_allclose(vectors.T @ vectors, np.eye(actual.shape[1]), atol=tol, rtol=tol, err_msg=label)


def function_case(reference, adjacency, label, *, n_components=2, solver="lobpcg",
                  normed=True, drop_first=True, degenerate=False):
    # For these small cases, the reference's LOBPCG path dispatches to LAPACK eigh.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = reference.spectral_embedding(
            adjacency, n_components=n_components, eigen_solver=solver, random_state=17,
            eigen_tol=1e-7, norm_laplacian=normed, drop_first=drop_first,
        )
    lines = native("function", adjacency, n_components, normed, drop_first)
    actual = read_matrix(lines, adjacency.dtype)
    assert not list(lines)
    eigenspace(actual, expected, adjacency, normed=normed, skip=int(drop_first),
               label=label, degenerate=degenerate)
    print(f"PASS {label}")


def estimator_case(reference, x, label, *, affinity, gamma=None, n_components=2, degenerate=False):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        expected = reference.SpectralEmbedding(
            n_components, affinity=affinity, gamma=gamma, eigen_solver="lobpcg", random_state=17,
        ).fit(x)
    lines = native("estimator", x, n_components, affinity, gamma)
    n_features, n_warnings = map(int, next(lines).split())
    assert n_features == expected.n_features_in_
    assert [(next(lines), next(lines)) for _ in range(n_warnings)] == [
        (entry.category.__name__, str(entry.message)) for entry in seen
    ]
    actual_gamma = next(lines)
    if affinity == "rbf":
        assert float(actual_gamma) == expected.gamma_
    else:
        assert actual_gamma == "None"
    actual_affinity = read_matrix(lines, x.dtype)
    actual_embedding = read_matrix(lines, x.dtype)
    assert not list(lines)
    if affinity == "precomputed":
        assert_array_equal(actual_affinity, expected.affinity_matrix_)
    else:
        tol = 3e-7 if x.dtype == np.float32 else 2e-13
        assert_allclose(actual_affinity, expected.affinity_matrix_, atol=tol, rtol=tol, err_msg=label)
    eigenspace(actual_embedding, expected.embedding_, expected.affinity_matrix_,
               normed=True, skip=1, label=label, degenerate=degenerate)
    print(f"PASS {label}")


def notebook_case(reference):
    checkout = ROOT.parents[1]
    notebook = checkout / "notebooks/manifold/plot_lle_digits.ipynb"
    cells = {
        cell["id"]: "".join(cell["source"])
        for cell in json.loads(notebook.read_text())["cells"]
    }
    sys.path.insert(0, str(checkout / "notebooks/_support"))
    from moxie_nb import Session

    session = Session("manifold/spectral_digits_focused")
    context = dict(
        nb=session, X=load_digits(n_class=6).data, SpectralEmbedding=reference.SpectralEmbedding,
    )
    exec(cells["spectral-digits-reference"], context)
    magic, source = cells["spectral-digits-native"].split("\n", 1)
    session.run_mojo(magic.split()[1], source)
    exec(cells["spectral-digits-parity"], context)
    for check in session.checks:
        print(check.status, check.name)
    assert [check.status for check in session.checks] == ["PASS", "PASS", "GAP"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="store_true", help="Run only the spectral digits cells")
    args = parser.parse_args()
    reference = pinned("sklearn/manifold/_spectral_embedding.py", "sklearn.manifold._pinned_spectral")
    if args.notebook:
        notebook_case(reference)
        return
    WORK.mkdir(parents=True, exist_ok=True)
    assemble(ROOT)
    compile_file(ROOT / "tests/spectral_driver.mojo", BINARY)
    rng = np.random.RandomState(28)
    affinity = rng.uniform(0.1, 2, (12, 12))
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 25)
    disconnected = np.zeros((12, 12))
    disconnected[:5, :5] = affinity[:5, :5]
    disconnected[5:11, 5:11] = affinity[5:11, 5:11]
    asymmetric = affinity.copy()
    asymmetric[2, 5] = 0.7
    x = rng.normal(size=(13, 4))
    digits = load_digits().data[::47][:28] / 16
    for dtype in (np.float32, np.float64):
        name = np.dtype(dtype).name
        graph = affinity.astype(dtype)
        for normed in (True, False):
            for drop in (True, False):
                function_case(reference, graph, f"{name} normalized={normed} drop_first={drop}",
                              normed=normed, drop_first=drop, solver="arpack")
        function_case(reference, disconnected.astype(dtype), f"{name} full disconnected null space",
                      n_components=3, drop_first=False, degenerate=True)
        function_case(reference, graph, f"{name} all eigenpairs", n_components=len(graph),
                      drop_first=False)
        function_case(reference, graph, f"{name} oversized request", n_components=20)
        function_case(reference, np.ones((1, 1), dtype=dtype), f"{name} singleton drops only mode")
        estimator_case(reference, graph, f"{name} estimator oversized request",
                       affinity="precomputed", n_components=20)
        estimator_case(reference, asymmetric.astype(dtype), f"{name} asymmetric precomputed",
                       affinity="precomputed")
        estimator_case(reference, x.astype(dtype), f"{name} default RBF gamma", affinity="rbf")
        estimator_case(reference, digits.astype(dtype), f"{name} digits RBF",
                       affinity="rbf", gamma=0.1, n_components=6)
        estimator_case(reference, x.astype(dtype), f"{name} zero gamma repeated eigenspace",
                       affinity="rbf", gamma=0.0, n_components=len(x) - 1, degenerate=True)
    print("PASS: 26 dense spectral comparisons against pinned sklearn")


if __name__ == "__main__":
    main()
