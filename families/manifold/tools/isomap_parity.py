"""Compare the dense Isomap entry points with pinned scikit-learn sources."""
import argparse
import importlib.util
import json
import subprocess
import sys
import warnings
from unittest.mock import patch

import numpy as np
from scipy.sparse import SparseEfficiencyWarning
from scipy.spatial.distance import cdist
from sklearn.pipeline import make_union
from sklearn.preprocessing import KernelCenterer, StandardScaler
from sklearn.utils import fixes

from check import ROOT, assemble, compile_file, environment

REFERENCE = ROOT.parents[2] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
WORK = ROOT / "builds/isomap-parity"
BINARY = WORK / "driver"
DISCONNECTED = "The number of connected components of the neighbors graph"


def pinned(relative, name):
    path = REFERENCE / relative
    expected = subprocess.check_output(["git", "-C", str(REFERENCE), "show", f"{REVISION}:{relative}"])
    assert path.read_bytes() == expected, f"{relative} differs from pinned revision"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def oracle():
    if hasattr(fixes, "_ensure_sparse_index_int32"):
        module = pinned("sklearn/manifold/_isomap.py", "sklearn.manifold._pinned_isomap")
    else:
        pinned_fixes = pinned("sklearn/utils/fixes.py", "sklearn.utils._pinned_isomap_fixes")
        with patch.object(fixes, "_ensure_sparse_index_int32", pinned_fixes._ensure_sparse_index_int32, create=True):
            module = pinned("sklearn/manifold/_isomap.py", "sklearn.manifold._pinned_isomap")
    module.KernelPCA = pinned(
        "sklearn/decomposition/_kernel_pca.py", "sklearn.decomposition._pinned_isomap_kernel_pca"
    ).KernelPCA
    module._fix_connected_components = pinned(
        "sklearn/utils/graph.py", "sklearn.utils._pinned_isomap_graph"
    )._fix_connected_components
    return module.Isomap


def matrix(name, values):
    path = WORK / f"{name}.txt"
    with path.open("w") as stream:
        stream.write(f"{values.shape[0]} {values.shape[1]}\n")
        np.savetxt(stream, values, fmt="%.17g")
    return str(path)


def read_matrix(lines, dtype):
    rows, columns, actual_dtype = next(lines).split()
    assert actual_dtype == str(dtype), actual_dtype
    shape = int(rows), int(columns)
    return np.fromiter(lines, dtype=dtype, count=np.prod(shape)).reshape(shape)


def native(X, query, settings, operation):
    arguments = [
        BINARY, X.dtype, matrix("train", X), matrix("query", query),
        settings.get("n_neighbors", 5), settings.get("radius"), settings.get("path_method", "auto"),
        settings.get("eigen_solver", "auto"), settings.get("neighbors_algorithm", "auto"),
        settings.get("metric", "minkowski"), settings.get("n_components", 2), settings.get("p", 2),
        operation,
    ]
    if "metric_params" in settings:
        arguments.append(matrix("weights", settings["metric_params"]["w"][None, :]))
    return subprocess.run(list(map(str, arguments)), text=True, capture_output=True, check=True,
                          env=environment(ROOT))


def case(sklearn_isomap, X, query, *, disconnected=False, **settings):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        warnings.simplefilter("always", UserWarning, append=True)
        expected = sklearn_isomap(**settings).fit(X)
    expected_warning = [str(w.message) for w in seen if DISCONNECTED in str(w.message)]
    assert bool(expected_warning) == disconnected
    result = native(X, query, settings, "model")
    assert not result.stderr
    lines = iter(result.stdout.splitlines())
    n_features, n_samples, error = next(lines).split()
    assert int(next(lines)) == len(expected_warning)
    for message in expected_warning:
        assert next(lines) == "UserWarning"
        assert next(lines) == message
    assert int(n_features) == expected.n_features_in_
    assert int(n_samples) == expected.nbrs_.n_samples_fit_
    assert [next(lines) for _ in expected.get_feature_names_out()] == list(expected.get_feature_names_out())
    paths = read_matrix(lines, expected.dist_matrix_.dtype)
    embedding = read_matrix(lines, expected.embedding_.dtype)
    projected = read_matrix(lines, expected.embedding_.dtype)
    assert not list(lines)

    float32 = X.dtype == np.float32
    np.testing.assert_allclose(paths, expected.dist_matrix_, atol=2e-6 if float32 else 1e-10,
                               rtol=2e-6 if float32 else 1e-10)
    signs = np.where(np.sum(embedding * expected.embedding_, axis=0) < 0, -1, 1)
    tolerance = 3e-5 if float32 else 3e-9
    np.testing.assert_allclose(embedding * signs, expected.embedding_, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(projected * signs, expected.transform(query), atol=tolerance, rtol=tolerance)
    if float32:
        # sklearn subtracts two nearly equal Float32 norms, which can lose
        # the remaining spectrum or return NaN. Resolve that reference residual
        # from the centered Float64 spectrum of the same training distances.
        kernel = KernelCenterer().fit_transform(-0.5 * expected.dist_matrix_.astype(np.float64) ** 2)
        omitted = np.linalg.eigvalsh(kernel)[: -embedding.shape[1]]
        expected_error = np.linalg.norm(omitted) / len(X)
    else:
        with np.errstate(invalid="ignore"):
            expected_error = expected.reconstruction_error()
        if np.isnan(expected_error):
            expected_error = 0.0
    np.testing.assert_allclose(float(error), expected_error,
                               atol=3e-6 if float32 else 1e-7, rtol=tolerance)


def composition(sklearn_isomap, X, query):
    settings = dict(n_neighbors=4, path_method="FW")
    expected = make_union(sklearn_isomap(**settings), StandardScaler())
    fitted = expected.fit_transform(X)
    result = native(X, query, settings, "union")
    assert not result.stderr
    lines = iter(result.stdout.splitlines())
    embedding = read_matrix(lines, X.dtype)
    transformed = read_matrix(lines, X.dtype)
    assert not list(lines)
    signs = np.ones(fitted.shape[1])
    signs[:2] = np.where(np.sum(embedding[:, :2] * fitted[:, :2], axis=0) < 0, -1, 1)
    tolerance = 3e-5 if X.dtype == np.float32 else 3e-9
    np.testing.assert_allclose(embedding * signs, fitted, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(transformed * signs, expected.transform(query), rtol=tolerance, atol=tolerance)


def estimator_parity(sklearn_isomap):
    WORK.mkdir(parents=True, exist_ok=True)
    assemble(ROOT)
    compile_file(ROOT / "tests/isomap_driver.mojo", BINARY)

    rng = np.random.default_rng(492)
    t = np.linspace(-1.9, 2.1, 31)
    curve = np.column_stack((t, np.sin(t), 0.4 * t**2, 0.08 * rng.normal(size=t.size)))
    queries = curve[1::5] + 0.01 * rng.normal(size=curve[1::5].shape)
    cases = 0
    for dtype in (np.float32, np.float64):
        X, Q = curve.astype(dtype), queries.astype(dtype)
        D = cdist(X, X).astype(dtype)
        DQ = cdist(Q, X).astype(dtype)
        for path in ("auto", "FW", "D"):
            case(sklearn_isomap, X, Q, n_neighbors=3, path_method=path)
            case(sklearn_isomap, X, Q, n_neighbors=None, radius=0.42, path_method=path,
                 neighbors_algorithm="ball_tree")
            case(sklearn_isomap, D, DQ, n_neighbors=4, metric="precomputed", path_method=path)
            cases += 3
        case(sklearn_isomap, X, Q, n_neighbors=4, metric="manhattan", neighbors_algorithm="kd_tree",
             eigen_solver="arpack")
        weights = np.linspace(0.9, 1.7, X.shape[1]).astype(dtype)
        case(sklearn_isomap, X, Q, n_neighbors=4, metric="minkowski", p=3,
             metric_params={"w": weights}, neighbors_algorithm="brute")
        separated = X.copy()
        separated[10:21, 0] += 7
        separated[21:, 0] += 16
        case(sklearn_isomap, separated, Q, n_neighbors=2, disconnected=True, path_method="D")
        case(sklearn_isomap, cdist(separated, separated).astype(dtype),
             cdist(Q, separated).astype(dtype), n_neighbors=2, metric="precomputed", disconnected=True)
        # Coincident rows must stay connected by zero-weight edges.
        repeated = np.array([[0.0], [0.0], [1.0], [1.0], [2.5], [4.3]], dtype=dtype)
        case(sklearn_isomap, repeated, repeated, n_neighbors=2, n_components=1, path_method="FW")
        composition(sklearn_isomap, X, Q)
        cases += 5
    print(f"PASS: {cases} dense Isomap fits and 2 FeatureUnion compositions against {REVISION}")


def notebook_parity(sklearn_isomap):
    from sklearn.datasets import load_digits

    notebooks = ROOT.parents[1] / "notebooks"
    sys.path.insert(0, str(notebooks / "_support"))
    from moxie_nb import Session

    path = notebooks / "manifold/plot_lle_digits.ipynb"
    cells = {cell["id"]: "".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]}
    X = load_digits(n_class=6).data
    S = X[:200]
    nb = Session("manifold/plot_lle_digits_isomap_validation")
    nb.share(S=S)
    namespace = dict(X=X, S=S, nb=nb, np=np, Isomap=sklearn_isomap)
    exec(cells["isomap-digits-reference"], namespace)
    signature, native = cells["6e9996e1"].split("\n", 1)
    assert signature == "%%mojo isomap_digits"
    nb.run_mojo("isomap_digits", native)
    exec(cells["isomap-digits-parity"], namespace)
    for check in nb.checks:
        print(check.name, check.status, check.max_abs)
    nb.assert_parity()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="store_true", help="Run the focused digits Isomap cells")
    args = parser.parse_args()
    if args.notebook:
        notebook_parity(oracle())
    else:
        estimator_parity(oracle())
