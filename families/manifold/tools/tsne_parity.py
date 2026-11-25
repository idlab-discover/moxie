"""Validate exact t-SNE against pinned Python and Cython sklearn sources."""
import importlib.util
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from sklearn.utils import _array_api

from check import ROOT, assemble, compile_file, environment

REFERENCE = ROOT.parents[2] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
FOLDER = ROOT / "builds/tsne-parity"
BINARY = ROOT / "builds/tsne_driver"


def source(relative):
    path = REFERENCE / relative
    expected = subprocess.check_output(["git", "-C", str(REFERENCE), "show", f"{REVISION}:{relative}"])
    assert path.read_bytes() == expected, f"{relative} differs from pinned revision"
    return path


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_module():
    # Isolate the pinned perplexity extension. Its sole non-libc Cython
    # dependency is a typedef header. All compilation output stays in builds.
    utils = source("sklearn/manifold/_utils.pyx")
    source("sklearn/utils/_typedefs.pxd")
    native = FOLDER / "reference"
    native.mkdir(exist_ok=True)
    pyx = native / "_pinned_tsne_perplexity.pyx"
    if not pyx.exists() or pyx.read_bytes() != utils.read_bytes():
        pyx.write_bytes(utils.read_bytes())
    setup = native / "build_reference.py"
    setup.write_text(
        "from Cython.Build import cythonize\n"
        "from setuptools import Extension, setup\n"
        "setup(name='pinned-tsne-perplexity', ext_modules=cythonize("
        "[Extension('_pinned_tsne_perplexity', ['_pinned_tsne_perplexity.pyx'])],"
        f" include_path=[{str(REFERENCE)!r}], compiler_directives={{'language_level': 3,"
        " 'boundscheck': False, 'wraparound': False, 'initializedcheck': False, 'cdivision': True}))\n"
    )
    subprocess.run(
        [sys.executable, str(setup), "build_ext", "--inplace"],
        cwd=native, env=environment(ROOT), check=True,
    )
    extension, = native.glob("_pinned_tsne_perplexity*.so")
    module = imported("sklearn.manifold._moxie_t_sne", source("sklearn/manifold/_t_sne.py"))
    module._utils = imported("_pinned_tsne_perplexity", extension)
    array_api = imported("sklearn.utils._moxie_tsne_array_api", source("sklearn/utils/_array_api.py"))
    with patch.multiple(_array_api, create=True, _cov=array_api._cov, array_device=array_api.array_device):
        module.PCA = imported("sklearn.decomposition._moxie_tsne_pca", source("sklearn/decomposition/_pca.py")).PCA
    return module


def matrix(name, values):
    path = FOLDER / f"{name}.txt"
    path.write_text(
        f"{values.shape[0]} {values.shape[1]}\n" + " ".join(repr(float(v)) for v in values.flat)
    )
    return str(path)


def native(*args):
    return subprocess.check_output([str(BINARY), *map(str, args)], text=True, env=environment(ROOT))


def native_fit(X, init, *, operation="fit", components=2, metric="euclidean", max_iter=300,
               exaggeration=1, lr=2, min_grad=0, patience=300, seed=0, perplexity=4):
    path = matrix("X", X)
    dtype = init.dtype if isinstance(init, np.ndarray) else np.dtype("float32")
    init = matrix("init", init) if isinstance(init, np.ndarray) else init
    lines = native(operation, X.dtype, dtype, path, init, components, metric, max_iter,
                   exaggeration, lr, min_grad, patience, seed, perplexity).splitlines()
    if operation == "init":
        return np.asarray(lines, dtype=dtype).reshape(len(X), components)
    iteration, kl, rate, features = lines[0].split()
    n, m, kind = lines[1].split()
    assert (int(n), int(m), kind) == (len(X), components, dtype.name)
    return int(iteration), float(kl), float(rate), int(features), np.asarray(lines[2:], dtype=dtype).reshape(len(X), components)


def objective_checks(reference):
    rng = np.random.RandomState(2)
    count = 0
    for dtype in (np.float32, np.float64):
        for components in (1, 2, 3, 4, 7):
            X = rng.standard_normal((12, 5))
            Y = rng.standard_normal((len(X), components)).astype(dtype)
            # Scale one fixture enough to trigger fallback when all row
            # Gaussian numerators underflow on the first beta trial.
            squared = squareform(pdist(X, "sqeuclidean")).astype(np.float32)
            if components == 4:
                squared *= 5000
            args = ("objective", np.dtype(dtype), matrix("Y", Y), matrix("squared", squared), 3)
            objective = np.fromstring(native(*args), sep="\n")
            P = reference._joint_probabilities(squared, 3, 0)
            dof = max(components - 1, 1)
            error, gradient = reference._kl_divergence(Y.ravel(), P, dof, len(Y), components)
            np.testing.assert_allclose(objective[1 + Y.size:], P, atol=2e-16, rtol=2e-14)
            np.testing.assert_allclose(objective[0], error, atol=2e-14, rtol=2e-14)
            np.testing.assert_allclose(objective[1:1 + Y.size], gradient, atol=3e-10 if dtype is np.float32 else 2e-16, rtol=2e-7 if dtype is np.float32 else 2e-14)
            count += 1
    return count


def patience_check(reference):
    coordinates = np.random.RandomState(37).normal(size=(8, 2))
    probabilities = np.zeros(28)
    y, loss, iteration = reference._gradient_descent(
        reference._kl_divergence, coordinates.ravel(), 0, 150, n_iter_check=50,
        n_iter_without_progress=-1, momentum=0.8, learning_rate=2, min_grad_norm=0,
        args=[probabilities, 1, 8, 2],
    )
    lines = native("descent", matrix("Y", coordinates)).splitlines()
    observed_loss, observed_iteration = lines[0].split()
    assert int(observed_iteration) == iteration == 99
    assert float(observed_loss) == loss == 0
    np.testing.assert_allclose(np.asarray(lines[1:], dtype=np.float64), y, rtol=2e-12, atol=2e-12)


def distance_checks():
    count = 0
    for dtype in (np.float64, np.float32):
        X = np.random.RandomState(3).normal(size=(12, 5)).astype(dtype)
        boolean_metrics = ("dice", "jaccard", "rogerstanimoto", "russellrao", "sokalsneath", "yule")
        for metric in (
            "euclidean", "l2", "manhattan", "l1", "cityblock", "cosine", "sqeuclidean",
            "minkowski", "chebyshev", "correlation", "mahalanobis", "seuclidean",
            "canberra", "precomputed", "hamming", "haversine", "nan_euclidean",
            *boolean_metrics,
        ):
            data = X
            args = []
            if metric == "precomputed":
                # Exact sklearn uses both input triangles in its perplexity search.
                distances = pairwise_distances(X)
                distances[1, 0] *= 1.25
                data = distances.copy()
            elif metric == "euclidean":
                distances = pairwise_distances(X, metric=metric, squared=True)
            elif metric == "minkowski":
                distances = pairwise_distances(X, metric=metric, p=1.5)
                args = [1.5]
            elif metric == "haversine":
                data = X[:, :2]
                distances = pairwise_distances(data, metric=metric)
            elif metric in boolean_metrics:
                mask = X > 0
                mask[:, 0] = True  # Keep every row out of the all-zero case.
                data = mask.astype(dtype)
                distances = pairwise_distances(mask, metric=metric)
            else:
                distances = pairwise_distances(X, metric=metric)
            if metric != "euclidean":
                distances **= 2
            got = np.fromstring(native("distances", X.dtype, matrix("X", data), metric, *args), sep="\n")
            np.testing.assert_allclose(got.reshape(distances.shape), distances.astype(np.float32), atol=2e-7, rtol=4e-7)
            count += 1
    return count


def fit_checks(reference):
    rng = np.random.RandomState(3)
    X = rng.standard_normal((16, 5))
    cases = (
        (X, rng.normal(scale=0.01, size=(16, 2)), {}),
        (X.astype(np.float32), rng.normal(scale=0.01, size=(16, 3)).astype(np.float32), dict(components=3)),
        (pairwise_distances(X), rng.normal(scale=0.01, size=(16, 1)), dict(components=1, metric="precomputed")),
        (X, np.zeros((16, 2)), dict(lr="auto", min_grad=1e-7, exaggeration=12)),
        (X, rng.normal(scale=0.01, size=(16, 2)), dict(max_iter=250)),
        (X, rng.normal(scale=0.01, size=(16, 2)), dict(max_iter=301, patience=-1, min_grad=100)),
    )
    for data, init, overrides in cases:
        settings = dict(components=2, max_iter=300, perplexity=4, exaggeration=1,
                        lr=2, min_grad=0, patience=300, metric="euclidean") | overrides
        fitted = reference.TSNE(
            n_components=settings["components"], method="exact", init=init, random_state=0,
            max_iter=settings["max_iter"], perplexity=settings["perplexity"], metric=settings["metric"],
            early_exaggeration=settings["exaggeration"], learning_rate=settings["lr"],
            min_grad_norm=settings["min_grad"], n_iter_without_progress=settings["patience"],
        )
        wanted = fitted.fit_transform(data.copy())
        iteration, kl, rate, features, got = native_fit(data, init, **settings)
        assert (iteration, rate, features) == (fitted.n_iter_, fitted.learning_rate_, fitted.n_features_in_)
        if init.dtype == np.float32:
            np.testing.assert_allclose(kl, fitted.kl_divergence_, atol=2e-7, rtol=2e-7)
            np.testing.assert_allclose(got, wanted, atol=2e-5, rtol=2e-6)
        else:
            np.testing.assert_allclose(kl, fitted.kl_divergence_, atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(got, wanted, atol=3e-9, rtol=3e-9)
    return len(cases)


def initialization_checks(reference):
    rng = np.random.RandomState(1)
    X = rng.standard_normal((25, 4))
    drawn = 0
    seeds = (0, 1, 42, 271, 2**32 - 1)
    for data in (X, X.astype(np.float32)):
        for seed in seeds:
            wanted = 1e-4 * np.random.RandomState(seed).standard_normal((len(data), 3)).astype(np.float32)
            got = native_fit(data, "random", operation="init", seed=seed, components=3)
            np.testing.assert_array_equal(got, wanted)
            drawn += got.size

        for pca_input in (data, np.vstack((data, 2 * data, 3 * data))):
            scores = reference.PCA(n_components=2).fit_transform(pca_input).astype(np.float32)
            expected = scores / np.std(scores[:, 0]) * np.float32(1e-4)
            got = native_fit(pca_input, "pca", operation="init")
            np.testing.assert_allclose(got, expected, atol=3e-10, rtol=3e-6)
            np.testing.assert_allclose(np.std(got[:, 0]), 1e-4, rtol=5e-7)

    # For naturally seeded fits, the exact optimizer can amplify tiny
    # reduction-order differences. Compare neighborhood structure and KL.
    X = rng.normal(scale=0.15, size=(30, 4)) + np.repeat(np.eye(3, 4) * 5, 10, axis=0)
    expected, actual = [], []
    for seed in seeds:
        options = dict(perplexity=5, method="exact", random_state=seed, init="random", max_iter=500)
        model = reference.TSNE(**options)
        y = model.fit_transform(X)
        _, kl, rate, _, z = native_fit(X, "random", max_iter=500, perplexity=5,
                                       exaggeration=12, lr="auto", min_grad=1e-7, seed=seed)
        assert rate == model.learning_rate_ == 50
        expected.append((model.kl_divergence_, reference.trustworthiness(X, y, n_neighbors=5)))
        actual.append((kl, reference.trustworthiness(X, z, n_neighbors=5)))
    expected, actual = np.array(expected), np.array(actual)
    assert np.min(actual[:, 1]) > 0.9, (actual, expected)
    np.testing.assert_allclose(np.mean(actual[:, 0]), np.mean(expected[:, 0]), atol=0.15, rtol=0)
    np.testing.assert_allclose(np.mean(actual[:, 1]), np.mean(expected[:, 1]), atol=0.04, rtol=0)
    return drawn, len(seeds)


def main():
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ.get(variable) == "1", f"set {variable}=1 for reproducible reductions"
    assemble(ROOT)
    FOLDER.mkdir(exist_ok=True)
    compile_file(ROOT / "tests/tsne_driver.mojo", BINARY)
    reference = reference_module()
    print(f"PASS: {objective_checks(reference)} exact P, KL and gradient comparisons")
    print(f"PASS: {distance_checks()} dense distance comparisons")
    print(f"PASS: {fit_checks(reference)} explicit-initialization fits, including stopping edges")
    patience_check(reference)
    print("PASS: no-progress stopping against pinned sklearn gradient descent")
    draws, seeds = initialization_checks(reference)
    print(f"PASS: {draws} identical Float32 RNG initialization draws, 4 PCA initializations and {seeds} seeded distribution comparisons")
    print(f"sklearn oracle revision {REVISION}")


if __name__ == "__main__":
    main()
