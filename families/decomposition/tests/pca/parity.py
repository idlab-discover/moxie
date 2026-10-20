"""Run the local sklearn PCA method bodies against the native implementation.

Only sklearn's Python integration/validation is replaced with NumPy plumbing;
fit/transform/covariance/precision/scoring and randomized SVD are local source.
"""
from pathlib import Path
import ast
import functools
import math
import subprocess
import warnings
import sys
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parents[2] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
assert subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip() == REVISION


def validate_data(estimator, x, reset=True, **kwargs):
    a = np.asarray(x)
    if reset: estimator.n_features_in_ = a.shape[1]
    else: assert a.shape[1] == estimator.n_features_in_
    return a


namespace = dict(np=np, linalg=linalg, sparse=sparse, issparse=sparse.issparse, svds=svds,
    warnings=warnings, partial=functools.partial, sqrt=math.sqrt, log=math.log, lgamma=math.lgamma,
    get_namespace=lambda *args, **kw: (np, False), array_device=lambda a: None,
    check_random_state=lambda s: s if isinstance(s, np.random.RandomState) else np.random.RandomState(s),
    validate_data=validate_data, check_array=lambda a, **kw: np.asarray(a),
    check_is_fitted=lambda e: None, check_same_namespace=lambda *a, **kw: None,
    _is_numpy_namespace=lambda xp: True,
    _validate_diagonal_args=lambda a, v, xp: (v, min(a.shape)))


def load(relative, functions=(), classes=()):
    path = SOURCE / relative
    assert not subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain", "--", str(path)], text=True)
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            node.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            node.bases = [ast.Name(id="_BasePCA", ctx=ast.Load())] if node.name == "PCA" else []
            node.keywords = []
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            for method in node.body:
                method.decorator_list = [d for d in method.decorator_list if isinstance(d, ast.Name) and d.id == "property"]
            nodes.append(node)
    assert len(nodes) == len(functions) + len(classes)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


load("sklearn/utils/_array_api.py", ("_cov", "_add_to_diagonal"))
load("sklearn/utils/_arpack.py", ("_init_arpack_v0",))
load("sklearn/utils/extmath.py", ("_randomized_svd", "_randomized_range_finder", "svd_flip", "fast_logdet"))
load("sklearn/decomposition/_base.py", classes=("_BasePCA",))
load("sklearn/decomposition/_pca.py", ("_infer_dimension", "_assess_dimension"), ("PCA",))
PCA = namespace["PCA"]
assert PCA._fit.__code__.co_filename == str(SOURCE / "sklearn/decomposition/_pca.py")


def native(a, components, solver, whiten, copy=True, score=True, power="auto", normalizer="auto"):
    kind = "none" if components is None else "mle" if components == "mle" else "int" if isinstance(components, int) else "fraction"
    args = [(ROOT / "builds/numax_pca_static_evaluate" if "--numax" in sys.argv else ROOT / "builds/pca_evaluate"), "f32" if a.dtype == np.float32 else "f64", *a.shape,
            kind, components if kind not in {"none", "mle"} else 0, solver, int(whiten), int(copy),
            "score" if score else "skip", -1 if power == "auto" else power, 10, normalizer, 42, *a.ravel()]
    process = subprocess.run(list(map(str, args)), text=True, capture_output=True)
    assert process.returncode == 0, (args[:14], process.stdout[-1000:], process.stderr[:3000])
    result = {}
    for line in process.stdout.splitlines():
        name, value = line.split()
        if name == "solver": result[name] = value
        else: result.setdefault(name, []).append(float(value))
    return {k: np.array(v) if k != "solver" else v for k, v in result.items()}


count = 0


def compare(a, components, solver="full", whiten=False, copy=True, score=True, power="auto", normalizer="auto", unique=True):
    global count
    reference = PCA(components, svd_solver=solver, whiten=whiten, copy=copy,
                    random_state=42, iterated_power=power, power_iteration_normalizer=normalizer)
    original = a.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected_fit = reference.fit_transform(a)
        expected_transform = reference.transform(original)
        expected_cov = reference.get_covariance()
    got = native(original, components, solver, whiten, copy, score, power, normalizer)
    tol = 6e-4 if a.dtype == np.float32 else 3e-9
    spectral_floor = reference.singular_values_.max(initial=0) * np.sqrt(np.finfo(a.dtype).eps) * 4 if reference._fit_svd_solver == "covariance_eigh" else 0
    assert int(got["k"][0]) == reference.n_components_
    assert got["solver"] == reference._fit_svd_solver
    for key, expected in [("variance", reference.explained_variance_), ("ratio", reference.explained_variance_ratio_),
                          ("singular", reference.singular_values_), ("mean", reference.mean_),
                          ("noise", reference.noise_variance_), ("input", a), ("covariance", expected_cov)]:
        np.testing.assert_allclose(got.get(key, []), np.asarray(expected).ravel(), atol=max(tol, spectral_floor) if key == "singular" else tol, rtol=tol, equal_nan=True,
                                   err_msg=f"{key}: {a.shape}, {components}, {solver}, whiten={whiten}")
    k = reference.n_components_
    native_components = got.get("components", np.array([])).reshape(k, a.shape[1])
    # A retained zero singular direction is an arbitrary vector in the null space.
    nonzero = reference.singular_values_ > (max(spectral_floor, reference.singular_values_.max(initial=0) * np.finfo(a.dtype).eps * max(a.shape) * 10))
    np.testing.assert_allclose(native_components[nonzero].T @ native_components[nonzero],
                               reference.components_[nonzero].T @ reference.components_[nonzero], atol=tol, rtol=tol)
    if unique:
        for key, expected in [("components", reference.components_), ("fit_transform", expected_fit), ("transform", expected_transform)]:
            np.testing.assert_allclose(got.get(key, []), expected.ravel(), atol=tol, rtol=tol, equal_nan=True, err_msg=key)
    np.testing.assert_allclose(got["inverse"], reference.inverse_transform(expected_transform).ravel(), atol=tol, rtol=tol, equal_nan=True)
    if score:
        for key, expected in [("precision", reference.get_precision()), ("score_samples", reference.score_samples(original)), ("score", reference.score(original))]:
            np.testing.assert_allclose(got[key], np.asarray(expected).ravel(), atol=tol * 10, rtol=tol * 10, err_msg=key)
    count += 1


rng = np.random.RandomState(73)
for dtype in (np.float64, np.float32):
    for shape in [(12, 4), (4, 12), (50, 3), (7, 1), (1, 5)]:
        x = rng.normal(size=shape).astype(dtype)
        for solver in ("full", "covariance_eigh", "auto"):
            # A single sample only produces NaN variances and arbitrary bases.
            if shape[0] == 1:
                if solver == "full": compare(x.copy(), None, solver, score=False, unique=False)
                continue
            for k in (0, 1, min(shape), 0.8, None):
                for whiten in (False, True):
                    # Wide full-rank whitened null components have arbitrary projections.
                    unique = not (shape[0] < shape[1] and (k is None or k == min(shape)))
                    if whiten and not unique: continue
                    compare(x.copy(), k, solver, whiten, score=shape[0] > shape[1], unique=unique)
    x = rng.normal(size=(24, 5)).astype(dtype)
    for solver in ("full", "covariance_eigh", "randomized"):
        for copy in (False, True):
            for whiten in (False, True): compare(x.copy(), 2, solver, whiten, copy=copy)
    for normalizer in ("auto", "none", "LU", "QR"):
        for power in (0, 2, 4): compare(x.copy(), 2, "randomized", power=power, normalizer=normalizer)
    # Dense ARPACK: exact leading factors versus scipy svds at tol=0.
    for shape in [(24, 5), (5, 24), (50, 3)]:
        y = rng.normal(size=shape).astype(dtype)
        for k in list(range(1, min(shape))) + [None]:
            for whiten in (False, True):
                compare(y.copy(), k, "arpack", whiten, score=shape[0] > shape[1])
        compare(y.copy(), 1, "arpack", copy=False, score=False)
    compare(x.copy(), "mle")
    compare(x.copy(), "mle", "covariance_eigh")
    # Distinct nonzero directions plus a constant feature: compare bases via projection.
    rank_deficient = np.column_stack([x[:, 0], x[:, 1], np.ones(24)]).astype(dtype)
    compare(rank_deficient.copy(), 2, score=False)
    compare(rank_deficient.copy(), None, score=False, unique=False)
    compare(np.ones((8, 3), dtype=dtype), None, score=False, unique=False)
    compare(np.zeros((8, 3), dtype=dtype), None, score=False, unique=False)

# Exercise auto -> randomized without a large feature-space factorization.
if "--numax" not in sys.argv:
    compare(rng.normal(size=(20, 501)), 2, "auto", score=False)
print(f"PASS: {count} dense PCA cases against local sklearn {REVISION}")
