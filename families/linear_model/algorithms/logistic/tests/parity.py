"""Dense logistic differential checks against the pinned sklearn estimator and loss."""
from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path
import runpy
import subprocess
import sys
from sysconfig import get_config_var
from unittest.mock import patch
import warnings

from Cython import Tempita
import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn import __version__ as support_version
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import _cd_fast
from sklearn.svm import _liblinear
from sklearn.utils import _array_api, fixes

FAMILY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FAMILY / "tools"))
from parity_io import REFERENCE, case_line, close, load_pinned, parse_results, pinned_code, verify_reference
from pinned_coordinate import load as load_pinned_coordinate

LOSS = "sklearn/linear_model/_linear_loss.py"
LOGISTIC = "sklearn/linear_model/_logistic.py"
NEWTON = "sklearn/linear_model/_glm/_newton_solver.py"
OPTIMIZE = "sklearn/utils/optimize.py"
SVM_BASE = "sklearn/svm/_base.py"
LIBLINEAR = "sklearn/svm/src/liblinear"
LINEAR_BASE = "sklearn/linear_model/_base.py"
SAG = "sklearn/linear_model/_sag.py"
SAG_FAST = "sklearn/linear_model/_sag_fast.pyx.tp"
DATASET = "sklearn/utils/_seq_dataset"
OUTPUT = FAMILY / "builds/logistic"


def pinned_sag():
    """Compile the pinned stochastic engine and its dataset sampler."""
    folder = OUTPUT / "sag_reference"
    templates = {
        SAG_FAST: "pinned_sag_fast.pyx",
        f"{DATASET}.pyx.tp": f"{DATASET}.pyx",
        f"{DATASET}.pxd.tp": f"{DATASET}.pxd",
    }
    for source, destination in templates.items():
        path = folder / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = Tempita.Template((REFERENCE / source).read_text()).substitute()
        if not path.exists() or path.read_text() != rendered:
            path.write_text(rendered)

    setup = folder / "setup.py"
    setup.write_text(
        "from setuptools import setup, Extension\n"
        "from Cython.Build import cythonize\n"
        "from numpy import get_include\n"
        "setup(ext_modules=cythonize([\n"
        f"    Extension('sklearn.utils._seq_dataset', ['{DATASET}.pyx'], "
        "include_dirs=[get_include()], extra_compile_args=['-ffp-contract=off']),\n"
        "    Extension('pinned_sag_fast', ['pinned_sag_fast.pyx'], "
        "include_dirs=[get_include()], extra_compile_args=['-ffp-contract=off']),\n"
        f"], include_path=['.', {str(REFERENCE)!r}], build_dir='cython'))\n"
    )
    with (folder / "compile.log").open("w") as log:
        subprocess.run(
            [sys.executable, setup.name, "build_ext", "--build-lib", "extensions"],
            cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True,
        )

    def load_extension(name):
        library = folder / "extensions" / (name.replace(".", "/") + get_config_var("EXT_SUFFIX"))
        spec = spec_from_file_location(name, library)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    base = load_pinned(LINEAR_BASE)
    dataset = load_extension("sklearn.utils._seq_dataset")
    with patch.dict(sys.modules, {"sklearn.utils._seq_dataset": dataset}):
        native = load_extension("pinned_sag_fast")
    base.ArrayDataset32 = dataset.ArrayDataset32
    base.ArrayDataset64 = dataset.ArrayDataset64
    sag = load_pinned(SAG)
    sag.make_dataset = base.make_dataset
    sag.sag32 = native.sag32
    sag.sag64 = native.sag64
    pinned_code(sag.sag_solver, SAG)
    pinned_code(sag.make_dataset, LINEAR_BASE)
    return sag.sag_solver


def pinned_liblinear():
    """Adapt the existing engine oracle to sklearn's dense train_wrap layout."""
    engine_tests = FAMILY.parent / "svm/shared/liblinear/tests"
    source = REFERENCE / LIBLINEAR
    binary = OUTPUT / "liblinear_oracle"
    subprocess.run(
        ["g++", "-O3", "-ffp-contract=off", "-std=c++17", f"-I{source}",
         engine_tests / "oracle.cpp", source / "linear.cpp", source / "tron.cpp", "-o", binary],
        check=True,
    )
    adapter = runpy.run_path(str(engine_tests / "parity.py"))

    def train_wrap(X, y, sparse, solver, tol, bias, C, cw, max_iter, seed, epsilon, sw):
        assert not sparse and solver in (0, 6, 7)
        if tol <= 0:
            raise ValueError("eps <= 0")
        fixture = OUTPUT / "liblinear_reference.txt"
        adapter["write_case"](
            fixture, X.astype(np.float64), y, sw, cw, solver, c=C, epsilon=epsilon,
            tol=tol, fit=bias > 0, scale=bias, limit=max_iter, seed=seed,
        )
        coef, intercept, iterations, _ = adapter["invoke"](binary, fixture)
        if bias > 0:
            coef = np.column_stack((coef, intercept / bias))
        return coef, iterations.astype(np.int32)

    return train_wrap


def unsupported_oracle_branch(*args, **kwargs):
    raise AssertionError("the pinned oracle reached an import used only by LogisticRegressionCV")


def oracle(*, coordinate=False):
    hashes = verify_reference([
        LOSS, LOGISTIC, NEWTON, OPTIMIZE, SVM_BASE, "sklearn/svm/_liblinear.pyx",
        f"{LIBLINEAR}/linear.cpp", f"{LIBLINEAR}/linear.h", f"{LIBLINEAR}/tron.cpp",
        f"{LIBLINEAR}/tron.h", "sklearn/svm/src/newrand/newrand.h",
        LINEAR_BASE, SAG, SAG_FAST, f"{DATASET}.pxd.tp", f"{DATASET}.pyx.tp",
        "sklearn/_loss/_loss.pxd", "sklearn/utils/_random.pxd", "sklearn/utils/_typedefs.pxd",
    ])
    print(f"scikit-learn source SHA256 {hashes}, support wheel {support_version}")
    # These symbols let the module import with the locked 1.9.1 support wheel.
    # The fitted solvers do not reach these compatibility shims.
    if not hasattr(_cd_fast, "enet_coordinate_descent_sparse"):
        _cd_fast.enet_coordinate_descent_sparse = _cd_fast.sparse_enet_coordinate_descent
    for module, name in [
        (_array_api, "_swapaxes"), (_array_api, "_unravel_index"), (fixes, "_is_gil_enabled")
    ]:
        if not hasattr(module, name):
            setattr(module, name, unsupported_oracle_branch)
    loss = load_pinned(LOSS, register=True)
    newton = load_pinned(NEWTON, register=True)
    optimize = load_pinned(OPTIMIZE, register=True)
    estimator = load_pinned(LOGISTIC)
    if coordinate:
        coordinate_module, coordinate_hashes = load_pinned_coordinate(FAMILY / "algorithms/logistic")
        print(f"scikit-learn coordinate source SHA256 {coordinate_hashes}")
        newton.enet_coordinate_descent = coordinate_module.cd_fast.enet_coordinate_descent
        newton._pre_fit = coordinate_module._pre_fit
        pinned_code(newton.NewtonCDSolver.inner_solve, NEWTON)
        assert estimator.NewtonCDSolver is newton.NewtonCDSolver
    estimator.sag_solver = pinned_sag()
    estimator._fit_liblinear = load_pinned(SVM_BASE)._fit_liblinear
    pinned_code(estimator._fit_liblinear, SVM_BASE)
    pinned_code(loss.LinearModelLoss.loss_gradient, LOSS)
    pinned_code(loss.LinearModelLoss.gradient_hessian, LOSS)
    pinned_code(newton.NewtonCholeskySolver.inner_solve, NEWTON)
    pinned_code(loss.LinearModelLoss.gradient_hessian_product, LOSS)
    pinned_code(optimize._newton_cg, OPTIMIZE)
    assert estimator._newton_cg is optimize._newton_cg
    assert estimator.NewtonCholeskySolver is newton.NewtonCholeskySolver
    pinned_code(estimator._logistic_regression_path, LOGISTIC)
    pinned_code(estimator.LogisticRegression.fit, LOGISTIC)
    assert estimator.LinearModelLoss is loss.LinearModelLoss
    return estimator.LogisticRegression


def fixtures():
    rng = np.random.default_rng(842)
    for bits in (32, 64):
        dtype = np.float32 if bits == 32 else np.float64
        for classes in (2, 3, 5):
            for scenario in (
                "default", "no-intercept", "weighted", "balanced", "explicit",
                "unpenalized", "warm", "zero-iter", "one-iter", "wide", "signed",
            ):
                n, p = (14, 24) if scenario == "wide" else (87, 5)
                labels = np.array([-3, 8, 23, 157, 201])
                indices = rng.permutation(np.arange(n) % classes)
                y = labels[indices]
                if scenario == "default":
                    y = y[:, None]
                X = rng.normal(size=(n, p)).astype(dtype)
                X[np.arange(n), indices] += 0.8
                if scenario == "no-intercept":
                    X += 2
                Z = rng.normal(size=(13, p)).astype(dtype)
                arrays = {"X": X, "y": y, "Z": Z}
                if scenario in ("weighted", "balanced", "explicit", "signed"):
                    sw = rng.uniform(0.35, 2, size=n).astype(dtype)
                    sw[0] = -0.05 if scenario == "signed" else 0
                    arrays["sw"] = sw
                params = {
                    "C": 1.0 if scenario == "default" else 0.7,
                    "tol": 1e-4 if scenario == "default" else 1e-8,
                    "max_iter": {"zero-iter": 0, "one-iter": 1}.get(scenario, 100),
                    "intercept": int(scenario != "no-intercept"),
                    "weight": scenario if scenario in ("balanced", "explicit") else "none",
                    "unpenalized": int(scenario == "unpenalized"),
                }
                if scenario == "warm":
                    params["refit_C"] = 1.8
                yield f"{bits}-{classes}-{scenario}", bits, params, arrays
                yield f"{bits}-{classes}-newton-{scenario}", bits, params | {"solver": "newton-cholesky"}, arrays
                yield f"{bits}-{classes}-cg-{scenario}", bits, params | {"solver": "newton-cg"}, arrays

        params = {
            "C": 1.0, "tol": 1e-4, "max_iter": 100, "intercept": 1,
            "weight": "none", "unpenalized": 0, "invalid": 1,
        }
        X = np.eye(4, dtype=dtype)
        y = np.array([-3, 8, -3, 8])
        for scenario, options in {
            "one-class": {"y": np.full(4, -3)},
            "bad-weight-shape": {"sw": np.ones(3, dtype=dtype)},
            "multilabel": {"y": np.eye(4, dtype=np.int64)},
        }.items():
            arrays = {"X": X, "y": y, "Z": X} | options
            yield f"{bits}-invalid-{scenario}", bits, params, arrays
            yield f"{bits}-newton-invalid-{scenario}", bits, params | {"solver": "newton-cholesky"}, arrays
            yield f"{bits}-cg-invalid-{scenario}", bits, params | {"solver": "newton-cg"}, arrays

        # Identical columns make the unpenalized Hessian exactly singular.
        X = np.array([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [3, 3]], dtype=dtype)
        y = np.array([-3, -3, 8, -3, 8, 8])
        params = {
            "C": 1.0, "tol": 1e-8, "max_iter": 100, "intercept": 1, "weight": "none",
            "unpenalized": 1, "solver": "newton-cholesky", "singular": 1,
        }
        yield f"{bits}-newton-singular", bits, params, {"X": X, "y": y, "Z": X}
        cg = {key: value for key, value in params.items() if key != "singular"} | {"solver": "newton-cg"}
        yield f"{bits}-cg-singular", bits, cg, {"X": X, "y": y, "Z": X}
        yield f"{bits}-cg-single-feature", bits, cg, {"X": X[:, :1], "y": y, "Z": X[:, :1]}
        X_zero = np.zeros((4, 1), dtype=dtype)
        yield f"{bits}-cg-stationary", bits, cg | {"exact_iterations": 1}, {
            "X": X_zero, "y": np.array([-3, -3, 8, 8]), "Z": X_zero,
        }
        yield f"{bits}-newton-singular-budget", bits, params | {"max_iter": 2, "exact_iterations": 1}, {"X": X, "y": y, "Z": X}
        X, y = make_classification(n_samples=10, n_features=20, random_state=42)
        X = X.astype(dtype)
        yield f"{bits}-newton-near-singular", bits, params | {
            "C": 1e30, "unpenalized": 0, "tol": 1e-4, "exact_iterations": 1,
        }, {"X": X, "y": y, "Z": X}

        # Signed sample weights make the initial directional curvature negative.
        X = np.array([[-8], [-2], [-1], [1], [2], [8]], dtype=dtype)
        y = np.array([-3, -3, -3, 8, 8, 8])
        sw = np.array([-1, 1, 1, 1, 1, -1], dtype=dtype)
        yield f"{bits}-cg-negative-curvature", bits, cg | {
            "C": 0.7, "unpenalized": 0, "intercept": 0, "cg_negative_curvature": 1,
        }, {"X": X, "y": y, "sw": sw, "Z": X[2:4]}

        line_search_rng = np.random.default_rng(41)
        X = line_search_rng.integers(-6, 7, (9, 3)) * 2 ** line_search_rng.integers(0, 6, (9, 3))
        X = X.astype(dtype)
        y = line_search_rng.permutation(np.arange(9) % 3)
        sw = (2 ** line_search_rng.integers(0, 6, 9)).astype(dtype)
        yield f"{bits}-newton-backtracking", bits, {
            "C": 0.01, "tol": 1e-4, "max_iter": 7, "intercept": 1, "weight": "none",
            "unpenalized": 0, "solver": "newton-cholesky", "backtracking": 1,
        }, {"X": X, "y": y, "sw": sw, "Z": X}


def liblinear_fixtures():
    rng = np.random.default_rng(481)
    for bits, dtype in [(32, np.float32), (64, np.float64)]:
        X = rng.normal(size=(53, 5)).astype(dtype)
        y = np.where(X[:, 0] - X[:, 1] + rng.normal(size=len(X)) > 0.4, 8, -3)
        Z = rng.normal(size=(13, X.shape[1])).astype(dtype)
        arrays = {"X": X, "y": y, "Z": Z}
        sw = rng.uniform(0.25, 2, size=len(X)).astype(dtype)
        sw[:2] = [0, -0.5]
        base = {
            "solver": "liblinear", "C": 0.7, "tol": 1e-8, "max_iter": 200,
            "intercept": 1, "weight": "none", "unpenalized": 0, "seed": 42,
        }
        for mode, options in [
            ("l2-primal", {}), ("l2-dual", {"dual": 1}), ("l1", {"ratio": 1.0}),
        ]:
            for scenario, params, data in [
                ("default", {}, arrays),
                ("wide-no-intercept", {"intercept": 0},
                 {"X": X[:8].T, "Z": X[8:16].T, "y": np.array([-3, 8, -3, 8, -3])}),
                ("signed-weights", {}, arrays | {"sw": sw}),
                ("balanced", {"weight": "balanced"}, arrays | {"sw": sw}),
                ("class-weight", {"weight": "explicit", "intercept_scaling": 2.3}, arrays | {"sw": sw}),
                ("drop-class", {}, arrays | {"sw": (y == -3).astype(dtype)}),
                ("warm-refit", {"refit_C": 0.4, "max_iter": 3, "exact_iterations": 1}, arrays),
                ("zero-iter", {"max_iter": 0}, arrays),
                ("one-iter", {"max_iter": 1, "seed": 19}, arrays),
            ]:
                yield f"{bits}-liblinear-{mode}-{scenario}", bits, base | options | params, data
        yield f"{bits}-liblinear-explicit-l1", bits, base | {"penalty": "l1"}, arrays
        limit_X = X.copy()
        limit_X[0, 0] = dtype(1e30)
        yield f"{bits}-liblinear-feature-limit", bits, base | {"max_iter": 0}, arrays | {"X": limit_X}
        limit_X = limit_X.copy()
        limit_X[0, 0] = np.nextafter(dtype(1e30), dtype(np.inf))
        for scenario, params, data in [
            ("multiclass", {}, arrays | {"y": np.where(np.arange(len(y)) == 0, 14, y)}),
            ("one-class", {}, arrays | {"y": np.full_like(y, -3)}),
            ("elasticnet", {"ratio": 0.5}, arrays),
            ("dual-l1", {"dual": 1, "ratio": 1.0}, arrays),
            ("unpenalized", {"unpenalized": 1}, arrays),
            ("no-penalty", {"penalty": "None"}, arrays),
            ("zero-tol", {"tol": 0}, arrays),
            ("large-feature", {}, arrays | {"X": limit_X}),
        ]:
            yield f"{bits}-liblinear-invalid-{scenario}", bits, base | params | {"invalid": 1}, data


def newton_cd_fixtures():
    rng = np.random.default_rng(1042)
    for bits, dtype in [(32, np.float32), (64, np.float64)]:
        X = rng.normal(size=(83, 7)).astype(dtype)
        X[:, -1] = 0
        y = np.where(X[:, 0] - 0.5 * X[:, 1] + rng.normal(size=len(X)) > 0.45, 8, -3)
        Z = rng.normal(size=(13, X.shape[1])).astype(dtype)
        sw = rng.uniform(0.25, 2, size=len(X)).astype(dtype)
        sw[:2] = [0, -0.05]
        arrays = {"X": X, "y": y, "Z": Z}
        base = {
            "solver": "newton-cd", "C": 0.42, "tol": 1e-6 if bits == 32 else 1e-8, "max_iter": 100,
            "intercept": 1, "weight": "none", "unpenalized": 0,
        }
        for ratio in (0, 0.65, 1):
            for scenario, params, data in [
                ("fit", {}, arrays),
                ("no-intercept", {"intercept": 0}, arrays),
                ("weighted", {"weight": "balanced"}, arrays | {"sw": np.maximum(sw, 0)}),
                ("warm", {"refit_C": 0.2}, arrays),
                ("zero-iter", {"max_iter": 0}, arrays),
                ("one-iter", {"max_iter": 1}, arrays),
            ]:
                yield f"{bits}-newton-cd-{ratio}-{scenario}", bits, base | {"ratio": ratio} | params, data
        yield f"{bits}-newton-cd-unpenalized", bits, base | {"unpenalized": 1}, arrays
        yield f"{bits}-newton-cd-signed", bits, base | {"weight": "balanced"}, arrays | {"sw": sw}
        if bits == 64:
            yield "64-newton-cd-l1-line-search-failure", bits, base | {
                "ratio": 1, "weight": "balanced", "line_search_failure": 1,
            }, arrays | {"sw": sw}
        yield f"{bits}-newton-cd-multiclass", bits, base | {"invalid": 1}, (
            arrays | {"y": np.where(np.arange(len(y)) == 0, 23, y)}
        )


def sag_fixtures():
    rng = np.random.default_rng(731)
    for bits, dtype in [(32, np.float32), (64, np.float64)]:
        for classes in (2, 3):
            X, encoded = make_classification(
                n_samples=73, n_features=7, n_informative=4, n_redundant=0,
                n_classes=classes, class_sep=0.85, random_state=313,
            )
            X = X.astype(dtype)
            Z = rng.normal(size=(11, X.shape[1])).astype(dtype)
            y = np.array([-3, 8, 23])[encoded]
            sw = rng.uniform(0.25, 2, size=len(X)).astype(dtype)
            sw[:2] = [0, -0.2]
            arrays = {"X": X, "y": y, "Z": Z}
            base = {
                "C": 0.14, "tol": 2e-6, "max_iter": 2000,
                "intercept": 1, "weight": "none", "unpenalized": 0, "seed": 42,
            }
            for solver, mode, options in [
                ("sag", "l2", {}), ("saga", "l2", {}),
                ("saga", "l1", {"ratio": 1.0}), ("saga", "elasticnet", {"ratio": 0.55}),
            ]:
                for scenario, params, data in [
                    ("fit", {}, arrays),
                    ("signed", {"weight": "explicit", "intercept": 0}, arrays | {"sw": sw}),
                    ("balanced", {"weight": "balanced"}, arrays | {"sw": sw}),
                    ("warm-refit", {"max_iter": 3, "refit_C": 0.4, "exact_iterations": 1}, arrays),
                    ("zero-iter", {"max_iter": 0}, arrays),
                    ("one-iter", {"max_iter": 1, "seed": 19}, arrays),
                ]:
                    yield f"{bits}-{classes}-{solver}-{mode}-{scenario}", bits, (
                        base | {"solver": solver} | options | params
                    ), data
            for solver in ("sag", "saga"):
                for scenario, params in [
                    ("no-penalty", {"penalty": "None"}),
                    ("infinite-C", {"unpenalized": 1, "ratio": 0.55}),
                ]:
                    yield f"{bits}-{classes}-{solver}-{scenario}", bits, (
                        base | {"solver": solver, "max_iter": 15, "exact_iterations": 1} | params
                    ), arrays
                zero_features = arrays | {"X": np.zeros_like(X)}
                yield f"{bits}-{classes}-{solver}-intercept-only", bits, (
                    base | {"solver": solver, "exact_iterations": 1}
                ), zero_features
                yield f"{bits}-{classes}-{solver}-zero-step-budget", bits, (
                    base | {"solver": solver, "max_iter": 0, "intercept": 0, "unpenalized": 1}
                ), zero_features
            yield f"{bits}-{classes}-saga-explicit-elasticnet", bits, (
                base | {"solver": "saga", "penalty": "elasticnet", "ratio": 0.55}
            ), arrays
        for scenario, params in [
            ("sag-l1", {"solver": "sag", "ratio": 1.0}),
            ("sag-elasticnet", {"solver": "sag", "ratio": 0.55}),
            ("saga-dual", {"solver": "saga", "dual": 1}),
        ]:
            yield f"{bits}-invalid-{scenario}", bits, base | params | {"invalid": 1}, arrays
        for solver in ("sag", "saga"):
            yield f"{bits}-{solver}-invalid-step", bits, (
                base | {"solver": solver, "intercept": 0, "unpenalized": 1, "invalid": 1}
            ), arrays | {"X": np.zeros_like(X)}


def reference(model_type, params, arrays):
    verbose = 2 if params.get("backtracking") or params.get("cg_negative_curvature") else 0
    model = model_type(
        penalty=None if params.get("penalty") == "None" else params.get("penalty", "deprecated"),
        dual=bool(params.get("dual", 0)), l1_ratio=params.get("ratio", 0),
        intercept_scaling=params.get("intercept_scaling", 1), random_state=params.get("seed"),
        solver=params.get("solver", "lbfgs"),
        C=np.inf if params["unpenalized"] else params["C"],
        tol=params["tol"], max_iter=params["max_iter"], fit_intercept=bool(params["intercept"]),
        class_weight={"balanced": "balanced", "explicit": {-3: 2.0}}.get(params["weight"]),
        warm_start="refit_C" in params, verbose=verbose,
    )
    X, y, Z = (arrays[key] for key in ("X", "y", "Z"))
    trace = StringIO()
    with warnings.catch_warnings(record=True) as diagnostics, redirect_stdout(trace):
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("always", LinAlgWarning)
        model.fit(X, y, sample_weight=arrays.get("sw"))
        if "refit_C" in params:
            model.set_params(C=params["refit_C"]).fit(X, y)
    if params.get("backtracking"):
        assert "line search iteration=2" in trace.getvalue()
    if params.get("cg_negative_curvature"):
        assert "detected a negative curvature" in trace.getvalue()
    if params.get("singular"):
        assert any(issubclass(item.category, LinAlgWarning) for item in diagnostics)
    intercept = np.atleast_1d(model.intercept_)
    if params.get("solver") == "liblinear":
        assert model.coef_.dtype == intercept.dtype == np.float64
    else:
        assert model.coef_.dtype == intercept.dtype == X.dtype
    assert model.n_iter_.dtype == np.int32 and model.classes_.dtype == y.dtype
    return {
        "classes": model.classes_, "coef": model.coef_, "intercept": intercept,
        "iterations": model.n_iter_, "decision": model.decision_function(Z),
        "predictions": model.predict(Z), "probability": model.predict_proba(Z),
        "log_probability": model.predict_log_proba(Z),
        "score": model.score(X, y, sample_weight=arrays.get("sw")), "features": model.n_features_in_,
    }


def main():
    model_type = oracle(coordinate=True)
    with patch.object(_liblinear, "train_wrap", pinned_liblinear()):
        compare(model_type)


def compare(model_type):
    cases = [*fixtures(), *liblinear_fixtures(), *sag_fixtures(), *newton_cd_fixtures()]
    output = OUTPUT
    fixture = output / "fixtures.txt"
    fixture.write_text("\n".join(case_line(*case) for case in cases) + "\n")
    completed = subprocess.run([output / "parity_driver", fixture], text=True, check=True, capture_output=True)
    (output / "native.txt").write_text(completed.stdout)
    observed = parse_results(completed.stdout)
    assert observed.keys() == {case[0] for case in cases}
    maxima = {}
    iteration_difference = 0
    fit_counts = {}
    widened_only = 0
    invalid = 0
    line_search_failures = 0
    for ident, bits, params, arrays in cases:
        if params.get("line_search_failure"):
            try:
                reference(model_type, params, arrays)
            except ConvergenceWarning as error:
                assert "fallback for non-zero L1" in str(error)
            else:
                raise AssertionError(f"pinned sklearn accepted {ident}")
            assert "newton-cd line search failed with a nonzero L1 penalty" in observed[ident]["error"]
            line_search_failures += 1
            continue
        if "invalid" in params:
            try:
                reference(model_type, params, arrays)
            except ValueError:
                pass
            else:
                raise AssertionError(f"pinned sklearn accepted {ident}")
            assert observed[ident]["error"].startswith("ValueError:"), observed[ident]
            invalid += 1
            continue

        solver = params.get("solver", "lbfgs")
        fit_counts[solver] = fit_counts.get(solver, 0) + 1
        assert "error" not in observed[ident], (ident, observed[ident])
        references = []
        try:
            expected = reference(model_type, params, arrays)
        except ValueError as error:
            # The pinned Newton solver pads the unpenalized multinomial
            # step with Float64 zeros, promoting Float32 raw predictions.
            if not (
                bits == 32 and solver == "newton-cholesky" and params["unpenalized"]
                and np.unique(arrays["y"]).size > 2
                and "Buffer dtype mismatch" in str(error)
            ):
                raise
            widened_only += 1
        else:
            references.append(("native-dtype", expected))
        if bits == 32 and solver != "liblinear":
            widened = {key: value if key == "y" else value.astype(np.float64) for key, value in arrays.items()}
            widened_result = reference(model_type, params, widened)
            if widened_result["classes"].size == 2:
                # Binary inference subtracts a rounded Float32 sigmoid
                # from one before taking the logarithm of either class.
                positive = widened_result["probability"][:, 1].astype(np.float32)
                probability = np.column_stack((1 - positive, positive))
                widened_result["log_probability"] = np.log(probability)
            references.append(("widened", widened_result))
        exact_iterations = (
            params["max_iter"] <= 1 or params.get("backtracking") or params.get("exact_iterations")
            or (bits == 64 and solver in {"sag", "saga"})
        )
        for kind, result in references:
            # Float32 CG can drift along the unpenalized shared class offset.
            # The widened reference still compares the raw coefficients and logits.
            compare_class_contrasts = (
                bits == 32 and kind == "native-dtype" and solver == "newton-cg"
                and params["unpenalized"] and result["classes"].size > 2
            )
            for key, wanted in result.items():
                actual = observed[ident][key]
                if key == "iterations" and not exact_iterations:
                    assert actual.shape == wanted.shape == (1,)
                    minimum = 0 if solver == "liblinear" else 1
                    assert minimum <= actual[0] <= params["max_iter"]
                    if kind == "native-dtype":
                        iteration_difference = max(iteration_difference, int(abs(actual[0] - wanted[0])))
                    continue
                if key in {"classes", "predictions", "features", "iterations"}:
                    atol = 0
                elif key == "score":
                    atol = 1e-7
                elif solver == "liblinear":
                    atol = 5e-6 if key in {"decision", "log_probability"} else 2e-6
                elif solver in {"sag", "saga"}:
                    if bits == 64:
                        atol = 3e-9 if key == "log_probability" else 5e-10
                    elif kind == "native-dtype":
                        atol = 2e-5 if key in {"decision", "log_probability"} else 5e-6
                    else:
                        atol = 3e-6
                    if key == "coef" and "-l1-fit" in ident:
                        np.testing.assert_array_equal(actual == 0, wanted == 0, err_msg=ident)
                        assert (actual == 0).any() and (actual != 0).any(), ident
                elif bits == 32 and kind == "native-dtype":
                    atol = 3e-3 if key in {"decision", "log_probability"} else 9e-4
                else:
                    atol = 5e-6 if key in {"decision", "log_probability"} else 2e-6
                if compare_class_contrasts and key in {"coef", "decision"}:
                    axis = 0 if key == "coef" else 1
                    actual = actual - actual.mean(axis=axis, keepdims=True)
                    wanted = wanted.astype(np.float64)
                    wanted -= wanted.mean(axis=axis, keepdims=True)
                if key == "coef" and solver == "newton-cd" and params.get("ratio") and ident.endswith("-fit"):
                    np.testing.assert_array_equal(actual == 0, wanted == 0, err_msg=ident)
                    assert (actual == 0).any() and (actual != 0).any(), ident
                error = close(actual, wanted, 0, atol, f"{ident}/{kind}/{key}")
                group = f"{bits}/{solver}/{kind}/{key}" if solver in {"liblinear", "newton-cd", "sag", "saga"} else f"{bits}/{kind}/{key}"
                maxima[group] = max(maxima.get(group, 0), error)
    print(f"PASS: {sum(fit_counts.values())} pinned dense logistic fits {fit_counts}, {invalid} input rejections"
          f" and {line_search_failures} line-search failure."
          f" {widened_only} fits used only the widened oracle."
          f" Max unconstrained iteration-count difference {iteration_difference}.")
    for key, error in sorted(maxima.items()):
        if error > 0:
            print(f"  {key}: {error:.8g}")


if __name__ == "__main__":
    main()
