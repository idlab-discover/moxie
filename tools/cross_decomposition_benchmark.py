"""Measure repeated Float64 fits on identical native and sklearn workloads."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import warnings

# pyperf starts each worker with a fresh environment. Set limits before imports.
THREAD_LIMITS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for variable in THREAD_LIMITS:
    os.environ[variable] = "1"

import numpy as np
import pyperf
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/cross_decomposition/pls"))
from parity import DEFAULT_TOL, REVISION, cases, datasets, load_reference, verify_reference

BINARY = ROOT / "builds/tests/cross_decomposition/benchmark"
FIXTURES = ROOT / "builds/cross_decomposition/benchmark/fixtures"
LARGE = "large-10000x100"
LARGE_SEED = 42
ESTIMATORS = ("PLSRegression", "PLSCanonical-nipals", "PLSCanonical-svd", "CCA", "PLSSVD")


def workloads():
    for name, X, y, _ in datasets():
        estimators = ESTIMATORS
        if name == "constant-y":
            # Pinned PLSCanonical SVD reaches a zero score denominator on
            # this block. The parity suite checks the error path.
            estimators = tuple(est for est in ESTIMATORS if est != "PLSCanonical-svd")
        yield name, X, y, estimators, None
    case = next(case for case in cases() if case["data"] == "cca-converges")
    yield case["data"], case["X"], case["y"], ("CCA",), case["params"]["n_components"]
    rng = np.random.default_rng(LARGE_SEED)
    X = rng.standard_normal((10_000, 100))
    # Distinct signal-to-noise ratios separate the leading canonical modes.
    y = X @ rng.standard_normal((100, 3)) + rng.standard_normal((10_000, 3)) * [1, 5, 15]
    yield LARGE, X, y, ESTIMATORS, 2


def fixture_paths(name):
    return FIXTURES / f"{name}-X.csv", FIXTURES / f"{name}-y.csv"


def native_output(loops, name, estimator, components, rows, columns, targets):
    X_path, y_path = fixture_paths(name)
    return subprocess.check_output(
        [BINARY, estimator, str(components), str(loops), str(rows),
         X_path, str(columns), y_path, str(targets)],
        text=True,
    )


def native_elapsed(loops, *args):
    return float(native_output(loops, *args))


def reference_estimator(module, name, components):
    class_name, _, algorithm = name.partition("-")
    options = {"algorithm": algorithm} if algorithm else {}
    return getattr(module, class_name)(n_components=components, **options)


def reference_elapsed(loops, module, estimator, components, X, y):
    model = reference_estimator(module, estimator, components)
    start = perf_counter()
    for _ in range(loops):
        model.fit(X, y)
    return perf_counter() - start


def verify_large_case(model, X, y, native_args):
    model.fit(X, y)
    expected = np.concatenate((model.x_weights_.ravel(), model.y_weights_.ravel()))
    actual = np.array(native_output(0, *native_args).splitlines(), dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=DEFAULT_TOL, atol=DEFAULT_TOL)
    return float(np.max(np.abs(actual - expected)))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_worker_args(command, args):
    command.extend(("--backend", args.backend))
    if args.dataset:
        command.extend(("--dataset", args.dataset))
    if args.parity_datasets:
        command.append("--parity-datasets")


def main():
    runner = pyperf.Runner(add_cmdline_args=add_worker_args)
    runner.argparser.add_argument("--backend", required=True, choices=("mojo", "sklearn"))
    selection = runner.argparser.add_mutually_exclusive_group()
    selection.add_argument("--dataset", help="Measure just one named dataset")
    selection.add_argument("--parity-datasets", action="store_true",
                           help="Skip the large synthetic workload")
    args = runner.parse_args()
    verify_reference()
    reference = load_reference()
    runner.metadata.update(
        reference_revision=REVISION,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        sklearn_infrastructure_version=sklearn.__version__,
        thread_limits=", ".join(f"{variable}=1" for variable in THREAD_LIMITS),
        timing_region="Fit-only batch beginning with a fresh estimator",
    )
    if args.backend == "mojo":
        runner.metadata.update(
            mojo_version=subprocess.check_output(["mojo", "--version"], text=True).strip(),
            binary_sha256=sha256(BINARY),
            assembly_sha256=sha256(ROOT / "builds/deps/ASSEMBLY.json"),
        )
        if not args.worker:
            FIXTURES.mkdir(parents=True, exist_ok=True)
    else:
        runner.metadata["threadpools"] = repr(threadpool_info())

    selected = False
    # Suppress expected warnings from constant/rank-deficient inputs so terminal
    # output stays outside the fit timing. Native diagnostics remain in memory.
    warnings.filterwarnings("ignore", message="y residual is constant at iteration")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for name, X, y, estimators, fixed_components in workloads():
        if args.dataset and args.dataset != name:
            continue
        if args.parity_datasets and name == LARGE:
            continue
        selected = True
        X, y = (np.ascontiguousarray(array, dtype=np.float64) for array in (X, y))
        data_hash = hashlib.sha256(X)
        data_hash.update(y)
        rows, columns = X.shape
        targets = y.shape[1] if y.ndim == 2 else 0
        if args.backend == "mojo" and not args.worker:
            for path, array in zip(fixture_paths(name), (X, y), strict=True):
                np.savetxt(path, array, fmt="%.17g", delimiter=",")
        for estimator in estimators:
            bound = min(rows - 1, columns)
            if estimator != "PLSRegression":
                bound = min(bound, max(1, targets))
            components = fixed_components or min(2, bound)
            bench_name = f"{name}/{estimator}"
            metadata = dict(
                n_samples=rows, n_features=columns, n_targets=max(1, targets),
                target_rank=y.ndim, n_components=components, input_dtype="float64",
                input_sha256=data_hash.hexdigest(),
            )
            if name == LARGE:
                metadata["seed"] = LARGE_SEED
            if args.backend == "mojo":
                native_args = (name, estimator, components, rows, columns, targets)
                if name == LARGE and not args.worker:
                    model = reference_estimator(reference, estimator, components)
                    error = verify_large_case(model, X, y, native_args)
                    print(f"{bench_name}: weights match (max absolute error {error:.3g})")
                runner.bench_time_func(bench_name, native_elapsed, *native_args, metadata=metadata)
            else:
                runner.bench_time_func(
                    bench_name, reference_elapsed, reference, estimator, components, X, y,
                    metadata=metadata,
                )
    if not selected:
        runner.argparser.error(f"unknown dataset {args.dataset!r}")


if __name__ == "__main__":
    main()
