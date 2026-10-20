"""Small repeatable fit-time baseline: native estimators versus sklearn.

Usage: pixi run python tools/benchmark_family.py [pyperf options] -o benchmarks/family.json
Env:   MOXIE_BENCH_SHAPES   e.g. "200x10,1000x20" (default)

Native timings run builds/benchmark_family (built by `pixi run build`) and
cover the fit region including allocations but not process startup. sklearn
timings run the pinned local estimator source (tests/support/oracle.py) with
OMP/BLAS threads fixed to 1. No speedup is claimed; this is a baseline.
"""
from pathlib import Path
import hashlib
import os
import platform
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pyperf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/support"))
from oracle import load_pinned

SHAPES = [tuple(map(int, s.split("x"))) for s in os.environ.get("MOXIE_BENCH_SHAPES", "200x10,1000x20").split(",")]
REFERENCE = {
    "pca": lambda m: m.PCA(3, svd_solver="full"),
    "incremental_pca": None,
    "truncated_svd": lambda m: m.TruncatedSVD(3, random_state=0),
    "factor_analysis": lambda m: m.FactorAnalysis(3),
    "fastica": lambda m: m.FastICA(3, random_state=0),
}
# The pinned _pca.py needs newer sklearn.utils than the support wheel, so PCA
# timings use the wheel's PCA (recorded in metadata); the others run pinned code.
MODULES = {"truncated_svd": "_truncated_svd", "factor_analysis": "_factor_analysis", "fastica": "_fastica"}


def native(loops, name, dtype, n, p):
    out = subprocess.check_output([ROOT / "builds/benchmark_family", name, dtype, str(n), str(p), str(loops)], text=True)
    return float(out.split()[0])


def data(n, p, dtype):
    i, j = np.meshgrid(np.arange(n), np.arange(p), indexing="ij")
    return (np.sin(i * (j + 1) * 0.37) + ((i * 13 + j * 7) % 17) / 17).astype(dtype)


runner = pyperf.Runner()
runner.metadata.update({
    "compiler": subprocess.check_output(["mojo", "--version"], text=True).strip(),
    "flags": "-O3 --fp-mode=contract=off", "platform": platform.platform(), "threads": "1 (OMP/BLAS)",
    "seed": 0, "sklearn_pca_source": "support wheel " + __import__("sklearn").__version__, "lock_sha256": hashlib.sha256((ROOT.parents[1] / "DEPENDENCIES.json").read_bytes()).hexdigest(),
})
for name in REFERENCE:
    if name == "pca":
        import sklearn.decomposition as module
    else:
        module = load_pinned(f"sklearn/decomposition/{MODULES[name]}.py") if name in MODULES else None
    for n, p in SHAPES:
        for dtype in ("f64", "f32"):
            runner.bench_time_func(f"native_{name}_{dtype}_{n}x{p}", native, name, dtype, n, p)
            if module is not None:
                x = data(n, p, np.float64 if dtype == "f64" else np.float32)
                runner.bench_func(f"sklearn_{name}_{dtype}_{n}x{p}", lambda: REFERENCE[name](module).fit(x))
