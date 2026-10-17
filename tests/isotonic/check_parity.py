#!/usr/bin/env python3
"""Compare native Mojo to the pinned local sklearn source and independent SciPy.

Validation only. Run with Moxie's Pixi environment through the root parity gate
or tools/isotonic_checks.py. Fixtures and environment metadata stay under
builds/isotonic/parity/. Python is never called by the Mojo library.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import warnings
import sys
import hashlib

import numpy as np
import scipy
from scipy.optimize import isotonic_regression as scipy_isotonic
import sklearn

ROOT = Path(__file__).resolve().parents[2]
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
SEED = 94217


def load_reference(checkout):
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected reference {REVISION}, found {revision}")
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--", "sklearn/isotonic.py", "sklearn/_isotonic.pyx"], text=True)
    if dirty:
        raise RuntimeError("Modified reference source: " + dirty)
    subprocess.run([sys.executable, str(ROOT / 'tests/isotonic/build_reference.py'),
                    'build_ext', '--build-lib', str(ROOT / 'builds/isotonic/reference'),
                    '--build-temp', str(ROOT / 'builds/isotonic/reference/temp')], cwd=ROOT, check=True)
    sys.path.insert(0, str(ROOT / 'builds/isotonic/reference'))
    import _moxie_isotonic_reference as native
    assert Path(native.__file__).resolve().is_relative_to(ROOT / 'builds/isotonic/reference')
    spec = importlib.util.spec_from_file_location(
        "sklearn._moxie_reference_isotonic", checkout / "sklearn/isotonic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._make_unique = native._make_unique
    module._inplace_contiguous_isotonic_regression = native._inplace_contiguous_isotonic_regression
    assert Path(module.IsotonicRegression.fit.__wrapped__.__code__.co_filename).resolve() == checkout / "sklearn/isotonic.py"
    return module


def encode(value):
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, str):
        return value
    if np.ndim(value) == 0:
        return repr(float(value))
    return ",".join(repr(float(x)) for x in value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sklearn-checkout", type=Path, default=ROOT.parent / "scikit-learn")
    args = parser.parse_args()
    reference = load_reference(args.sklearn_checkout.resolve())
    rng = np.random.default_rng(SEED)
    records = []
    dtype = np.float64

    def add(*values):
        records.append("|".join(map(encode, values)))

    def kernel(y, w=None, low=None, high=None, increasing=True):
        y = np.asarray(y, dtype=dtype)
        w = None if w is None else np.asarray(w, dtype=dtype)
        expected = reference.isotonic_regression(
            y, sample_weight=w, y_min=low, y_max=high, increasing=increasing
        )
        independent = scipy_isotonic(y, weights=w, increasing=increasing).x
        independent = independent.astype(dtype)
        independent = np.clip(independent, -np.inf if low is None else low,
                              np.inf if high is None else high)
        np.testing.assert_allclose(expected, independent, rtol=2e-12, atol=2e-12)
        add("kernel", y, w, low, high, increasing, expected)

    def estimator(x, y, w=None, low=None, high=None, direction=True, mode="clip", query=None, score=False):
        x, y = np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype)
        w = None if w is None else np.asarray(w, dtype=dtype)
        model = reference.IsotonicRegression(
            y_min=low, y_max=high, increasing=direction, out_of_bounds=mode
        ).fit(x, y, sample_weight=w)
        if query is None:
            query = np.linspace(model.X_min_, model.X_max_, 31)
        query = np.asarray(query, dtype=dtype)
        expected = model.predict(query)
        fields = ["score" if score else "estimator", x, y, w, low, high, direction,
                  mode, model.X_thresholds_, model.y_thresholds_, query, expected,
                  model.increasing_, [model.X_min_, model.X_max_]]
        if score:
            target = rng.normal(size=len(query)).astype(dtype)
            sw = rng.uniform(0.01, 2, size=len(query)).astype(dtype)
            fields += [target, sw, [model.score(query, target, sample_weight=sw)]]
        add(*fields)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(240):
            n = int(rng.integers(1, 180))
            y = rng.normal(size=n) * (10.0 ** int(rng.integers(-6, 7)))
            if i % 4 == 0:
                y = np.round(y)
            w = None if i % 3 == 0 else 10.0 ** rng.uniform(-5, 5, size=n)
            low, high = [(None, None), (-0.5, 0.5), (0.0, None), (None, 0.0), (2.0, 1.0)][i % 5]
            kernel(y, w, low, high, bool(i % 2))
        # Long chains exercise stack backtracking and descending / sorted inputs.
        for direction in (True, False):
            kernel(np.arange(10000, dtype=float), increasing=direction)
            kernel(np.arange(10000, dtype=float)[::-1], increasing=direction)
        for i in range(240):
            n = int(rng.integers(2, 150))
            x = rng.integers(-20, 20, size=n).astype(float) if i % 2 else rng.normal(size=n)
            y = rng.normal(size=n)
            w = rng.uniform(0.1, 10, size=n)
            w[rng.random(n) < 0.15] = 0
            w[rng.random(n) < 0.05] = -1
            w[0] = 1
            if i % 4 == 0:
                w = None
            direction = [True, False, "auto"][(i // 3) % 3]
            low, high = [(None, None), (-0.5, 0.5), (0.0, None), (None, 0.0)][i % 4]
            mode = ["nan", "clip", "raise"][i % 3]
            query = None if mode == "raise" else np.linspace(-25, 25, 51)
            estimator(x, y, w, low, high, direction, mode, query, score=mode == "clip")
        for mode in ("nan", "clip", "raise"):
            estimator([1.0], [3.0], mode=mode, query=[-10.0, 1.0, 10.0])
            estimator([1.0, 1.0, 1.0], [0.0, 3.0, 8.0], [1.0, 2.0, 1.0], mode=mode, query=[-10.0, 10.0])
        for x, y in [
            ([0.0, 1e-16, 1.0, 1.0 + 1e-14], [0.0, 2.0, 3.0, 4.0]),
            ([0.0, 4.1e-320, 4.4e-314, 1.0], [0.42, 0.42, 0.44, 0.44]),
            ([0.0, 1.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            ([8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14], [21, 23.5, 23, 24, 21, 25, 21.5, 22, 19, 23.5, 25]),
        ]:
            for direction in (True, False, "auto"):
                estimator(x, y, direction=direction)
        # Direction uses all rows, including those filtered by weights.
        estimator([0, 1, 2, 3], [100, 3, 4, -100], [0, 1, 1, -1], direction="auto")
        for i in range(120):
            n = int(rng.integers(1, 100))
            x = rng.integers(0, max(1, n // 3), size=n).astype(float)
            y = rng.integers(0, max(1, n // 3), size=n).astype(float)
            add("direction", x, y, reference.check_increasing(x, y))

    float64_records = records
    records = []
    dtype = np.float32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(150):
            n = int(rng.integers(2, 150))
            x = rng.integers(-20, 20, size=n).astype(dtype) if i % 2 else rng.normal(size=n).astype(dtype)
            y = rng.normal(size=n).astype(dtype)
            w = rng.uniform(0.01, 10, size=n).astype(dtype)
            low, high = [(None, None), (-0.5, 0.5), (0.0, None), (None, 0.0), (2.0, 1.0)][i % 5]
            kernel(y, w, low, high, bool(i % 2))
            w[1::7] = 0
            estimator(x, y, w, low, high, [True, False, "auto"][i % 3], ["nan", "clip", "raise"][i % 3], score=i % 3 == 1)
            add("direction", x, y, reference.check_increasing(x, y))
        for direction in (True, False, "auto"):
            estimator([0, 1e-7, 1e-6, 2e-6, 1], [0, 2, 3, 1, 4], direction=direction)
        for mode in ("nan", "clip", "raise"):
            estimator([1], [3], mode=mode, query=[-10, 1, 10])
    float32_records = records
    records = float64_records

    output = ROOT / "builds/isotonic/parity"
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "reference.txt"
    fixture.write_text("\n".join(records) + "\n")
    metadata = {"seed": SEED, "cases": len(records), "reference_revision": REVISION,
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "sklearn_support_wheel": sklearn.__version__,
                "platform": platform.platform(), "reference_hashes": {f: hashlib.sha256((args.sklearn_checkout / f).read_bytes()).hexdigest() for f in ["sklearn/isotonic.py", "sklearn/_isotonic.pyx"]}, "rtol": 2e-12, "atol": 2e-12}
    metadata["float32"] = {"cases": len(float32_records), "rtol": 2e-6, "atol": 2e-6}
    metadata["mojo"] = subprocess.check_output(["mojo", "--version"], cwd=ROOT, text=True).strip()
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    binary = output / "runner"
    subprocess.run(["mojo", "build", "-O3", "--fp-mode=contract=off", "-I", "builds/deps", "tests/isotonic/parity_runner.mojo", "-o", str(binary)], cwd=ROOT, check=True)
    result = subprocess.run([str(binary), str(fixture)], cwd=ROOT, text=True, capture_output=True)
    (output / "result.txt").write_text(result.stdout + result.stderr)
    print(result.stdout, end="")
    print(result.stderr, end="")
    result.check_returncode()
    fixture32 = output / "reference32.txt"
    fixture32.write_text("\n".join(float32_records) + "\n")
    result32 = subprocess.run([str(binary), str(fixture32), "float32"], cwd=ROOT, text=True, capture_output=True)
    (output / "result32.txt").write_text(result32.stdout + result32.stderr)
    print(result32.stdout + result32.stderr, end="")
    result32.check_returncode()


if __name__ == "__main__":
    main()
