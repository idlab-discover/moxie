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
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
SEED = 94217
REFERENCE_SOURCES = (
    "sklearn/isotonic.py",
    "sklearn/_isotonic.pyx",
    "sklearn/metrics/_regression.py",
    "sklearn/utils/_metadata_requests.py",
    "sklearn/utils/_set_output.py",
    "sklearn/utils/_tags.py",
)


def load_reference(checkout):
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected reference {REVISION}, found {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--", *REFERENCE_SOURCES],
        text=True,
    )
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
    return ",".join(repr(float(x)) for x in value) or "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sklearn-checkout", type=Path, default=ROOT.parent / "scikit-learn")
    args = parser.parse_args()
    reference = load_reference(args.sklearn_checkout.resolve())
    regression_source = args.sklearn_checkout / "sklearn/metrics/_regression.py"
    spec = importlib.util.spec_from_file_location(
        "sklearn.metrics._moxie_reference_regression", regression_source
    )
    regression = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regression)
    output_source = args.sklearn_checkout / "sklearn/utils/_set_output.py"
    spec = importlib.util.spec_from_file_location(
        "sklearn.utils._moxie_reference_output", output_source
    )
    reference_output = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_output)
    rng = np.random.default_rng(SEED)
    records = []
    dtype = np.float64

    def add(*values):
        records.append("|".join(map(encode, values)))

    def estimator_tags():
        model = reference.IsotonicRegression()
        for fitted in (False, True):
            if fitted:
                x = np.asarray([0, 1], dtype=dtype)
                model.fit(x.reshape(-1, 1), x)
            tags = model.__sklearn_tags__()
            add("tags", fitted, tags.input_tags.one_d_array, tags.input_tags.two_d_array,
                tags._skip_test)

    def model_selection_rejects_misaligned_weights():
        x = np.arange(10, dtype=dtype)
        with sklearn.config_context(enable_metadata_routing=True):
            for weight_count in (1, 11):
                metadata = {"sample_weight": np.ones(weight_count, dtype=dtype)}
                for fit_request in (True, False):
                    model = reference.IsotonicRegression(out_of_bounds="clip")
                    model.set_fit_request(sample_weight=fit_request)
                    model.set_score_request(sample_weight=not fit_request)
                    with np.testing.assert_raises_regex(ValueError, "inconsistent numbers of samples"):
                        cross_val_score(model, x, x, params=metadata, error_score="raise")
                    search = GridSearchCV(model, {"y_max": [4.0]}, error_score="raise")
                    with np.testing.assert_raises_regex(ValueError, "inconsistent numbers of samples"):
                        search.fit(x, x, **metadata)
        print(f"PASS: pinned isotonic rejects misaligned CV and search weights ({dtype.__name__})")

    def routing_requests():
        source = args.sklearn_checkout / "sklearn/utils/_metadata_requests.py"
        spec = importlib.util.spec_from_file_location("sklearn.utils._moxie_reference_metadata", source)
        routing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(routing)
        model = reference.IsotonicRegression()
        for method in ("fit", "score"):
            request = routing.MethodMetadataRequest(owner=model, method=method)
            for setting in (None, False, True, "_weights2", "sample_weight", routing.UNCHANGED):
                if setting != routing.UNCHANGED:
                    request.add_request(param="sample_weight", alias=setting)
                for metadata in ({}, {"sample_weight": 1}, {"_weights2": 2}):
                    try:
                        resolved = request._route_params(metadata, parent=model, caller=method)
                    except UnsetMetadataPassedError:
                        expected = "reject"
                    else:
                        expected = str(resolved.get("sample_weight", "-"))
                    add("routing", method, setting, next(iter(metadata), "-"), expected)

    def serialized_requests():
        for fitted in (False, True):
            for fit, score in [(None, None), (False, True), (True, "_score2"), ("_fit2", False)]:
                with sklearn.config_context(enable_metadata_routing=True):
                    model = reference.IsotonicRegression().set_fit_request(sample_weight=fit)
                    model.set_score_request(sample_weight=score)
                    if fitted:
                        x = np.asarray([0, 1], dtype=dtype)
                        model.fit(x, x)
                    restored = reference.IsotonicRegression()
                    restored.__setstate__(model.__getstate__())
                    routing = restored.get_metadata_routing()
                    add("serialized_requests", fitted, fit, score,
                        routing.fit.requests["sample_weight"], routing.score.requests["sample_weight"])

    def warning_fields(caught):
        return [
            text for item in caught
            for text in (item.category.__name__, str(item.message))
        ]

    def direction_check(x, y):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                expected = reference.check_increasing(x, y)
            except ValueError:
                expected = "error"
        add("direction", x, y, expected, *warning_fields(caught))
        add("dense_direction", x, y, expected, *warning_fields(caught))

    def score_warnings():
        model = reference.IsotonicRegression().fit(
            np.asarray([0, 1], dtype=dtype), np.asarray([1, 2], dtype=dtype)
        )
        for kind, values, weights in [
            ("list", [0], None),
            ("column", [0], None),
            ("array", [0], None),
            ("array_weighted", [0], [2]),
            ("array_weighted", [0], [0]),
            ("array", [0, 1], None),
        ]:
            x = np.asarray(values, dtype=dtype)
            y = x + 1
            weights = None if weights is None else np.asarray(weights, dtype=dtype)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    expected = regression.r2_score(y, model.predict(x), sample_weight=weights)
                except ValueError as error:
                    if "at least one non-zero number" not in str(error):
                        raise
                    expected = "error"
            add("score_warnings", x, y, weights, kind, expected, *warning_fields(caught))

    def kernel(y, w=None, low=None, high=None, increasing=True):
        y = np.asarray(y, dtype=dtype)
        w = None if w is None else np.asarray(w, dtype=dtype)
        expected = reference.isotonic_regression(
            y, sample_weight=w, y_min=low, y_max=high, increasing=increasing
        )
        assert expected.shape == y.shape and expected.dtype == y.dtype
        independent = scipy_isotonic(y, weights=w, increasing=increasing).x
        independent = independent.astype(dtype)
        independent = np.clip(independent, -np.inf if low is None else low,
                              np.inf if high is None else high)
        np.testing.assert_allclose(expected, independent, rtol=2e-12, atol=2e-12)
        add("kernel", y, w, low, high, increasing, expected)

    def estimator(x, y, w=None, low=None, high=None, direction=True, mode="clip",
                  query=None, score=False, restored_mode=None):
        x, y = np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype)
        w = None if w is None else np.asarray(w, dtype=dtype)
        model = reference.IsotonicRegression(
            y_min=low, y_max=high, increasing=direction, out_of_bounds=mode
        ).fit(x, y, sample_weight=w)
        if query is None:
            query = np.linspace(model.X_min_, model.X_max_, 31)
        query = np.asarray(query, dtype=dtype)
        kind = "score" if score else "estimator"
        if restored_mode is not None:
            kind = "serialization"
            model.set_params(out_of_bounds=restored_mode)
            restored = reference.IsotonicRegression()
            restored.__setstate__(model.__getstate__())
            model = restored
        try:
            expected = model.predict(query)
        except ValueError as error:
            if restored_mode != "raise" or "interpolation range" not in str(error):
                raise
            expected = None
        else:
            if restored_mode is not None:
                assert expected.dtype == dtype
        fields = [kind, x, y, w, low, high, direction,
                  mode, model.X_thresholds_, model.y_thresholds_, query, expected,
                  model.increasing_, [model.X_min_, model.X_max_]]
        if score:
            target = rng.normal(size=len(query)).astype(dtype)
            sw = rng.uniform(0.01, 2, size=len(query)).astype(dtype)
            fields += [target, sw, [model.score(query, target, sample_weight=sw)]]
        if restored_mode is not None:
            fields.append(restored_mode)
        add(*fields)

    def serialization():
        for mode, restored_mode in [("raise", "nan"), ("nan", "raise"), ("raise", "clip")]:
            estimator([0, 1, 2], [3, 2, 1], direction="auto", mode=mode,
                      query=[-1, 0.5, 1.5, 3], restored_mode=restored_mode)

    def pipeline():
        x = np.asarray([7, -2, 3, 3, 10, 16], dtype=dtype)
        y = np.asarray([9, -1, 6, 2, 8, 13], dtype=dtype)
        query = np.asarray([18, 5, -4, 3, 12, 0], dtype=dtype)
        model = make_pipeline(
            StandardScaler(), reference.IsotonicRegression(out_of_bounds="clip")
        ).fit(x[:, None], y)
        expected = model.predict(query[:, None])
        assert expected.dtype == dtype
        add("pipeline", x, y, query, expected)

    def default_output():
        x = np.asarray([0, 1, 2], dtype=dtype)
        y = np.asarray([1, 4, 2], dtype=dtype)
        model = reference.IsotonicRegression()
        set_output = reference_output._SetOutputMixin.set_output
        assert set_output(model, transform="default") is model
        fitted = model.fit_transform(x[:, None], y)
        assert set_output(model, transform=None) is model
        transformed = model.transform(x)
        predicted = model.predict(x)
        for result in (fitted, transformed, predicted):
            assert result.dtype == dtype and result.shape == x.shape
        add("default_output", x, y, fitted, transformed, predicted)

    def labeled_output():
        x = np.asarray([0, 1, 1, 2, 3], dtype=dtype)
        y = np.asarray([1, 4, 2, 2, 6], dtype=dtype)
        query = np.asarray([4, 1.5, -1, 1], dtype=dtype)
        for mode, weights in (
            ("clip", None),
            ("nan", np.asarray([0, 1, 3, 2, 0], dtype=dtype)),
        ):
            model = reference.IsotonicRegression(out_of_bounds=mode)
            reference_output._SetOutputMixin.set_output(model, transform="pandas")
            # Apply the pinned wrapper to the pinned methods. The class was
            # otherwise auto-wrapped by the installed supporting sklearn wheel.
            transform = reference_output._wrap_method_output(
                reference.IsotonicRegression.transform.__wrapped__, "transform"
            )
            fit_transform = reference_output._wrap_method_output(
                reference.IsotonicRegression.fit_transform.__wrapped__, "transform"
            )
            fitted = fit_transform(model, x[:, None], y, sample_weight=weights)
            transformed = transform(model, query)
            name, = fitted.columns
            assert transformed.columns.tolist() == [name]
            for table, inputs in ((fitted, x), (transformed, query)):
                assert table.shape == (len(inputs), 1)
                assert table.dtypes.iloc[0] == dtype
                np.testing.assert_array_equal(table.index, np.arange(len(inputs)))
            add("labeled_output", x, y, weights, query, mode, name,
                fitted[name].to_numpy(), transformed[name].to_numpy())

    def fit_warnings():
        for x, y, mode, weights in [
            ([0], [1], "auto", None),
            ([0, 0], [1, 2], "auto", None),
            ([0, 0, 0, 0], [1, 2, 3, 4], "auto", None),
            ([0, 1, 2, 3], [1, 1, 1, 1], "auto", None),
            ([0, 1, 2], [0, 2, 1], "auto", None),
            ([0, 1, 2, 3], [0, 2, 1, 3], "auto", None),
            ([0, 1, 2, 3], [3, 1, 2, 0], "auto", None),
            ([0, 1, 2, 3], [0, 1, 2, 3], "auto", None),
            ([0, 1, 2, 3], [3, 2, 1, 0], "auto", None),
            (list(range(8)), [0, 2, 1, 3, 4, 5, 6, 7], "auto", None),
            ([0, 1, 2, 3], [0, 2, 1, 3], "fixed", None),
            ([0, 1, 2, 3], [0, 2, 1, 3], "auto", [1, 1, 0, 1]),
        ]:
            x, y = np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype)
            weights = None if weights is None else np.asarray(weights, dtype=dtype)
            model = reference.IsotonicRegression(
                increasing="auto" if mode == "auto" else True,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(x, y, sample_weight=weights)
            add("fit_warnings", x, y, mode, weights, model.increasing_, *warning_fields(caught))

    def validation_edges():
        for x, y in [
            ([], []),
            ([np.nan], [1]),
            ([0, 1, 2, 3], [0, 1, 2, np.nan]),
            ([0, 0, 0, 0], [0, 1, 2, np.nan]),
            ([0, 0, 0, 0], [0, 1, 2, 3]),
            ([0, np.nan, 2], [0, 1, 2]),
            ([0, 1, 2], [1, 2, np.inf]),
            ([-np.inf, 0, np.inf, np.inf], [3, 2, 1, 1]),
            ([0, 1, 2], [np.inf, np.inf, np.inf]),
            ([0, 1, 2, 3], [0, 1, 2, 3]),
            ([0, 1, 2, 3], [0, 2, 1, 3]),
            ([], [1, 2]),
            ([0, 1], [1]),
        ]:
            direction_check(np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype))
        for mode in ("nan", "clip", "raise"):
            for n in (1, 2):
                x = np.arange(n, dtype=dtype)
                y = x + 3
                model = reference.IsotonicRegression(out_of_bounds=mode).fit(x, y)
                for values in ([], [np.nan, 0.5], [-np.inf, np.nan, 0, np.inf]):
                    query = np.asarray(values, dtype=dtype)
                    try:
                        with np.errstate(invalid="ignore"):
                            expected = model.f_(query)
                    except ValueError as error:
                        if mode != "raise" or "interpolation range" not in str(error):
                            raise
                        expected = "error"
                    else:
                        assert expected.shape == query.shape and expected.dtype == dtype
                    add("interpolator", x, y, mode, query, expected)

    pipeline()
    default_output()
    labeled_output()
    serialization()
    validation_edges()
    fit_warnings()
    score_warnings()
    routing_requests()
    serialized_requests()
    estimator_tags()
    model_selection_rejects_misaligned_weights()
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
            direction_check(x, y)

    float64_records = records
    records = []
    dtype = np.float32
    pipeline()
    default_output()
    labeled_output()
    serialization()
    validation_edges()
    fit_warnings()
    score_warnings()
    routing_requests()
    serialized_requests()
    estimator_tags()
    model_selection_rejects_misaligned_weights()
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
            direction_check(x, y)
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
                "platform": platform.platform(), "reference_hashes": {f: hashlib.sha256((args.sklearn_checkout / f).read_bytes()).hexdigest() for f in REFERENCE_SOURCES}, "rtol": 2e-12, "atol": 2e-12}
    metadata["float32"] = {"cases": len(float32_records), "rtol": 2e-6, "atol": 2e-6}
    metadata["mojo"] = subprocess.check_output(["mojo", "--version"], cwd=ROOT, text=True).strip()
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    binary = output / "runner"
    subprocess.run(["mojo", "build", "-O3", "--fp-mode=contract=off", "-I", "builds/deps", "tests/isotonic/parity_runner.mojo", "-o", str(binary)], cwd=ROOT, check=True)
    fixture32 = output / "reference32.txt"
    fixture32.write_text("\n".join(float32_records) + "\n")

    def check_runner(fixture, result_path, cases, *args):
        result = subprocess.run(
            [str(binary), str(fixture), *args], cwd=ROOT, text=True, capture_output=True
        )
        result_path.write_text(result.stdout + result.stderr)
        if result.returncode:
            print(result.stdout + result.stderr, end="")
            result.check_returncode()
        expected = []
        for case in cases:
            kind, *fields = case.split("|")
            if kind in ("direction", "dense_direction", "score_warnings"):
                warning_fields = fields[5:] if kind == "score_warnings" else fields[3:]
                expected.extend(
                    f"{category}: {message}"
                    for category, message in zip(
                        warning_fields[::2], warning_fields[1::2], strict=True
                    )
                )
        np.testing.assert_array_equal(result.stderr.splitlines(), expected)
        print(result.stdout, end="")
        print(f"PASS: {len(expected)} stderr warnings match in order")

    check_runner(fixture, output / "result.txt", records)
    check_runner(fixture32, output / "result32.txt", float32_records, "float32")


if __name__ == "__main__":
    main()
