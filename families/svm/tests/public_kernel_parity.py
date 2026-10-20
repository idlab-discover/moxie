"""Compare public native estimators with pinned sklearn wrappers and LIBSVM.

Installed sklearn supplies validation infrastructure. The fit and presentation
classes are executed from the pinned local checkout, with calls to LIBSVM
redirected to the C++ oracle compiled from that same checkout.
"""
from pathlib import Path
import ast
import importlib.util
import json
import subprocess
import warnings

import numpy as np
import sklearn.svm._base as installed_base
import sklearn.svm._classes as installed_classes

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[2] / "scikit-learn"
REVISION = "4358542a1bd39af2e7e1cbace8b9fed067473c5b"
ENGINE = ROOT / "shared/libsvm"


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in tree.body} == set(names)
    exec(compile(tree, str(path), "exec"), namespace)


class PinnedBackend:
    """Validation-only bridge; all optimization and inference use the oracle."""

    def __init__(self, oracle):
        self.oracle = oracle

    def set_verbosity_wrap(self, value):
        pass

    def fit(self, X, y, **options):
        self.x = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.sw = options.pop("sample_weight")
        if not len(self.sw):
            self.sw = np.ones(len(y))
        self.cw = options.pop("class_weight")
        self.options = options
        got = self.run(X)
        support = np.asarray(got.get("raw_support", got["support"]), dtype=np.int32)
        n_support = np.asarray(got["n_support"], dtype=np.int32)
        if options["svm_type"] >= 2:
            n_support = np.array([len(support), len(support)], dtype=np.int32)
        vectors = np.empty((0, 0)) if options["kernel"] == "precomputed" else self.x[np.asarray(got["support"], dtype=int)].copy()
        return (support, vectors, n_support, np.asarray(got["dual"]),
                -np.asarray(got["rho"]), np.asarray(got["prob_a"]),
                np.asarray(got["prob_b"]), int(np.asarray(got["status"]).item()),
                np.asarray(got["iterations"], dtype=np.int32))

    def run(self, query):
        options = dict(self.options)
        options["formulation"] = ("c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr")[options.pop("svm_type")]
        for old, new in (("C", "c"), ("cache_size", "cache_mb"), ("random_seed", "seed")):
            options[new] = options.pop(old)
        return self.oracle.run_reference(self.x, self.y, self.sw, self.cw, options, query)

    def predict(self, X, *args, **options):
        return np.asarray(self.run(X)["predictions"])

    def decision_function(self, X, *args, **options):
        return np.asarray(self.run(X)["decisions"])

    def predict_proba(self, X, *args, **options):
        return np.asarray(self.run(X)["probabilities"])


def reference_classes():
    actual = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip()
    assert actual == REVISION
    assert not subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain", "--", "sklearn/svm"], text=True)
    spec = importlib.util.spec_from_file_location("solver_parity", ENGINE / "tests/solver_parity.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    oracle.compile_reference()
    base = dict(vars(installed_base))
    base["libsvm"] = PinnedBackend(oracle)
    load_definitions(REFERENCE / "sklearn/svm/_base.py", {"BaseLibSVM", "BaseSVC", "_one_vs_one_coef"}, base)
    classes = dict(vars(installed_classes))
    classes.update(BaseLibSVM=base["BaseLibSVM"], BaseSVC=base["BaseSVC"])
    load_definitions(REFERENCE / "sklearn/svm/_classes.py", {"SVC", "NuSVC", "SVR", "NuSVR", "OneClassSVM"}, classes)
    return classes


def native(x, y, sw, query, kind, config, single=False, expect_error=False):
    weights = config.get("class_weight")
    mode = "balanced" if weights == "balanced" else "explicit" if isinstance(weights, dict) else "none"
    gamma = config.get("gamma", "scale")
    gamma_mode = gamma if isinstance(gamma, str) else "value"
    gamma_value = 0 if isinstance(gamma, str) else gamma
    values = [len(x), x.shape[1], len(query), kind, config.get("kernel", "rbf"),
              gamma_mode, gamma_value, config.get("degree", 3), config.get("coef0", 0),
              config.get("C", 1), config.get("nu", .5), config.get("epsilon", .1),
              config.get("tol", .001), int(config.get("shrinking", True)),
              int(config.get("probability", False)), config.get("max_iter", -1),
              config.get("random_state", 42), mode, config.get("decision_function_shape", "ovr"),
              int(config.get("break_ties", False)), int(single), len(weights) if isinstance(weights, dict) else 0,
              *np.asarray(x, dtype=float).ravel(), *y, *sw, *np.asarray(query, dtype=float).ravel()]
    if isinstance(weights, dict):
        for label, weight in weights.items():
            values.extend([label, weight])
    case = ROOT / "builds/public-kernel-case.txt"
    case.write_text(" ".join(map(str, values)) + "\n")
    result = subprocess.run([str(ROOT / "builds/public_kernel_driver"), str(case)], text=True, capture_output=True, timeout=120)
    if expect_error:
        assert result.returncode != 0 and "ValueError" in result.stdout + result.stderr, result.stdout + result.stderr
        return None
    assert result.returncode == 0, result.stdout + result.stderr
    lines = iter(result.stdout.splitlines())
    arrays = {}
    for line in lines:
        words = line.split()
        if not words or words[0] not in ("ARRAY", "LIST", "SCALAR"):
            continue
        tag, name = words[:2]
        if tag == "SCALAR":
            arrays[name] = float(words[2])
            continue
        shape = tuple(map(int, words[2:]))
        arrays[name] = np.array([float(next(lines)) for _ in range(int(np.prod(shape)))]).reshape(shape)
    return arrays


def compare(expected, got, query, kind, config):
    for field in ("support_", "n_support_"):
        np.testing.assert_array_equal(got[field], getattr(expected, field), err_msg=field)
    for field in ("support_vectors_", "dual_coef_", "intercept_"):
        np.testing.assert_allclose(got[field], getattr(expected, field), rtol=3e-5, atol=3e-6, err_msg=field)
    assert got["fit_status_"] == expected.fit_status_
    assert got["n_features_in_"] == expected.n_features_in_
    np.testing.assert_array_equal(got["n_iter_"], np.atleast_1d(expected.n_iter_))
    if kind in (0, 1):
        np.testing.assert_array_equal(got["classes_"], expected.classes_)
        np.testing.assert_allclose(got["class_weight_"], expected.class_weight_, rtol=1e-14)
        np.testing.assert_allclose(got["decision"], expected.decision_function(query), rtol=3e-5, atol=3e-6)
        np.testing.assert_array_equal(got["prediction"], expected.predict(query))
        if config.get("probability"):
            for field in ("probA_", "probB_"):
                np.testing.assert_allclose(got[field], getattr(expected, field), rtol=3e-5, atol=3e-6)
            np.testing.assert_allclose(got["probability"], expected.predict_proba(query), rtol=3e-5, atol=3e-6)
            np.testing.assert_allclose(got["log_probability"], expected.predict_log_proba(query), rtol=3e-5, atol=3e-6)
    elif kind == 2:
        np.testing.assert_allclose(got["decision"], expected.decision_function(query), rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(got["score_samples"], expected.score_samples(query), rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(got["offset_"], expected.offset_, rtol=3e-5, atol=3e-6)
        np.testing.assert_array_equal(got["prediction"], expected.predict(query))
    else:
        np.testing.assert_allclose(got["prediction"], expected.predict(query), rtol=3e-5, atol=3e-6)
    if config["kernel"] == "linear":
        np.testing.assert_allclose(got["coef_"], expected.coef_, rtol=3e-5, atol=3e-6)


def main():
    classes = reference_classes()
    rng = np.random.RandomState(194)
    results = []
    names = ("SVC", "NuSVC", "OneClassSVM", "SVR", "NuSVR")
    for kind, name in enumerate(names):
        for kernel in ("linear", "poly", "rbf", "sigmoid", "precomputed"):
            for variant in range(4):
                x = rng.normal(size=(18, 3)) * .4
                query = rng.normal(size=(7, 3)) * .4
                y = np.array([-7, 4, 19])[np.arange(18) % (2 if variant < 2 else 3)] if kind < 2 else rng.normal(size=18)
                sw = np.ones(18) if variant % 2 == 0 else rng.uniform(.7, 1.4, 18)
                config = dict(kernel=kernel, gamma="scale" if variant == 0 else "auto" if variant == 1 else .7,
                              degree=2, coef0=.2, tol=1e-6, shrinking=bool(variant % 2), max_iter=-1)
                if kind < 2:
                    config.update(probability=variant == 3, random_state=42,
                                  decision_function_shape="ovo" if variant == 2 else "ovr",
                                  break_ties=variant == 3,
                                  class_weight="balanced" if variant == 1 else {-7:.8, 4:1.3, 19:1.1} if variant == 2 else None)
                if kind in (0, 3, 4):
                    config["C"] = .8
                if kind in (1, 2, 4):
                    config["nu"] = .3
                if kind == 3:
                    config["epsilon"] = .12
                single = variant == 1
                if kernel == "precomputed":
                    query, x = query @ x.T, x @ x.T
                if single:
                    x, query = x.astype(np.float32), query.astype(np.float32)
                print(name, kernel, variant, flush=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    expected = classes[name](**config)
                    expected.fit(x, sample_weight=sw) if kind == 2 else expected.fit(x, y, sample_weight=sw)
                    got = native(x, y, sw, query, kind, config, single)
                    compare(expected, got, query, kind, config)
                    if kind != 2:
                        np.testing.assert_allclose(got["score"], expected.score(x, y, sample_weight=sw), rtol=3e-5, atol=3e-6)
                results.append(dict(estimator=name, kernel=kernel, variant=variant, iterations_native=got["n_iter_"].tolist(), iterations_reference=np.atleast_1d(expected.n_iter_).tolist()))
    for kind, name in enumerate(names):
        for edge in ("limit0", "limit1", "limit3", "masked", "constant"):
            x = rng.normal(size=(12, 3)) * .4
            query = rng.normal(size=(5, 3)) * .4
            if edge == "constant":
                x[:] = .5
            y = np.array([-7, 4])[np.arange(12) % 2] if kind < 2 else rng.normal(size=12)
            sw = np.ones(12)
            if edge == "masked":
                sw[0], sw[3] = 0, -.1
            config = dict(kernel="rbf", gamma="scale", tol=1e-6,
                          max_iter=int(edge[-1]) if edge.startswith("limit") else -1)
            if kind < 2:
                config["random_state"] = 17
            if kind in (1, 2, 4):
                config["nu"] = .3
            print(name, edge, flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected = classes[name](**config)
                try:
                    expected.fit(x, sample_weight=sw) if kind == 2 else expected.fit(x, y, sample_weight=sw)
                except ValueError:
                    native(x, y, sw, query, kind, config, expect_error=True)
                    results.append(dict(estimator=name, edge=edge, rejected=True))
                    continue
                got = native(x, y, sw, query, kind, config)
                compare(expected, got, query, kind, config)
                if kind != 2:
                    np.testing.assert_allclose(got["score"], expected.score(x, y, sample_weight=sw), rtol=3e-5, atol=3e-6)
                results.append(dict(estimator=name, edge=edge, iterations_native=got["n_iter_"].tolist(), iterations_reference=np.atleast_1d(expected.n_iter_).tolist()))
    (ROOT / "builds/public-kernel-parity.json").write_text(json.dumps(dict(revision=REVISION, cases=results), indent=2) + "\n")
    print(f"PASS {len(results)} public kernel estimator comparisons against pinned sklearn wrappers and LIBSVM")


if __name__ == "__main__":
    main()
