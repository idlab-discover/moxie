"""Readiness audit, not an assertion of complete parity.

Run `pixi run audit`; JSON observations and inputs are retained in builds/audit.
Use --strict to exit nonzero for observed differences. Normal numerical
comparisons use 2e-8 Float64 / 5e-5 Float32 relative and absolute tolerances;
state preservation uses exact equality. No widened or tie-normalized oracle
is used here. Existing algorithm parity suites provide broader numeric coverage.
"""
from pathlib import Path
import contextlib
import io as textio
import json
import os
import platform
import subprocess
import sys
import warnings

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import parity_io as io
import pinned_sklearn

OUT = ROOT / 'builds/audit'
OUT.mkdir(parents=True, exist_ok=True)
MODULES = pinned_sklearn.load()
EMP = MODULES['_empirical_covariance']
GL = MODULES['_graph_lasso']
CLASSES = [EMP.EmpiricalCovariance,
           MODULES['_shrunk_covariance'].ShrunkCovariance,
           MODULES['_shrunk_covariance'].LedoitWolf,
           MODULES['_shrunk_covariance'].OAS,
           MODULES['_robust_covariance'].MinCovDet,
           MODULES['_elliptic_envelope'].EllipticEnvelope,
           GL.GraphicalLasso, GL.GraphicalLassoCV]
assert all(pinned_sklearn.executed_from_pinned(cls.fit) for cls in CLASSES)
REPORT = dict(reference=pinned_sklearn.REVISION, cases=[], differences=[], comparisons=0)


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, Exception):
        return f'{type(value).__name__}: {value}'
    return value


def observe(call):
    stdout = textio.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(stdout):
        warnings.simplefilter('always')
        try:
            result = call()
        except Exception as error:
            result = error
    return result, [dict(category=type(w.message).__name__, message=str(w.message)) for w in caught], stdout.getvalue()


def compare(case, key, got, want, dtype, exact=False):
    REPORT['comparisons'] += 1
    if isinstance(got, Exception) or isinstance(want, Exception):
        same = isinstance(got, Exception) and isinstance(want, Exception)
        if same:
            # Native errors use prefixed text, not Python exception objects.
            same = type(want).__name__ in str(got)
    elif got is None or want is None:
        same = got is None and want is None
    else:
        got, want = np.asarray(got).ravel(), np.asarray(want).ravel()
        tol = 0 if exact else (2e-8 if dtype == np.float64 else 5e-5)
        same = got.shape == want.shape and np.allclose(got, want, rtol=tol, atol=tol, equal_nan=True)
    if not same:
        REPORT['differences'].append(dict(case=case, method=key, native=serial(got), reference=serial(want)))


def snapshot(model, X):
    result = {}
    attrs = ['covariance', 'location', 'precision', 'n_features_in', 'shrinkage',
             'raw_covariance', 'raw_location', 'raw_support', 'support', 'dist', 'offset', 'alpha']
    for attr in attrs:
        if hasattr(model, attr + '_'):
            result['features' if attr == 'n_features_in' else attr] = getattr(model, attr + '_')
    result['precision'] = model.get_precision()
    result['mahalanobis'] = model.mahalanobis(X)
    result['error'] = model.error_norm(model.covariance_)
    if isinstance(model, CLASSES[5]):
        result.update(predict=model.predict(X), decision=model.decision_function(X), samples=model.score_samples(X))
        result['score'] = model.score(X, model.predict(X))
    else:
        result['score'] = model.score(X)
    if hasattr(model, 'n_iter_'):
        result['iterations'] = model.n_iter_
        result['cost_count'] = len(model.costs_)
        for i, (cost, gap) in enumerate(model.costs_):
            result[f'objective{i}'], result[f'gap{i}'] = cost, gap
    if hasattr(model, 'cv_results_'):
        cv = model.cv_results_
        result.update(alphas=cv['alphas'], mean=cv['mean_test_score'], std=cv['std_test_score'],
                      splits=np.column_stack([cv[f'split{i}_test_score'] for i in range(3)]))
    return {k: np.asarray(v).copy() for k, v in result.items()}


def refits():
    rng = np.random.default_rng(872)
    original = rng.normal(size=(40, 2)) + [1, -2]
    for dtype in (np.float64, np.float32):
        X = original.astype(dtype)
        for kind, cls in enumerate(CLASSES):
            options = dict(random_state=7) if kind in (4, 5) else (dict(alphas=[.3, .1], cv=3) if kind == 7 else {})
            scenarios = ['invalid', 'numerical', 'success']
            if kind not in (0, 3):
                scenarios.append('parameter')
            for scenario in scenarios:
                Y = (X * 1.2 + .5).copy()
                if scenario == 'invalid':
                    Y = np.full((40, 3), np.nan, dtype=dtype)
                elif scenario == 'numerical':
                    Y = np.zeros((40, 2), dtype=dtype) if kind >= 4 else np.full_like(X, 1e200 if dtype == np.float64 else 1e30)
                case = f'{cls.__name__}-{dtype.__name__}-{scenario}'
                native = io.run(ROOT / 'builds/audit_driver', OUT / (case + '.txt'),
                                f'{32 if dtype == np.float32 else 64} {kind} {io.matrix(X)} {io.matrix(Y)} {scenario}')
                model = cls(**options)
                fitted, initial_warnings, _ = observe(lambda: model.fit(X))
                assert not isinstance(fitted, Exception), fitted
                before = snapshot(model, X)
                model.set_params(assume_centered=True)
                updated = snapshot(model, X)
                assert native['clone_fitted'][0] == 0
                for key in before:
                    compare(case, 'initial.' + key, native['0_' + key], before[key], dtype)
                    compare(case, 'set_params.' + key, native['1_' + key], updated[key], dtype)
                    np.testing.assert_array_equal(native['0_' + key], native['1_' + key])
                if scenario == 'parameter':
                    invalid_params = {1: dict(shrinkage=-1.), 2: dict(block_size=0),
                                      4: dict(support_fraction=2.), 5: dict(contamination=.7),
                                      6: dict(tol=-1.), 7: dict(tol=-1.)}
                    model.set_params(**invalid_params[kind])
                fitted, caught, stdout = observe(lambda: model.fit(Y))
                after, _, _ = observe(lambda: snapshot(model, X))
                native_error = native.get('refit')
                compare(case, 'fit.outcome', native_error, fitted if isinstance(fitted, Exception) else None, dtype)
                if native_error is not None and scenario in ('invalid', 'parameter'):
                    for key in before:
                        np.testing.assert_array_equal(native['1_' + key], native['2_' + key])
                if isinstance(after, dict):
                    for key, value in after.items():
                        compare(case, 'refit.' + key, native.get('2_' + key), value, dtype)
                else:
                    REPORT['differences'].append(dict(case=case, method='post_failure_inference',
                                                      native='usable prior state', reference=serial(after)))
                REPORT['cases'].append(dict(case=case, native=serial(native), reference_after=serial(after),
                                             reference_error=serial(fitted) if isinstance(fitted, Exception) else None,
                                             warnings=initial_warnings + caught, stdout=stdout))


def graphical_edges():
    rng = np.random.default_rng(310)
    base = rng.normal(size=(30, 3))
    base[:, 1] += .8 * base[:, 0]
    for dtype in (np.float64, np.float32):
        for mode in ('cd', 'lars'):
            for scenario in ('zero_iterations', 'one_iteration', 'ill_conditioned', 'constant'):
                X = base.copy().astype(dtype)
                iterations = 0 if scenario == 'zero_iterations' else 1 if scenario == 'one_iteration' else 100
                if scenario == 'ill_conditioned':
                    X[:, 1] = X[:, 0] + 1e-7 * X[:, 1]
                if scenario == 'constant':
                    X[:, 1] = 0
                emp = np.cov(X.T.astype(np.float64), bias=True).astype(dtype)
                case = f'graphical-{mode}-{dtype.__name__}-{scenario}'
                native = io.run(ROOT / 'builds/audit_graphical_driver', OUT / (case + '.txt'),
                                f'{32 if dtype == np.float32 else 64} {io.matrix(X)} {io.matrix(X[:8])} {io.matrix(emp)} {io.real(.01)} {iterations} {mode}')
                observations = []
                for centered in (False, True):
                    prefix = f'gl_assume{int(centered)}'
                    model = GL.GraphicalLasso(alpha=.01, max_iter=iterations, mode=mode, assume_centered=centered)
                    result, caught, _ = observe(lambda: model.fit(X))
                    observations.append(dict(branch=prefix, warnings=caught, error=serial(result) if isinstance(result, Exception) else None))
                    compare(case, prefix + '.outcome', native.get(prefix), result if isinstance(result, Exception) else None, dtype)
                    if isinstance(result, Exception):
                        continue
                    for key, want in dict(location=model.location_, covariance=model.covariance_, precision=model.precision_,
                                          costs=model.costs_, n_iter=model.n_iter_, score=model.score(X[:8]), mahalanobis=model.mahalanobis(X[:8])).items():
                        compare(case, prefix + '.' + key, native[prefix + '_' + key], want, dtype, exact=key == 'n_iter')
                for zero, alpha in enumerate((.01, 0.)):
                    prefix = f'fn_alpha{zero}'
                    result, caught, _ = observe(lambda: GL.graphical_lasso(emp, alpha, mode=mode, max_iter=iterations, return_costs=True, return_n_iter=True))
                    observations.append(dict(branch=prefix, warnings=caught, error=serial(result) if isinstance(result, Exception) else None))
                    compare(case, prefix + '.outcome', native.get(prefix), result if isinstance(result, Exception) else None, dtype)
                    if not isinstance(result, Exception):
                        for key, want in zip(('covariance', 'precision', 'costs', 'n_iter'), result):
                            compare(case, prefix + '.' + key, native[prefix + '_' + key], want, dtype, exact=key == 'n_iter')
                REPORT['cases'].append(dict(case=case, native=serial(native), observations=observations))


def diagnostics():
    X = np.random.default_rng(44).normal(size=(40, 2)) + [1, -2]
    for kind in (0, 1, 2, 3, 4, 5, 6, 7):
        cls = CLASSES[kind]
        data = X[:1] if kind < 4 else (np.column_stack([X[:, 0], X[:, 0]]) if kind < 6 else X)
        path = OUT / f'diagnostics-{cls.__name__}.txt'
        path.write_text(f'64 {kind} {io.matrix(data)} {io.matrix(data)} verbose')
        native = subprocess.run([str(ROOT / 'builds/audit_driver'), str(path)], capture_output=True, text=True, check=True)
        options = dict(random_state=7) if kind in (4, 5) else (dict(verbose=True, max_iter=1) if kind >= 6 else {})
        if kind == 7:
            options.update(alphas=[.3, .1], cv=3)
        stderr = textio.StringIO()
        with contextlib.redirect_stderr(stderr):
            _, caught, stdout = observe(lambda: cls(**options).fit(data))
        # Native warning transport is category-prefixed stderr, not Python
        # warning objects. Check each expected category/message stem plus output.
        missing = [w for w in caught if w['category'] + ': ' not in native.stderr
                   or w['message'].split(':')[0] not in native.stderr]
        if missing or bool(stdout) != bool(native.stdout):
            REPORT['differences'].append(dict(case=cls.__name__, method='diagnostics', native_stdout=native.stdout,
                                              native_stderr=native.stderr, reference_stdout=stdout,
                                              reference_stderr=stderr.getvalue(), reference_warnings=caught))


def cv_edges():
    for dtype in (np.float64, np.float32):
        for mode in ('cd', 'lars'):
            for scenario, iterations in (('zero_iterations', 0), ('one_iteration', 1), ('float_grid', 20), ('singular_baseline', 100)):
                X = np.random.default_rng(1510).normal(size=(53, 4))
                X[:, 1] += .6 * X[:, 0]
                X += np.linspace(-.7, 1.3, 4)
                if scenario == 'singular_baseline':
                    X = X[:6]
                X = X.astype(dtype)
                case = f'cv-{mode}-{dtype.__name__}-{scenario}'
                native = io.run(ROOT / 'builds/audit_cv_driver', OUT / (case + '.txt'),
                                f'{32 if dtype == np.float32 else 64} {io.matrix(X)} {io.matrix(X[:5])} 3 0 2 3 {iterations} 0 {mode}')
                observations = []
                for centered in (False, True):
                    prefix = f'cv{int(centered)}'
                    model = GL.GraphicalLassoCV(alphas=3, n_refinements=2, cv=3, max_iter=iterations, mode=mode, assume_centered=centered)
                    result, caught, _ = observe(lambda: model.fit(X))
                    observations.append(dict(branch=prefix, warnings=caught, error=serial(result) if isinstance(result, Exception) else None))
                    compare(case, prefix + '.outcome', native.get(prefix), result if isinstance(result, Exception) else None, dtype)
                    if isinstance(result, Exception):
                        continue
                    cv = model.cv_results_
                    expected = dict(alpha=model.alpha_, alphas=cv['alphas'], mean=cv['mean_test_score'], std=cv['std_test_score'],
                                    splits=np.column_stack([cv[f'split{i}_test_score'] for i in range(3)]),
                                    covariance=model.covariance_, precision=model.precision_, location=model.location_,
                                    costs=model.costs_, n_iter=model.n_iter_, score=model.score(X[:5]), mahalanobis=model.mahalanobis(X[:5]))
                    for key, value in expected.items():
                        compare(case, prefix + '.' + key, native.get(prefix + '_' + key), value, dtype, exact=key == 'n_iter')
                    # Compare selection independently of nearby alpha values.
                    selected = np.argmin(abs(native[prefix + '_alphas'] - native[prefix + '_alpha'][0]))
                    expected_selected = np.argmin(abs(cv['alphas'] - model.alpha_))
                    compare(case, prefix + '.selected_index', selected, expected_selected, dtype, exact=True)
                REPORT['cases'].append(dict(case=case, native=serial(native), observations=observations))


def helper_edges():
    for dtype in (np.float64, np.float32):
        for name, data, precision in (
            ('regular', [[2., .5], [.5, 1.]], [[1., 0.], [0., 1.]]),
            ('asymmetric', [[1., 0.], [0., 1.]], [[2., 1.], [.5, 2.]]),
            ('nonfinite', [[1., np.nan], [0., 1.]], [[1., 0.], [0., 1.]]),
            ('singleton', [[1., 2.]], [[1., 0.], [0., 1.]]),
            ('constant', [[1., 1.], [1., 1.], [1., 1.]], [[1., 0.], [0., 1.]]),
        ):
            X, P = np.array(data, dtype=dtype), np.array(precision, dtype=dtype)
            case = f'helpers-{dtype.__name__}-{name}'
            native = io.run(ROOT / 'builds/audit_driver', OUT / (case + '.txt'),
                            f'{32 if dtype == np.float32 else 64} 8 {io.matrix(X)} {io.matrix(P)}')
            assert 'empirical' in native, 'rebuild audit_driver before running this audit'
            shrink = MODULES['_shrunk_covariance']
            calls = dict(empirical=lambda: EMP.empirical_covariance(X),
                         empirical_centered=lambda: EMP.empirical_covariance(X, assume_centered=True),
                         shrunk=lambda: shrink.shrunk_covariance(X),
                         likelihood=lambda: EMP.log_likelihood(X, P),
                         lw=lambda: shrink.ledoit_wolf(X),
                         lw_coefficient=lambda: shrink.ledoit_wolf_shrinkage(X),
                         oas=lambda: shrink.oas(X),
                         mcd=lambda: MODULES['_robust_covariance'].fast_mcd(X, random_state=7))
            observations = {}
            for key, call in calls.items():
                result, caught, stdout = observe(call)
                observations[key] = dict(result=serial(result), warnings=caught, stdout=stdout)
                if isinstance(result, Exception):
                    compare(case, key, native.get(key), result, dtype)
                elif key in ('lw', 'oas', 'mcd'):
                    fields = ('location', 'covariance', 'support', 'dist') if key == 'mcd' else ('covariance', 'shrinkage')
                    for field, value in zip(fields, result):
                        compare(case, key + '_' + field, native.get(key + '_' + field, native.get(key)), value, dtype)
                else:
                    compare(case, key, native.get(key), result, dtype)
            REPORT['cases'].append(dict(case=case, native=serial(native), observations=observations))


def main():
    import scipy
    import sklearn
    REPORT['environment'] = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                 sklearn_support=sklearn.__version__, platform=platform.platform(),
                                 compiler_flags='-O3 --fp-mode=contract=off', threads=1,
                                 seeds=[872, 310, 1510, 44, 7])
    refits()
    graphical_edges()
    cv_edges()
    helper_edges()
    diagnostics()
    (OUT / 'results.json').write_text(json.dumps(serial(REPORT), indent=2, allow_nan=False) + '\n')
    print(f"AUDIT: {len(REPORT['cases'])} cases, {REPORT['comparisons']} comparisons, {len(REPORT['differences'])} observed differences; builds/audit/results.json")
    if '--strict' in sys.argv and REPORT['differences']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
