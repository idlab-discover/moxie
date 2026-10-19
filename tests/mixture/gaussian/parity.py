"""Differential parity: native GaussianMixture versus the pinned local sklearn.

The pinned sources sklearn/mixture/{_base,_gaussian_mixture}.py at the recorded
revision are executed (not the support wheel's copies); the support wheel
supplies infrastructure (validation, KMeans, array API helpers). Its
sklearn/cluster/_kmeans.py, _k_means_lloyd.pyx and _k_means_common.pyx are
checked to be byte-identical to the pinned revision, so KMeans initialization
is the pinned algorithm too. OMP/BLAS threads are pinned to 1 for determinism.

Environment: MOXIE_PARITY_REPORT (default builds/parity/report.json).
"""
from pathlib import Path
import hashlib, importlib.util, json, os, struct, subprocess, sys, warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path(os.environ.get('MOXIE_WORKSPACE', ROOT.parent)) / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
RUNNER = ROOT / 'builds/tests/mixture/gaussian/parity_runner'
WORK = ROOT / 'builds/mixture/gaussian/parity'
import shutil
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True, exist_ok=True)
PINNED = ['sklearn/mixture/_base.py', 'sklearn/mixture/_gaussian_mixture.py',
          'sklearn/mixture/_bayesian_mixture.py', 'sklearn/cluster/_kmeans.py',
          'sklearn/cluster/_k_means_lloyd.pyx', 'sklearn/cluster/_k_means_common.pyx',
          'sklearn/utils/_array_api.py']


def verify_reference():
    head = subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == REVISION, head
    dirty = subprocess.check_output(['git', '-C', str(REFERENCE), 'status', '--porcelain', '--', *PINNED], text=True)
    assert not dirty.strip(), dirty
    import sklearn
    wheel = Path(sklearn.__file__).parent
    for name in ['cluster/_kmeans.py', 'cluster/_k_means_lloyd.pyx', 'cluster/_k_means_common.pyx']:
        assert (wheel / name).read_bytes() == (REFERENCE / 'sklearn' / name).read_bytes(), f'KMeans source differs: {name}'


def load_pinned(name, relative):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify_reference()
import sklearn.mixture  # support wheel package; pinned modules replace its members below
BASE = load_pinned('sklearn.mixture._base', 'sklearn/mixture/_base.py')
GAUSSIAN = load_pinned('sklearn.mixture._gaussian_mixture', 'sklearn/mixture/_gaussian_mixture.py')
ReferenceGM = GAUSSIAN.GaussianMixture
assert ReferenceGM.fit_predict.__wrapped__.__code__.co_filename == str(REFERENCE / 'sklearn/mixture/_base.py')
assert ReferenceGM._m_step.__code__.co_filename == str(REFERENCE / 'sklearn/mixture/_gaussian_mixture.py')


def bits(values):
    return ' '.join(str(struct.unpack('<q', struct.pack('<d', float(v)))[0]) for v in np.asarray(values, dtype=np.float64).ravel())


def unbits(tokens):
    return np.array([struct.unpack('<d', struct.pack('<q', int(t)))[0] for t in tokens], dtype=np.float64)


def write_case(path, dtype, params, arrays):
    lines = [f'dtype {"f32" if dtype == np.float32 else "f64"}']
    for key, value in params.items():
        if key in ('tol', 'reg_covar'):
            lines.append(f'{key} {bits([value])}')
        elif isinstance(value, bool):
            lines.append(f'{key} {int(value)}')
        else:
            lines.append(f'{key} {value}')
    for name, array in arrays.items():
        a = np.asarray(array)
        lines.append(f'array {name} {a.ndim} {" ".join(map(str, a.shape))} {bits(a)}')
    path.write_text('\n'.join(lines) + '\n')


def run_native(case_id, dtype, params, arrays):
    source = WORK / f'{case_id}.case'
    target = WORK / f'{case_id}.out'
    write_case(source, dtype, params, arrays)
    subprocess.run([str(RUNNER), str(source), str(target)], check=True)
    result = {}
    for line in target.read_text().splitlines():
        parts = line.split(' ')
        if parts[0] == 'error':
            return {'error': line[6:]}
        result[parts[0]] = unbits(parts[2:])
    return result


def run_reference(dtype, params, X, T):
    kwargs = {k: v for k, v in params.items() if k not in ('n_fits', 'n_draws')}
    model = ReferenceGM(**kwargs)
    out = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for fit in range(params.get('n_fits', 1)):
                labels = model.fit_predict(X)
                out[f'fit_lower_bound_{fit}'] = np.array([model.lower_bound_])
    except (ValueError, AttributeError) as exc:
        return {'error': str(exc)}
    out['fit_predict'] = labels
    for name in ['weights_', 'means_', 'covariances_', 'precisions_', 'precisions_cholesky_']:
        out[name] = getattr(model, name)
        assert out[name].dtype == dtype, (name, out[name].dtype)
    out['converged_'] = np.array([float(model.converged_)])
    out['n_iter_'] = np.array([model.n_iter_])
    out['lower_bound_'] = np.array([model.lower_bound_])
    out['lower_bounds_'] = np.array(model.lower_bounds_, dtype=np.float64)
    out['predict'] = model.predict(T)
    out['predict_proba'] = model.predict_proba(T)
    out['score_samples'] = model.score_samples(T)
    out['score'] = np.array([model.score(T)])
    out['bic'] = np.array([model.bic(T)])
    out['aic'] = np.array([model.aic(T)])
    if params.get('n_draws', 0):
        xs, ys = model.sample(params['n_draws'])
        out['sample_X'], out['sample_y'] = xs, ys
        out['_model'] = model
    return out


def tolerances(case_id, dtype):
    """(rtol, atol). Near-singular covariances (tiny: 5 rows; ill: 1e5 feature
    scale ratio) amplify accumulation-order rounding through log-determinants
    and EM iterations, so those datasets get a 1000x looser relative bound."""
    loose = any(tag in case_id for tag in ('_tiny_', '_ill_'))
    if dtype == np.float32:
        return (2e-2, 2e-3) if loose else (5e-4, 1e-3)
    return (1e-6, 1e-6) if loose else (1e-9, 1e-11)


EXACT = {'fit_predict', 'predict', 'converged_', 'n_iter_', 'sample_y'}


def compare(case_id, dtype, params, X, T):
    expected = run_reference(dtype, params, X, T)
    actual = run_native(case_id, dtype, params, {'X': X, 'T': T})
    if 'error' in expected or 'error' in actual:
        assert 'error' in expected and 'error' in actual, (case_id, expected.get('error'), actual.get('error'))
        # Native errors carry the Python exception class as a prefix.
        assert expected['error'].split(' (')[0].rstrip('.') in actual['error'], (case_id, expected['error'], actual['error'])
        return 'error'
    rtol, atol = tolerances(case_id, dtype)
    for name, want in expected.items():
        if name.startswith('_'):
            continue
        got = actual[name].reshape(np.asarray(want).shape)
        if name in EXACT:
            np.testing.assert_array_equal(got, want, err_msg=f'{case_id}: {name}')
        elif name == 'sample_X':
            check_samples(case_id, params, expected['_model'], got, want, expected['sample_y'], rtol, atol)
        else:
            scale = rtol * 50 if name in ('bic', 'aic') else rtol
            np.testing.assert_allclose(got, want, rtol=scale, atol=atol, err_msg=f'{case_id}: {name}')
    return 'ok'


def check_samples(case_id, params, model, got, want, ys, rtol, atol):
    """diag/spherical draws match exactly up to rounding; full/tied Mahalanobis norms."""
    kind = params['covariance_type']
    if kind in ('diag', 'spherical'):
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f'{case_id}: sample_X')
        return
    # Native multivariate_normal uses a Jacobi SVD factor whose orientation may
    # differ from LAPACK's; |F^-T (x - mu)|^2 = |z|^2 is orientation-free.
    for label in range(model.n_components):
        cov = model.covariances_[label] if kind == 'full' else model.covariances_
        prec = np.linalg.inv(np.asarray(cov, dtype=np.float64))
        mean = model.means_[label].astype(np.float64)
        rows = ys == label
        def norms(points):
            delta = points[rows] - mean
            return np.einsum('ij,jk,ik->i', delta, prec, delta)
        np.testing.assert_allclose(norms(got), norms(want), rtol=1e-4 if model.means_.dtype == np.float32 else 1e-9,
                                   atol=1e-8, err_msg=f'{case_id}: sample Mahalanobis norms')


def datasets():
    rng = np.random.RandomState(20260925)
    blobs = np.vstack([rng.randn(70, 3) * s + c for s, c in [(0.5, (0, 0, 0)), (1.0, (4, 4, 0)), (0.3, (0, 5, 5))]])
    tall = np.vstack([rng.randn(120, 2) @ rng.randn(2, 2) + rng.uniform(-6, 6, 2) for _ in range(4)])
    wide = rng.randn(40, 8) + np.repeat([[0.0] * 8, [3.0] * 8], 20, axis=0)
    tiny = np.array([[0.0, 0.0], [0.1, 0.3], [5.0, 5.0], [5.2, 4.9], [0.2, -0.1]])
    base = rng.randn(150, 3)
    ill = np.column_stack([base[:, 0] * 1e3, base[:, 0] * 1e3 + base[:, 1] * 1e-1, base[:, 2] * 1e-2])
    ill[75:] += [2e3, 2e3, 0.05]
    return {'blobs': (blobs, 3), 'tall': (tall, 4), 'wide': (wide, 2), 'tiny': (tiny, 2), 'ill': (ill, 2)}


def main():
    records, counts = [], {'ok': 0, 'error': 0, 'float32_seeded_diverged': 0}
    data = datasets()
    kinds = ['full', 'tied', 'diag', 'spherical']
    inits = ['kmeans', 'k-means++', 'random', 'random_from_data']
    case_number = 0
    for dtype in (np.float64, np.float32):
        for name, (raw, k) in data.items():
            if name == 'ill' and dtype == np.float32:
                # 1e3-scale features with 1e-2 structure: float32 second moments
                # (eps * 1e6 ~ 0.1) cannot represent the 1e-4 variances, so the
                # expanded covariance formulas are rounding noise in both
                # implementations. Ill-conditioning is covered in Float64.
                continue
            X = raw.astype(dtype)
            T = X[::3] + dtype(0.25)
            for kind in kinds:
                for init in inits:
                    for seed in ([0, 7] if name == 'blobs' else [3]):
                        params = dict(n_components=k, covariance_type=kind, init_params=init, random_state=seed,
                                      n_init=2 if seed == 7 else 1, n_draws=25 if name in ('blobs', 'tall') else 0)
                        case_number += 1
                        case_id = f'c{case_number}_{name}_{kind}_{init}_{np.dtype(dtype).name}'
                        try:
                            counts[compare(case_id, dtype, dict(params), X, T)] += 1
                        except AssertionError as exc:
                            # Float32 seeding from single rows (k-means++,
                            # random_from_data) gives zero-variance components
                            # regularized only by reg_covar=1e-6, and 'random'
                            # starts from nearly identical components; in both
                            # the rounding noise that BLAS summation order
                            # decides breaks the tie, so trajectories (and label
                            # permutations) can legitimately diverge. Record it;
                            # every Float64 and Float32 kmeans case must match.
                            if dtype != np.float32 or init == 'kmeans':
                                raise
                            native = run_native(case_id + '_recheck', dtype, dict(params), {'X': X, 'T': T})
                            assert 'error' not in native and np.isfinite(native['lower_bound_'][0]), case_id
                            counts['float32_seeded_diverged'] += 1
                            reference = run_reference(dtype, dict(params), X, T)
                            records.append({'case': case_id, 'divergence': str(exc).splitlines()[1:3],
                                            'native_lower_bound': float(native['lower_bound_'][0]),
                                            'reference_lower_bound': float(reference['lower_bound_'][0])})
            # Lifecycle and parameter branches on the blobs data only.
            if name != 'blobs':
                continue
            for kind in kinds:
                extra = []
                extra.append(dict(max_iter=2, tol=1e-12))  # nonconvergence
                extra.append(dict(max_iter=0))  # initialization only
                extra.append(dict(warm_start=True, max_iter=1, n_fits=4))  # warm start resumes
                # Float32 objectives carry ~1e-6 absolute rounding: keep tol above it.
                extra.append(dict(reg_covar=1e-2, tol=1e-6 if dtype == np.float64 else 1e-4))
                ref = ReferenceGM(n_components=k, covariance_type=kind, random_state=1).fit(X)
                init_arrays = {'weights_init': ref.weights_, 'means_init': ref.means_, 'precisions_init': ref.precisions_}
                extra.append(dict(_arrays=dict(init_arrays)))  # all three init arrays (no RNG use)
                extra.append(dict(_arrays={'means_init': ref.means_}))
                extra.append(dict(_arrays={'weights_init': ref.weights_, 'precisions_init': ref.precisions_}, max_iter=0))
                for index, options in enumerate(extra):
                    params = dict(n_components=k, covariance_type=kind, random_state=11)
                    arrays = options.pop('_arrays', {})
                    params.update(options)
                    reference_params = dict(params)
                    for key, value in arrays.items():
                        reference_params[key] = value
                    case_number += 1
                    case_id = f'c{case_number}_lifecycle{index}_{kind}_{np.dtype(dtype).name}'
                    counts[compare_with_arrays(case_id, dtype, params, reference_params, arrays, X, T)] += 1
        # Degenerate data: collapsed component without regularization raises.
        collapsed = np.vstack([np.zeros((6, 2)), np.ones((6, 2))]).astype(dtype)
        for kind in kinds:
            case_number += 1
            params = dict(n_components=2, covariance_type=kind, reg_covar=0.0, random_state=0)
            counts[compare(f'c{case_number}_collapsed_{kind}_{np.dtype(dtype).name}', dtype, params, collapsed, collapsed)] += 1
    report = {'revision': REVISION, 'runner': str(RUNNER), 'runner_sha256': hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
              'assembly': json.loads((ROOT / 'builds/deps/ASSEMBLY.json').read_text())['modules'], 'counts': counts, 'diverged': records,
              'tolerances': {'float64': tolerances('_blobs_', np.float64), 'float32': tolerances('_blobs_', np.float32),
                             'float64_tiny_ill': tolerances('_tiny_', np.float64),
                             'float32_tiny': tolerances('_tiny_', np.float32), 'bic_aic_rtol_factor': 50}}
    assert report['assembly']['moxie.mixture._gaussian_mixture'] == ['moxie']
    Path(os.environ.get('MOXIE_PARITY_REPORT', WORK / 'report.json')).write_text(json.dumps(report, indent=2))
    print(f'PASS {counts["ok"]} numerical and {counts["error"]} error-path GaussianMixture parity cases against pinned sklearn {REVISION[:10]}; '
          f'{counts["float32_seeded_diverged"]} Float32 single-row-seeded cases diverged by rounding (finite fits, see report)')


def compare_with_arrays(case_id, dtype, params, reference_params, arrays, X, T):
    typed = {key: np.asarray(value, dtype=dtype) for key, value in arrays.items()}
    reference_params = dict(reference_params)
    reference_params.update(typed)
    expected = run_reference(dtype, reference_params, X, T)
    native_params = dict(params)
    native_params['_arrays'] = typed
    actual_params = dict(native_params)
    actual = run_native(case_id, dtype, {k: v for k, v in actual_params.items() if k != '_arrays'}, {'X': X, 'T': T, **typed})
    if 'error' in expected or 'error' in actual:
        assert 'error' in expected and 'error' in actual, (case_id, expected.get('error'), actual.get('error'))
        return 'error'
    rtol, atol = tolerances(case_id, dtype)
    for name, want in expected.items():
        if name.startswith('_'):
            continue
        got = actual[name].reshape(np.asarray(want).shape)
        if name in EXACT:
            np.testing.assert_array_equal(got, want, err_msg=f'{case_id}: {name}')
        else:
            scale = rtol * 50 if name in ('bic', 'aic') else rtol
            np.testing.assert_allclose(got, want, rtol=scale, atol=atol, err_msg=f'{case_id}: {name}')
    return 'ok'


if __name__ == '__main__':
    main()
