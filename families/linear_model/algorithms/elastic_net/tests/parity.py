"""Dense coordinate parity using the pinned Python AND Cython sources.

Float64 1e-8; Float32 3e-4 for BLAS reduction-order differences. Absolute
error scales with max(1, |y|)^2. Float64 iteration counts must be exact;
Float32 threshold-sensitive counts are recorded with coefficient/gap agreement.
"""
from pathlib import Path
import json, subprocess, sys, warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'tools'))
import parity_io as io
from pinned_coordinate import load
from childcheck import environment
reference, hashes = load(ROOT)
rng = np.random.default_rng(20260930)
cases = []
for bits, dtype in ((64, np.float64), (32, np.float32)):
    for name, n, p, outputs in [('tall', 30, 5, 1), ('wide', 8, 15, 1), ('multi', 30, 4, 3), ('tiny', 2, 1, 1), ('constant', 8, 3, 1)]:
        X = rng.normal(size=(n, p)).astype(dtype)
        if name == 'constant': X[:] = 1
        y = (X @ rng.normal(size=(p, outputs)) + rng.normal(size=(n, outputs)) * .1 + 2).astype(dtype)
        if outputs == 1: y = y[:, 0]
        for ratio in (0., .5, 1.):
            for mode in ('plain', 'weighted', 'positive', 'no-intercept', 'warm', 'one-iteration', 'zero-target', 'large-alpha', 'column'):
                if mode == 'column' and outputs != 1: continue
                target = np.zeros_like(y) if mode == 'zero-target' else y
                if mode == 'column': target = target.reshape(-1, 1)
                sw = rng.uniform(.2, 2, n).astype(dtype) if mode == 'weighted' else None
                if sw is not None and n > 2: sw[0] = 0
                cases.append(dict(id=f'{name}-{ratio}-{mode}-{bits}', bits=bits, name=name, mode=mode, X=X, y=target, Z=rng.normal(size=(4, p)).astype(dtype),
                    alpha=100. if mode == 'large-alpha' else .1, ratio=ratio, intercept=mode != 'no-intercept',
                    positive=mode == 'positive', warm=mode == 'warm', max_iter=1 if mode == 'one-iteration' else 1000, tol=1e-4, sw=sw))
path_cases = []
for bits, dtype in ((64, np.float64), (32, np.float32)):
    for name, n, p in [('tall', 31, 5), ('wide', 9, 13), ('singleton', 5, 1)]:
        X = rng.normal(size=(n, p)).astype(dtype)
        X = X if p == 1 else np.column_stack((X[:, :-1], np.zeros(n, dtype=dtype)))
        y = (X @ rng.normal(size=p) + rng.normal(size=n) * .1 + 1).astype(dtype)
        for ratio, mode in [
            (0.5, 'default'), (1., 'default'), (0.3, 'explicit'),
            (1., 'explicit'), (0., 'explicit'), (1., 'one'),
            (0.5, 'positive'), (1., 'positive'), (1., 'negative-correlations'),
            (0.5, 'zero-target'), (1., 'one-iteration'), (0.3, 'increasing'),
        ]:
            design, target = X, y
            count, eps, max_iter = 100, 1e-3, 1000
            alphas, initial = None, None
            positive = mode in ('positive', 'negative-correlations')
            if mode in ('explicit', 'one-iteration'):
                alphas = np.array([0.15, 4.0, 0.4, 0.15])
                initial = rng.normal(size=p).astype(dtype)
                if p > 1:
                    initial[-1] = 0
            if mode == 'one':
                count = 1
            if mode == 'positive':
                count = 11
            if mode == 'negative-correlations':
                design, target, count = np.abs(X), -np.abs(y), 7
            if mode == 'zero-target':
                target, count = np.zeros_like(y), 7
            if mode == 'one-iteration':
                max_iter = 1
            if mode == 'increasing':
                count, eps = 7, 3.
            path_cases.append(dict(
                id=f'path-{name}-{ratio}-{mode}-{bits}', bits=bits, mode=mode,
                X=design, y=target, ratio=ratio, count=count, eps=eps,
                alphas=alphas, initial=initial, max_iter=max_iter, positive=positive,
            ))

# Short sweeps expose the sampled coordinate order. Warm and weighted fits,
# positive constraints, pure L2 (no screening) and independent targets reuse it.
cases += [
    dict(case, id=f"random-{seed}-{case['id']}", seed=seed)
    for case in cases
    for seed in (0, 4294967295)
    if case['name'] in ('tall', 'multi')
    and case['mode'] in ('one-iteration', 'warm', 'weighted', 'positive')
]
path_cases += [
    dict(case, id=f"random-{seed}-{case['id']}", seed=seed)
    for case in path_cases
    for seed in (0, 4294967295)
    if case['mode'] in ('one-iteration', 'positive', 'explicit')
]

# Exercise computed and supplied Gram fits, including both outcomes when
# centering invalidates the caller's matrix and the weighted gallery workflow.
gram_cases = []
for c in cases:
    if c['mode'] not in ('weighted', 'positive', 'no-intercept', 'warm', 'one-iteration', 'zero-target'):
        continue
    gram_cases.append(dict(c, id='computed-gram-' + c['id'], precompute=True))
    if c['name'] == 'constant' or c['mode'] not in ('no-intercept', 'weighted', 'warm', 'one-iteration'):
        continue
    X = c['X']
    sw = c['sw']
    weights = np.ones(X.shape[0], dtype=X.dtype) if sw is None else sw * (len(sw) / sw.sum())
    if c['mode'] in ('weighted', 'warm', 'one-iteration'):
        # Use symmetric pairs to keep the weighted mean inside sklearn's
        # absolute 1e-8 check, even with its Float32 reduction order.
        X = np.stack([X, -X], axis=1).reshape(-1, X.shape[1])
        y = np.repeat(c['y'], 2, axis=0)
        y[1::2] = -y[1::2]
        weights = np.repeat(weights, 2)
        sw = np.repeat(sw, 2) if sw is not None else None
    else:
        y = c['y']
    scaled = X * np.sqrt(weights)[:, None]
    gram_cases.append(dict(c, id='supplied-gram-' + c['id'], X=X, y=y, sw=sw, gram=scaled.T @ scaled))
    if c['mode'] == 'warm' and 'seed' not in c and c['name'] in ('tall', 'wide'):
        X = c['X'] + 2
        gram_cases.append(dict(c, id='replaced-gram-' + c['id'], X=X, gram=X.T @ X))
cases += gram_cases

# Replay varied paths with both Gram sources. Supply cross products in
# explicit-Gram cases with a generated alpha grid.
gram_path_cases = []
for case in path_cases:
    gram_path_cases.append(dict(case, id='computed-gram-' + case['id'], precompute=True))
    X = case['X']
    gram_path_cases.append(dict(
        case, id='supplied-gram-' + case['id'], gram=X.T @ X,
        Xy=X.T @ case['y'] if case['alphas'] is None else None,
    ))
    if case['mode'] == 'default':
        gram_path_cases.append(dict(case, id='auto-' + case['id'], precompute='auto'))
path_cases += gram_path_cases

folder = ROOT / 'builds/parity'
folder.mkdir(parents=True, exist_ok=True)
lines=[]
for c in cases:
    params={k: (int(c[k]) if isinstance(c[k], bool) else c[k]) for k in ('alpha','ratio','intercept','positive','warm','max_iter','tol')}
    if 'seed' in c: params['seed'] = c['seed']
    if c.get('precompute'): params['precompute'] = 1
    arrays={k:c[k] for k in ('X','y','Z','gram') if k in c}
    if c['sw'] is not None: arrays['sw']=c['sw']
    lines.append(io.case_line(c['id'],c['bits'],params,arrays))
for case in path_cases:
    params = {k: case[k] for k in ('ratio', 'count', 'eps', 'max_iter')}
    params.update(path=1, positive=int(case['positive']))
    if 'seed' in case: params['seed'] = case['seed']
    if case.get('precompute') == 'auto':
        params['auto'] = 1
    elif case.get('precompute'):
        params['precompute'] = 1
    arrays = {
        k: case[k] for k in ('X', 'y', 'alphas', 'initial', 'gram', 'Xy')
        if case.get(k) is not None
    }
    lines.append(io.case_line(case['id'], case['bits'], params, arrays))
fixtures=folder/'fixtures.txt'; fixtures.write_text('\n'.join(lines)+'\n')
output=subprocess.run([str(ROOT/'builds/parity_driver'),str(fixtures)],env=environment(ROOT),text=True,capture_output=True,check=True).stdout
(folder/'native.txt').write_text(output)
results=io.parse_results(output)
assert len(results) == len(cases) + len(path_cases)
assert set(results) == {c['id'] for c in cases + path_cases}
report=[]
for c in cases:
    got=results[c['id']]
    assert 'error' not in got,(c['id'],got)
    kwargs=dict(alpha=c['alpha'],fit_intercept=c['intercept'],positive=c['positive'],warm_start=c['warm'],max_iter=c['max_iter'],tol=c['tol'])
    kwargs['precompute'] = c.get('gram', c.get('precompute', False))
    if 'seed' in c: kwargs.update(selection='random', random_state=c['seed'])
    if c['ratio'] == 1:
        model = reference.Lasso()
    else:
        model = reference.ElasticNet()
        kwargs['l1_ratio'] = c['ratio']
    model.set_params(**kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(c['X'],c['y'],sample_weight=c['sw'])
        if c['warm']:
            caught.clear()
            model.set_params(alpha=model.alpha * .5)
            model.fit(c['X'],c['y'],sample_weight=c['sw'])
    t=1e-8 if c['bits']==64 else 3e-4
    scale=max(1.,float(np.max(np.abs(c['y']))))
    errors={}
    for key,expected in [('coef',model.coef_),('intercept',np.broadcast_to(np.atleast_1d(model.intercept_), got['intercept'].shape)),('pred',model.predict(c['Z'])),('gap',np.atleast_1d(model.dual_gap_))]:
        errors[key]=io.close(got[key],expected,t,t*scale**2,c['id']+' '+key)
    errors['score']=io.close(got['score'], model.score(c['X'],c['y']), t, t*scale**2, c['id']+' score')
    warning_count=sum(issubclass(w.category,ConvergenceWarning) for w in caught)
    assert got['unconverged']==warning_count,(c['id'],got['unconverged'],warning_count)
    iterations=np.atleast_1d(model.n_iter_)
    if c['bits']==64:
        assert np.array_equal(got['iterations'],iterations),(c['id'],got['iterations'],iterations)
    report.append(dict(case=c['id'],**errors,native_iterations=got['iterations'].tolist(),reference_iterations=iterations.tolist()))
path_report = []
for case in path_cases:
    ident = case['id']
    got = results[ident]
    assert 'error' not in got, (ident, got)
    kwargs = dict(
        eps=case['eps'], precompute=case.get('gram', case.get('precompute', False)),
        Xy=case.get('Xy'), positive=case['positive'],
        max_iter=case['max_iter'], return_n_iter=True,
        alphas=case['count'] if case['alphas'] is None else case['alphas'],
    )
    if 'seed' in case:
        kwargs.update(selection='random', random_state=case['seed'])
    if case['initial'] is not None:
        kwargs['coef_init'] = case['initial'].copy()
    if case['ratio'] == 1:
        path = reference.lasso_path
    else:
        path = reference.enet_path
        kwargs['l1_ratio'] = case['ratio']
    io.pinned_code(path, 'sklearn/linear_model/_coordinate_descent.py')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        alphas, coefs, gaps, iterations = path(case['X'], case['y'], **kwargs)

    is_f64 = case['bits'] == 64
    scale = max(1., float(np.max(np.abs(case['y']))))
    tol = 1e-8 if is_f64 else 3e-4
    errors = {
        'alphas': io.close(got['alphas'], alphas, 2e-9 if is_f64 else 2e-6, 1e-17, ident + ' alphas'),
        'coefs': io.close(got['coefs'], coefs, tol, tol * scale, ident + ' coefs'),
        'dual_gaps': io.close(got['dual_gaps'], gaps, tol, tol * scale**2, ident + ' dual_gaps'),
    }
    counts = got['iterations']
    assert counts.shape == (len(alphas),), (ident, counts.shape)
    assert np.all((0 <= counts) & (counts <= case['max_iter'])), (ident, counts)
    if is_f64:
        np.testing.assert_array_equal(counts, iterations, err_msg=ident)
    path_report.append(dict(
        case=ident, **errors, native_iterations=got['iterations'].tolist(),
        reference_iterations=iterations,
    ))
(folder/'path-report.json').write_text(
    json.dumps(dict(revision=io.REVISION, sources=hashes, results=path_report), indent=2) + '\n'
)
print(f'Dense ElasticNet/Lasso path parity passed: {len(path_cases)} cases ({len(gram_path_cases)} Gram or auto)')
(folder/'report.json').write_text(json.dumps(dict(revision=io.REVISION,sources=hashes,cases=len(cases),results=report),indent=2)+'\n')
print(f'Dense ElasticNet/Lasso parity passed: {len(cases)} cases ({len(gram_cases)} Gram); pinned sklearn Python and Cython {io.REVISION[:10]}')
