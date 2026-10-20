"""Cyclic parity using the pinned Python AND Cython sources.

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
                cases.append(dict(id=f'{name}-{ratio}-{mode}-{bits}', bits=bits, X=X, y=target, Z=rng.normal(size=(4, p)).astype(dtype),
                    alpha=100. if mode == 'large-alpha' else .1, ratio=ratio, intercept=mode != 'no-intercept',
                    positive=mode == 'positive', warm=mode == 'warm', max_iter=1 if mode == 'one-iteration' else 1000, tol=1e-4, sw=sw))
folder = ROOT / 'builds/parity'
folder.mkdir(parents=True, exist_ok=True)
lines=[]
for c in cases:
    params={k: (int(c[k]) if isinstance(c[k], bool) else c[k]) for k in ('alpha','ratio','intercept','positive','warm','max_iter','tol')}
    arrays={k:c[k] for k in ('X','y','Z')}
    if c['sw'] is not None: arrays['sw']=c['sw']
    lines.append(io.case_line(c['id'],c['bits'],params,arrays))
fixtures=folder/'fixtures.txt'; fixtures.write_text('\n'.join(lines)+'\n')
output=subprocess.run([str(ROOT/'builds/parity_driver'),str(fixtures)],env=environment(ROOT),text=True,capture_output=True,check=True).stdout
(folder/'native.txt').write_text(output)
results=io.parse_results(output)
assert set(results)=={c['id'] for c in cases}
report=[]
for c in cases:
    got=results[c['id']]
    assert 'error' not in got,(c['id'],got)
    kwargs=dict(alpha=c['alpha'],fit_intercept=c['intercept'],positive=c['positive'],warm_start=c['warm'],max_iter=c['max_iter'],tol=c['tol'])
    model=reference.Lasso(**kwargs) if c['ratio']==1 else reference.ElasticNet(l1_ratio=c['ratio'],**kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(c['X'],c['y'],sample_weight=c['sw'])
        if c['warm']:
            caught.clear()
            model.alpha *= .5
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
(folder/'report.json').write_text(json.dumps(dict(revision=io.REVISION,sources=hashes,cases=len(cases),results=report),indent=2)+'\n')
print(f'Cyclic ElasticNet/Lasso parity passed: {len(cases)} cases; pinned sklearn Python and Cython {io.REVISION[:10]}')
