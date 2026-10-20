"""Differential engine checks against the pinned, unmodified LIBSVM C++ source."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[4] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FORMS = ('c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr')
KINDS = ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')


def compile_reference():
    assert subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', str(REFERENCE), 'status', '--porcelain', '--', 'sklearn/svm/src/libsvm', 'sklearn/svm/src/newrand'], text=True)
    (ROOT / 'builds').mkdir(exist_ok=True)
    command = ['c++', '-O2', '-ffp-contract=off', '-std=c++17', '-I', str(REFERENCE / 'sklearn/svm/src/libsvm'), str(ROOT / 'tests/oracle.cpp'), '-o', str(ROOT / 'builds/oracle')]
    subprocess.run(command, check=True)
    return command


def parse(output):
    return {parts[0]: np.array([float(v) for v in parts[1:]]) for line in output.splitlines() if (parts := line.split())}


def run_reference(x, y, sw, cw, options, query=None):
    """Execute actual LIBSVM; options use KernelOptions names plus kernel fields.

    `formulation` and `kernel` are string names; gamma/degree/coef0 are resolved.
    Returned dual and decisions are 2D, support and n_support are integer arrays.
    """
    executable = ROOT / 'builds/oracle'
    if not executable.exists():
        compile_reference()
    x = np.asarray(x, dtype=float)
    query = x if query is None else np.asarray(query, dtype=float)
    n, p = x.shape
    m = len(query)
    form = FORMS.index(options['formulation'])
    kind = KINDS.index(options.get('kernel', 'rbf'))
    k = len(cw) if form < 2 else 0
    header = [n, p, m, k, form, kind, options.get('degree', 3), options.get('gamma', .4), options.get('coef0', .2), options.get('cache_mb', 200), options.get('tol', 1e-7), options.get('c', 1.7), options.get('nu', .35), options.get('epsilon', .12), int(options.get('shrinking', True)), options.get('max_iter', -1), int(options.get('probability', False)), options.get('seed', 42)]
    values = header + [*x.ravel(), *np.asarray(y).ravel(), *np.asarray(sw).ravel(), *np.asarray(cw).ravel(), *query.ravel()]
    result = parse(subprocess.check_output([str(executable)], input=' '.join(map(str, values)), text=True))
    for name in ('support', 'raw_support', 'n_support', 'iterations', 'status'):
        result[name] = result[name].astype(np.int64)
    result['dual'] = result['dual'].reshape(k-1 if form < 2 else 1, -1)
    result['decisions'] = result['decisions'].reshape(m, k*(k-1)//2 if form < 2 else 1)
    if result['probabilities'].size:
        result['probabilities'] = result['probabilities'].reshape(m, k)
    return result


def fixture(n, p, k, weighted, form):
    i, j = np.arange(n)[:, None], np.arange(p)[None, :]
    x = ((i*17+j*11) % 37-18)/13 + (i%k)*2.0 + (2.0 if form == 2 else 0.0)
    y = np.arange(n)%k if form < 2 else (np.ones(n) if form == 2 else ((np.arange(n)*13)%23-11)/7)
    w = np.where(np.arange(n)%7 == 0, -.5 if weighted == 2 else 0, .5+(np.arange(n)%5)*.4) if weighted else np.ones(n)
    cw = .7+np.arange(k)*.4 if weighted else np.ones(k)
    return x, y, w, cw


def check_layout(case, model):
    form, kind, n, p, k, weighted, shrinking, max_iter, probability, seed, cache = case
    support = model['support'].astype(int)
    counts = model['n_support'].astype(int)
    assert support.size == len(set(support)) and np.all((support >= 0) & (support < n))
    assert counts.sum() == support.size
    x, y, w, cw = fixture(n,p,k,weighted,form)
    assert np.all(w[support] > 0)
    gram = x @ x.T
    if kind == 1: gram = (.4*gram+.2)**3
    if kind == 2: gram = np.exp(-.4*np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(x*x,axis=1)[None,:]-2*gram,0))
    if kind == 3: gram = np.tanh(.4*gram+.2)
    training_gram = gram.astype(np.float32).astype(np.float64)
    dual = model['dual'].reshape(k-1 if form<2 else 1,-1)
    decisions = model['decisions'].reshape(n,-1)
    objectives = []
    if form >= 2:
        objective = .5*dual[0] @ gram[np.ix_(support,support)] @ dual[0]
        if form in (3,4): objective -= y[support] @ dual[0]
        if form == 3: objective += .12*np.abs(dual[0]).sum()
        objectives.append(float(objective))
        np.testing.assert_allclose(gram[:,support]@dual[0]-model['rho'][0],decisions[:,0],atol=2e-10,rtol=2e-10)
        if form in (3,4):
            assert abs(dual.sum()) < 1e-8
            assert np.all(np.abs(dual[0]) <= 1.7*w[support]+1e-8)
        if form == 2:
            assert np.all((dual[0]>=0)&(dual[0]<=w[support]+1e-8))
            np.testing.assert_allclose(dual.sum(), .35*w[w>0].sum(),atol=1e-8,rtol=0)
            if max_iter == -1 and kind != 3:
                free=(dual[0]>1e-7)&(dual[0]<w[support]-1e-7)
                np.testing.assert_allclose((training_gram[:,support]@dual[0]-model['rho'][0])[support[free]],0,atol=2e-5,rtol=0)
        if form == 4 and max_iter == -1 and kind != 3:
            free=(np.abs(dual[0])>1e-7)&(np.abs(dual[0])<1.7*w[support]-1e-7)
            residual=y[support]-(training_gram[:,support]@dual[0]-model['rho'][0])[support]
            tube=residual[free]*np.sign(dual[0,free])
            if tube.size:
                assert tube.min() >= -2e-5
                np.testing.assert_allclose(tube,tube.mean(),atol=2e-5,rtol=0)
        if form == 3 and max_iter == -1 and kind != 3:
            free=(np.abs(dual[0])>1e-7)&(np.abs(dual[0])<1.7*w[support]-1e-7)
            residual=y[support]-(training_gram[:,support]@dual[0]-model['rho'][0])[support]
            np.testing.assert_allclose(residual[free],.12*np.sign(dual[0,free]),atol=2e-5,rtol=0)
    else:
        offsets = np.r_[0,counts.cumsum()]
        for a in range(k):
            assert np.all(y[support[offsets[a]:offsets[a+1]]] == a)
        pair=0
        for a in range(k):
            for b in range(a+1,k):
                ia=slice(offsets[a],offsets[a+1]);ib=slice(offsets[b],offsets[b+1])
                coef=np.r_[dual[b-1,ia],dual[a,ib]]
                indices=np.r_[support[ia],support[ib]]
                objective=.5*coef @ gram[np.ix_(indices,indices)] @ coef
                if form == 0: objective -= np.abs(coef).sum()
                objectives.append(float(objective))
                np.testing.assert_allclose(gram[:,indices]@coef-model['rho'][pair],decisions[:,pair],atol=2e-10,rtol=2e-10)
                assert abs(coef.sum()) < 1e-8
                if form == 1 and max_iter == -1 and kind != 3 and coef.size:
                    # Nu rescaling hides the upper bound; the largest alpha/W
                    # is an upper bound on all free coefficients, saturated or not.
                    ratio=np.abs(coef)/w[indices]
                    free=(np.abs(coef)>1e-7)&(ratio<ratio.max()-1e-7)
                    margin=np.where(y[indices]==a,1,-1)*(training_gram[:,indices]@coef-model['rho'][pair])[indices]
                    np.testing.assert_allclose(margin[free],1,atol=2e-5,rtol=0)
                if form == 0:
                    bound=1.7*w[indices]*cw[y[indices].astype(int)]
                    assert np.all(np.abs(coef) <= bound+1e-8)
                    if max_iter == -1 and kind != 3:
                        margin=np.where(y[indices]==a,1,-1)*(training_gram[:,indices]@coef-model['rho'][pair])[indices]
                        free=(np.abs(coef)>1e-7)&(np.abs(coef)<bound-1e-7)
                        np.testing.assert_allclose(margin[free],1,atol=2e-5,rtol=0)
                pair+=1
    if model['probabilities'].size:
        probabilities=model['probabilities'].reshape(n,k)
        assert np.all((probabilities>=0)&(probabilities<=1))
        np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=2e-12,rtol=0)
    return objectives


def run():
    command=compile_reference()
    cases=[]
    for form in range(5):
        for kind in range(5):
            for weighted in (0,1):
                for shrinking in (0,1):
                    cases.append((form,kind,30,4,3 if form<2 else 2,weighted,shrinking,-1,0,42,.001))
    for form in range(5):
        cases.append((form,2,31,7,2,2,1,-1,0,42,10))
        for limit in (0,3,40):
            cases.append((form,2,31,7,2,1,1,limit,0,42,10))
    for form in (0,1):
        for k in (2,3,4):
            for seed in (0,42):
                for weighted in (0,1):
                    cases.append((form,2,40,5,k,weighted,1,-1,1,seed,10))
    evidence=[]
    for case in cases:
        arguments=list(map(str,case))
        expected=parse(subprocess.check_output([str(ROOT/'builds/oracle'),*arguments],text=True,timeout=300))
        actual=parse(subprocess.check_output([str(ROOT/'builds/driver'),*arguments],text=True,timeout=300))
        try:
            for name in ('support','n_support','status','iterations'):
                np.testing.assert_array_equal(actual[name],expected[name],err_msg=name)
            for name in ('dual','rho','decisions','prob_a','prob_b','probabilities'):
                np.testing.assert_allclose(actual[name],expected[name],rtol=2e-6,atol=2e-6,err_msg=name)
            np.testing.assert_allclose(check_layout(case,actual),check_layout(case,expected),atol=2e-6,rtol=2e-6,err_msg='dual objective')
        except AssertionError as error:
            raise AssertionError(f'case={case}: {error}') from error
        evidence.append({'case':case,'iterations_native':actual['iterations'].tolist(),'iterations_reference':expected['iterations'].tolist(),'native':{key:value.tolist() for key,value in actual.items()},'reference':{key:value.tolist() for key,value in expected.items()}})
    destination=ROOT/'docs/private/libsvm/solver-parity.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps({'reference':REVISION,'compile':command,'source_sha256':{str(p.relative_to(REFERENCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (REFERENCE/'sklearn/svm/src/libsvm').glob('*') if p.is_file()},'cases':evidence,'tolerance':'2e-6 parameter/decision tolerance at solver eps=1e-7, float32 cached columns; exact support/layout/status/iterations, independent reconstruction 2e-10, free-support KKT 2e-5 against Float32 training Gram (not Float64 prediction Gram)'},indent=2)+'\n')
    print(f'PASS: {len(cases)} actual pinned-LIBSVM solver, layout, KKT and probability comparisons')


if __name__=='__main__':
    run()
