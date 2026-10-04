"""Execute pinned local sklearn functions; support wheel supplies infrastructure."""
from reference import *
rng=np.random.default_rng(286041)
records=[]
def enc(a):
    a=np.asarray(a)
    return f'{a.shape[0]}:{a.shape[1] if a.ndim==2 else 0}:'+','.join(repr(float(x)) for x in a.ravel())
def nums(a): return ','.join(repr(float(x)) for x in np.atleast_1d(a))
def case(dtype,name,X,y,Z,yt,alpha=1,weights=None,gamma=None,degree=3,coef0=1,updates=None):
    X=np.asarray(X,dtype=dtype);y=np.asarray(y,dtype=dtype);Z=np.asarray(Z,dtype=dtype);yt=np.asarray(yt,dtype=dtype)
    alpha=np.asarray(alpha,dtype=dtype)
    if isinstance(weights,(list,np.ndarray)):weights=np.asarray(weights,dtype=np.float64)
    elif weights is not None:
        # Python scalar zero bypasses sklearn's array validator. np.float32(0)
        # instead enters that validator, which rejects all-zero arrays.
        weights=0.0 if weights == 0 else float(weights)
    metric=(lambda x,z,scale=1,factor=1,enabled=False,mode="":
            scale*(factor if enabled and mode == "scaled" else 1)*(x[0]+2*z[0]+2)) if name in ('callable', 'callable_params') else name
    model=KR(alpha=alpha,kernel=metric,gamma=gamma,degree=degree,coef0=coef0)
    if name == "callable_params":
        model.set_params(kernel_params={"scale": 1.75, "factor": 2, "enabled": True, "mode": "scaled"})
    failed=False
    try:
        model.fit(X,y,sample_weight=weights)
        if updates:
            model.set_params(**updates)
        coef=model.dual_coef_;pred=model.predict(Z);score=model.score(Z,yt)
        if np.isscalar(weights) and weights == 0:
            assert np.all(coef == 0) and np.all(pred == 0)
    except np.linalg.LinAlgError:
        if updates:
            raise
        failed=True;coef=np.zeros_like(y);pred=np.zeros_like(yt);score=0
    tolerance=2e-4 if dtype==np.float32 else 3e-9
    records.append('|'.join([str(np.dtype(dtype).itemsize*8),name,enc(X),enc(y),enc(Z),enc(yt),nums(alpha),'none' if weights is None else ('s:'+nums(weights) if np.isscalar(weights) else 'v:'+nums(weights)),'none' if gamma is None else str(gamma),str(degree),str(coef0),enc(coef),enc(pred),repr(float(score)),str(tolerance),str(int(failed))]))
    if updates:
        records[-1] += '|' + '|'.join(
            'keep' if key not in updates else
            'none' if updates[key] is None else
            'ignored' if key == 'kernel_params' else str(updates[key])
            for key in ('kernel', 'gamma', 'degree', 'coef0', 'alpha', 'kernel_params')
        )
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for dtype in [np.float32,np.float64]:
        for name in [*ns['PAIRWISE_KERNEL_FUNCTIONS'],'callable', 'callable_params']:
            for rank in [1,2,3]:
                n=7;t=rank-1
                X=rng.uniform(0,2,(n,3));Z=rng.uniform(0,2,(4,3))
                y=rng.normal(size=n if rank==1 else (n,t));yt=rng.normal(size=4 if rank==1 else (4,t))
                for mode in range(4):
                    a=2 if mode<2 or t<2 else ([2,2] if mode==2 else [1,2])
                    w=None if mode==0 else (2.0 if mode==1 else [0,.2,1,2,1,.5,1])
                    case(dtype,name,X,y,Z,yt,a,w,.3 if name=='chi2' or mode%2 else None,2.5 if mode==3 else 3,3)
        # tiny, default, zero/all-zero weights and singular/indefinite/nonsymmetric
        for K in [np.ones((3,3)),np.zeros((3,3)),np.diag([-1,1,0]),[[2,.2,0],[9,2,0],[3,2,1]],[[0,1,0],[2,0,0],[0,0,-1]]]:
            for a in [0,[0,0],[0,.1]]:
                for w in [None,0.0,[0,1,2]]:
                    case(dtype,'precomputed',K,[[1,2],[3,4],[5,6]],np.asarray(K)[:2],[[2,1],[1,2]],a,w)
        case(dtype,'linear',[[1]], [2], [[1],[2]],[2,4])
        case(dtype,'linear',[[0],[0]], [1,1], [[0],[0]],[1,1])
        case(dtype,'linear',[[0],[0]], [0,0], [[0],[0]],[0,0])
        for n in [16,32]:
            X=rng.normal(size=(n,5));Z=rng.normal(size=(5,5))
            case(dtype,'linear',X,rng.normal(size=(n,2)),Z,rng.normal(size=(5,2)),[.3,.7])
        # Dtype rank cutoff on either side of epsilon, avoiding equality boundary.
        for ratio in [.25,2,100]:
            eps=np.finfo(dtype).eps
            K=np.diag([-1,ratio*eps,0])
            case(dtype,'precomputed',K,[1,ratio*eps,3],K,[1,ratio*eps,3],0)
# Large nearby Float64 values must retain sklearn's staged distance arithmetic.
large = [[95306579.74822405], [95306575.48487124]]
case(np.float64, 'rbf', large, [1., 2.], large[::-1], [2., 1.], gamma=.1)
case(np.float64, 'rbf', large, [[1., 2.], [2., 1.]], large[::-1],
     [[2., 1.], [1., 2.]], gamma=.1, weights=[.5, 2.])
# Predict filters unused settings; fit validates all constructor parameters.
# Each mutation happens after a valid fit, so the learned coefficients stay fixed.
mutation_start = len(records)
for dtype in [np.float32, np.float64]:
    for name in [*ns['PAIRWISE_KERNEL_FUNCTIONS'], 'precomputed', 'callable']:
        X = [[1., .2, .3], [.2, 1., .4], [.3, .4, 1.]]
        for key, value in [('alpha', -1.), ('gamma', -1.),
                           ('degree', -1.), ('coef0', float('inf')),
                           ('kernel_params', {'unused': 19.})]:
            used = support.KERNEL_PARAMS.get(name, ())
            if key in used or (name == 'callable' and key == 'kernel_params'):
                continue
            case(dtype, name, X, [[1., 2.], [2., 0.], [0., 1.]],
                 X[:2], [[1., 2.], [2., 0.]], gamma=.3, updates={key: value})
    for updates in [{'kernel': 'rbf', 'gamma': .7}, {'gamma': None},
                    {'gamma': .8}, {'degree': 2.}, {'coef0': .5}]:
        case(dtype, 'poly', [[0., 1.], [1., 2.], [2., 0.]], [1., 2., 0.],
             [[.5, 1.], [1., .5]], [1., 0.], gamma=.3, updates=updates)
mutation_cases = len(records) - mutation_start
out=ROOT/'builds/parity';out.mkdir(parents=True,exist_ok=True)
(out/'fixtures.txt').write_text('\n'.join(records)+'\n')
(out/'metadata.json').write_text(json.dumps({'revision':REV,'seed':286041,'cases':len(records),'mutation_cases':mutation_cases,'numpy':np.__version__,'scipy':scipy.__version__,'support_sklearn':__import__('sklearn').__version__,'source_hashes':{p:hashlib.sha256((REFERENCE/p).read_bytes()).hexdigest() for p in paths},'adaptations':['score uses widened accumulation']},indent=2))
compile(ROOT/'tests/parity_runner.mojo',ROOT/'builds/parity_runner')
p=run(ROOT/'builds/parity_runner',out/'fixtures.txt',capture=True)
(out/'result.txt').write_text(p.stdout+p.stderr)
print(p.stdout,end='')

# Weighted score oracle: execute pinned R2, its target validation, and weight
# validation rather than allowing the installed support wheel to define them.
import sklearn.metrics._regression as regression_support
regression_ns = dict(vars(regression_support))
load_functions(REFERENCE / 'sklearn/metrics/_regression.py',
               ['r2_score', '_check_reg_targets',
                '_check_reg_targets_with_floating_dtype',
                '_assemble_fraction_of_explained_deviance'], regression_ns)
regression_ns['_check_sample_weight'] = validation_ns['_check_sample_weight']
score_records = []
for dtype in [np.float32, np.float64]:
    for n in [1, 2, 5]:
        X = np.eye(n, dtype=dtype)
        for rank in [1, 2, 3]:
            target = np.arange(n * max(rank - 1, 1), dtype=dtype)
            if rank > 1:
                target = target.reshape(n, rank - 1)
            for constant in [False, True]:
                y = np.ones_like(target) if constant else target
                for alpha in [0., 1.]:
                    model = KR(kernel='precomputed', alpha=alpha).fit(X, y)
                    weight_cases = [None, np.ones(n), np.arange(1, n + 1),
                                    np.zeros(n), np.full(n, -1.),
                                    np.full(n, np.nan), np.full(n, np.inf),
                                    np.ones(n + 1)]
                    if n > 1:
                        weight_cases += [np.r_[1., np.zeros(n - 1)],
                                         np.r_[-1., np.ones(n - 1) * 2],
                                         np.r_[-(n - 1.), np.ones(n - 1)]]
                    for weights in weight_cases:
                        weights = None if weights is None else weights.astype(dtype)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            try:
                                expected = repr(float(regression_ns['r2_score'](
                                    y, model.predict(X), sample_weight=weights)))
                            except (ValueError, ZeroDivisionError):
                                expected = 'error'
                        score_records.append('|'.join([
                            str(np.dtype(dtype).itemsize * 8), enc(X), enc(y),
                            str(alpha), 'none' if weights is None else nums(weights),
                            expected]))
(out / 'score-fixtures.txt').write_text('\n'.join(score_records) + '\n')
(out / 'score-metadata.json').write_text(json.dumps({
    'revision': REV, 'cases': len(score_records),
    'source_hashes': {p: hashlib.sha256((REFERENCE / p).read_bytes()).hexdigest()
                      for p in ['sklearn/metrics/_regression.py',
                                'sklearn/utils/validation.py']},
    'tolerances': {'float32': 2e-5, 'float64': 1e-12},
    'adaptation': 'Native score accumulates in Float64',
}, indent=2))
compile(ROOT / 'tests/score_parity_runner.mojo', ROOT / 'builds/score_parity_runner')
p = run(ROOT / 'builds/score_parity_runner', out / 'score-fixtures.txt', capture=True)
(out / 'score-result.txt').write_text(p.stdout + p.stderr)
print(p.stdout, end='')
