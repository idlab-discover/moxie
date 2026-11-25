"""Dense coding versus the pinned _dict_learning.py public APIs."""
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'support'))
from oracle import ROOT, REVISION, load_pinned, run_native, assert_close
sys.path.insert(0, str(ROOT.parent / 'linear_model/tools'))
from pinned_coordinate import load as load_coordinate

coordinate, _ = load_coordinate(ROOT)
module = load_pinned('sklearn/decomposition/_dict_learning.py')
module.Lasso = coordinate.Lasso

rng=np.random.RandomState(12)
count=0
for dtype in (np.float32,np.float64):
    for n,p,k in ((9,5,3),(4,3,7),(1,1,1)):
        x=rng.normal(size=(n,p)).astype(dtype)
        d=rng.normal(size=(k,p)).astype(dtype)
        for alpha in (None,0.,.2,5.):
            for positive in (False,True):
                for split in ((False,True) if dtype==np.float64 else (False,)):
                    # Perturbed precomputed covariance verifies it is actually consumed.
                    cov=(d@x.T+.125).astype(dtype)
                    for use_cov in (False,True):
                        expected=module.sparse_encode(x,d,algorithm='threshold',alpha=alpha,positive=positive,
                                                     cov=cov if use_cov else None)
                        m=module.SparseCoder(d,transform_algorithm='threshold',transform_alpha=alpha,
                                            positive_code=positive,split_sign=split)
                        z=m.transform(x)
                        got=run_native(ROOT/'builds/sparse_coding_driver',dtype,[x,d,cov,d,d],
                                       'threshold', -1 if alpha is None else alpha,
                                       int(positive), int(split), 1000, int(use_cov), 0, 0)
                        tol=2e-6 if dtype==np.float32 else 2e-12
                        fit = m.fit_transform(x)
                        for key,value in dict(functional=expected,code=z,inverse=m.inverse_transform(z),
                                              fit=fit,restored_fit=fit).items():
                            assert_close(got[key],value,tol,f'{dtype} {(n,p,k)} {alpha=} {positive=} {split=} {use_cov=} {key}')
                        count+=1
print(f'PASS: {count} threshold coding cases against local sklearn {REVISION}')

count = 0
for dtype in (np.float32, np.float64):
    for fixture, n, p, k in (('tall', 9, 6, 3), ('wide', 4, 3, 7),
                             ('single-sample', 1, 5, 3), ('singleton', 1, 1, 1),
                             ('zero', 3, 4, 4), ('orthogonal', 6, 4, 4)):
        x = rng.normal(size=(n, p)).astype(dtype)
        d = rng.normal(size=(k, p)).astype(dtype)
        if fixture == 'zero':
            x.fill(0)
            d[0].fill(0)
        if fixture == 'orthogonal':
            d = np.eye(k, dtype=dtype)
        for positive in (False, True):
            for mode in ('defaults', 'gram', 'warm-one-step', 'large-alpha', 'rank-deficient', 'unpenalized'):
                if mode == 'unpenalized' and fixture != 'orthogonal':
                    continue
                dictionary = d.copy()
                if mode == 'rank-deficient':
                    dictionary[-1].fill(0)
                if mode == 'warm-one-step':
                    initial = rng.uniform(0, .3, size=(n, k)).astype(dtype)
                else:
                    initial = None
                alpha = {'defaults': None, 'large-alpha': 10., 'unpenalized': 0.}.get(mode, .35)
                max_iter = 1 if mode == 'warm-one-step' else 1000
                gram = dictionary @ dictionary.T
                # Perturb the supplied system in a bounded one-sweep case to verify
                # that it drives the updates. The reference calls Lasso with check_input=False.
                if mode == 'warm-one-step':
                    gram *= dtype(1.25)
                use_gram = mode in ('gram', 'warm-one-step')
                # The reference Lasso computes its own correlations for this algorithm.
                cov = np.full((k, n), 50, dtype=dtype)
                split = dtype == np.float64 and mode == 'gram' and n > 1
                model = module.SparseCoder(dictionary, transform_algorithm='lasso_cd',
                    transform_alpha=alpha, transform_max_iter=max_iter,
                    positive_code=positive, split_sign=split)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    warnings.filterwarnings('ignore', message='With alpha=0')
                    warnings.filterwarnings('ignore', message='Coordinate descent with no regularization')
                    expected = module.sparse_encode(
                        x, dictionary, algorithm='lasso_cd', alpha=alpha, cov=cov,
                        gram=gram if use_gram else None,
                        init=initial.copy() if initial is not None else None,
                        positive=positive, max_iter=max_iter,
                    )
                    z = model.transform(x)
                    fit = model.fit_transform(x)
                assert expected.dtype == z.dtype == dtype
                reconstruction = model.inverse_transform(np.atleast_2d(z))
                got = run_native(
                    ROOT / 'builds/sparse_coding_driver', dtype,
                    [x, dictionary, cov, gram, initial if initial is not None else gram],
                    'lasso_cd', -1 if alpha is None else alpha, int(positive),
                    int(split), max_iter, 1, int(use_gram), int(initial is not None),
                )
                tol = 2e-5 if dtype == np.float32 else 3e-8
                for key, value in dict(functional=expected, code=z, inverse=reconstruction,
                                       fit=fit, restored_fit=fit).items():
                    assert_close(got[key], value, tol, f'{dtype} {fixture=} {positive=} {mode=} {key}')
                count += 1
print(f'PASS: {count} Lasso CD coding cases against local sklearn {REVISION}')
