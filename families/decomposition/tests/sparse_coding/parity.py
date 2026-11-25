"""Dense coding versus the pinned _dict_learning.py public APIs."""
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'support'))
from oracle import ROOT, REVISION, load_pinned, run_native, assert_close, compile_pinned_extension
sys.path.insert(0, str(ROOT.parent / 'linear_model/tools'))
from pinned_coordinate import load as load_coordinate

coordinate, _ = load_coordinate(ROOT)
module = load_pinned('sklearn/decomposition/_dict_learning.py')
module.Lasso = coordinate.Lasso
module.orthogonal_mp_gram = load_pinned('sklearn/linear_model/_omp.py').orthogonal_mp_gram

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

count = 0
for dtype in (np.float32, np.float64):
    fixtures = {
        'tall': (rng.normal(size=(9, 24)), rng.normal(size=(5, 24))),
        'overcomplete': (rng.normal(size=(4, 3)), rng.normal(size=(7, 3))),
        'single-sample': (rng.normal(size=(1, 5)), rng.normal(size=(3, 5))),
        'singleton': ([[2.]], [[-4.]]),
        'zero-correlation': (np.zeros((3, 4)), np.eye(4)),
        'dependent': ([[2., -1.], [0., 0.]], [[1., 0.], [1., 0.], [0., 1.], [0., 0.]]),
        'tie-after-swap': ([[1., 1., 2.]], np.eye(3)),
        'small-correlation': ([[1e-9, -1e-9]], np.eye(2)),
    }
    for fixture, (x, dictionary) in fixtures.items():
        x = np.asarray(x, dtype=dtype)
        dictionary = np.asarray(dictionary, dtype=dtype)
        k = len(dictionary)
        for atom_count in dict.fromkeys((None, min(k, 2), k)):
            # Independent perturbations demonstrate that both borrowed inputs drive OMP.
            gram = (dictionary @ dictionary.T) * dtype(1.25)
            cov = dictionary @ (x + dtype(.125)).T
            for use_gram, use_cov in ((False, False), (True, False), (False, True), (True, True)):
                split = dtype == np.float64 and use_gram and use_cov
                model = module.SparseCoder(dictionary, transform_n_nonzero_coefs=atom_count,
                                           split_sign=split)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Orthogonal matching pursuit ended prematurely')
                    expected = module.sparse_encode(
                        x, dictionary, algorithm='omp', n_nonzero_coefs=atom_count,
                        gram=gram if use_gram else None, cov=cov if use_cov else None,
                    )
                    z = model.transform(x)
                    fit = model.fit_transform(x)
                assert expected.shape == (len(x), k)
                assert expected.dtype == z.dtype == dtype
                got = run_native(
                    ROOT / 'builds/sparse_coding_driver', dtype,
                    [x, dictionary, cov, gram, gram],
                    'omp', -1, 0, int(split), 0, int(use_cov), int(use_gram), 0,
                    atom_count or 0,
                )
                tol = 2e-5 if dtype == np.float32 else 2e-12
                for key, value in dict(functional=expected, code=z, inverse=model.inverse_transform(z),
                                       fit=fit, restored_fit=fit).items():
                    assert_close(got[key], value, tol, f'{dtype} {fixture=} {atom_count=} {use_gram=} {use_cov=} {key}')
                count += 1
print(f'PASS: {count} OMP coding cases against local sklearn {REVISION}')

least_angle = load_pinned('sklearn/linear_model/_least_angle.py')
least_angle.arrayfuncs = compile_pinned_extension('sklearn/utils/arrayfuncs.pyx')
module.Lars = least_angle.Lars
module.LassoLars = least_angle.LassoLars

drop_rng = np.random.RandomState(120)
for _ in range(4):
    drop_dictionary = drop_rng.randn(5, 7)
    drop_x = drop_rng.randn(1, 7)
_, _, path = least_angle.lars_path_gram(
    (drop_dictionary @ drop_x.T).ravel(), drop_dictionary @ drop_dictionary.T,
    n_samples=7, method='lasso',
)
assert np.any((np.abs(path[:, 1:]) < 1e-10) & (np.abs(path[:, :-1]) > 1e-5))

count = 0
for dtype in (np.float32, np.float64):
    fixtures = {
        'drop-and-reentry': (drop_x, drop_dictionary),
        'underdetermined': (rng.normal(size=(5, 4)), rng.normal(size=(7, 4))),
        'determined': (rng.normal(size=(7, 25)), rng.normal(size=(5, 25))),
        'single-sample': (rng.normal(size=(1, 5)), rng.normal(size=(3, 5))),
        'singleton': ([[2.]], [[-4.]]),
        'positive-endpoint': ([[3., -2., 0.]], np.eye(3)[:2]),
        'zero': (np.zeros((2, 3)), np.eye(3)),
        'dependent': ([[3., -2.]], [[1., 0.], [1., 0.], [0., 1.], [0., 0.]]),
        'tie': ([[1., -1., 2.]], np.eye(3)),
    }
    for fixture, (x, dictionary) in fixtures.items():
        x = np.asarray(x, dtype=dtype)
        dictionary = np.asarray(dictionary, dtype=dtype)
        k = len(dictionary)
        modes = [
            ('lars', None, False, 0, None),
            ('lars', None, False, 0, min(k, 3)),
            ('lasso_lars', None, False, 1000, None),
            ('lasso_lars', .35, True, 1000, None),
            ('lasso_lars', .35, False, 1, None),
            ('lasso_lars', .35, True, 0, None),
            ('lasso_lars', 10., False, 1000, None),
        ]
        # For dense underdetermined Float32 Gram inputs, requesting knots beyond
        # the rank makes sklearn's active set sensitive to round-off in pivots.
        # Exercise oversized counts on full-rank and exactly dependent fixtures.
        if fixture != 'underdetermined':
            modes.append(('lars', None, False, 0, k + 1))
        if fixture == 'drop-and-reentry':
            modes += [('lasso_lars', alpha, False, 1000, None) for alpha in (.5, .04, 0.)]
        for algorithm, alpha, positive, max_iter, atom_count in modes:
            for use_gram, use_cov in ((False, False), (True, True)):
                gram = dictionary @ dictionary.T * dtype(1.25)
                cov = dictionary @ (x + dtype(.125)).T
                split = dtype == np.float64 and use_gram
                model = module.SparseCoder(
                    dictionary, transform_algorithm=algorithm, transform_alpha=alpha,
                    transform_max_iter=max_iter, transform_n_nonzero_coefs=atom_count,
                    positive_code=positive, split_sign=split,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    expected = module.sparse_encode(
                        x, dictionary, algorithm=algorithm, alpha=alpha, positive=positive,
                        max_iter=max_iter, n_nonzero_coefs=atom_count,
                        gram=gram if use_gram else None, cov=cov if use_cov else None,
                    )
                    z = model.transform(x)
                    fit = model.fit_transform(x)
                assert expected.shape == (len(x), k)
                assert expected.dtype == z.dtype == dtype
                got = run_native(
                    ROOT / 'builds/sparse_coding_driver', dtype, [x, dictionary, cov, gram, x],
                    algorithm, -1 if alpha is None else alpha, int(positive), int(split),
                    max_iter, int(use_cov), int(use_gram), 1, atom_count or 0,
                )
                tol = 1e-4 if dtype == np.float32 else 2e-10
                for key, value in dict(functional=expected, code=z, inverse=model.inverse_transform(z),
                                       fit=fit, restored_fit=fit).items():
                    assert_close(
                        got[key], value, tol,
                        f'{dtype} {fixture=} {algorithm=} {alpha=} {positive=} '
                        f'{max_iter=} {atom_count=} {use_gram=} {key}',
                    )
                count += 1
print(f'PASS: {count} least-angle coding cases against local sklearn {REVISION}')
