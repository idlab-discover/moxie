"""Differential checks against pinned _kernel_pca.py, including held-out rows."""
import sys
import warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'support'))
from oracle import ROOT, REVISION, load_pinned, run_native, assert_close
module = load_pinned('sklearn/decomposition/_kernel_pca.py')
count = 0


def compare(x, y, k=3, kernel='rbf', solver='dense', inverse=False, remove=False, gamma=.2, power=4, subspace=False):
    global count
    m = module.KernelPCA(k, kernel=kernel, eigen_solver=solver, fit_inverse_transform=inverse,
                        remove_zero_eig=remove, gamma=gamma, random_state=42, iterated_power=power if power >= 0 else "auto")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        z = m.fit_transform(x)
        test = m.transform(y)
    got = run_native(ROOT / 'builds/kernel_pca_driver', x.dtype, [x,y], k or 0, kernel, solver,
                     int(inverse), int(remove), 42, gamma if gamma is not None else -1, power)
    tol = 2e-7 if x.dtype == np.float64 else 8e-4
    label = f'{x.dtype} {x.shape} {kernel} {solver} {k=} {inverse=} {remove=}'
    assert_close(got.get('values'), m.eigenvalues_, tol, label+' values')
    # Signs are public; compare directly for these nondegenerate fixtures.
    # Zero-eigenvalue vectors have no unique basis, compare the embedding.
    assert_close(got['gamma'], m.gamma_, tol, label+' gamma')
    if subspace:
        fit = got['fit'].reshape(z.shape)
        train = got['transform'].reshape(z.shape)
        held = got['test'].reshape(test.shape)
        assert_close(fit @ fit.T, z @ z.T, tol, label+' embedding Gram')
        assert_close(train @ fit.T, m.transform(x) @ z.T, tol, label+' train subspace')
        assert_close(held @ fit.T, test @ z.T, tol, label+' held-out subspace')
    else:
        assert_close(got.get('fit'), z, tol, label+' fit')
        assert_close(got.get('transform'), m.transform(x), tol, label+' transform')
        assert_close(got.get('test'), test, tol, label+' held-out')
    positive = m.eigenvalues_ > 1e-6
    if positive.any() and not subspace:
        v = got['vectors'].reshape(m.eigenvectors_.shape)
        assert_close(v[:,positive], m.eigenvectors_[:,positive], tol, label+' vectors')
    if inverse:
        assert_close(got['inverse'], m.inverse_transform(z), tol*4, label+' inverse')
        assert_close(got['inverse_test'], m.inverse_transform(test), tol*4, label+' inverse held-out')
        assert_close(got['dual'], m.dual_coef_, tol*4, label+' dual')
    count += 1


rng = np.random.RandomState(4)
for dtype in (np.float64, np.float32):
    x = rng.uniform(.1, 2, (17, 5)).astype(dtype)
    y = rng.uniform(.1, 2, (6, 5)).astype(dtype)
    for kernel in ('linear','poly','rbf','cosine'):
        for solver in ('dense','arpack','randomized','auto'):
            compare(x,y,kernel=kernel,solver=solver)
        compare(x,y,kernel=kernel,inverse=True)
    repeated=(np.eye(4)-np.ones((4,4))/4).astype(dtype)
    compare(repeated,repeated[:2],k=3,kernel="precomputed",subspace=True)
    compare(x,y,k=1,gamma=None,power=-1)
    compare(x,y,k=None,remove=True)
    compare(x,y,k=30)
    compare(x@x.T,y@x.T,kernel='precomputed')
    compare(np.zeros((4,3),dtype=dtype),np.zeros((2,3),dtype=dtype),k=None,kernel='linear')
    compare(np.zeros((4,3),dtype=dtype),np.zeros((2,3),dtype=dtype),k=3,kernel='linear')
compare(rng.normal(size=(201,3)),rng.normal(size=(2,3)),k=2,kernel='linear',solver='auto')
print(f'PASS: {count} KernelPCA cases against local sklearn {REVISION}')
