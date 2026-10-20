"""NCA objective, initialization, optimization, and pinned original test evidence.

Run with a Python containing the compiled local sklearn revision (see docs/nca.md).
All native numerical outputs are produced by evaluate_nca, never Python runtime.
"""
import inspect
import itertools
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import check_grad
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parents[2] / 'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'],text=True).strip() == REV
for path, cls in [('sklearn/neighbors/_nca.py', NCA),
                  ('sklearn/decomposition/_pca.py', PCA),
                  ('sklearn/discriminant_analysis.py', LDA)]:
    expected = subprocess.check_output(['git','-C',str(REF),'show',f'{REV}:{path}'])
    assert (REF/path).read_bytes() == expected
    assert Path(inspect.getfile(cls)).read_bytes() == expected


def native(X, y, mode='identity', k=None, A=None, iterations=50, tol=1e-5):
    X = np.asarray(X)
    n, d = X.shape
    k = d if k is None else k
    A = np.eye(k,d) if A is None else A
    values = [n,d,*X.ravel(),*y,*A.ravel()]
    with tempfile.NamedTemporaryFile(mode='w',dir=ROOT/'builds') as f:
        f.write('\n'.join(format(float(v),'.17g') for v in values)); f.flush()
        output = subprocess.check_output([str(ROOT/'builds/evaluate_nca'), f.name,
            'f32' if X.dtype == np.float32 else 'f64',mode,str(k),str(iterations),str(tol)],text=True)
    result = {}
    for line in output.splitlines():
        key, *values = line.split()
        result.setdefault(key,[]).extend(map(float,values))
    result = {key:np.array(value) for key,value in result.items()}
    if 'A' in result:
        result['A'] = result['A'].reshape(tuple(result['S'].astype(int)))
        if 'T' in result: result['T'] = result['T'].reshape(n,-1)
    return result


def objective(X,y,A):
    ref = NCA(); ref.n_iter_ = 1
    return ref._loss_grad_lbfgs(A.ravel(), X, y[:,None] == y[None,:], -1.)


def main():
    rng = np.random.RandomState(1729)
    counts = dict(gradient=0, initialization=0, optimization=0, trajectory=0)
    for dtype, shape, k in itertools.product([np.float32,np.float64],[(9,3),(3,7),(17,5)],[1,2]):
        X = rng.normal(size=shape).astype(dtype)
        y = np.arange(shape[0]) % 3
        A = rng.normal(size=(k,shape[1]))
        got = native(X,y,'gradient',k,A)
        loss, grad = objective(X.astype(float),y,A)
        assert_allclose(got['L'],loss,atol=3e-12,rtol=3e-12)
        assert_allclose(got['G'],grad,atol=3e-11,rtol=3e-11)
        # Independent finite differences of the pinned original objective.
        error = check_grad(lambda a: objective(X.astype(float),y,a.reshape(A.shape))[0],
                           lambda a: native(X,y,'gradient',k,a.reshape(A.shape))['G'], A.ravel())
        assert error < 3e-5, error
        counts['gradient'] += 1
    for dtype, shape, k in itertools.product([np.float32,np.float64],[(12,3),(7,9),(25,5)],[1,2]):
        X = rng.normal(size=shape).astype(dtype)
        y = np.arange(shape[0]) % 3
        for mode in ['pca','lda']:
            got = native(X,y,mode+'_init',k)['A']
            ref = NCA(n_components=k,init=mode,random_state=42)
            ref.random_state_ = np.random.RandomState(42)
            expected = ref._initialize(X.astype(float),y,mode)
            # SVD signs are arbitrary for LDA; PCA applies sklearn's sign rule.
            if mode == 'lda':
                assert_allclose(got.T@got,expected.T@expected,atol=2e-9,rtol=2e-9)
            else:
                assert_allclose(got,expected,atol=2e-10,rtol=2e-10)
            counts['initialization'] += 1
    for dtype, mode, k in itertools.product([np.float32,np.float64],['identity','random','pca','lda','auto','array'],[1,2]):
        X = rng.normal(scale=.25,size=(18,3))
        y = np.arange(18)%3
        X[:,0] += 2*y
        X = X.astype(dtype)
        A = rng.normal(size=(k,3))
        got = native(X,y,mode,k,A,iterations=100,tol=1e-8)
        ref = NCA(n_components=k,init=A if mode == 'array' else mode,
                  random_state=42,max_iter=100,tol=1e-8).fit(X.astype(float),y)
        loss,_ = objective(X.astype(float),y,got['A'])
        reference_loss,_ = objective(X.astype(float),y,ref.components_)
        # These separated fixtures include SVD sign/basis differences.
        # Trajectory parity from identical initial matrices is checked below.
        assert -loss > 17.99, (dtype,mode,k,loss)
        assert abs(loss-reference_loss) < .01, (mode,loss,reference_loss)
        assert_allclose(got['T'],X.astype(float)@got['A'].T,rtol=3e-12,atol=3e-12)
        assert 1 <= got['I'][0] <= 101
        counts['optimization'] += 1
    # Mixed-label, nonseparable problems compare every accepted iterate.
    # Explicit initial matrices isolate the solver from SVD sign conventions.
    trajectory_rng = np.random.RandomState(857)
    for dtype, scale, k in itertools.product(
            [np.float32, np.float64], [.1, 1., 3.], [1, 2]):
        X = (scale * trajectory_rng.normal(size=(15, 3))).astype(dtype)
        y = np.arange(15) % 3
        A = trajectory_rng.normal(size=(k, 3))
        got = native(X, y, 'array', k, A, iterations=60, tol=1e-7)
        callbacks, indices = [], []
        def callback(a, i):
            callbacks.append(a.copy())
            indices.append(i)
        ref = NCA(k, init=A, max_iter=60, tol=1e-7,
                  callback=callback).fit(X.astype(float), y)
        assert got['I'][0] == ref.n_iter_
        assert np.array_equal(got['CI'], indices)
        # BLAS reductions and compact-vs-two-loop Hessian rounding accumulate
        # along nonconvex trajectories; tolerance remains on coordinates, not
        # merely objective scores (max observed absolute error below 2e-6).
        assert_allclose(got['C'].reshape(-1, k*3), callbacks,
                        rtol=3e-7, atol=3e-6)
        assert_allclose(got['A'], ref.components_, rtol=3e-7, atol=3e-6)
        counts['trajectory'] += 1
    # Auto initialization: LDA, PCA, identity, and the n_samples boundary.
    for n,d,c,k in [(12,4,3,2),(12,4,2,2),(12,4,2,4),(3,7,2,4)]:
        X = rng.normal(size=(n,d)); y = np.arange(n)%c
        got = native(X,y,'auto',k,iterations=1,tol=1e10)['A']
        ref = NCA(k,init='auto',random_state=42)
        ref.random_state_ = np.random.RandomState(42)
        expected = ref._initialize(X,y,'auto')
        assert_allclose(got.T@got,expected.T@expected,atol=2e-9,rtol=2e-9)
        counts['initialization'] += 1
    # Original test_nca.py toy geometry and degenerate objective cases.
    X = np.array([[0.,0],[0,1],[2,0],[2,1]])
    y = np.array([0,1,0,1])
    got = native(X,y)
    transformed = got['T']
    distances = ((transformed[:,None]-transformed[None,:])**2).sum(axis=2)
    assert np.array_equal(np.argsort(distances,axis=1)[:,1],[2,3,0,1])
    points = np.random.RandomState(42).randn(2,5)
    X = np.vstack([points,points.mean(axis=0)])
    y = np.array([0,0,1])
    got = native(X,y)
    assert abs(objective(X,y,got['A'])[0]+1) < 1e-9
    assert_allclose(got['T']-got['T'][0],0,atol=1e-7)
    for y in [np.zeros(3,dtype=int), np.arange(3)]:
        got = native(X,y)
        assert_allclose(got['A'],np.eye(5),atol=1e-14)
        assert got['I'][0] == 1
    # Large distances test the stabilized off-diagonal softmax.
    X = np.array([[0.,0],[10000,1],[10000,2]])
    y = np.array([0,1,1])
    got = native(X,y,'gradient')
    expected = objective(X,y,np.eye(2))
    assert_allclose(got['L'],expected[0],atol=1e-12)
    assert_allclose(got['G'],expected[1],atol=1e-10)
    print('PASS: NCA',counts,'plus toy collapse, nearest-neighbor geometry, one class, singleton classes and extreme distances')


if __name__ == '__main__': main()
