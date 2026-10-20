"""Native KDTree vs pinned sklearn tests and independent SciPy distances.

Always runs deterministic dense oracle cases. With a compiled pinned sklearn
available, also calls its original tree test functions through a native adapter.
Set SKLEARN_ORACLE_PYTHON when running pixi run parity to select that environment.
"""
import itertools
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
SOURCES = ['sklearn/neighbors/_kd_tree.pyx.tp', 'sklearn/neighbors/_binary_tree.pxi.tp',
           'sklearn/neighbors/tests/test_neighbors_tree.py', 'sklearn/neighbors/tests/test_kd_tree.py']
assert not subprocess.check_output(['git', '-C', str(REFERENCE), 'diff', 'HEAD', '--', *SOURCES], text=True)


def native(X, Y, metric='euclidean', p=2, leaf=2, k=3, dual=False, breadth=False,
           radii=0.3, kernel='gaussian', h=0.4, return_log=False, weights=None,
           atol=0, rtol=0):
    X = np.asarray(X)
    Y = np.asarray(Y, dtype=X.dtype)
    radii = np.atleast_1d(radii).astype(float)
    with tempfile.NamedTemporaryFile(mode='w', dir=ROOT/'builds', suffix='.txt') as f:
        values = [*X.shape, *X.ravel(), *Y.shape, *Y.ravel(), len(radii), *radii,
                  *(np.ones(len(X)) if weights is None else weights)]
        f.write('\n'.join(format(float(value),'.17g') for value in values))
        f.flush()
        args = [f.name, 'f32' if X.dtype == np.float32 else 'f64', leaf, metric, p,
                min(k, len(X)), int(dual), int(breadth), kernel, h, int(return_log),
                int(weights is not None), atol, rtol]
        output = subprocess.check_output([str(ROOT/'builds/evaluate_kdtree'), *map(str, args)], text=True)
    result = {tag: [] for tag in 'QORCDTNIB'}
    for line in output.splitlines():
        tag, *values = line.split()
        result[tag].append(list(map(float, values)))
    return {tag: np.array(rows) for tag, rows in result.items()}


def independent_cases():
    cases = 0
    rng = np.random.RandomState(123)
    for dtype, metric, p, leaf, dual, breadth in itertools.product(
            [np.float32, np.float64], ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            [1.5, 3], [1, 4, 40], [False, True], [False, True]):
        X = rng.normal(size=(31, 3)).astype(dtype)
        X[2] = X[1]
        Y = rng.normal(size=(7, 3)).astype(dtype)
        Y[0] = X[1]
        radii = np.array([0, 0.5, 1, 2, -1, np.inf, 3.0])
        got = native(X, Y, metric, p, leaf, 5, dual, breadth, radii)
        metric_name = {'manhattan': 'cityblock'}.get(metric, metric)
        D = cdist(Y, X, metric_name, **({'p': p} if metric == 'minkowski' else {}))
        order = np.argsort(D, axis=1, kind='stable')[:, :5]
        tol = 3e-5 if dtype == np.float32 else 2e-11
        distances = got['Q'][:, 1].reshape(7, 5)
        ids = got['Q'][:, 0].astype(int).reshape(7, 5)
        assert_allclose(distances, np.take_along_axis(D, order, axis=1), rtol=tol, atol=tol*1e-3)
        assert_allclose(distances, np.take_along_axis(D, ids, axis=1), rtol=tol, atol=tol*1e-3)
        offsets = got['O'].ravel().astype(int)
        for i in range(7):
            row = got['R'][offsets[i]:offsets[i+1]]
            ids_r = row[:, 0].astype(int) if len(row) else np.array([], dtype=int)
            assert_array_equal(np.sort(ids_r), np.flatnonzero(D[i] <= radii[i]))
            if len(row): assert_allclose(row[:, 1], D[i, ids_r], rtol=tol, atol=tol*1e-3)
        assert_array_equal(got['C'].ravel(), (D <= radii[:, None]).sum(axis=1))
        assert_array_equal(got['T'].ravel(), [(D <= r).sum() for r in radii])
        # Validate inspection as geometry, not a particular tied permutation.
        nodes = got['N']
        bounds = got['B'].astype(dtype).reshape(2, len(nodes), 3)
        indices = got['I'].ravel().astype(int)
        assert_array_equal(np.sort(indices), np.arange(len(X)))
        for node, (start, end, is_leaf, radius) in enumerate(nodes):
            points = X[indices[int(start):int(end)]]
            assert_array_equal(bounds[0, node], points.min(axis=0))
            assert_array_equal(bounds[1, node], points.max(axis=0))
            if is_leaf: assert 1 <= len(points) <= 2*leaf or len(X) < leaf
        cases += 1
    # Weights, six kernels, bandwidth and log-space tails, both traversal orders.
    for dtype, kernel, h, breadth, weighted in itertools.product(
            [np.float32, np.float64], ['gaussian', 'exponential', 'tophat', 'epanechnikov', 'linear', 'cosine'],
            [0.01, 0.2, 2], [False, True], [False, True]):
        X = rng.uniform(size=(23, 3)).astype(dtype)
        Y = rng.uniform(size=(5, 3)).astype(dtype)
        weights = np.array([0 if i%4 == 0 else (i+1)/7 for i in range(len(X))], dtype=dtype) if weighted else None
        got = native(X,Y,kernel=kernel,h=h,breadth=breadth,weights=weights,return_log=True)
        distances = cdist(Y,X)
        z = distances/h
        volume = np.exp(1.5*np.log(np.pi)-gammaln(2.5))
        factors = dict(gaussian=(2*np.pi)**1.5, exponential=volume*6,
                       tophat=volume, epanechnikov=volume*2/5,
                       linear=volume/4, cosine=volume*3*(2/np.pi-16/np.pi**3))
        if kernel == 'gaussian': logs = -0.5*z*z
        elif kernel == 'exponential': logs = -z
        else:
            with np.errstate(divide='ignore',invalid='ignore'):
                values = dict(tophat=np.ones_like(z), epanechnikov=1-z*z,
                              linear=1-z, cosine=np.cos(np.pi*z/2))[kernel]
                logs = np.where(z<1,np.log(values),-np.inf)
        from scipy.special import logsumexp
        expected = logsumexp(logs,b=None if weights is None else weights,axis=1)-np.log(factors[kernel])-3*np.log(h)
        tol = 3e-5 if dtype==np.float32 else 2e-9
        assert_allclose(got['D'].ravel(),expected,rtol=tol,atol=tol,err_msg=str((dtype,kernel,h,breadth,weighted)))
        cases += 1
    print(f'PASS KDTree {cases} independent dense parity configurations')


class NativeTree:
    """Adapt only Mojo's typed result/shape boundary to the sklearn test API."""
    def __init__(self, X, leaf_size=40, metric='euclidean', **kwargs):
        self.X = np.asarray(X)
        self.leaf = leaf_size
        self.metric = metric
        self.p = kwargs.get('p',2)
    def query(self, Y, k=1, dualtree=False, breadth_first=False, **kwargs):
        Y = np.atleast_2d(Y)
        result = native(self.X,Y,self.metric,self.p,self.leaf,k,dualtree,breadth_first)
        return (result['Q'][:,1].astype(self.X.dtype).reshape(len(Y),k),result['Q'][:,0].astype(int).reshape(len(Y),k))
    def query_radius(self, Y, r, return_distance=False, **kwargs):
        Y = np.atleast_2d(Y)
        result = native(self.X,Y,self.metric,self.p,self.leaf,radii=r)
        offsets = result['O'].ravel().astype(int)
        ids, distances = [], []
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            row = result['R'][lo:hi]
            ids.append(row[:,0].astype(int) if len(row) else np.array([],dtype=int))
            distances.append(row[:,1].astype(self.X.dtype) if len(row) else np.array([],dtype=self.X.dtype))
        return (ids,distances) if return_distance else ids
    def kernel_density(self,Y,h,kernel='gaussian',atol=0,rtol=1e-8,breadth_first=True):
        return native(self.X,np.atleast_2d(Y),self.metric,self.p,self.leaf,kernel=kernel,h=h,atol=atol,rtol=rtol,breadth=breadth_first)['D'].ravel().astype(self.X.dtype)
    def two_point_correlation(self,Y,r,dualtree=False):
        return native(self.X,np.atleast_2d(Y),self.metric,self.p,self.leaf,radii=r,dual=dualtree)['T'].ravel()


def original_tests():
    try:
        import sklearn
    except ImportError:
        print('NOTE: compiled sklearn unavailable; original local test execution requires SKLEARN_ORACLE_PYTHON (see docs/kdtree.md)')
        return
    assert sklearn.__version__ == '1.10.dev0', 'Use the compiled pinned local sklearn checkout'
    installed = Path(sklearn.__file__).resolve().parent.parent
    for source in SOURCES[:2]:
        assert (installed/source).read_bytes() == (REFERENCE/source).read_bytes(), source
    import importlib.util
    # The build provenance is checked separately; test bodies always come from
    # the exact local checkout, not whichever tests shipped in an installed wheel.
    spec = importlib.util.spec_from_file_location('local_tree_tests', REFERENCE/SOURCES[2])
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    cases = 0
    for metric,k,dual,breadth in itertools.product(tests.KD_TREE_METRICS,[1,3,5],[False,True],[False,True]):
        tests.test_nn_tree_query(NativeTree,metric,k,dual,breadth)
        cases += 1
    for kernel,h,rtol,atol,breadth in itertools.product(
            ['gaussian','tophat','epanechnikov','exponential','linear','cosine'],
            [0.01,0.1,1],[0,1e-5],[1e-6,1e-2],[True,False]):
        tests.test_kernel_density(NativeTree,kernel,h,rtol,atol,breadth)
        cases += 1
    tests.test_neighbor_tree_query_radius(NativeTree)
    tests.test_neighbor_tree_query_radius_distance(NativeTree)
    for dual in [False,True]: tests.test_neighbor_tree_two_point(NativeTree,dual)
    tests.test_gaussian_kde(NativeTree)
    spec_kd = importlib.util.spec_from_file_location('local_kd_tests', REFERENCE/SOURCES[3])
    kd_tests = importlib.util.module_from_spec(spec_kd)
    spec_kd.loader.exec_module(kd_tests)
    kd_tests.KDTree64 = lambda X, **kw: NativeTree(np.asarray(X,dtype=np.float64),**kw)
    kd_tests.KDTree32 = lambda X, **kw: NativeTree(np.asarray(X,dtype=np.float32),**kw)
    for metric in kd_tests.METRICS:
        kd_tests.test_kd_tree_numerical_consistency(42,metric)
        kd_tests.test_kernel_density_numerical_consistency(42,metric)
        cases += 2
    print(f'PASS {cases+5} original local sklearn test invocations through native adapter; oracle {sklearn.__version__}')


if __name__ == '__main__':
    independent_cases()
    original_tests()
