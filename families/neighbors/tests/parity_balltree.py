"""Native BallTree vs pinned sklearn tests and independent SciPy distances.

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
SOURCES = ['sklearn/neighbors/_ball_tree.pyx.tp', 'sklearn/neighbors/_binary_tree.pxi.tp',
           'sklearn/neighbors/tests/test_neighbors_tree.py', 'sklearn/neighbors/tests/test_ball_tree.py']
assert not subprocess.check_output(['git', '-C', str(REFERENCE), 'diff', 'HEAD', '--', *SOURCES], text=True)


def native(X, Y, metric='euclidean', p=2, leaf=2, k=3, dual=False, breadth=False,
           radii=0.3, kernel='gaussian', h=0.4, return_log=False, weights=None,
           atol=0, rtol=0, metric_params=None):
    X = np.asarray(X)
    Y = np.asarray(Y, dtype=X.dtype)
    radii = np.atleast_1d(radii).astype(float)
    with tempfile.NamedTemporaryFile(mode='w', dir=ROOT/'builds', suffix='.txt') as f:
        values = [*X.shape, *X.ravel(), *Y.shape, *Y.ravel(), len(radii), *radii,
                  *(np.ones(len(X)) if weights is None else weights)]
        for key in ['w','V','VI']:
            array = np.asarray((metric_params or {}).get(key, []), dtype=X.dtype).ravel()
            values.extend([len(array), *array])
        f.write('\n'.join(format(float(value),'.17g') for value in values))
        f.flush()
        args = [f.name, 'f32' if X.dtype == np.float32 else 'f64', leaf, metric, p,
                min(k, len(X)), int(dual), int(breadth), kernel, h, int(return_log),
                int(weights is not None), atol, rtol, 'params']
        output = subprocess.check_output([str(ROOT/'builds/evaluate_balltree'), *map(str, args)], text=True)
    result = {tag: [] for tag in 'QORCDTNIB'}
    for line in output.splitlines():
        tag, *values = line.split()
        result[tag].append(list(map(float, values)))
    return {tag: np.array(rows) for tag, rows in result.items()}


def check_queries(X,Y,metric,p,leaf,dual,breadth,params=None):
    from sklearn.metrics import DistanceMetric
    params = params or {}
    # The compiled pinned metric is an independent scalar oracle, including
    # legacy Boolean semantics removed from newer SciPy versions.
    kwargs = dict(params)
    if metric in ('minkowski','wminkowski','p'): kwargs['p']=p
    D=DistanceMetric.get_metric(metric, **kwargs).pairwise(Y.astype(float),X.astype(float))
    radii=np.array([0,0.3,1,2,-1,np.inf,3])
    got=native(X,Y,metric,p,leaf,5,dual,breadth,radii,metric_params=params)
    distances=got['Q'][:,1].reshape(len(Y),5)
    ids=got['Q'][:,0].astype(int).reshape(len(Y),5)
    tol=3e-5 if X.dtype==np.float32 else 2e-11
    assert_allclose(distances,np.sort(D,axis=1)[:,:5],rtol=tol,atol=tol*1e-3)
    assert_allclose(distances,np.take_along_axis(D,ids,axis=1),rtol=tol,atol=tol*1e-3)
    offsets=got['O'].ravel().astype(int)
    for i in range(len(Y)):
        row=got['R'][offsets[i]:offsets[i+1]]
        ids_r=row[:,0].astype(int) if len(row) else np.array([],dtype=int)
        assert_array_equal(np.sort(ids_r),np.flatnonzero(D[i]<=radii[i]))
    assert_array_equal(got['C'].ravel(),(D<=radii[:,None]).sum(axis=1))
    assert_array_equal(got['T'].ravel(),[(D<=r).sum() for r in radii])
    nodes=got['N']; centers=got['B'].reshape(1,len(nodes),X.shape[1])[0]
    indices=got['I'].ravel().astype(int)
    assert_array_equal(np.sort(indices),np.arange(len(X)))
    for node,(start,end,leaf_node,radius) in enumerate(nodes):
        points=X[indices[int(start):int(end)]]
        assert_allclose(centers[node],points.astype(float).mean(axis=0),rtol=tol,atol=tol)
        cd=DistanceMetric.get_metric(metric,**kwargs).pairwise(centers[node:node+1].astype(float),points.astype(float))
        assert_allclose(radius,np.max(cd),rtol=tol,atol=tol)
    return got


def independent_cases():
    rng=np.random.RandomState(456)
    boolean=['jaccard','dice','rogerstanimoto','russellrao','sokalmichener','sokalsneath']
    metrics=['euclidean','manhattan','chebyshev','minkowski','seuclidean','mahalanobis','haversine','hamming','canberra','braycurtis',*boolean]
    count=0
    for dtype,metric,leaf,dual,breadth in itertools.product([np.float32,np.float64],metrics,[1,4,40],[False,True],[False,True]):
        dim=2 if metric=='haversine' else 5
        X=rng.uniform(-1,1,(31,dim)).astype(dtype)
        Y=rng.uniform(-1,1,(7,dim)).astype(dtype)
        if metric in boolean:
            X=(X>0).astype(dtype);Y=(Y>0).astype(dtype)
            X[:,0]=1;Y[:,0]=1  # defined Dice/Sokal-Sneath denominators
        elif metric in ['hamming','braycurtis']:
            X=np.round(3*(X+1))+1;Y=np.round(3*(Y+1))+1
        X[2]=X[1];Y[0]=X[1]
        params={}
        if metric=='seuclidean':params={'V':np.arange(1,dim+1,dtype=dtype)}
        if metric=='mahalanobis':params={'VI':(np.eye(dim)+np.ones((dim,dim))/dim).astype(dtype)}
        check_queries(X,Y,metric,3,leaf,dual,breadth,params)
        count+=1
    for dtype,p,dual,breadth in itertools.product([np.float32,np.float64],[1,1.5,2,3],[False,True],[False,True]):
        X=rng.normal(size=(31,5)).astype(dtype);Y=rng.normal(size=(7,5)).astype(dtype)
        check_queries(X,Y,'minkowski',p,1,dual,breadth,{'w':np.array([0,1,2,3,4],dtype=dtype)})
        count+=1
    for dtype,dual,breadth in itertools.product([np.float32,np.float64],[False,True],[False,True]):
        X=rng.normal(size=(31,5)).astype(dtype);Y=rng.normal(size=(7,5)).astype(dtype)
        covariance=(np.eye(5)+np.ones((5,5))/5).astype(dtype)
        check_queries(X,Y,'mahalanobis',2,1,dual,breadth,{'V':covariance})
        count+=1
    for dtype in [np.float32,np.float64]:
        for scale in [1e-15,1.0,1e15]:
            X=np.array([[0,0],[3,4],[6,8],[3,4]],dtype=dtype)*scale
            Y=X[:1]
            boundary=float(np.linalg.norm(X[1].astype(float)))
            for radius in [np.nextafter(boundary,0),boundary,np.nextafter(boundary,np.inf)]:
                got=native(X,Y,leaf=1,k=4,radii=radius)
                D=cdist(Y.astype(float),X.astype(float))
                assert_array_equal(got['C'].ravel(),(D<=radius).sum(axis=1))
                count+=1
    from sklearn.metrics import DistanceMetric
    antipodal_rng=np.random.RandomState(419)
    for dtype,scale in itertools.product([np.float32,np.float64],[1e-10,1e-8,1e-6,1e-4]):
        X=(antipodal_rng.normal(size=(31,2))*scale).astype(dtype)
        Y=np.array([[0,np.pi]],dtype=dtype)
        D=DistanceMetric.get_metric('haversine').pairwise(Y.astype(float),X.astype(float))
        radius=float(np.sort(D[0])[15])
        for dual in [False,True]:
            got=native(X,Y,'haversine',leaf=1,k=5,dual=dual,radii=radius)
            tol=2e-7 if dtype==np.float32 else 2e-11
            assert_allclose(got['Q'][:,1],np.sort(D[0])[:5],atol=tol,rtol=tol)
            assert_array_equal(got['C'].ravel(),(D<=radius).sum(axis=1))
            count+=1
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
        count += 1
    print(f'PASS BallTree {count} independent metric/query/density configurations')


def original_tests():
    import importlib.util
    import sklearn
    assert sklearn.__version__=='1.10.dev0'
    from sklearn.neighbors import VALID_METRICS
    names={key:[] for key in VALID_METRICS}
    for line in subprocess.check_output([str(ROOT/'builds/evaluate_balltree'),'registry'],text=True).splitlines():
        backend,name=line.split();names[backend].append(name)
    for key in names: assert set(names[key])==set(VALID_METRICS[key])
    installed=Path(sklearn.__file__).resolve().parent.parent
    for source in SOURCES[:2]: assert (installed/source).read_bytes()==(REFERENCE/source).read_bytes()
    import parity_kdtree as kd
    # Reuse only the typed result adapter, pointed at the BallTree executable.
    kd.native=native
    class NativeTree(kd.NativeTree):
        def __init__(self,X,leaf_size=40,metric='euclidean',**kwargs):
            super().__init__(X,leaf_size,metric,**kwargs)
    spec=importlib.util.spec_from_file_location('local_ball_tests',REFERENCE/SOURCES[3])
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    count=0
    for dtype in [np.float64,np.float32]:
        factory=lambda X,**kw:NativeTree(np.asarray(X,dtype=dtype),**kw)
        for metric in tests.BOOLEAN_METRICS+tests.DISCRETE_METRICS:
            tests.test_ball_tree_query_metrics(metric,'array',factory);count+=1
        tests.test_query_haversine(factory,6 if dtype==np.float64 else 5);count+=1
    tests.BallTree64=lambda X,**kw:NativeTree(np.asarray(X,dtype=np.float64),**kw)
    tests.BallTree32=lambda X,**kw:NativeTree(np.asarray(X,dtype=np.float32),**kw)
    for metric in [*tests.METRICS,*tests.BOOLEAN_METRICS]:
        tests.test_ball_tree_numerical_consistency(42,metric)
        tests.test_kernel_density_numerical_consistency(42,metric)
        count+=2
    tests.test_two_point_correlation_numerical_consistency(42);count+=1
    spec=importlib.util.spec_from_file_location('local_tree_tests',REFERENCE/SOURCES[2])
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    for metric,k,dual,breadth in itertools.product(common.KD_TREE_METRICS,[1,3,5],[False,True],[False,True]):
        common.test_nn_tree_query(NativeTree,metric,k,dual,breadth);count+=1
    for kernel,h,rtol,atol,breadth in itertools.product(['gaussian','tophat','epanechnikov','exponential','linear','cosine'],[0.01,0.1,1],[0,1e-5],[1e-6,1e-2],[True,False]):
        common.test_kernel_density(NativeTree,kernel,h,rtol,atol,breadth);count+=1
    common.test_neighbor_tree_query_radius(NativeTree)
    common.test_neighbor_tree_query_radius_distance(NativeTree)
    for dual in [False,True]:common.test_neighbor_tree_two_point(NativeTree,dual)
    common.test_gaussian_kde(NativeTree)
    print(f'PASS BallTree {count+5} original pinned sklearn test invocations')

if __name__=='__main__':
    independent_cases()
    original_tests()
