"""Compare public native predictions with pinned sklearn generic method bodies.

Compiled sklearn accelerators are bypassed, not substituted for native inference.
Neighbor arrays supplied to the reference methods are independent SciPy distances.
"""
import ast,itertools,subprocess,warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor
from sklearn.neighbors import _classification, _regression
ROOT=Path(__file__).resolve().parents[1]
SK=ROOT.parents[2]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(SK),'rev-parse','HEAD'],text=True).strip()==REV
for module in (_classification, _regression):
    loaded = Path(module.__file__)
    assert loaded.read_bytes() == (SK/'sklearn/neighbors'/loaded.name).read_bytes(), loaded
ns=dict(np=np,warnings=warnings,check_is_fitted=lambda *a:None,_num_samples=len,
        _adjusted_metric=lambda **kw:(kw['metric'],{}),
        _all_with_any_reduction_axis_1=lambda x,value:np.any(np.all(x==value,axis=1)))
def method(file,cls,name):
    path=SK/'sklearn/neighbors'/file
    assert not subprocess.check_output(['git','-C',str(SK),'diff','HEAD','--',str(path)],text=True)
    tree=ast.parse(path.read_text())
    if cls:tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns[name]
method('_base.py',None,'_get_weights')
methods={(rad,cl):method('_classification.py' if cl else '_regression.py',('RadiusNeighbors' if rad else 'KNeighbors')+('Classifier' if cl else 'Regressor'),'predict_proba' if cl else 'predict') for rad in (False,True) for cl in (False,True)}
count=0
for dtype,rad,weight,multi,selfq,pre,algorithm in itertools.product(['f32','f64'],[False,True],['uniform','distance','callable'],[False,True],[False,True],[False,True],['auto','kd_tree','ball_tree']):
    if pre and algorithm != 'auto': continue
    typ=np.float32 if dtype=='f32' else np.float64
    x=np.zeros((7,2),dtype=typ)
    for i in range(2,7):x[i]=[i,i*i%5]
    q=np.array([[0,0],[2.5,1.5],[30,0],[40,0]],dtype=typ)
    y=np.array([[i*i/3,i/7] for i in range(7)],dtype=typ)
    labels=np.array([[2+i*2%3*3,-2+i%2*5] for i in range(7)],dtype=typ)
    if not multi:y=y[:,0];labels=labels[:,0]
    query=x if selfq else q
    d=cdist(query,x)
    if pre:d=d.astype(typ).astype(float)
    # Engine outputs have the feature dtype before weighting.
    d=d.astype(typ).astype(float)
    if selfq:np.fill_diagonal(d,np.inf)
    indices=np.argsort(d,axis=1,kind='stable')[:,:3]
    kd=np.take_along_axis(d,indices,axis=1)
    ri=np.empty(len(d),dtype=object);rd=np.empty(len(d),dtype=object)
    for i in range(len(d)):
        ri[i]=np.flatnonzero(d[i]<=3.5);rd[i]=d[i,ri[i]]
    def kneighbors(X,return_distance=True):return (kd.copy(),indices.copy()) if return_distance else indices.copy()
    def radius_neighbors(X):return rd.copy(),ri.copy()
    def smooth(dist):
        if dist.dtype==object:
            out=np.empty(len(dist),dtype=object)
            for i,row in enumerate(dist):out[i]=1/(1+row)
            return out
        return 1/(1+dist)
    args=dict(weights=smooth if weight=='callable' else weight,_fit_method='generic',_fit_X=x,
              metric='euclidean',metric_params=None,p=2,kneighbors=kneighbors,radius_neighbors=radius_neighbors)
    with warnings.catch_warnings(record=True) as regression_warnings:
        warnings.simplefilter('always')
        expected_r=methods[rad,False](SimpleNamespace(_y=y,**args),query)
    with warnings.catch_warnings(record=True) as classification_warnings:
        warnings.simplefilter('always')
        c=[np.unique(labels[:,j] if multi else labels) for j in range(2 if multi else 1)]
        encoded=np.column_stack([np.searchsorted(a,labels[:,j] if multi else labels) for j,a in enumerate(c)])
        outliers=[typ(-99+j) for j in range(len(c))]
        if not multi:encoded=encoded.ravel()
        expected_p=methods[rad,True](SimpleNamespace(_y=encoded,classes_=c if multi else c[0],outputs_2d_=multi,outlier_label_=outliers,**args),query)
    if not multi:expected_p=[expected_p]
    expected_c=np.column_stack([c[j][p.argmax(axis=1)] for j,p in enumerate(expected_p)])
    if rad:
        for j,p in enumerate(expected_p):
            expected_c[p.sum(axis=1)==0,j]=outliers[j]
    native = subprocess.run(
        [str(ROOT/'builds/evaluate_estimators'), dtype, 'radius' if rad else 'knn',
         weight, 'multi' if multi else 'single', 'self' if selfq else 'query',
         'precomputed' if pre else 'features', algorithm],
        text=True, capture_output=True, check=True,
    )
    records=[line.split() for line in native.stdout.splitlines()]
    def array(key,shape):return np.array([float(r[3]) for r in records if r[0]==key]).reshape(shape)
    tol=3e-6 if dtype=='f32' else 2e-12
    np.testing.assert_array_equal(array('C',expected_c.shape),expected_c)
    np.testing.assert_allclose(array('R',expected_r.shape),expected_r,atol=tol,rtol=tol,equal_nan=True)
    for j,p in enumerate(expected_p):np.testing.assert_allclose(array('P'+str(j),p.shape),p,atol=tol,rtol=tol)
    operations = [('predict classifier', classification_warnings),
                  ('predict probabilities', classification_warnings),
                  ('predict regressor', regression_warnings)] if rad else []
    expected_stderr = []
    for marker, captured in operations:
        expected_stderr.append(marker)
        expected_stderr.extend(f'{w.category.__name__}: {w.message}' for w in captured)
    assert native.stderr.splitlines() == expected_stderr, (native.stderr, expected_stderr)
    if rad:
        # Also check warning triggers through the compiled estimator's own queries.
        # Use integer labels for sklearn's scalar outlier-label configuration.
        opts = dict(radius=3.5, weights=args['weights'], algorithm=algorithm, leaf_size=1,
                    metric='precomputed' if pre else 'euclidean')
        fit_x = cdist(x, x).astype(typ) if pre else x
        query_x = None if selfq else (cdist(q, x).astype(typ) if pre else q)
        reference_outliers = [int(label) for label in outliers]
        clf = RadiusNeighborsClassifier(
            outlier_label=reference_outliers if multi else reference_outliers[0], **opts,
        ).fit(fit_x, labels.astype(np.int64))
        reg = RadiusNeighborsRegressor(**opts).fit(fit_x, y)
        for operation, expected in [(clf.predict, classification_warnings),
                                    (clf.predict_proba, classification_warnings),
                                    (reg.predict, regression_warnings)]:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                operation(query_x)
            assert [w.category for w in captured] == [w.category for w in expected], (
                operation, dtype, weight, multi, selfq, pre, algorithm, captured, expected,
            )
    count+=1
print(f'PASS {count} public estimator configurations: KNN/radius, both dtypes, uniform/distance/native callable, multi-output, self queries, precomputed, duplicates, empty neighborhoods and prediction warnings; reference {REV}')
