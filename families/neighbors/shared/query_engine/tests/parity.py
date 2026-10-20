"""Execute pinned sklearn reducers; independently check native metric kernels.

AST isolation bypasses the unbuilt checkout's extension imports. SciPy is used
only for its reference distance formulas; sklearn's reducers and weights execute
from the local pinned source, never a support wheel.
"""
import ast,json,subprocess,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1]
SK=ROOT.parents[4]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(SK),'rev-parse','HEAD'],text=True).strip()==REV
source=SK/'sklearn/neighbors/_base.py'
assert not subprocess.check_output(['git','-C',str(SK),'diff','HEAD','--',str(source)],text=True)
mod=ast.parse(source.read_text());ns={'np':np}
for node in mod.body:
    if isinstance(node,ast.ClassDef):
        for f in node.body:
            if isinstance(f,ast.FunctionDef) and f.name in ('_kneighbors_reduce_func','_radius_neighbors_reduce_func'):
                exec(compile(ast.Module(body=[f],type_ignores=[]),str(source),'exec'),ns)
    elif isinstance(node,ast.FunctionDef) and node.name=='_get_weights':
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
for name in ['_kneighbors_reduce_func','_radius_neighbors_reduce_func','_get_weights']:
    assert ns[name].__code__.co_filename==str(source)
boolean={'jaccard','dice','rogerstanimoto','russellrao','sokalmichener','sokalsneath','yule'}
metrics=['euclidean','l2','sqeuclidean','manhattan','l1','cityblock','chebyshev','cosine','correlation','canberra','braycurtis','hamming','haversine','nan_euclidean','seuclidean','mahalanobis',*sorted(boolean)]
configs=[(name,2,False) for name in metrics]+[('minkowski',p,w) for p in [.1,.5,.7,1,1.1,1.5,2,2.3,3,7.1,31.4,np.inf] for w in [False,True]]+[('wminkowski',3,True)]
records=[]
def matrix(a,b,name,p,w):
    name={'l1':'cityblock','manhattan':'cityblock','l2':'euclidean','sokalmichener':'rogerstanimoto'}.get(name,name)
    cols=a.shape[1]
    kw={}
    if name in ('minkowski','wminkowski'):
        if name=='wminkowski':
            return cdist(a*np.arange(cols),b*np.arange(cols),'minkowski',p=p)
        kw={'p':p}
        if w:kw['w']=np.arange(cols,dtype=float)
    if name=='seuclidean':kw['V']=np.arange(1,cols+1,dtype=float)
    if name=='mahalanobis':kw['VI']=np.diag(1/np.arange(1,cols+1,dtype=float))
    if name=='haversine':
        d=a[:,None,:]-b[None,:,:]
        return 2*np.arcsin(np.sqrt(np.clip(np.sin(d[:,:,0]/2)**2+np.cos(a[:,None,0])*np.cos(b[None,:,0])*np.sin(d[:,:,1]/2)**2,0,1)))
    if name=='nan_euclidean':
        common=~(np.isnan(a[:,None,:])|np.isnan(b[None,:,:]))
        with np.errstate(invalid='ignore',divide='ignore'):
            return np.sqrt(np.where(common,(a[:,None,:]-b[None,:,:])**2,0).sum(2)*cols/common.sum(2))
    return cdist(a,b,name,**kw)
for dtype in ['f32','f64']:
    typ=np.float32 if dtype=='f32' else np.float64
    tol=3e-6 if dtype=='f32' else 2e-12
    for name,p,weighted in configs:
        cols=2 if name=='haversine' else 5
        x=np.array([[(i*7+j*3+1)%13/7 for j in range(cols)] for i in range(9)],dtype=typ)
        q=np.array([[(i*11+j*5+2)%17/9 for j in range(cols)] for i in range(3)],dtype=typ)
        if name in boolean:
            x=np.array([[(i+j)%3!=0 for j in range(cols)] for i in range(9)],dtype=typ)
            q=np.array([[(i+j)%2!=0 for j in range(cols)] for i in range(3)],dtype=typ)
        if name=='nan_euclidean':q[0,0]=np.nan;x[-1,:]=np.nan
        # Widen after construction, matching the documented accumulation policy.
        d=matrix(q.astype(float),x.astype(float),name,p,weighted)
        sd=matrix(x.astype(float),x.astype(float),name,p,weighted)
        raw=subprocess.check_output([str(ROOT/'builds/evaluate'),dtype,name,str(p),str(int(weighted))],text=True)
        lines=[s.split() for s in raw.splitlines()]
        actual=np.array([float(s[3]) for s in lines if s[0]=='D']).reshape(3,9)
        np.testing.assert_allclose(actual,d,rtol=tol,atol=tol,equal_nan=True,err_msg=f'{dtype} {name} {p} {weighted}')
        # Execute the pinned generic reducer on actual distances (not squared).
        fixture=SimpleNamespace(effective_metric_=name if name!='euclidean' else 'l2')
        expected,idx=ns['_kneighbors_reduce_func'](fixture,d,0,4,True)
        k=[s for s in lines if s[0]=='K']
        ki=np.array([int(s[1]) for s in k]).reshape(3,4)
        kd=np.array([float(s[2]) for s in k]).reshape(3,4)
        np.testing.assert_allclose(kd,expected,rtol=tol,atol=tol,equal_nan=True)
        np.testing.assert_allclose(kd,np.take_along_axis(d,ki,axis=1),rtol=tol,atol=tol,equal_nan=True)
        assert all(len(set(row))==4 for row in ki)
        rd,ri=ns['_radius_neighbors_reduce_func'](fixture,d,0,1.2,True)
        for i in range(3):
            rr=[s for s in lines if s[0]=='R' and int(s[1])==i]
            assert sorted(int(s[2]) for s in rr)==sorted(ri[i])
            np.testing.assert_allclose([float(s[3]) for s in rr],np.sort(rd[i]),rtol=tol,atol=tol)
        ss=[s for s in lines if s[0]=='S']
        si=np.array([int(s[1]) for s in ss]).reshape(9,3)
        sv=np.array([float(s[2]) for s in ss]).reshape(9,3)
        for i in range(9):
            assert i not in si[i]
            expected=np.sort(np.delete(sd[i],i))[:3]
            np.testing.assert_allclose(sv[i],expected,rtol=tol,atol=tol,equal_nan=True)
        records.append(dict(dtype=dtype,metric=name,p=str(p),weighted=weighted))
# Inferred Mahalanobis VI: execute the pinned sklearn parameter preparation body.
pairwise=SK/'sklearn/metrics/pairwise.py'
assert not subprocess.check_output(['git','-C',str(SK),'diff','HEAD','--',str(pairwise)],text=True)
node=next(n for n in ast.parse(pairwise.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_precompute_metric_params')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(pairwise),'exec'),ns)
assert ns['_precompute_metric_params'].__code__.co_filename==str(pairwise)
def inferred_fixture(fixture):
    if fixture==0: return np.array([[0.,0],[1,0],[0,1]])
    if fixture==1: return np.array([[((i*7+j*5+i*j*j+1)%13)/7 for j in range(4)] for i in range(12)])
    rows=[]
    for i in range(25):
        a=((i*5+1)%11)/3;b=a*0.5+((i*3)%7)/5
        rows.append([a,b,((i*i)%13)/4-b*0.25])
    return np.array(rows)
for metric in ['mahalanobis','seuclidean']:
    for dtype in ['f32','f64']:
        typ=np.float32 if dtype=='f32' else np.float64
        # Covariance condition numbers are below 12; inverse error stays far inside these bounds.
        tol=3e-6 if dtype=='f32' else 2e-12
        for fixture in range(3):
            x=inferred_fixture(fixture).astype(typ)
            kw=ns['_precompute_metric_params'](x,x,metric)
            sd=cdist(x.astype(float),x.astype(float),metric,**kw)
            raw=subprocess.check_output([str(ROOT/'builds/evaluate'),dtype,metric+'_inferred',str(fixture)],text=True)
            lines=[s.split() for s in raw.splitlines()]
            n=len(x);k=min(3,n-1)
            ss=[s for s in lines if s[0]=='S']
            si=np.array([int(s[1]) for s in ss]).reshape(n,k);sv=np.array([float(s[2]) for s in ss]).reshape(n,k)
            for i in range(n):
                assert i not in si[i]
                np.testing.assert_allclose(sv[i],np.sort(np.delete(sd[i],i))[:k],rtol=tol,atol=tol,err_msg=f'{dtype} inferred {fixture}')
                np.testing.assert_allclose(sv[i],sd[i,si[i]],rtol=tol,atol=tol)
                # Radius membership away from the boundary; sorted distances match.
                row=np.delete(np.arange(n),i);inside=row[sd[i,row]<=1.5]
                assert not np.any(np.abs(sd[i,row]-1.5)<1e-6)
                tt=[s for s in lines if s[0]=='T' and int(s[1])==i]
                assert sorted(int(s[2]) for s in tt)==sorted(inside)
                np.testing.assert_allclose([float(s[3]) for s in tt],np.sort(sd[i,inside]),rtol=tol,atol=tol)
            records.append(dict(dtype=dtype,metric=metric+'_inferred',fixture=fixture,cond=float(np.linalg.cond(np.cov(x.T)))))
np.testing.assert_array_equal(ns['_get_weights'](np.array([[0.,0.,2.]]),'distance'),[[1.,1.,0.]])
report={'revision':REV,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':records,'pairwise_sha256':hashlib.sha256(pairwise.read_bytes()).hexdigest(),'scope':'Explicit metric parameters and inferred self-query Mahalanobis VI; pinned reducers and SciPy metric formulas; not full estimator parity'}
(ROOT/'docs/private/query_engine').mkdir(parents=True, exist_ok=True)
(ROOT/'docs/private/query_engine/parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS {len(records)} dtype/metric configurations (incl. inferred Mahalanobis); pinned top-k/radius reducers, ties, self exclusion and weights')
