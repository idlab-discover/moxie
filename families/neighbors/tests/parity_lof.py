"""Execute the pinned sklearn reachability/score method bodies on independent queries."""
import ast, subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1]
path=ROOT.parents[2]/'scikit-learn/sklearn/neighbors/_lof.py'
assert subprocess.check_output(['git','-C',str(path.parent),'rev-parse','HEAD'],text=True).strip()=='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert not subprocess.check_output(['git','-C',str(path.parent),'diff','HEAD','--',str(path)],text=True)
ns=dict(np=np,check_is_fitted=lambda *a:None,check_array=lambda x,**kw:x)
cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LocalOutlierFactor')
for name in ['_local_reachability_density','score_samples']:
 f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name);f.decorator_list=[]
 exec(compile(ast.Module(body=[f],type_ignores=[]),str(path),'exec'),ns)
for dtype in ['f32','f64']:
 typ=np.float32 if dtype=='f32' else np.float64
 x=np.array([[0],[1],[2],[3],[4],[20]],dtype=typ);q=np.array([[2.5],[30]],dtype=typ)
 d=cdist(x,x);np.fill_diagonal(d,np.inf)
 idx=np.argsort(d,axis=1,kind='stable')[:,:2];dd=np.take_along_axis(d,idx,axis=1).astype(typ)
 fixture=SimpleNamespace(n_neighbors_=2,_distances_fit_X_=dd)
 fixture._local_reachability_density=lambda a,b:ns['_local_reachability_density'](fixture,a,b)
 fixture._lrd=fixture._local_reachability_density(dd,idx)
 train=-(fixture._lrd[idx]/fixture._lrd[:,None]).mean(axis=1)
 dq=cdist(q,x);iq=np.argsort(dq,axis=1,kind='stable')[:,:2]
 fixture.kneighbors=lambda X,n_neighbors:(np.take_along_axis(dq,iq,axis=1).astype(typ),iq)
 score=ns['score_samples'](fixture,q)
 raw=subprocess.check_output([str(ROOT/'builds/test_lof'),dtype],text=True)
 lines=[s.split() for s in raw.splitlines()]
 tol=3e-6 if dtype=='f32' else 2e-12
 np.testing.assert_allclose([float(s[1]) for s in lines if s[0]=='F'],train,rtol=tol,atol=tol)
 got=np.array([[float(v) for v in s[1:]] for s in lines if s[0]=='S'])
 np.testing.assert_allclose(got[:,0],score,rtol=tol,atol=tol)
 np.testing.assert_allclose(got[:,1],score+1.5,rtol=tol,atol=tol)
 np.testing.assert_array_equal(got[:,2],np.where(score+1.5<0,-1,1))
 offset=np.percentile(train,25)
 np.testing.assert_allclose([float(s[1]) for s in lines if s[0]=='O'],[offset],rtol=tol,atol=tol)
 np.testing.assert_array_equal([float(s[1]) for s in lines if s[0]=='L'],np.where(train<offset,-1,1))
print('PASS LOF: both dtypes, training factors, novelty scores/decisions/labels, contamination percentile and native lifecycle/duplicate checks')
