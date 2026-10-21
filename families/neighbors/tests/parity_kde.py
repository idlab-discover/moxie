"""KDE formulas from the pinned binary-tree reference and its actual sample method."""
import ast,itertools,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp,gammaln
ROOT=Path(__file__).resolve().parents[1]; SK=ROOT.parents[2]/'scikit-learn'
assert subprocess.check_output(['git','-C',str(SK),'rev-parse','HEAD'],text=True).strip()=='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
for source in ['sklearn/neighbors/_kde.py','sklearn/neighbors/_binary_tree.pxi.tp']:
 assert not subprocess.check_output(['git','-C',str(SK),'diff','HEAD','--',source],text=True)
path=SK/'sklearn/neighbors/_kde.py';tree=ast.parse(path.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KernelDensity')
f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='sample');f.decorator_list=[]
ns=dict(np=np,check_is_fitted=lambda *a:None,check_random_state=np.random.RandomState)
exec(compile(ast.Module(body=[f],type_ignores=[]),str(path),'exec'),ns)
count=0;maximum=0
for dtype,kernel,weighted,rule,algorithm,metric in itertools.product(['f32','f64'],['gaussian','tophat','epanechnikov','exponential','linear','cosine'],[False,True],['','scott','silverman'],['auto','kd_tree','ball_tree'],['euclidean','manhattan','canberra']):
 if algorithm=='kd_tree' and metric=='canberra': continue
 x=np.array([[i,i%2] for i in range(4)],dtype=float);q=np.array([[0,0],[1.5,.75],[20,0]])
 w=np.array([1,0,3,2],dtype=float) if weighted else np.ones(4)
 h=1.5 if not rule else (4 if rule=='scott' else 4*(2+2)/4)**(-1/6)
 z=cdist(q,x,{'manhattan':'cityblock'}.get(metric,metric))/h;d=2;logv=d/2*np.log(np.pi)-gammaln(d/2+1)
 with np.errstate(divide='ignore',invalid='ignore'):
  if kernel=='gaussian':lk=-.5*z*z;factor=d/2*np.log(2*np.pi)
  elif kernel=='exponential':lk=-z;factor=logv+np.log(d)+gammaln(d)
  elif kernel=='tophat':lk=np.where(z<1,0,-np.inf);factor=logv
  elif kernel=='epanechnikov':lk=np.where(z<1,np.log(1-z*z),-np.inf);factor=logv+np.log(2/(d+2))
  elif kernel=='linear':lk=np.where(z<1,np.log(1-z),-np.inf);factor=logv-np.log(d+1)
  else:
   lk=np.where(z<1,np.log(np.cos(np.pi*z/2)),-np.inf)
   term=2/np.pi;value=0
   for k in range(1,d+1,2):value+=term;term*=-(d-k)*(d-k-1)*(2/np.pi)**2
   factor=logv+np.log(d)+np.log(value)
  expected=logsumexp(lk+np.log(w),axis=1)-np.log(w.sum())-factor-d*np.log(h)
 raw=subprocess.check_output([str(ROOT/'builds/test_kde'),dtype,kernel,'weighted' if weighted else 'uniform',rule,algorithm,metric],text=True)
 lines=[s.split() for s in raw.splitlines()]
 got=np.array([float(s[1]) for s in lines if s[0]=='S'])
 finite=np.isfinite(got)&np.isfinite(expected)
 if finite.any():maximum=max(maximum,float(np.max(np.abs(got[finite]-expected[finite]))))
 # Numax fdlibm exp/log (within one ulp) and summation order: a few ulps on log densities.
 np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-12)
 np.testing.assert_allclose([float(s[1]) for s in lines if s[0]=='T'],[expected.sum()],rtol=1e-12,atol=1e-12)
 np.testing.assert_allclose([float(s[1]) for s in lines if s[0]=='B'],[h],rtol=1e-14,atol=0)
 if kernel=='gaussian':
  fixture=SimpleNamespace(kernel=kernel,bandwidth_=h,tree_=SimpleNamespace(data=x,sample_weight=w if weighted else None))
  samples=ns['sample'](fixture,n_samples=3,random_state=42)
  np.testing.assert_allclose([float(s[1]) for s in lines if s[0]=='G'],samples.ravel(),rtol=2e-9,atol=2e-9)
 count+=1
print(f'PASS {count} KDE configurations: six kernels, both input dtypes, weighted/uniform, numeric/Scott/Silverman bandwidth; Gaussian seeded samples execute pinned sklearn sample; maximum log-density absolute error {maximum:.3g}')
