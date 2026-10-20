"""Differential tests: execute pinned sklearn AST bodies, never a wheel target."""
from pathlib import Path
import ast, functools, json, os, subprocess, warnings
import numpy as np
from scipy import linalg, sparse
ROOT=Path(__file__).resolve().parents[1]
ORACLE=ROOT.parents[4]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(ORACLE),'rev-parse','HEAD'],text=True).strip()==REV
source=ORACLE/'sklearn/utils/extmath.py'
assert not subprocess.check_output(['git','-C',str(ORACLE),'status','--porcelain','--',str(source)],text=True)
names={'_randomized_svd','_randomized_range_finder','svd_flip'}
tree=ast.parse(source.read_text()); selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(selected)==len(names)
ns=dict(np=np,linalg=linalg,sparse=sparse,warnings=warnings,partial=functools.partial,get_namespace=lambda *a:(np,False),check_random_state=lambda s:s if isinstance(s,np.random.RandomState) else np.random.RandomState(s),array_device=lambda a:None)
exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
assert ns['_randomized_svd'].__code__.co_filename==str(source)

def native(a,mode='svd',k=2,over=3,it=2,seed=42,normalizer='auto',transpose='auto',driver='gesdd'):
    args=[ROOT/'builds/evaluate','f32' if a.dtype==np.float32 else 'f64',mode,*a.shape,k,over,it,seed,normalizer,transpose,driver,*a.ravel()]
    p=subprocess.run(list(map(str,args)),capture_output=True,text=True)
    assert p.returncode==0,(args[:12],p.returncode,p.stdout,p.stderr,a)
    out={}
    for line in p.stdout.splitlines():
        name,value=line.split();out.setdefault(name,[]).append(float(value))
    out={key:np.array(v) for key,v in out.items()}
    if 'v' in out:
        count=len(out['s']);out['u']=out['u'].reshape(a.shape[0],count);out['v']=out['v'].reshape(count,a.shape[1])
    else:out['u']=out['u'].reshape(a.shape[0],-1)
    return out

count=0
rng=np.random.RandomState(182)
for dtype,tol in [(np.float64,3e-11),(np.float32,3e-5)]:
    cases=[np.zeros((4,3)),np.ones((5,3)),np.eye(4),np.diag([1,1e-5,1e-10]),np.array([[0,2],[1,0],[0,0]]),np.array([[3.]]),rng.normal(size=(1,5)),rng.normal(size=(6,1)),rng.normal(size=(9,5)),rng.normal(size=(5,9))]
    for a in cases:
        a=a.astype(dtype); k=min(a.shape)
        for kernel in ('svd','gesvd'):
            r=native(a,mode=kernel)
            np.testing.assert_allclose(r['s'],linalg.svdvals(a),atol=tol,rtol=tol)
            np.testing.assert_allclose((r['u']*r['s'])@r['v'],a,atol=tol,rtol=tol)
            np.testing.assert_allclose(r['u'].T@r['u'],np.eye(k),atol=tol,rtol=tol)
            np.testing.assert_allclose(r['v']@r['v'].T,np.eye(k),atol=tol,rtol=tol)
            count+=1
        r=native(a)
        np.testing.assert_allclose((r['u']*r['s'])@r['v'],a,atol=tol,rtol=tol)
        np.testing.assert_allclose(r['u'].T@r['u'],np.eye(k),atol=tol,rtol=tol)
        np.testing.assert_allclose(r['v']@r['v'].T,np.eye(k),atol=tol,rtol=tol)
        q=native(a,mode='qr')['u'];np.testing.assert_allclose(q.T@q,np.eye(k),atol=tol,rtol=tol)
        np.testing.assert_allclose(q@(q.T@a),a,atol=tol,rtol=tol)
        count+=1
    for n in [1,2,6,13]:
        a=rng.normal(size=(n,n)).astype(dtype);a=a+a.T
        r=native(a,mode='eigh')
        np.testing.assert_allclose(r['s'],linalg.eigvalsh(a)[::-1],atol=tol*10,rtol=tol*10)
        np.testing.assert_allclose(a@r['u'],r['u']*r['s'],atol=tol*10,rtol=tol*10)
        np.testing.assert_allclose(r['u'].T@r['u'],np.eye(n),atol=tol,rtol=tol)
        count+=1
    for shape in [(12,7),(7,12),(1,4),(4,1)]:
      a=rng.normal(size=shape).astype(dtype)
      for mode in ['none','QR','LU','auto']:
       for it in [0,2,4,-1]:
        for trans,driver in [('auto','gesdd'),('true','gesdd'),('false','gesdd'),('auto','gesvd'),('true','gesvd'),('false','gesvd')]:
         r=native(a,mode='random',normalizer=mode,it=it,transpose=trans,driver=driver)
         state=np.random.RandomState(42)
         expected=ns['_randomized_svd'](a,2,n_oversamples=3,n_iter='auto' if it==-1 else it,power_iteration_normalizer=mode,transpose='auto' if trans=='auto' else trans=='true',random_state=state,svd_lapack_driver=driver)
         # Compare approximate reconstruction rather than arbitrary individual bases.
         np.testing.assert_allclose(r['s'],expected[1],atol=tol*30,rtol=tol*30)
         np.testing.assert_allclose((r['u']*r['s'])@r['v'],(expected[0]*expected[1])@expected[2],atol=tol*100,rtol=tol*100)
         np.testing.assert_allclose(r['u'].T@r['u'],np.eye(len(r['s'])),atol=tol*10,rtol=tol*10)
         np.testing.assert_allclose(r['draw'],state.standard_normal(),atol=2e-14,rtol=2e-14)
         count+=1

# Dynamic range, rank deficiency and repeated eigenspaces are independent of
# sign/basis choices; compare singular values relatively even at tiny scales.
for a in [np.diag([1.,1e-100,1e-200]), np.diag([1e150,1.,1e-150]),
          np.eye(4)*1e-200, np.eye(4)*1e200, np.diag([1e300,1e-300])]:
    r=native(a)
    # Analytic diagonal spectrum: SciPy itself underflows the 1e-300 value
    # in diag(1e300,1e-300), so it is not an accurate oracle for this case.
    np.testing.assert_allclose(r['s'],np.sort(np.abs(np.diag(a)))[::-1],rtol=1e-12,atol=0)
    np.testing.assert_allclose(r['u'].T@r['u'],np.eye(a.shape[0]),atol=1e-12)
    eigen=native(a,mode='eigh')
    np.testing.assert_allclose(eigen['s'],np.sort(np.diag(a))[::-1],rtol=1e-12,atol=0)
    count+=1
# Golub-Kahan uses one uniform scale, so only moderate dynamic range is exact.
for a in [np.diag([1.,1e-5,1e-10]), np.eye(4)*1e-200, np.eye(4)*1e200, np.diag([1e150,1.,1e-150])]:
    r=native(a,mode='gesvd')
    np.testing.assert_allclose(r['s'],np.sort(np.abs(np.diag(a)))[::-1],rtol=1e-12,atol=0)
    np.testing.assert_allclose(r['u'].T@r['u'],np.eye(a.shape[0]),atol=1e-12)
    count+=1
# Ill-conditioned and rank-deficient dense inputs for the bidiagonal kernel.
for dtype,tol in [(np.float64,1e-10),(np.float32,2e-4)]:
    base=rng.normal(size=(12,6))
    for a in [base@np.diag(10.0**-np.arange(6))@rng.normal(size=(6,6)), np.column_stack([base[:,:3],base[:,:3]]), np.vander(np.linspace(0,1,10),6)]:
        a=a.astype(dtype)
        for kernel in ('svd','gesvd'):
            for m in (a, a.T):
                r=native(np.ascontiguousarray(m),mode=kernel)
                np.testing.assert_allclose(r['s'],linalg.svdvals(m),atol=tol*np.abs(m).max(),rtol=tol)
                np.testing.assert_allclose((r['u']*r['s'])@r['v'],m,atol=tol*np.abs(m).max(),rtol=tol)
                count+=1
for dtype,tol in [(np.float64,1e-10),(np.float32,2e-4)]:
    for a in [np.zeros((7,4)),np.ones((4,7)),np.diag([3,3,0,0])]:
        a=a.astype(dtype)
        for normalizer in ['none','QR','LU','auto']:
            r=native(a,mode='random',k=4,over=7,it=4,normalizer=normalizer)
            np.testing.assert_allclose((r['u']*r['s'])@r['v'],a,atol=tol,rtol=tol)
            np.testing.assert_allclose(r['u'].T@r['u'],np.eye(len(r['s'])),atol=tol,rtol=tol)
            count+=1
    # Oversampling smaller than matrix rank and many stable power iterations.
    a=rng.normal(size=(40,25)).astype(dtype)
    for normalizer in ['QR','LU','auto']:
        for iterations in [5,20]:
            r=native(a,mode='random',k=3,over=2,it=iterations,normalizer=normalizer)
            expected=ns['_randomized_svd'](a,3,n_oversamples=2,n_iter=iterations,power_iteration_normalizer=normalizer,random_state=42)
            np.testing.assert_allclose(r['s'],expected[1],atol=tol*10,rtol=tol*10)
            np.testing.assert_allclose((r['u']*r['s'])@r['v'],(expected[0]*expected[1])@expected[2],atol=tol*10,rtol=tol*10)
            count+=1
    # gesdd maps to native one-sided Jacobi, gesvd to native Golub-Kahan QR.
    a=rng.normal(size=(8,6)).astype(dtype)
    for driver in ['gesdd','gesvd']:
        r=native(a,mode='random',driver=driver)
        expected=ns['_randomized_svd'](a,2,n_oversamples=3,n_iter=2,random_state=42,svd_lapack_driver=driver)
        np.testing.assert_allclose(r['s'],expected[1],atol=tol,rtol=tol)
        count+=1
    # Range bases have arbitrary signs; compare their orthogonal projectors.
    for normalizer in ['none','QR','LU','auto']:
        a=rng.normal(size=(9,7)).astype(dtype)
        r=native(a,mode='range',k=3,it=4,normalizer=normalizer)
        state=np.random.RandomState(42)
        expected=ns['_randomized_range_finder'](a,size=3,n_iter=4,power_iteration_normalizer=normalizer,random_state=state)
        np.testing.assert_allclose(r['u']@r['u'].T,expected@expected.T,atol=tol*10,rtol=tol*10)
        np.testing.assert_allclose(r['draw'],state.standard_normal(),atol=2e-14,rtol=2e-14)
        count+=1
print(f'PASS {count} differential cases against {source} at {REV}')
(ROOT/'docs/private/svd_engine').mkdir(parents=True, exist_ok=True)
(ROOT/'docs/private/svd_engine/parity.json').write_text(json.dumps({'cases':count,'oracle':REV,'status':'pass'},indent=2)+'\n')
