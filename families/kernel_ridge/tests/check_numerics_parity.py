"""Execute pinned local sklearn functions; support wheel supplies infrastructure."""
from pathlib import Path
import ast, builtins, hashlib, importlib.util, json, subprocess, sys, warnings
import numpy as np
import scipy, scipy.linalg
import sklearn.metrics.pairwise as support
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check import run, compile
REFERENCE=ROOT.parents[2]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(REFERENCE),'rev-parse','HEAD'],text=True).strip()==REV
paths=['sklearn/metrics/pairwise.py','sklearn/linear_model/_ridge.py','sklearn/metrics/_pairwise_fast.pyx']
assert not subprocess.check_output(['git','-C',str(REFERENCE),'status','--porcelain','--',*paths],text=True).strip()

def load_functions(path,names,namespace):
    tree=ast.parse(path.read_text(),filename=str(path))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    for n in nodes: n.decorator_list=[]
    exec(builtins.compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    for name in names: assert namespace[name].__code__.co_filename==str(path)

# Compile the unedited pinned chi2 implementation. The support wheel supplies
# OpenMP thread plumbing, not the kernel under comparison.
fast=ROOT/'builds/numerics-parity/fast';fast.mkdir(parents=True,exist_ok=True)
source=fast/'_pinned_pairwise_fast.pyx'
source.write_bytes((REFERENCE/paths[2]).read_bytes())
setup=fast/'setup.py'
setup.write_text("from setuptools import setup, Extension\nfrom Cython.Build import cythonize\nimport numpy\nsetup(ext_modules=cythonize([Extension('_pinned_pairwise_fast', ['_pinned_pairwise_fast.pyx'], include_dirs=[numpy.get_include()])], include_path=["+repr(str(REFERENCE))+"], compiler_directives={'language_level': 3}))\n")
from check import environment
compiled=subprocess.run([sys.executable,str(setup),'build_ext','--inplace'],cwd=fast,env=environment(),text=True,capture_output=True)
(fast/'build.log').write_text(compiled.stdout+compiled.stderr)
compiled.check_returncode()
so=next(fast.glob('_pinned_pairwise_fast*.so'))
spec=importlib.util.spec_from_file_location('_pinned_pairwise_fast',so)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ns=dict(vars(support))
ns['_chi2_kernel_fast']=module._chi2_kernel_fast

names=['linear_kernel','polynomial_kernel','rbf_kernel','laplacian_kernel','sigmoid_kernel','cosine_similarity','additive_chi2_kernel','chi2_kernel','_euclidean_distances','_euclidean_distances_upcast','euclidean_distances','manhattan_distances','_pairwise_callable']
load_functions(REFERENCE/paths[0],names,ns)
solver={'np':np,'linalg':scipy.linalg,'warnings':warnings}
load_functions(REFERENCE/paths[1],['_solve_cholesky_kernel'],solver)
solve=solver['_solve_cholesky_kernel']
NAMES=dict(linear='linear_kernel',poly='polynomial_kernel',polynomial='polynomial_kernel',rbf='rbf_kernel',laplacian='laplacian_kernel',sigmoid='sigmoid_kernel',cosine='cosine_similarity',chi2='chi2_kernel',additive_chi2='additive_chi2_kernel')
rng=np.random.default_rng(681094)
records=[]
def enc(a):
    a=np.asarray(a);return f'{a.shape[0]}:{a.shape[1]}:'+','.join(repr(float(x)) for x in a.ravel())
def nums(a): return ','.join(repr(float(x)) for x in np.atleast_1d(a))
def solve_case(K,y,alpha,weights,tol,dtype,kind='solve'):
    K=np.asarray(K,dtype=dtype);y=np.asarray(y,dtype=dtype)
    alpha=np.atleast_1d(alpha).astype(dtype);weights=np.atleast_1d(weights).astype(dtype)
    try:
        expected=scipy.linalg.lstsq(K,y)[0] if kind=='lstsq' else solve(K.copy(),y.copy(),alpha,weights,copy=True)
    except np.linalg.LinAlgError:
        kind='error';expected=np.zeros_like(y)
    records.append('|'.join([str(np.dtype(dtype).itemsize*8),kind,enc(K),enc(y),nums(alpha),nums(weights),'0','0',enc(expected),str(tol)]))

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for dtype in [np.float32,np.float64]:
        ktol=4e-6 if dtype==np.float32 else 3e-12
        stol=8e-5 if dtype==np.float32 else 2e-9
        for name,fn in NAMES.items():
            for i in range(8):
                X=rng.uniform(0,2,(20 if i >= 6 else 7,5)).astype(dtype);Y=rng.uniform(0,2,(16 if i >= 6 else 4,5)).astype(dtype)
                X[0]=0;Y[0]=0
                if name not in ['chi2','additive_chi2']:
                    X[1:]-=.8;Y[1:]-=.8
                if name=='cosine': X[1]=np.finfo(dtype).eps/10
                same=i%2==0
                if same: Y=X
                gamma=None if i%3==0 and name!='chi2' else (0.0 if i==1 else .3)
                degree=2.5 if i==2 else 3.0;coef0=3.0
                kw={}
                if name in ['rbf','laplacian','poly','polynomial','sigmoid','chi2']:kw['gamma']=gamma
                if name in ['poly','polynomial']:kw['degree']=degree
                if name in ['poly','polynomial','sigmoid']:kw['coef0']=coef0
                expected=ns[fn](X,Y,**kw)
                if name == "laplacian":
                    expected=expected.astype(dtype)  # documented typed-output adaptation
                else:
                    assert expected.dtype == np.dtype(dtype)
                records.append('|'.join([str(np.dtype(dtype).itemsize*8),name,enc(X),enc(Y),'none' if gamma is None else str(gamma),str(degree),str(coef0),str(int(same)),enc(expected),str(ktol)]))
        for same in [False,True]:
            X=np.array([[1,2],[3,4],[5,6]],dtype=dtype)
            Y=X if same else np.array([[7,8],[9,10]],dtype=dtype)
            expected=ns['_pairwise_callable'](X,Y,lambda x,y:x[0]+2*y[0]+2)
            records.append('|'.join([str(np.dtype(dtype).itemsize*8),'callable',enc(X),enc(Y),'0','0','0',str(int(same)),enc(expected),str(ktol)]))
        for base in [1e4,1e6]:
            X=np.array([[base,base],[base+.25,base-.125]],dtype=dtype)
            Y=X.copy()
            expected=ns['rbf_kernel'](X,Y,gamma=.3)
            records.append('|'.join([str(np.dtype(dtype).itemsize*8),'rbf',enc(X),enc(Y),'.3','3','1','0',enc(expected),str(ktol)]))
        for i in range(45):
            n=1+i%15;t=1+i%4
            X=rng.normal(size=(n,n+2)).astype(dtype)
            K=X@X.T+dtype(.2)*np.eye(n,dtype=dtype)
            y=rng.normal(size=(n,t))
            alpha=np.full(t,.3) if i%3==0 else rng.uniform(.2,1,t)
            weights=rng.uniform(.1,2,n) if i%3 else np.array([2.5])
            if i%4==0:weights[0]=0
            solve_case(K,y,alpha,weights,stol,dtype)
        for n in [2,4,10]:
            y=rng.normal(size=(n,3))
            solve_case(np.ones((n,n)),y,0,1,stol,dtype)
            solve_case(np.zeros((n,n)),y,0,0,stol,dtype)
            solve_case(np.ones((n,n)),y,[0,.1,.2],1,stol,dtype)
        for i in range(15):
            X=rng.integers(-3,4,(10,3)).astype(dtype);K=X@X.T
            w=rng.integers(1,4,10).astype(dtype)**2;w[0]=0
            solve_case(K,rng.normal(size=(10,2)),0,w,stol,dtype)
            A=rng.normal(size=(7,7));A[0,0]=-2
            solve_case(A,rng.normal(size=(7,2)),0,1,stol,dtype)
        for K in [[[2,.2],[9,2]],[[0,1],[2,0]],[[1,2],[2,1]]]:
            for a in [[0],[0,.1]]:solve_case(K,[[1,3],[2,1]],a,1,stol,dtype)
        # Orthogonal rank threshold cases, forced into least squares by negative diagonal.
        eps=np.finfo(dtype).eps
        for ratio in [.25,2,100]:
            K=np.diag([-1,ratio*eps,0]).astype(dtype)
            solve_case(K,[[1],[ratio*eps],[3]],0,1,stol,dtype)
        for condition in [100,10000]:
            q,_=np.linalg.qr(rng.normal(size=(6,6)))
            K=(q@np.diag(np.geomspace(1,1/condition,6))@q.T).astype(dtype)
            tol=5e-3 if dtype==np.float32 else 2e-8
            solve_case(K,rng.normal(size=(6,2)),0,1,tol,dtype)
        # Direct fallback exercises lower triangle and strided-compatible matrix contract.
        solve_case([[1,2],[3,4]],[[1],[2]],0,1,stol,dtype,'lstsq')

out=ROOT/'builds/numerics-parity';out.mkdir(parents=True,exist_ok=True)
(out/'fixtures.txt').write_text('\n'.join(records)+'\n')
(out/'metadata.json').write_text(json.dumps({'revision':REV,'seed':681094,'numpy':np.__version__,'scipy':scipy.__version__,'support_sklearn':__import__('sklearn').__version__,'cases':len(records),'source_hashes':{p:hashlib.sha256((REFERENCE/p).read_bytes()).hexdigest() for p in paths}},indent=2))
compile(ROOT/'tests/numerics_parity_runner.mojo',ROOT/'builds/numerics-parity_runner')
p=subprocess.run([str(ROOT/'builds/numerics-parity_runner'),str(out/'fixtures.txt')],env=environment(),text=True,capture_output=True)
(out/'result.txt').write_text(p.stdout+p.stderr)
p.check_returncode()
print(p.stdout,end='')
