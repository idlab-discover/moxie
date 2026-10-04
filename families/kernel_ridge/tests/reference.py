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
paths=['sklearn/metrics/pairwise.py','sklearn/linear_model/_ridge.py','sklearn/metrics/_pairwise_fast.pyx','sklearn/kernel_ridge.py','sklearn/base.py','sklearn/utils/validation.py','sklearn/metrics/_regression.py']
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
fast=ROOT/'builds/parity/fast';fast.mkdir(parents=True,exist_ok=True)
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

names=['check_pairwise_arrays','_return_float_dtype','_find_floating_dtype_allow_sparse','linear_kernel','polynomial_kernel','rbf_kernel','laplacian_kernel','sigmoid_kernel','cosine_similarity','additive_chi2_kernel','chi2_kernel','_euclidean_distances','_euclidean_distances_upcast','euclidean_distances','manhattan_distances','_pairwise_callable','pairwise_kernels']
load_functions(REFERENCE/paths[0],names,ns)
solver={'np':np,'linalg':scipy.linalg,'warnings':warnings}
load_functions(REFERENCE/paths[1],['_solve_cholesky_kernel'],solver)
solve=solver['_solve_cholesky_kernel']

ns['PAIRWISE_KERNEL_FUNCTIONS']={name:ns[fn] for name,fn in dict(linear='linear_kernel',poly='polynomial_kernel',polynomial='polynomial_kernel',rbf='rbf_kernel',laplacian='laplacian_kernel',sigmoid='sigmoid_kernel',cosine='cosine_similarity',chi2='chi2_kernel',additive_chi2='additive_chi2_kernel').items()}
estimator_path=REFERENCE/'sklearn/kernel_ridge.py'
estimator_ns={'__name__':'pinned_kernel_ridge'}
exec(builtins.compile(estimator_path.read_text(),str(estimator_path),'exec'),estimator_ns)
estimator_ns['_solve_cholesky_kernel']=solve
import sklearn.utils.validation as validation_support
validation_ns=dict(vars(validation_support))
load_functions(REFERENCE/'sklearn/utils/validation.py', ['_check_sample_weight'], validation_ns)
estimator_ns['_check_sample_weight']=validation_ns['_check_sample_weight']
estimator_ns['pairwise_kernels']=ns['pairwise_kernels']
KR=estimator_ns['KernelRidge']
assert KR.predict.__code__.co_filename==str(estimator_path)
assert KR.fit.__wrapped__.__code__.co_filename==str(estimator_path)
# Execute inherited score body from the pinned base, with r2 infrastructure.
base=REFERENCE/'sklearn/base.py'
cls=next(n for n in ast.parse(base.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RegressorMixin')
score_node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='score')
score_ns={}
exec(builtins.compile(ast.Module(body=[score_node],type_ignores=[]),str(base),'exec'),score_ns)
KR.score=score_ns['score']
