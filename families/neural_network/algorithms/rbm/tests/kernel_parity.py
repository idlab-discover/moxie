"""Run original pinned sklearn RBM method bodies, using wheel infrastructure only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.sparse as sp
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin
from sklearn.utils import check_random_state, gen_even_slices
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, validate_data

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT.parents[4]/'scikit-learn'
SOURCE = ORACLE/'sklearn/neural_network/_rbm.py'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(ORACLE),'rev-parse','HEAD'],text=True).strip() == REVISION
assert not subprocess.check_output(['git','-C',str(ORACLE),'status','--porcelain','--',str(SOURCE)],text=True)
# Remove only validation decorators and parameter-constraint declarations; every
# numerical method body, including RNG consumption and update order, is original.
tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
cls = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='BernoulliRBM')
cls.body = [x for x in cls.body if not isinstance(x,ast.AnnAssign)]
for method in cls.body:
    if isinstance(method,ast.FunctionDef):
        method.decorator_list=[]
mod = ast.Module(body=[cls],type_ignores=[])
namespace = dict(globals())
exec(compile(ast.fix_missing_locations(mod),str(SOURCE),'exec'),namespace)
RBM = namespace['BernoulliRBM']
assert RBM._fit.__code__.co_filename == str(SOURCE)

def compare():
    lines = iter(subprocess.check_output([ROOT/'builds/kernel_dump'],text=True).splitlines())
    cases = [(1,1,1,1,0),(7,3,2,4,5),(2,10,5,8,3)]
    worst = 0.0
    for dtype in (np.float64,np.float32):
        for n,d,h,batch,steps in cases:
            rng=np.random.RandomState(42)
            model=RBM(n_components=h,batch_size=batch,random_state=rng)
            X=(np.arange(n*d).reshape(n,d)%7).astype(dtype)/dtype(6)
            model.components_=rng.normal(0,.01,(h,d)).astype(dtype,order='F')
            model.intercept_hidden_=np.zeros(h,dtype=dtype)
            model.intercept_visible_=np.zeros(d,dtype=dtype)
            model.h_samples_=np.zeros((batch,h),dtype=dtype)
            model.n_features_in_=d
            for _ in range(steps):
                model._fit(X,rng)
            expected=[('weights',model.components_),('hidden_bias',model.intercept_hidden_),
                      ('visible_bias',model.intercept_visible_),('particles',model.h_samples_),
                      ('probabilities',model._mean_hiddens(X)),('energy',model._free_energy(X)),
                      ('gibbs',model.gibbs(X)),('scores',model.score_samples(X))]
            for name, array in expected:
                parts=next(lines).split()
                assert parts[0]==name
                actual=np.array(parts[1:],dtype=float)
                wanted=array.ravel(order='C')
                worst=max(worst,float(np.max(np.abs(actual-wanted))))
                if name in ('particles','gibbs'):
                    assert_array_equal(actual,wanted)
                else:
                    tol=3e-5 if dtype==np.float32 else 2e-12
                    assert_allclose(actual,wanted,rtol=tol,atol=tol,err_msg=f'{dtype} {n,d,h,batch,steps} {name}')
            parts=next(lines).split()
            state=rng.get_state()
            assert parts[:1]==['rng'] and int(parts[1])==state[2]
            assert (parts[2]=='True')==bool(state[3])
            assert_allclose(float(parts[3]),state[4],rtol=2e-14,atol=2e-14)
            for word in state[1]:
                parts=next(lines).split()
                assert parts[0]=='word' and int(parts[1])==int(word)
    assert next(lines,None) is None
    report={'oracle':REVISION,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'cases':6,'maximum_absolute_error':worst,'scope':'private kernels only; no estimator lifecycle parity'}
    (ROOT/'docs/private/rbm').mkdir(parents=True, exist_ok=True)
    (ROOT/'docs/private/rbm/kernel-parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS private kernel parity',report)

if __name__=='__main__':
    compare()
