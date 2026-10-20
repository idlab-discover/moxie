"""Threshold coding versus the pinned _dict_learning.py public APIs."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'support'))
from oracle import ROOT, REVISION, load_pinned, run_native, assert_close
module=load_pinned('sklearn/decomposition/_dict_learning.py')
rng=np.random.RandomState(12)
count=0
for dtype in (np.float32,np.float64):
    for n,p,k in ((9,5,3),(4,3,7),(1,1,1)):
        x=rng.normal(size=(n,p)).astype(dtype)
        d=rng.normal(size=(k,p)).astype(dtype)
        for alpha in (None,0.,.2,5.):
            for positive in (False,True):
                for split in ((False,True) if dtype==np.float64 else (False,)):
                    # Perturbed precomputed covariance verifies it is actually consumed.
                    cov=(d@x.T+.125).astype(dtype)
                    for use_cov in (False,True):
                        expected=module.sparse_encode(x,d,algorithm='threshold',alpha=alpha,positive=positive,
                                                     cov=cov if use_cov else None)
                        m=module.SparseCoder(d,transform_algorithm='threshold',transform_alpha=alpha,
                                            positive_code=positive,split_sign=split)
                        z=m.transform(x)
                        got=run_native(ROOT/'builds/sparse_coding_driver',dtype,[x,d,cov],
                                       -1 if alpha is None else alpha,int(positive),int(split),int(use_cov))
                        tol=2e-6 if dtype==np.float32 else 2e-12
                        for key,value in dict(functional=expected,code=z,inverse=m.inverse_transform(z),
                                              fit=m.fit_transform(x)).items():
                            assert_close(got[key],value,tol,f'{dtype} {(n,p,k)} {alpha=} {positive=} {split=} {use_cov=} {key}')
                        count+=1
print(f'PASS: {count} threshold coding cases against local sklearn {REVISION}')
