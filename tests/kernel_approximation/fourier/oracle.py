"""Execute original local sampler methods; NumPy supplies validation plumbing only."""
import ast, subprocess, types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
REF=ROOT.parent/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
def load():
    assert subprocess.check_output(['git','-C',str(REF),'rev-parse','HEAD'],text=True).strip()==REV
    paths=['sklearn/kernel_approximation.py','sklearn/base.py','sklearn/utils/validation.py']
    assert not subprocess.check_output(['git','-C',str(REF),'status','--porcelain','--',*paths],text=True).strip()
    source=REF/paths[0]
    def validate_data(est,X,reset=True,**kwargs):
        X=np.array(X,copy=True)
        if X.ndim!=2 or 0 in X.shape or not np.isfinite(X).all():raise ValueError('invalid matrix')
        if reset:est.n_features_in_=X.shape[1]
        elif X.shape[1]!=est.n_features_in_:raise ValueError('features')
        return X
    def check_random_state(seed):
        return seed if isinstance(seed,np.random.RandomState) else np.random.RandomState(seed)
    namespace={'np':np,'sp':types.SimpleNamespace(issparse=lambda x:False),
      'validate_data':validate_data,'check_random_state':check_random_state,
      'check_is_fitted':lambda est:None,'safe_sparse_dot':np.dot}
    nodes=[]
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.ClassDef) and node.name in ['RBFSampler','SkewedChi2Sampler']:
            node.bases=[];node.decorator_list=[]
            node.body=[x for x in node.body if isinstance(x,ast.FunctionDef) and x.name in ['__init__','fit','transform']]
            for method in node.body:method.decorator_list=[]
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),namespace)
    for name in ['RBFSampler','SkewedChi2Sampler']:
        assert namespace[name].fit.__code__.co_filename==str(source)
    return namespace
