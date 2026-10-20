"""Execute pinned sklearn bodies, with dense-only host validation plumbing."""
from pathlib import Path
import ast, contextlib, hashlib, math, subprocess, warnings
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parents[4]/'scikit-learn'
REV='4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES=['sklearn/cluster/_affinity_propagation.py','sklearn/metrics/pairwise.py','sklearn/base.py']

def verify():
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*FILES],text=True).strip()
    return {f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES}

def extract(file,names,env):
    path=SOURCE/file
    tree=ast.parse(path.read_text())
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            node.decorator_list=[]
            nodes.append(node)
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    for name in names: assert env[name].__code__.co_filename==str(path)

class ConvergenceWarning(UserWarning): pass

def load():
    verify()
    env={'np':np,'warnings':warnings,'ConvergenceWarning':ConvergenceWarning,'math':math,
         'get_namespace_and_device':lambda *a:(np,False,None),'issparse':lambda x:False,
         'safe_sparse_dot':lambda a,b,**kw: a@b,
         'row_norms':lambda a,squared=True: np.einsum('ij,ij->i',a,a),
         'gen_batches':lambda n,b:(slice(i,min(n,i+b)) for i in range(0,n,b)),
         '_max_precision_float_dtype':lambda **kw:np.float64,
         '_modify_in_place_if_numpy':lambda xp,f,*a,**kw:f(*a,**kw),
         '_fill_diagonal':lambda a,v,**kw:np.fill_diagonal(a,v)}
    extract(FILES[1],['_euclidean_distances','_euclidean_distances_upcast'],env)
    env['euclidean_distances']=lambda x,squared=True:env['_euclidean_distances'](x,x,squared=squared)
    env['check_random_state']=lambda s: s if isinstance(s,np.random.RandomState) else np.random.RandomState(s)
    extract(FILES[0],['_equal_similarities_and_preferences','_affinity_propagation','affinity_propagation'],env)
    tree=ast.parse((SOURCE/FILES[0]).read_text())
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AffinityPropagation')
    methods=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','fit','predict')]
    for n in methods:n.decorator_list=[]
    cls=ast.ClassDef(name='AffinityPropagation',bases=[],keywords=[],body=methods,decorator_list=[])
    ast.copy_location(cls,original)
    def validate_data(model,x,reset=True,copy=False,**kwargs):
        x=np.array(x,copy=True) if copy else np.asarray(x)
        if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all(): raise ValueError('invalid dense input')
        if reset:model.n_features_in_=x.shape[1]
        elif model.n_features_in_!=x.shape[1]: raise ValueError('feature mismatch')
        return x
    def check_is_fitted(m):
        if not hasattr(m,'labels_'):raise ValueError('unfitted')
    env.update(validate_data=validate_data,check_is_fitted=check_is_fitted,
               config_context=lambda **kw:contextlib.nullcontext(),
               pairwise_distances_argmin=lambda x,c:((x.astype('d')[:,None,:]-c.astype('d')[None,:,:])**2).sum(axis=2).argmin(axis=1))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(SOURCE/FILES[0]),'exec'),env)
    return env
