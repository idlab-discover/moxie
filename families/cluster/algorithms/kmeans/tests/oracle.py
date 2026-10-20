"""Unedited pinned sklearn Python bodies and its compiled native kernels."""
import ast
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy import sparse
ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT.parents[1]
SOURCE = FAMILY.parents[2] / 'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load():
    engine = load_module('centroid_oracle', FAMILY/'shared/centroid_engine/tests/oracle.py')
    common, lloyd, elkan, _ = engine.load()
    affinity = load_module('affinity_oracle', FAMILY/'algorithms/affinity/tests/oracle.py')
    env = affinity.load()
    env.update(sp=sparse, _is_arraylike_not_scalar=lambda x: not isinstance(x, str) and not callable(x),
               _openmp_effective_n_threads=lambda: 1,
               _is_same_clustering=common._is_same_clustering,
               lloyd_iter_chunked_dense=lloyd.lloyd_iter_chunked_dense,
               elkan_iter_chunked_dense=elkan.elkan_iter_chunked_dense,
               init_bounds_dense=elkan.init_bounds_dense,
               _inertia_dense=common._inertia_dense,
               _check_sample_weight=lambda w,x,dtype=None: np.ones(x.shape[0],dtype=x.dtype) if w is None else np.asarray(w,dtype=x.dtype),
               check_array=lambda x,dtype=None,copy=False,**kw: np.array(x,dtype=None if isinstance(dtype,list) else dtype,copy=True) if copy else np.asarray(x,dtype=None if isinstance(dtype,list) else dtype))
    env['euclidean_distances'] = lambda x,y=None,squared=False: env['_euclidean_distances'](x,x if y is None else y,squared=squared)
    env['_labels_inertia_threadpool_limit'] = lambda x,w,c,n_threads=1,return_inertia=True: env['_labels_inertia'](x,w,c,n_threads,return_inertia)
    file = 'sklearn/cluster/_kmeans.py'
    import subprocess
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',file],text=True)
    affinity.extract(file, ['_kmeans_plusplus','kmeans_plusplus','_tolerance','_kmeans_single_lloyd','_kmeans_single_elkan','_labels_inertia','k_means'],env)
    tree = ast.parse((SOURCE/file).read_text())
    classes = []
    for name, methods in [('_BaseKMeans', ['__init__','_check_params_vs_input','_validate_center_shape','_check_test_data','_init_centroids','predict','transform','_transform','score','fit_predict','fit_transform']),('KMeans',['__init__','_check_params_vs_input','fit'])]:
        original = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
        nodes = [n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in methods]
        for n in nodes: n.decorator_list=[]
        cls = ast.ClassDef(name=name,bases=[] if name=='_BaseKMeans' else [ast.Name(id='_BaseKMeans',ctx=ast.Load())],keywords=[],body=nodes,decorator_list=[])
        ast.copy_location(cls,original);classes.append(cls)
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),str(SOURCE/file),'exec'),env)
    env['_BaseKMeans']._check_mkl_vcomp = lambda *a: None
    env['source_hash'] = hashlib.sha256((SOURCE/file).read_bytes()).hexdigest()
    return env
