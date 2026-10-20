"""Pinned sklearn bodies and Cython MST/label kernels; SciPy metric adapter."""
import ast
import hashlib
import heapq
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings
import numpy as np
from scipy.spatial.distance import cdist

FAMILY = Path(__file__).resolve().parents[3]
SOURCE = FAMILY.parents[2]/'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES = ['sklearn/cluster/'+f for f in ['_agglomerative.py','_feature_agglomeration.py','_hierarchical_fast.pyx','_hierarchical_fast.pxd']]

def native_reference():
    source=(SOURCE/FILES[2]).read_text()
    # Keep the original MST, union-find and descendant algorithms. Substitute
    # only the metric call (SciPy cdist) for sklearn's unavailable extension ABI.
    header='''import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport INFINITY
ctypedef cnp.float64_t float64_t
ctypedef cnp.intp_t intp_t
ctypedef cnp.uint8_t uint8_t
'''
    descendant=source[source.index('def _hc_get_descendent'):source.index('def hc_get_heads')]
    union=source[source.index('cdef class UnionFind'):source.index('def mst_linkage_core')]
    union=union.replace('cdef class UnionFind(object):','cdef class UnionFind(object):\n    cdef intp_t next_label\n    cdef intp_t[:] parent\n    cdef intp_t[:] size')
    mst=source[source.index('def mst_linkage_core'):]
    mst=mst.replace('DistanceMetric64 dist_metric','object dist_metric')
    old='''dist_metric.dist(&raw_data[current_node, 0],
                                          &raw_data[j, 0],
                                          num_features)'''
    assert old in mst
    mst=mst.replace(old,'dist_metric(np.asarray(raw_data[current_node]), np.asarray(raw_data[j]))')
    directory=FAMILY/'builds/agglomerative-oracle';directory.mkdir(parents=True,exist_ok=True)
    generated=header+descendant+union+mst
    path=directory/'hierarchical_reference.pyx'
    if not path.exists() or path.read_text()!=generated or not list(directory.glob('hierarchical_reference*.so')):
        path.write_text(generated)
        (directory/'setup.py').write_text('from setuptools import setup, Extension\nfrom Cython.Build import cythonize\nimport numpy as np\nsetup(ext_modules=cythonize([Extension("hierarchical_reference", ["hierarchical_reference.pyx"], include_dirs=[np.get_include()])], compiler_directives={"language_level":3}))\n')
        subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    sys.path.insert(0,str(directory))
    module=importlib.import_module('hierarchical_reference')
    module.max_merge=None;module.average_merge=None
    return module

class DistanceMetric:
    @staticmethod
    def get_metric(metric):
        return lambda x,y: cdist(x[None,:],y[None,:],metric=metric)[0,0]


def validate(model,x,reset=True,ensure_min_samples=1,ensure_min_features=1,**kw):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[0]<ensure_min_samples or x.shape[1]<ensure_min_features or not np.isfinite(x).all():
        raise ValueError('invalid X')
    if reset:model.n_features_in_=x.shape[1]
    elif x.shape[1]!=model.n_features_in_:raise ValueError('feature mismatch')
    return x


def load():
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',*FILES],text=True)
    env=dict(np=np,warnings=warnings,_hierarchical=native_reference(),DistanceMetric=DistanceMetric,
        METRIC_MAPPING64={'euclidean','cityblock','chebyshev'},validate_data=validate,
        check_memory=lambda _:SimpleNamespace(cache=lambda f:f),check_is_fitted=lambda _:None,
        issparse=lambda _:False,**{n:getattr(heapq,n) for n in ['heappush','heappushpop']})
    transform=ast.parse((SOURCE/FILES[1]).read_text())
    for node in transform.body:
        if isinstance(node,ast.ClassDef):
            node.bases=[]
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE/FILES[1]),'exec'),env)
    nodes=[]
    for node in ast.parse((SOURCE/FILES[0]).read_text()).body:
        if isinstance(node,ast.FunctionDef):
            node.decorator_list=[];nodes.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_TREE_BUILDERS' for t in node.targets):nodes.append(node)
        elif isinstance(node,ast.ClassDef):
            if node.name=='AgglomerativeClustering':node.bases=[]
            else:node.bases=[ast.Name(id='AgglomerationTransform',ctx=ast.Load()),ast.Name(id='AgglomerativeClustering',ctx=ast.Load())]
            node.body=[n for n in node.body if isinstance(n,ast.FunctionDef)]
            for method in node.body:
                method.decorator_list=[d for d in method.decorator_list if isinstance(d,ast.Name) and d.id=='property']
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE/FILES[0]),'exec'),env)
    return env,{f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES}
