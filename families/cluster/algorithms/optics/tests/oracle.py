"""Pinned sklearn OPTICS bodies, with independent SciPy distance queries."""
import ast
from contextlib import nullcontext
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
import warnings
import numpy as np
from scipy.spatial.distance import cdist

FAMILY = Path(__file__).resolve().parents[3]
SOURCE = FAMILY.parents[2]/'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILE = 'sklearn/cluster/_optics.py'


def distances(x, y, metric='minkowski', **params):
    if metric == 'precomputed': return x
    metric = {'manhattan':'cityblock', 'l1':'cityblock', 'l2':'euclidean'}.get(metric,metric)
    return cdist(x,y,metric=metric,**params)


class Neighbors:
    def __init__(self, n_neighbors, metric, p, metric_params=None, **kwargs):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.params = dict(metric_params or {})
        if metric == 'minkowski': self.params.setdefault('p',p)
    def fit(self,x): self.x=x; return self
    def kneighbors(self,x,n_neighbors):
        d = distances(x,self.x,self.metric,**self.params)
        ids=np.argsort(d,axis=1,kind='stable')[:,:n_neighbors]
        return np.take_along_axis(d,ids,axis=1),ids
    def radius_neighbors(self,x,radius,return_distance=False):
        d=distances(x,self.x,self.metric,**self.params)
        return [np.flatnonzero(row<=radius) for row in d]


def load():
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',FILE],text=True)
    nodes=[]
    for node in ast.parse((SOURCE/FILE).read_text()).body:
        if isinstance(node,ast.FunctionDef):
            node.decorator_list=[];nodes.append(node)
        elif isinstance(node,ast.ClassDef) and node.name=='OPTICS':
            node.bases=[]
            node.body=[n for n in node.body if isinstance(n,ast.FunctionDef)]
            for method in node.body: method.decorator_list=[]
            nodes.append(node)
    env=dict(np=np,warnings=warnings,NearestNeighbors=Neighbors,
        pairwise_distances=lambda x,y,metric,n_jobs=None,**kw: distances(x,y,metric,**kw),
        validate_data=lambda model,x,**kw: np.asarray(x,dtype=float),
        PAIRWISE_BOOLEAN_FUNCTIONS=[],issparse=lambda x:False,
        check_memory=lambda x:SimpleNamespace(cache=lambda f:f),
        get_chunk_n_rows=lambda **kw:kw['max_n_rows'],
        gen_batches=lambda n,b:(slice(i,min(i+b,n)) for i in range(0,n,b)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE/FILE),'exec'),env)
    return env,{FILE:hashlib.sha256((SOURCE/FILE).read_bytes()).hexdigest()}
