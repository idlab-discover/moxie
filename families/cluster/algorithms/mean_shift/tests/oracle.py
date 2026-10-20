"""Execute pinned sklearn MeanShift code with independent SciPy query primitives."""
import ast
from collections import defaultdict
from contextlib import nullcontext
import hashlib
from pathlib import Path
import subprocess
import warnings
import numpy as np
from scipy.spatial.distance import cdist

FAMILY = Path(__file__).resolve().parents[3]
SOURCE = FAMILY.parents[2] / 'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILE = 'sklearn/cluster/_mean_shift.py'

class Neighbors:
    def __init__(self, n_neighbors=5, radius=1., **kwargs):
        self.n_neighbors, self.radius = n_neighbors, radius
    def get_params(self): return dict(radius=self.radius)
    def fit(self, x): self.x = x; return self
    def radius_neighbors(self, x, radius=None, return_distance=False):
        return [np.flatnonzero(row <= (self.radius if radius is None else radius)) for row in cdist(x, self.x)]
    def kneighbors(self, x, return_distance=True):
        distances = cdist(x, self.x).astype(self.x.dtype)
        indices = np.argsort(distances, axis=1, kind='stable')[:, :self.n_neighbors]
        return np.take_along_axis(distances, indices, axis=1), indices

class Parallel:
    def __init__(self, **kwargs): pass
    def __call__(self, tasks): return [task() for task in tasks]

def load():
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==REV
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',FILE],text=True)
    tree=ast.parse((SOURCE/FILE).read_text())
    nodes=[]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list=[]; nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name=='MeanShift':
            node.bases=[]
            node.body=[n for n in node.body if isinstance(n,ast.FunctionDef)]
            for method in node.body: method.decorator_list=[]
            nodes.append(node)
    env=dict(np=np,defaultdict=defaultdict,warnings=warnings,NearestNeighbors=Neighbors,
             Parallel=Parallel,delayed=lambda f: lambda *a,**k: lambda: f(*a,**k),
             check_array=np.asarray,validate_data=lambda model,x,**kw: np.asarray(x),
             check_random_state=lambda seed: np.random.RandomState(seed),
             gen_batches=lambda n,b: (slice(i,min(i+b,n)) for i in range(0,n,b)),
             check_is_fitted=lambda model: None,config_context=lambda **kw: nullcontext(),
             pairwise_distances_argmin=lambda x,c: cdist(x,c).argmin(axis=1))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE/FILE),'exec'),env)
    return env, {FILE:hashlib.sha256((SOURCE/FILE).read_bytes()).hexdigest()}
