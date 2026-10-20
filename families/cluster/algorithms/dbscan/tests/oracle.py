"""Original pinned sklearn fit and compiled traversal; SciPy radius oracle."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import warnings
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

FAMILY = Path(__file__).resolve().parents[3]
SOURCE = FAMILY.parents[2] / 'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES = ['sklearn/cluster/_dbscan.py', 'sklearn/cluster/_dbscan_inner.pyx', 'sklearn/utils/_typedefs.pxd']


def load():
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == REV
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--', *FILES], text=True)
    target = FAMILY / 'builds/dbscan-oracle'
    for package in ['sklearn', 'sklearn/cluster', 'sklearn/utils']:
        (target/package).mkdir(parents=True, exist_ok=True)
        (target/package/'__init__.py').write_text('')
    for file in FILES:
        dest = target/file
        if not dest.exists() or dest.read_bytes() != (SOURCE/file).read_bytes():
            shutil.copyfile(SOURCE/file, dest)
    (target/'setup.py').write_text("from setuptools import setup, Extension\nfrom Cython.Build import cythonize\nsetup(ext_modules=cythonize([Extension('sklearn.cluster._dbscan_inner',['sklearn/cluster/_dbscan_inner.pyx'],language='c++')],compiler_directives={'language_level':3}))\n")
    with (target/'build.log').open('w') as log:
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=target, stdout=log, stderr=log, check=True)
    sys.path.insert(0, str(target))
    inner = importlib.import_module('sklearn.cluster._dbscan_inner')
    assert Path(inner.__file__).is_relative_to(target)

    class RadiusOracle:
        """Independent distance membership, not a substituted clustering algorithm."""
        def __init__(self, radius, metric, p, **kwargs):
            self.radius, self.metric, self.p = radius, metric, p
        def fit(self, x):
            self.x = x
        def radius_neighbors(self, x, return_distance=False):
            if self.metric == 'precomputed':
                distances = x
            else:
                kwargs = {'p': self.p or 2} if self.metric == 'minkowski' else {}
                metric = 'cityblock' if self.metric == 'manhattan' else self.metric
                distances = cdist(x, self.x, metric=metric, **kwargs)
            result = np.empty(len(x), dtype=object)
            for i, row in enumerate(distances): result[i] = np.flatnonzero(row <= self.radius)
            return result

    env = dict(np=np, sparse=sparse, warnings=warnings, NearestNeighbors=RadiusOracle,
               validate_data=lambda model, x, **kwargs: np.asarray(x),
               _check_sample_weight=lambda w, x: np.asarray(w, dtype=x.dtype),
               dbscan_inner=inner.dbscan_inner)
    tree = ast.parse((SOURCE/FILES[0]).read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DBSCAN')
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in ['__init__', 'fit', 'fit_predict']]
    for method in methods: method.decorator_list = []
    cls = ast.ClassDef(name='DBSCAN', bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE/FILES[0]), 'exec'), env)
    return env['DBSCAN'], {f: hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES}
