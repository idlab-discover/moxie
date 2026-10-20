"""Unedited sklearn CF-tree and Birch bodies; dense validation adapters only."""
import ast
import hashlib
import importlib.util
from math import sqrt
from numbers import Integral
from pathlib import Path
import subprocess

import numpy as np
from scipy import sparse

FAMILY = Path(__file__).resolve().parents[3]
SOURCE = FAMILY.parents[2] / 'scikit-learn'
REV = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def load():
    affinity = module('birch_affinity_reference', FAMILY/'algorithms/affinity/tests/oracle.py')
    env = affinity.load()
    agglomerative = module('birch_agglomerative_reference', FAMILY/'algorithms/agglomerative/tests/oracle.py')
    agg_env, hashes = agglomerative.load()
    agg = agg_env['AgglomerativeClustering']
    agg.fit_predict = lambda self, x: self.fit(x).labels_
    env.update(AgglomerativeClustering=agg, Integral=Integral, sqrt=sqrt, sparse=sparse)
    env['check_is_fitted'] = lambda model: getattr(model, 'subcluster_centers_')
    env['euclidean_distances'] = lambda x, y=None, **kw: env['_euclidean_distances'](x, x if y is None else y, **kw)
    env['pairwise_distances_argmin'] = lambda x, centers, **kw: (
        (x.astype('d')[:, None, :] - centers.astype('d')[None, :, :])**2
    ).sum(axis=2).argmin(axis=1)
    path = SOURCE/'sklearn/cluster/_birch.py'
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--',str(path)],text=True)
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == '_split_node':
            nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            node.bases = []
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            for method in node.body:
                method.decorator_list = [d for d in method.decorator_list if isinstance(d, ast.Name) and d.id == 'property']
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),env)
    hashes['sklearn/cluster/_birch.py'] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes['sklearn/metrics/pairwise.py'] = hashlib.sha256((SOURCE/'sklearn/metrics/pairwise.py').read_bytes()).hexdigest()
    return env, hashes
