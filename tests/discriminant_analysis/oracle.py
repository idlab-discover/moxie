"""Pinned scikit-learn source is the oracle; the installed wheel supplies infrastructure.

Executes the unmodified pinned `sklearn/discriminant_analysis.py`, the covariance
helpers it calls (`empirical_covariance`, `shrunk_covariance`, `ledoit_wolf`),
`StandardScaler`, the inherited `LinearClassifierMixin` decision/predict bodies,
`ClassifierMixin.score` and `softmax`. The wheel (1.9.1) lacks four newer private
array-API helpers; those are taken from the pinned `_array_api.py` as well.
"""
import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import sklearn.base
import sklearn.linear_model._base
import sklearn.utils._array_api as wheel_array_api
import sklearn.utils.extmath

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / '../scikit-learn').resolve()
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES = ['sklearn/discriminant_analysis.py', 'sklearn/covariance/_empirical_covariance.py',
         'sklearn/covariance/_shrunk_covariance.py', 'sklearn/preprocessing/_data.py',
         'sklearn/utils/_array_api.py', 'sklearn/utils/extmath.py', 'sklearn/linear_model/_base.py',
         'sklearn/base.py', 'sklearn/tests/test_discriminant_analysis.py']


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _function(relative, names, namespace, cls=None):
    """Compile named top-level (or class-level) functions from a pinned file into namespace."""
    path = SOURCE / relative
    body = ast.parse(path.read_text()).body
    if cls is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), (relative, names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return [namespace[n] for n in names]


def load():
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--', *FILES], text=True)
    pinned_api = _module('sklearn.utils._da_pinned_array_api', 'sklearn/utils/_array_api.py')
    for name in ['_cov', '_swapaxes', '_unravel_index', 'array_device']:
        if not hasattr(wheel_array_api, name):
            setattr(wheel_array_api, name, getattr(pinned_api, name))
    empirical = _module('sklearn.covariance._da_pinned_empirical', 'sklearn/covariance/_empirical_covariance.py')
    shrunk = _module('sklearn.covariance._da_pinned_shrunk', 'sklearn/covariance/_shrunk_covariance.py')
    shrunk.empirical_covariance = empirical.empirical_covariance
    data = _module('sklearn.preprocessing._da_pinned_data', 'sklearn/preprocessing/_data.py')
    da = _module('sklearn._da_pinned_discriminant_analysis', 'sklearn/discriminant_analysis.py')
    da.empirical_covariance = empirical.empirical_covariance
    da.shrunk_covariance = shrunk.shrunk_covariance
    da.ledoit_wolf = shrunk.ledoit_wolf
    da.StandardScaler = data.StandardScaler
    (softmax,) = _function('sklearn/utils/extmath.py', ['softmax'], vars(sklearn.utils.extmath).copy())
    da.softmax = softmax
    # Inherited LinearClassifierMixin bodies and ClassifierMixin.score from the pinned sources.
    lin_ns = vars(sklearn.linear_model._base).copy()
    decision, predict = _function('sklearn/linear_model/_base.py', ['decision_function', 'predict'], lin_ns,
                                  cls='LinearClassifierMixin')
    sklearn.linear_model._base.LinearClassifierMixin.decision_function = decision
    sklearn.linear_model._base.LinearClassifierMixin.predict = predict
    (score,) = _function('sklearn/base.py', ['score'], vars(sklearn.base).copy(), cls='ClassifierMixin')
    sklearn.base.ClassifierMixin.score = score
    for cls in [da.LinearDiscriminantAnalysis, da.QuadraticDiscriminantAnalysis]:
        assert cls.fit.__wrapped__.__code__.co_filename == str(SOURCE / FILES[0])
    return dict(da=da, empirical=empirical, shrunk=shrunk, data=data,
                hashes={f: hashlib.sha256((SOURCE / f).read_bytes()).hexdigest() for f in FILES})
