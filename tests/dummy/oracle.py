"""Pinned source is the oracle; installed sklearn supplies supporting infrastructure."""
import sys
sys.dont_write_bytecode = True
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sklearn
import sklearn.utils.multiclass
import sklearn.utils.validation
import sklearn.metrics
import sklearn.base

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / '../scikit-learn').resolve()
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
FILES = ['sklearn/dummy.py', 'sklearn/utils/stats.py', 'sklearn/utils/multiclass.py',
         'sklearn/tests/test_dummy.py', 'sklearn/base.py', 'sklearn/utils/validation.py', 'sklearn/metrics/_classification.py',
         'sklearn/metrics/_regression.py', 'doc/modules/model_evaluation.rst']

def load():
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--', *FILES], text=True)
    def module(name, relative):
        spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    stats = module('sklearn.utils._dummy_pinned_stats', 'sklearn/utils/stats.py')
    dummy = module('sklearn._dummy_pinned_oracle', 'sklearn/dummy.py')
    dummy._weighted_percentile = stats._weighted_percentile
    # Pinned inherited scores and metric bodies, with support-wheel infrastructure.
    classification = module('sklearn.metrics._dummy_pinned_classification', 'sklearn/metrics/_classification.py')
    regression = module('sklearn.metrics._dummy_pinned_regression', 'sklearn/metrics/_regression.py')
    sklearn.metrics.accuracy_score = classification.accuracy_score
    sklearn.metrics.r2_score = regression.r2_score
    path = SOURCE / 'sklearn/base.py'
    tree = ast.parse(path.read_text())
    for name in ['ClassifierMixin', 'RegressorMixin']:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        score = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'score')
        ns = vars(sklearn.base).copy()
        exec(compile(ast.Module(body=[score], type_ignores=[]), str(path), 'exec'), ns)
        setattr(getattr(sklearn.base, name), 'score', ns['score'])
    path = SOURCE / 'sklearn/utils/validation.py'
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['_check_sample_weight', 'check_random_state']]
    ns = vars(sklearn.utils.validation).copy()
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), ns)
    dummy._check_sample_weight = ns['_check_sample_weight']
    dummy.check_random_state = ns['check_random_state']
    # Execute exact dense class_distribution body with the wheel's module globals.
    path = SOURCE / 'sklearn/utils/multiclass.py'
    body = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'class_distribution')
    namespace = vars(sklearn.utils.multiclass).copy()
    exec(compile(ast.Module(body=[body], type_ignores=[]), str(path), 'exec'), namespace)
    dummy.class_distribution = namespace['class_distribution']
    assert dummy.DummyClassifier.predict.__code__.co_filename == str(SOURCE / 'sklearn/dummy.py')
    assert dummy.DummyRegressor.fit.__wrapped__.__code__.co_filename == str(SOURCE / 'sklearn/dummy.py')
    evidence = dict(revision=REVISION, support_sklearn=sklearn.__version__, files={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in FILES})
    (ROOT/'docs/private/dummy.estimators/oracle.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return dummy
