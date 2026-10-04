"""Execute the pinned local scikit-learn covariance sources for parity.

The support wheel (scikit-learn from the child's Pixi environment) supplies
validation, parameter checking and compiled infrastructure. The covariance
modules under comparison, and `fast_logdet`, are compiled from the pinned
checkout with their original filenames, so every reference estimator method
runs the pinned code. Python is used for validation only.

Environment: MOXIE_WORKSPACE (default: parent of the Moxie repository).
"""
from pathlib import Path
import ast
import os
import subprocess
import sys
import types

FAMILY = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get('MOXIE_WORKSPACE', FAMILY.parents[2])).resolve()
REFERENCE = WORKSPACE / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
MODULES = ['_empirical_covariance', '_shrunk_covariance', '_robust_covariance', '_elliptic_envelope', '_graph_lasso']


def _verify():
    head = subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == REVISION, f'scikit-learn reference at {head}, expected {REVISION}'
    paths = ['sklearn/covariance', 'sklearn/utils/extmath.py', 'sklearn/base.py', 'sklearn/linear_model/_least_angle.py']
    dirty = subprocess.check_output(['git', '-C', str(REFERENCE), 'status', '--porcelain', '--', *paths], text=True)
    assert not dirty.strip(), f'pinned reference sources modified: {dirty}'


def _pinned_function(path, name):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


INJECTED = []


def _supply_missing_imports(path):
    """Add pinned definitions the support wheel lacks (e.g. newer private helpers).

    Each missing name is compiled from the pinned source of its own module,
    inside the support module's namespace; the list is kept in INJECTED.
    """
    import importlib
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not (node.module or '').startswith('sklearn.'):
            continue
        if node.module.startswith('sklearn.covariance'):
            continue
        support = importlib.import_module(node.module)
        missing = [a.name for a in node.names if not hasattr(support, a.name)]
        if not missing:
            continue
        source = REFERENCE / (node.module.replace('.', '/') + '.py')
        definitions = [n for n in ast.parse(source.read_text(), filename=str(source)).body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in missing]
        assert {n.name for n in definitions} == set(missing), (node.module, missing)
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), support.__dict__)
        INJECTED.extend(f'{node.module}.{name}' for name in missing)


def load(modules=MODULES):
    """Return {module name: pinned module}; installs them into sklearn.covariance."""
    _verify()
    import sklearn.covariance as package
    import sklearn.utils._array_api as array_api
    if '_graph_lasso' in modules:
        import sklearn.linear_model as linear_model
        path = REFERENCE / 'sklearn/linear_model/_least_angle.py'
        angle = types.ModuleType('sklearn.linear_model._least_angle')
        angle.__file__ = str(path)
        angle.__package__ = 'sklearn.linear_model'
        _supply_missing_imports(path)
        exec(compile(path.read_text(), str(path), 'exec'), angle.__dict__)
        linear_model.lars_path_gram = angle.lars_path_gram
        assert executed_from_pinned(angle._lars_path_solver)
    fast_logdet = _pinned_function(REFERENCE / 'sklearn/utils/extmath.py', 'fast_logdet')
    fast_logdet.__globals__['get_namespace'] = array_api.get_namespace
    loaded = {}
    for name in modules:
        path = REFERENCE / 'sklearn/covariance' / f'{name}.py'
        module = types.ModuleType(f'sklearn.covariance.{name}')
        module.__file__ = str(path)
        module.__package__ = 'sklearn.covariance'
        sys.modules[module.__name__] = module
        _supply_missing_imports(path)
        exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
        if 'fast_logdet' in module.__dict__:
            module.fast_logdet = fast_logdet
        for attr, value in list(module.__dict__.items()):
            if not attr.startswith('__') and getattr(value, '__module__', None) == module.__name__:
                setattr(package, attr, value)
        loaded[name] = module
    return loaded


def executed_from_pinned(function):
    """True when `function` was compiled from the pinned checkout."""
    code = getattr(function, '__wrapped__', function).__code__
    return Path(code.co_filename).resolve().is_relative_to(REFERENCE.resolve())
