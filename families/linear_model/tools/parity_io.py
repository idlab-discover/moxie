"""Parity tooling shared by linear_model children (validation only, never runtime).

- `load_pinned(path)` executes an estimator module from the pinned local
  scikit-learn checkout (revision and cleanliness verified). The installed
  support wheel only provides the utilities that module imports; the classes and
  functions under comparison are compiled from the pinned file, which callers
  assert through `pinned_code(obj)`.
- `case_line`/`parse_results` implement the text exchange read by
  tests/support/parity_io.mojo.
"""
from pathlib import Path
import builtins
import hashlib
import importlib
import subprocess
import sys
import types

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
REFERENCE = FAMILY.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'


def verify_reference(paths):
    head = subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == REVISION, f'scikit-learn checkout at {head}, expected {REVISION}'
    dirty = subprocess.check_output(['git', '-C', str(REFERENCE), 'status', '--porcelain', '--', *paths], text=True)
    assert not dirty.strip(), f'pinned sources modified: {dirty}'
    return {p: hashlib.sha256((REFERENCE / p).read_bytes()).hexdigest() for p in paths}


# Names the pinned (1.10.dev0) sources import that the 1.9.1 support wheel only
# provides under an older name. Each alias is device plumbing, not numerics.
COMPAT_ALIASES = {'sklearn.utils._array_api': {'array_device': 'device'}}


def _install_compat_aliases():
    for module_name, aliases in COMPAT_ALIASES.items():
        module = importlib.import_module(module_name)
        for new, old in aliases.items():
            if not hasattr(module, new):
                setattr(module, new, getattr(module, old))


def load_pinned(relative, name=None, register=False):
    """Execute sklearn/<...>.py from the pinned checkout as a fresh module.

    With `register=True` the module replaces its real dotted name in
    sys.modules, so later pinned modules importing it bind the pinned code.
    """
    path = REFERENCE / relative
    parts = relative.removesuffix('.py').split('/')
    package = '.'.join(parts[:-1])
    _install_compat_aliases()
    if register:
        importlib.import_module(package)  # import the support package first (avoids cycles)
    module = types.ModuleType('.'.join(parts) if register else (name or 'pinned_' + '_'.join(parts)))
    module.__file__ = str(path)
    module.__package__ = package
    sys.modules[module.__name__] = module
    exec(builtins.compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


def pinned_code(function, relative):
    """Assert that a function (or a decorated wrapper) was compiled from the pinned file."""
    target = function
    while hasattr(target, '__wrapped__'):
        target = target.__wrapped__
    assert target.__code__.co_filename == str(REFERENCE / relative), (target, target.__code__.co_filename)


def encode(name, array):
    a = np.asarray(array, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1)
    rows = a.shape[0]
    cols = a.shape[1] if a.ndim == 2 else 0
    return f'{name}:{rows}:{cols}:' + ','.join(repr(float(v)) for v in a.ravel())


def case_line(ident, bits, params, arrays):
    text = ';'.join(f'{k}={v}' for k, v in params.items())
    return '|'.join(['case', ident, str(bits), text] + [encode(k, v) for k, v in arrays.items()])


def decode(text):
    name, rows, cols, body = text.split(':', 3)
    values = np.array([float(v) for v in body.split(',')] if body else [], dtype=np.float64)
    rows, cols = int(rows), int(cols)
    return name, (values.reshape(rows, cols) if cols else values.reshape(rows))


def parse_results(output):
    results = {}
    for line in output.splitlines():
        fields = line.split('|')
        if fields[0] == 'error':
            results[fields[1]] = {'error': '|'.join(fields[2:])}
        elif fields[0] == 'result':
            record = {}
            for field in fields[2:]:
                if '=' in field and ':' not in field.split('=')[0]:
                    key, value = field.split('=', 1)
                    record[key] = float(value)
                else:
                    key, value = decode(field)
                    record[key] = value
            results[fields[1]] = record
    return results


def close(actual, expected, rtol, atol, label):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, f'{label}: shape {actual.shape} != {expected.shape}'
    if actual.size == 0:
        return 0.0
    both_nan = np.isnan(actual) & np.isnan(expected)
    if both_nan.all():
        return 0.0  # e.g. R2 of a single sample: NaN on both sides
    actual = np.where(both_nan, 0.0, actual)
    expected = np.where(both_nan, 0.0, expected)
    error = np.abs(actual - expected)
    limit = atol + rtol * np.abs(expected)
    assert np.all(np.isfinite(actual)), f'{label}: nonfinite {actual}'
    assert np.all(error <= limit), f'{label}: max error {error.max()} (limit {limit.max()})\nactual={actual}\nexpected={expected}'
    return float(error.max())
