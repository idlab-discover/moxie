"""Reproduce a shared-contract limitation with the pinned sklearn source.

Only NumPy namespace discovery and array input validation are supplied locally.
The weight validator, accumulator, and incremental arithmetic execute their
actual AST bodies from the pinned checkout; no sklearn wheel is substituted.
This is a blocker diagnostic, not estimator parity.
"""
import ast
from contextlib import nullcontext
import hashlib
import inspect
import json
import numbers
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIN = json.loads((ROOT / 'tests/REFERENCE.json').read_text())
SKLEARN = (ROOT / PIN['path']).resolve()
assert subprocess.check_output(['git', '-C', str(SKLEARN), 'rev-parse', 'HEAD'], text=True).strip() == PIN['revision']
for name, digest in PIN['files'].items():
    assert hashlib.sha256((SKLEARN / name).read_bytes()).hexdigest() == digest, name
    assert not subprocess.check_output(['git', '-C', str(SKLEARN), 'diff', 'HEAD', '--', name]), name

def checked_array(value, *, dtype, **kwargs):
    # Infrastructure for this finite Float64 vector case only.
    result = np.array(value, dtype=dtype, copy=True)
    assert result.dtype in (np.float32, np.float64) and result.ndim == 1
    assert np.isfinite(result).all()
    return result

namespace = {
    'np': np, 'numbers': numbers, 'inspect': inspect, 'nullcontext': nullcontext,
    'get_namespace_and_device': lambda *args, **kwargs: (np, False, None),
    '_max_precision_float_dtype': lambda *args, **kwargs: np.float64,
    '_is_numpy_namespace': lambda xp: xp is np,
    'xpx': SimpleNamespace(nansum=np.nansum),
    '_num_samples': len, 'check_array': checked_array,
}

def load(relative, names):
    source = SKLEARN / relative
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    for name in names:
        assert namespace[name].__code__.co_filename == str(source)

load('sklearn/utils/validation.py', ['_check_sample_weight'])
load('sklearn/utils/extmath.py', ['_safe_accumulator_op', '_incremental_mean_and_var'])


load('sklearn/preprocessing/_data.py', ['_is_constant_feature'])
update = namespace['_incremental_mean_and_var']
constant = namespace['_is_constant_feature']
