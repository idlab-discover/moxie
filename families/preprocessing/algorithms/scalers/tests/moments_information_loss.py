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
    assert result.dtype == np.float64 and result.ndim == 1
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

# Verify StandardScaler uses this validator without enabling nonnegative checks.
tree = ast.parse((SKLEARN / 'sklearn/preprocessing/_data.py').read_text())
standard = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'StandardScaler')
partial = next(node for node in standard.body if isinstance(node, ast.FunctionDef) and node.name == 'partial_fit')
checks = [node for node in ast.walk(partial) if isinstance(node, ast.Call)
          and isinstance(node.func, ast.Name) and node.func.id == '_check_sample_weight']
assert checks and all('ensure_non_negative' not in {kw.arg for kw in call.keywords} for call in checks)

update = namespace['_incremental_mean_and_var']
zero, one = np.zeros(1), np.ones(1)
records = []
with np.errstate(all='ignore'):
    for values in ([1., 2.], [2., 4.]):
        x = np.array(values).reshape(2, 1)
        weights = namespace['_check_sample_weight'](np.array([1., -1.]), x, dtype=x.dtype)
        np.testing.assert_array_equal(weights, [1., -1.])
        mean, variance, count = update(x, zero, zero, zero, sample_weight=weights)
        next_mean, next_variance, next_count = update(x, zero, zero, one, sample_weight=weights)
        records.append({
            'input': values, 'weights': weights.tolist(),
            'count': count, 'mean': mean, 'variance': variance,
            'updated_mean': next_mean, 'updated_variance': next_variance,
            'updated_count': next_count,
        })

for field in ('count', 'mean', 'variance'):
    np.testing.assert_array_equal(records[0][field], records[1][field])
np.testing.assert_array_equal(records[0]['count'], [0.])
np.testing.assert_array_equal(records[0]['mean'], [-np.inf])
assert np.isnan(records[0]['variance'][0])
np.testing.assert_array_equal(records[0]['updated_mean'], [-1.])
np.testing.assert_array_equal(records[1]['updated_mean'], [-2.])
for record in records:
    print('batch:', record['input'], 'FeatureMoments:',
          {key: record[key].tolist() for key in ('count', 'mean', 'variance')},
          'required updated mean:', record['updated_mean'].tolist())

# JSON uses strings for nonfinite IEEE values, preserving strict JSON syntax.
def serial(value):
    if isinstance(value, np.ndarray):
        return [float(v) if np.isfinite(v) else str(float(v)) for v in value]
    return value

report = {
    'reference': PIN, 'numpy': np.__version__,
    'records': [{key: serial(value) for key, value in record.items()} for record in records],
    'result': 'identical merge inputs require distinct outputs; contract insufficient',
    'scope': 'Float64 signed sample weights, one feature, exact arithmetic expectations',
}
destination = ROOT / 'docs/private/scalers/moments_information_loss.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print('REPRODUCED: shared FeatureMoments loses a nonzero weighted sum when count is zero.')
if '--require-sufficient-contract' in sys.argv:
    raise SystemExit('BLOCKED: FeatureStatistics.merge cannot distinguish these batches; shared-owner change required.')
