"""Positive owner API regression for functional scale's historical collision.

This compares native nanmean_std_at plus the transform owner to pinned scale.
It certifies no public estimator. --require-parity is retained for gate callers.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIN = json.loads((ROOT / 'tests/REFERENCE.json').read_text())
ORACLE = (ROOT / PIN['path']).resolve()
assert subprocess.check_output(
    ['git', '-C', str(ORACLE), 'rev-parse', 'HEAD'], text=True
).strip() == PIN['revision']
for relative, digest in PIN['files'].items():
    assert hashlib.sha256((ORACLE / relative).read_bytes()).hexdigest() == digest
    assert not subprocess.check_output(
        ['git', '-C', str(ORACLE), 'diff', 'HEAD', '--', relative]
    )


def check_array(x, *, copy, **kwargs):
    # Only finite rank-2 Float32/64 ndarrays enter this diagnostic. The original
    # scale arithmetic and zero-scale policy below run without modifications.
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert x.dtype in (np.dtype('float32'), np.dtype('float64'))
    assert np.isfinite(x).all()
    return x.copy() if copy else x


source = ORACLE / 'sklearn/preprocessing/_data.py'
names = {'scale', '_handle_zeros_in_scale'}
nodes = [n for n in ast.parse(source.read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in nodes} == names
for node in nodes:
    node.decorator_list = []  # Parameter validation is infrastructure, not math.
namespace = dict(np=np, warnings=warnings, check_array=check_array,
                 FLOAT_DTYPES=(np.float64, np.float32),
                 sparse=SimpleNamespace(issparse=lambda x: False),
                 get_namespace=lambda x: (np, False))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
for name in names:
    assert namespace[name].__code__.co_filename == str(source)

output = subprocess.check_output([ROOT / 'builds/test_functional_statistics'], text=True)
records = []
for line in output.splitlines():
    tag, dtype, permutation, tile, *values = line.split()
    assert tag == 'functional_statistics'
    permutation, tile = int(permutation), int(tile)
    dtype = np.float32 if dtype == 'f32' else np.float64
    x = np.array([100000000, 100000008, 100000016], dtype=dtype)
    if permutation:
        x = x[[0, 2, 1]]
    x = x.reshape(3, 1)
    expected = namespace['scale'](x, with_mean=False)
    values = np.array(values, dtype=np.float64)
    state, divisor, got = values[:4], values[4], values[5:]
    matches = bool(np.allclose(got, expected.ravel(),
                              rtol=3e-6 if dtype == np.float32 else 3e-12, atol=0))
    records.append(dict(dtype=np.dtype(dtype).name, permutation=permutation,
                        tile_rows=tile, state=state.tolist(),
                        owner_divisor=float(divisor),
                        reference_divisor=float(np.nanstd(x, axis=0)[0]),
                        native=got.tolist(), reference=expected.ravel().tolist(),
                        matches=matches))
    print(f'{np.dtype(dtype).name} permutation={permutation} tile={tile}: '
          f'owner divisor={divisor}; reference divisor={np.nanstd(x, axis=0)[0]}; '
          f'matches={matches}')

assert len(records) == 12
for tile in (1, 3, 256):
    a, b = [r for r in records if r['dtype'] == 'float32' and r['tile_rows'] == tile]
    assert a['state'] == b['state'], 'Expected identical owner summaries'
    assert a['reference_divisor'] != b['reference_divisor']
    assert a['owner_divisor'] != b['owner_divisor']
assert all(r['matches'] for r in records)
assert all(r['owner_divisor'] == r['reference_divisor'] for r in records)

report = dict(reference=PIN, numpy=np.__version__, records=records,
              dependencies_sha256=hashlib.sha256(
                  (ROOT / 'DEPENDENCIES.json').read_bytes()).hexdigest(),
              result='Input-precision owner API resolves the historical collision')
destination = ROOT / 'docs/private/scalers/functional_statistics.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print('PASS: input-precision statistics and composed scale outputs match all 12 cases')
