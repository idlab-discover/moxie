"""Pinned shared-owner regressions; not scaler estimator parity."""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
from moments_information_loss import update, PIN

ROOT = Path(__file__).resolve().parents[1]
output = subprocess.check_output([ROOT / 'builds/test_owner_updates'], text=True)
print(output, end='')
records = []
for line in output.splitlines():
    if not line.startswith('cancellation '):
        continue
    _, dtype, tile, count, mean, variance = line.split()
    dtype = np.float32 if dtype == 'f32' else np.float64
    x = np.arange(1, 9, dtype=dtype).reshape(8, 1)
    weights = np.array([1e16, 1, -1e16, 1, 0, 0, 0, 0], dtype=dtype)
    with np.errstate(all='ignore'):
        expected_mean, expected_variance, expected_count = update(
            x, np.zeros(1), np.zeros(1), np.ones(1), sample_weight=weights)
    got = np.array([float(count), float(mean), float(variance)])
    expected = np.array([expected_count[0], expected_mean[0], expected_variance[0]])
    matches = bool(np.allclose(got, expected, rtol=3e-11, atol=2e-10, equal_nan=True))
    records.append(dict(dtype=np.dtype(dtype).name, tile_rows=int(tile),
                        native=got.tolist(), reference=expected.tolist(), matches=matches))
    print(f'{np.dtype(dtype).name}, tile={tile}: native count={count}; '
          f'reference count={expected_count[0]}, mean={expected_mean[0]}')
assert len(records) == 6
# JSON encodes IEEE exceptional results as strings, not invalid JSON NaN literals.
def encode(value):
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value

report = dict(reference=PIN, numpy=np.__version__, input=list(range(1, 9)),
              weights=[1e16, 1, -1e16, 1, 0, 0, 0, 0],
              previous=dict(count=1, mean=0, variance=0), records=records,
              dependencies_sha256=hashlib.sha256((ROOT/'DEPENDENCIES.json').read_bytes()).hexdigest())
path = ROOT / 'docs/private/scalers/owner_update_parity.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(encode(report), indent=2, allow_nan=False) + '\n')
if any(not r['matches'] or r['native'][0] != r['reference'][0] for r in records):
    raise SystemExit('BLOCKED: statistics owner changes zero-effective signed count; see owner_update_parity.json')
print('PASS: signed-weight count, mean and variance match the pinned reference in all six cases')
