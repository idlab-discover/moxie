"""Reproduce a statistics-owner overflow mismatch, not estimator acceptance.

Default mode verifies the known failure plus finite controls. --require-parity
exits nonzero while the owner disagrees with the pinned sklearn update body.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from moments_information_loss import update, PIN

ROOT = Path(__file__).resolve().parents[1]
output = subprocess.check_output([ROOT / 'builds/test_unweighted_reduction'], text=True)
records = []
for line in output.splitlines():
    tag, magnitude, tile, track, *values = line.split()
    assert tag == 'unweighted'
    magnitude, tile, track = float(magnitude), int(tile), bool(int(track))
    x = np.array([magnitude, 0, magnitude, -magnitude, -magnitude, 0, 0, 0]).reshape(8, 1)
    with np.errstate(all='ignore'):
        mean, variance, count = update(
            x, np.zeros(1), np.zeros(1) if track else None, np.zeros(1))
        expected = np.array([count[0], mean[0], variance[0] if track else 0,
                             np.sum(x, axis=0)[0], (x - mean)[0, 0], (x - mean)[1, 0]])
    got = np.array(values, dtype=np.float64)
    matches = bool(np.allclose(got, expected, rtol=3e-11, atol=2e-10, equal_nan=True))
    records.append(dict(magnitude=magnitude, tile_rows=tile, track_variance=track,
                        native=got.tolist(), reference=expected.tolist(), matches=matches))
    print(f'magnitude={magnitude:g}, tile={tile}, variance={track}: '
          f'native mean={got[1]}, reference mean={mean[0]}, matches={matches}')

assert len(records) == 12
# Pairwise fast-axis sums reproduce the reference zero mean without overflow.
assert all(r['matches'] for r in records)
assert all(r['native'][1] == 0 and r['reference'][1] == 0
           for r in records if r['magnitude'] == 1e308)


def encode(value):
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


report = dict(reference=PIN, numpy=np.__version__, records=records,
              fields=['count', 'mean', 'variance_or_zero', 'new_sum', 'centered_0', 'centered_1'],
              dependencies_sha256=hashlib.sha256(
                  (ROOT / 'DEPENDENCIES.json').read_bytes()).hexdigest(),
              result='Pairwise unweighted owner sums match the reference, including the overflow case')
destination = ROOT / 'docs/private/scalers/unweighted_reduction.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(encode(report), indent=2, allow_nan=False) + '\n')
print('PASS unweighted reductions: pairwise owner sums match the pinned reference at 1e100 and 1e308')
