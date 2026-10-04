"""Python side of tests/support/parity_io.mojo (validation tooling only)."""
import struct
import subprocess
import numpy as np


def bits(value):
    return str(struct.unpack('<q', struct.pack('<d', float(value)))[0])


def matrix(a):
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    return f'{a.shape[0]} {a.shape[1]} ' + ' '.join(bits(v) for v in a.ravel())


def real(value):
    return bits(value)


def run(driver, path, text):
    path.write_text(text)
    proc = subprocess.run([str(driver), str(path)], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f'{driver} failed on {path}: {proc.stderr}')
    results = {}
    for line in proc.stdout.splitlines():
        key, count, *rest = line.split()
        if count == 'error':
            results[key] = RuntimeError(' '.join(rest).replace('_', ' '))
        else:
            results[key] = np.array([struct.unpack('<d', struct.pack('<q', int(v)))[0] for v in rest])
            assert len(results[key]) == int(count)
    return results
